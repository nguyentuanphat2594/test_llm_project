"""
train.py
--------
Fine-tune một model nhỏ (miễn phí) bằng LoRA, dùng dữ liệu train.jsonl
đã được build sẵn từ pipeline/build_train_dataset.py

Yêu cầu cài đặt (chạy 1 lần):
    pip install -r requirements.txt

Cách chạy:
    python -m pipeline.train

Sau khi train xong, model LoRA sẽ được lưu vào src.config.ADAPTER_DIR
(mặc định: output/output_model). Ngay sau đó, script tự chạy inference
trên tập TEST và xuất CSV (question, reference, prediction, ROUGE/BLEU)
vào src.config.PREDICTIONS_CSV -- CSV này là input cho bước LLM Judge ở
pipeline/evaluate.py (KHÔNG generate lại câu trả lời lần 2 ở đó).
"""
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from transformers import EarlyStoppingCallback, ProgressCallback
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from src.config import (
    ADAPTER_DIR,
    BASE_MODEL_NAME,
    MAX_NEW_TOKENS_TRAIN_GEN,
    PREDICTIONS_CSV,
    PROMPT_STYLE,
    SUMMARY_CSV,
    SYSTEM_PROMPT,
    TEST_FILE,
    TRAIN_FILE,
    VAL_FILE,
)
from src.evaluation import compute_perplexity, save_predictions_csv

USE_GPU = torch.cuda.is_available()


# ============================================================
# 1. LOAD MODEL
#    - Có GPU  -> dùng 4-bit quantization (nhẹ VRAM, nhanh)
#    - Không GPU -> load bình thường, train bằng CPU (chậm hơn nhưng vẫn chạy được)
# ============================================================

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if USE_GPU:
        print("Phát hiện GPU -> load model ở chế độ 4-bit")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map={"": 0},
        )
        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False
    else:
        print("Không có GPU -> load model ở CPU (float32, sẽ train chậm hơn)")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
        )

    return model, tokenizer


# ============================================================
# 2. GẮN LORA VÀO MODEL (chỉ train 1 phần nhỏ tham số, rất nhẹ)
# ============================================================

def build_lora_config():
    """
    CHỈ trả về LoraConfig, KHÔNG tự gọi get_peft_model()/gradient_checkpointing
    ở đây nữa -- truyền config này vào SFTTrainer(peft_config=...) để nó tự lo
    toàn bộ chuỗi prepare_model_for_kbit_training -> get_peft_model ->
    gradient_checkpointing_enable() -> enable_input_require_grads() theo ĐÚNG
    thứ tự nội bộ của nó. Tự làm 1 phần bên ngoài (như code cũ) rồi để Trainer
    wrap model thêm 1 lần nữa (đặc biệt khi fp16=True, accelerate.prepare bọc
    model) làm 2 bên không đồng bộ -> graph đứt, không param nào nhận gradient
    -> lỗi "No inf checks were recorded prior to update." khi optimizer.step().
    """
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


# ============================================================
# 3. LOAD DATASET (train.jsonl / val.jsonl) VÀ ÁP DỤNG CHAT TEMPLATE
# ============================================================

def load_and_format_dataset(tokenizer, path, required=True):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                f"Không tìm thấy {path}. "
                f"Hãy chạy `python -m pipeline.build_train_dataset` trước để tạo file này."
            )
        return None

    dataset = load_dataset("json", data_files=str(path), split="train")

    if len(dataset) == 0:
        return None

    def format_sample(sample):
        text = tokenizer.apply_chat_template(
            sample["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = dataset.map(format_sample)
    return dataset


# ============================================================
# 3b. LOAD TẬP TEST "THÔ" (question/answer) ĐỂ XUẤT CSV ROUGE/BLEU
#     -- khác với load_and_format_dataset() ở trên (dataset đó đã bị
#     format thành "text" theo chat template, không dùng để inference
#     câu-hỏi-riêng được nữa).
# ============================================================

def load_raw_test_for_export(path):
    """
    Đọc thẳng test.jsonl, chuẩn hoá về 2 cột "question"/"answer" để
    save_predictions_csv() dùng làm ground truth khi tính ROUGE/BLEU.
    test.jsonl có thể ở dạng {question, contexts, ground_truth} (build từ
    pipeline/build_train_dataset.py) -- map ground_truth -> answer.
    """
    if not os.path.exists(path):
        return None

    raw_samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw_samples.append({
                "question": row["question"],
                "answer": row.get("ground_truth", row.get("answer", "")),
            })

    if not raw_samples:
        return None

    return Dataset.from_list(raw_samples)


def get_mean_token_accuracy(trainer):
    """
    SFTTrainer tự log 'mean_token_accuracy' mỗi logging_steps. Lấy trung bình
    các lần log trong lúc TRAIN (bỏ qua các dict log của eval, không có key
    này) để có 1 con số đại diện cho cả quá trình train.
    """
    values = [
        log["mean_token_accuracy"]
        for log in trainer.state.log_history
        if "mean_token_accuracy" in log
    ]
    return sum(values) / len(values) if values else None


def print_and_save_summary(summary: dict):
    """In bảng tổng hợp ra console và append thêm 1 dòng vào SUMMARY_CSV
    (mỗi lần chạy train.py là 1 dòng mới, dễ so sánh giữa các lần)."""
    print("=" * 50)
    print("BẢNG TỔNG HỢP KẾT QUẢ")
    print("=" * 50)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key:<20}: {value:.4f}")
        else:
            print(f"  {key:<20}: {value}")

    df_row = pd.DataFrame([summary])
    if os.path.exists(SUMMARY_CSV):
        df_row.to_csv(SUMMARY_CSV, mode="a", header=False, index=False)
    else:
        os.makedirs(os.path.dirname(SUMMARY_CSV), exist_ok=True)
        df_row.to_csv(SUMMARY_CSV, mode="w", header=True, index=False)
    print(f"\nĐã lưu (append) dòng tổng hợp -> {SUMMARY_CSV}")


def sanity_check_lora_gradients(model, tokenizer, train_dataset):
    """
    Chạy 1 forward+backward THỬ (1 mẫu, vài giây) để xác nhận gradient có
    lan tới LoRA adapters hay không -- TRƯỚC khi train thật (25+ phút mới
    biết qua val_loss đứng yên). Nếu phát hiện lỗi, dừng ngay lập tức với
    thông báo rõ ràng, đỡ tốn thời gian train rồi mới nghi ngờ.
    """
    print("Đang chạy sanity check (gradient có lan tới LoRA không)...")
    was_training = model.training
    model.train()

    sample_text = train_dataset[0]["text"]
    inputs = tokenizer(
        sample_text, return_tensors="pt", truncation=True, max_length=512
    ).to(model.device)
    inputs["labels"] = inputs["input_ids"].clone()

    model.zero_grad()
    outputs = model(**inputs)
    loss = outputs.loss
    loss.backward()

    lora_params = [
        (n, p) for n, p in model.named_parameters()
        if "lora_" in n and p.requires_grad
    ]
    total = len(lora_params)
    with_grad = sum(
        1 for _, p in lora_params if p.grad is not None and p.grad.abs().sum().item() > 0
    )

    model.zero_grad()
    model.train(was_training)

    print(f"  Loss thử (1 mẫu)        : {loss.item():.4f}")
    print(f"  LoRA params có gradient : {with_grad}/{total}")

    if total == 0:
        raise RuntimeError(
            "SANITY CHECK THẤT BẠI: không tìm thấy param nào có 'lora_' trong "
            "tên và requires_grad=True. LoRA có thể chưa được gắn đúng "
            "(kiểm tra lại target_modules / apply_lora())."
        )

    if with_grad == 0:
        raise RuntimeError(
            "SANITY CHECK THẤT BẠI: 0/%d LoRA params nhận được gradient. "
            "Train sẽ VÔ NGHĨA (val_loss sẽ đứng yên suốt, giống lỗi đã gặp "
            "trước đó). Nguyên nhân thường gặp: gradient checkpointing chặn "
            "gradient trước khi tới LoRA -- kiểm tra lại "
            "enable_input_require_grads() / gradient_checkpointing_kwargs "
            "trong apply_lora()." % total
        )

    if with_grad < total:
        print(
            f"  CẢNH BÁO: chỉ {with_grad}/{total} LoRA params có gradient "
            f"(không phải tất cả). Có thể vẫn train được nhưng nên kiểm tra "
            f"lại target_modules nếu kết quả cuối không như mong đợi."
        )

    print("  -> OK, gradient lan tới LoRA bình thường. Bắt đầu train thật.\n")


# ============================================================
# 4. TRAIN
# ============================================================

def main():
    print("=" * 50)
    print("Loading model + tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    print("Chuẩn bị LoRA config (SFTTrainer sẽ tự gắn LoRA)...")
    lora_config = build_lora_config()

    print("Loading train dataset...")
    train_dataset = load_and_format_dataset(tokenizer, TRAIN_FILE, required=True)
    print(f"Total training samples: {len(train_dataset)}")

    print("Loading val dataset...")
    val_dataset = load_and_format_dataset(tokenizer, VAL_FILE, required=False)
    if val_dataset is not None:
        print(f"Total validation samples: {len(val_dataset)}")
    else:
        print(
            "Val set rỗng hoặc không tồn tại -- bỏ qua eval trong lúc train "
            "(dataset quá nhỏ, hoặc chưa chạy build_train_dataset). "
            "Kết quả train vẫn ra bình thường, chỉ là không theo dõi được eval loss."
        )

    # Eval theo STEPS (không phải mỗi epoch) để có nhiều điểm so sánh
    # eval_loss trong lúc train -- cần thiết để EarlyStopping + chọn
    # checkpoint tốt nhất hoạt động, vì chỉ chạy 1 epoch nên "epoch" chỉ
    # cho đúng 1 điểm đo (không đủ để biết đã qua điểm tối ưu hay chưa).
    EVAL_STEPS = 200

    callbacks = [ProgressCallback()]
    if val_dataset is not None:
        # Dừng sớm nếu eval_loss không cải thiện sau 3 lần eval liên tiếp
        # -> tránh train quá điểm tối ưu (overfitting) mà không cần tự canh.
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))

    training_args = SFTConfig(
        output_dir=str(ADAPTER_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,          # log mỗi step
        # save_strategy PHẢI khớp tần suất với eval_strategy để
        # load_best_model_at_end tìm đúng checkpoint tương ứng.
        save_strategy="steps" if val_dataset is not None else "epoch",
        save_steps=EVAL_STEPS,
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=EVAL_STEPS,
        # Sau khi train xong, tự động load lại checkpoint có eval_loss
        # THẤP NHẤT (không nhất thiết là checkpoint cuối cùng) -> đây là
        # điểm tối ưu, tránh trường hợp train "đi quá" rồi mới dừng.
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_loss" if val_dataset is not None else None,
        greater_is_better=False if val_dataset is not None else None,
        save_total_limit=3,       # chỉ giữ 3 checkpoint gần nhất, đỡ tốn ổ đĩa
        fp16=USE_GPU,              # khớp dtype fp16 thật của model -> bật GradScaler, tránh underflow gradient
        use_cpu=not USE_GPU,
        # Để Trainer TỰ bật gradient checkpointing (đồng bộ với fp16/accelerate.prepare),
        # không tự gọi model.gradient_checkpointing_enable() thủ công bên ngoài nữa.
        gradient_checkpointing=True,
        # use_reentrant=True (mặc định cũ) có thể làm đứt gradient tới LoRA
        # khi kết hợp với model 4-bit -> val_loss đứng yên, không học được gì.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataset_text_field="text",
        max_length=512,
        disable_tqdm=False,       # bật progress bar
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,   # SFTTrainer tự lo prepare_model_for_kbit_training
                                    # -> get_peft_model -> gradient_checkpointing_enable()
                                    # -> enable_input_require_grads() đúng thứ tự nội bộ.
        callbacks=callbacks,
    )

    # In ra % param trainable để XÁC NHẬN LoRA đã gắn đúng.
    trainer.model.print_trainable_parameters()

    # Kiểm tra nhanh (vài giây) trước khi train thật -- tránh tốn 25+ phút
    # train rồi mới phát hiện gradient không lan tới LoRA. Check trên
    # trainer.model (model THẬT SỰ sẽ được train), không phải model gốc.
    sanity_check_lora_gradients(trainer.model, tokenizer, train_dataset)

    print("Bắt đầu train...")
    if val_dataset is not None:
        print(
            f"Eval mỗi {EVAL_STEPS} step trên val set -- sẽ tự dừng sớm nếu "
            f"eval_loss không cải thiện sau 3 lần eval liên tiếp, và tự "
            f"chọn checkpoint có eval_loss thấp nhất (điểm tối ưu) khi xong."
        )
    train_result = trainer.train()
    train_loss = train_result.metrics.get("train_loss")

    val_loss = None
    if val_dataset is not None:
        best_ckpt = trainer.state.best_model_checkpoint
        val_loss = trainer.state.best_metric
        print(f"Checkpoint tốt nhất: {best_ckpt} (eval_loss = {val_loss:.4f})")
    else:
        # Không có val set trong lúc train -> chạy 1 lần evaluate cuối
        # cùng trên chính train set chỉ để có 1 con số tham khảo (không
        # đáng tin bằng val_loss thật, vì model đã thấy dữ liệu này rồi).
        pass

    mean_token_acc = get_mean_token_accuracy(trainer)

    print(f"Train xong. Lưu model vào {ADAPTER_DIR}")
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    import subprocess
    subprocess.run(
        ["zip", "-r", "output.zip", "output"],
        cwd="/kaggle/working/MockProject_062026_NhomAI/Training",
        check=True,
    )
    print("Đã zip -> Training/output.zip (tải về ngay!)")

    # --------------------------------------------------------
    # Xuất CSV dự đoán (ROUGE/BLEU) trên tập TEST -- input cho
    # bước LLM Judge (Prometheus / API) ở pipeline/evaluate.py.
    # Không dùng eval_loss/perplexity ở đây vì đó là số đo trong
    # --------------------------------------------------------
    print("Đang tải tập test (thô) để xuất CSV ROUGE/BLEU...")
    test_raw = load_raw_test_for_export(TEST_FILE)

    rouge1 = rouge2 = rougeL = bleu = None

    if test_raw is None:
        print(
            f"Không tìm thấy/ rỗng {TEST_FILE} -- bỏ qua bước xuất CSV. "
            f"Hãy chạy `python -m pipeline.build_train_dataset` trước nếu cần."
        )
    else:
        print(f"Số câu hỏi test: {len(test_raw)}")
        print("Đang chạy inference trên tập test để tính ROUGE/BLEU...")
        save_predictions_csv(
            model=model,
            tokenizer=tokenizer,
            dataset=test_raw,
            output_path=str(PREDICTIONS_CSV),
            system_prompt=SYSTEM_PROMPT,
            prompt_style=PROMPT_STYLE,
            max_new_tokens=MAX_NEW_TOKENS_TRAIN_GEN,
        )
        print(f"Đã lưu CSV dự đoán -> {PREDICTIONS_CSV}")
        print(
            "Bước tiếp theo: chạy `python -m pipeline.evaluate` để đưa CSV này "
            "qua LLM Judge (RAGAs + Prometheus/API)."
        )

        # Đọc lại CSV vừa xuất để lấy điểm trung bình ROUGE/BLEU cho bảng
        # tổng hợp cuối cùng (save_predictions_csv chỉ in ra console, không
        # trả về giá trị).
        pred_df = pd.read_csv(PREDICTIONS_CSV)
        rouge1 = pred_df["rouge1"].mean()
        rouge2 = pred_df["rouge2"].mean()
        rougeL = pred_df["rougeL"].mean()
        bleu = pred_df["bleu"].mean()

    # --------------------------------------------------------
    # Bảng tổng hợp cuối cùng
    # --------------------------------------------------------
    perplexity = compute_perplexity(val_loss) if val_loss is not None else None

    print_and_save_summary({
        "model": BASE_MODEL_NAME,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "bleu": bleu,
        "perplexity": perplexity,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "mean_token_accuracy": mean_token_acc,
    })

    print("=" * 50)
    print("HOÀN TẤT")
    print("=" * 50)


if __name__ == "__main__":
    main()