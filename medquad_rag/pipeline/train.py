"""
train.py
--------
Fine-tune NHANH, KHÔNG chạy HPO -- dùng để test độc lập 1 bộ hyperparameter
cụ thể (vd: debug pipeline, thử nhanh trước khi chạy hpo_train.py tốn thời
gian hơn). Luồng chính của project nên dùng pipeline/hpo_train.py (Optuna
search + final training), không phải file này.

Cách chạy:
    python -m pipeline.train

Sau khi train xong, model LoRA sẽ được lưu vào src.config.ADAPTER_DIR
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    ProgressCallback,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from src.config import (
    ADAPTER_DIR,
    BASE_MODEL_NAME,
    EARLY_STOPPING_PATIENCE,
    EVAL_MAX_SAMPLES,
    EVAL_STEPS,
    FINAL_MAX_EPOCHS,
    LORA_ALPHA_MULT,
    MAX_NEW_TOKENS_TRAIN_GEN,
    PREDICTIONS_CSV,
    SUMMARY_CSV,
    TEST_FILE,
    TRAIN_FILE,
    VAL_FILE,
    WARMUP_RATIO,
)
from src.evaluation import (
    append_summary_row,
    compute_perplexity,
    load_raw_test_samples,
    save_predictions_csv,
)
from src.utils import format_chat_dataset

USE_GPU = torch.cuda.is_available()

# Bộ hyperparameter mặc định cho lần chạy nhanh này (không qua Optuna).
# Nếu muốn dùng bộ đã tìm được từ HPO, xem output/best_hyperparameters.json
# rồi tự điền lại các giá trị dưới đây, hoặc dùng thẳng pipeline/hpo_train.py.
LR = 2e-4
LORA_R = 16
LORA_DROPOUT = 0.05
WEIGHT_DECAY = 0.0


def build_lora_config():
    """
    CHỈ trả về LoraConfig -- KHÔNG tự gọi get_peft_model() ở đây. Truyền
    config này vào SFTTrainer(peft_config=...) để nó tự lo toàn bộ chuỗi
    prepare_model_for_kbit_training -> get_peft_model ->
    gradient_checkpointing_enable() -> enable_input_require_grads() đúng
    thứ tự nội bộ. Tự gọi get_peft_model() thủ công trước rồi để Trainer
    wrap model thêm 1 lần nữa (đặc biệt khi fp16=True) làm 2 bên không đồng
    bộ -> graph đứt, gradient không tới LoRA (xem bug grad_norm=0 đã gặp).
    """
    return LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA_MULT * LORA_R,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


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
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False
    else:
        print("Không có GPU -> load model ở CPU (float32, sẽ train chậm hơn)")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
        )

    return model, tokenizer


def main():
    print("=" * 50)
    print("Loading model + tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    print("Chuẩn bị LoRA config (SFTTrainer sẽ tự gắn LoRA)...")
    lora_config = build_lora_config()

    print("Loading train dataset...")
    train_dataset = format_chat_dataset(tokenizer, TRAIN_FILE, required=True)
    print(f"Total training samples: {len(train_dataset)}")

    print("Loading val dataset...")
    val_dataset = format_chat_dataset(tokenizer, VAL_FILE, required=False)
    if val_dataset is not None:
        print(f"Total validation samples: {len(val_dataset)}")
    else:
        print("Val set rỗng/không tồn tại -- bỏ qua eval + early stopping trong lần chạy này.")

    training_args = SFTConfig(
        output_dir=str(ADAPTER_DIR),
        num_train_epochs=FINAL_MAX_EPOCHS,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=20,
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=EVAL_STEPS,
        save_strategy="steps" if val_dataset is not None else "epoch",
        save_steps=EVAL_STEPS,
        save_total_limit=3,
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_loss" if val_dataset is not None else None,
        greater_is_better=False,
        fp16=USE_GPU,       # bật loss scaling khi chạy fp16 thật trên GPU
        use_cpu=not USE_GPU,
        report_to="none",
        dataset_text_field="text",
        max_length=1024,
        disable_tqdm=False,
    )

    callbacks = [ProgressCallback()]
    if val_dataset is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE))

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        callbacks=callbacks,
    )
    trainer.model.print_trainable_parameters()

    print("Bắt đầu train...")
    trainer.train()

    print(f"Train xong. Lưu model vào {ADAPTER_DIR}")
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    val_loss = None
    if val_dataset is not None:
        eval_result = trainer.evaluate()
        val_loss = eval_result.get("eval_loss")

    print("Đang tải tập test để tính ROUGE/BLEU...")
    test_dataset = load_raw_test_samples(TEST_FILE)
    if test_dataset is not None:
        print(f"Số câu hỏi test: {len(test_dataset)}")
        df = save_predictions_csv(
            model=trainer.model,
            tokenizer=tokenizer,
            dataset=test_dataset,
            output_path=str(PREDICTIONS_CSV),
            max_new_tokens=MAX_NEW_TOKENS_TRAIN_GEN,
            max_samples=EVAL_MAX_SAMPLES,
        )
        append_summary_row(
            {
                "model": BASE_MODEL_NAME,
                "lr": LR,
                "lora_r": LORA_R,
                "lora_dropout": LORA_DROPOUT,
                "weight_decay": WEIGHT_DECAY,
                "rouge1": df["rouge1"].mean(),
                "rouge2": df["rouge2"].mean(),
                "rougeL": df["rougeL"].mean(),
                "bleu": df["bleu"].mean(),
                "perplexity": compute_perplexity(val_loss) if val_loss is not None else None,
                "val_loss": val_loss,
            },
            SUMMARY_CSV,
        )
    else:
        print(f"{TEST_FILE} rỗng/không tồn tại -- bỏ qua bước ROUGE/BLEU. "
              f"Chạy `python -m pipeline.build_train_dataset` trước nếu cần.")

    print("=" * 50)
    print("HOÀN TẤT")
    print("=" * 50)


if __name__ == "__main__":
    main()