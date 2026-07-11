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
(mặc định: output/output_model)
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from datasets import load_dataset
from transformers import ProgressCallback
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from src.config import ADAPTER_DIR, BASE_MODEL_NAME, TRAIN_FILE, VAL_FILE

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

def apply_lora(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(model, lora_config)


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
# 4. TRAIN
# ============================================================

def main():
    print("=" * 50)
    print("Loading model + tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    print("Applying LoRA...")
    model = apply_lora(model)

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

    training_args = SFTConfig(
        output_dir=str(ADAPTER_DIR),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=1,          # log mỗi step
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset is not None else "no",
        fp16=False,
        use_cpu=not USE_GPU,
        report_to="none",
        dataset_text_field="text",
        max_length=1024,
        disable_tqdm=False,       # bật progress bar
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[ProgressCallback()],   # callback hiển thị tiến độ
    )

    print("Bắt đầu train...")
    trainer.train()

    print(f"Train xong. Lưu model vào {ADAPTER_DIR}")
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    print("=" * 50)
    print("HOÀN TẤT")
    print("=" * 50)


if __name__ == "__main__":
    main()
