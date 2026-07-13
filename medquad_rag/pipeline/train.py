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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

from src.config import (
    ADAPTER_DIR,
    BASE_MODEL_NAME,
    EARLY_STOPPING_PATIENCE,
    EVAL_STEPS,
    FINAL_MAX_EPOCHS,
    LORA_ALPHA_MULT,
    TRAIN_FILE,
    VAL_FILE,
    WARMUP_RATIO,
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
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        model.config.use_cache = False
    else:
        print("Không có GPU -> load model ở CPU (float32, sẽ train chậm hơn)")
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
        )

    return model, tokenizer


def apply_lora(model):
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA_MULT * LORA_R,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    return get_peft_model(model, lora_config)


def main():
    print("=" * 50)
    print("Loading model + tokenizer...")
    model, tokenizer = load_model_and_tokenizer()

    print("Applying LoRA...")
    model = apply_lora(model)

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
        fp16=False,
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
        callbacks=callbacks,
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