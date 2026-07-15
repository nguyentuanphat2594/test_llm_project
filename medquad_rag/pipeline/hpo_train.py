"""
hpo_train.py
------------
Pipeline chính để tìm hyperparameter tối ưu VÀ train model cuối cùng.
Đây là file nên chạy trong luồng chính (thay cho pipeline/train.py, file đó
chỉ để test nhanh 1 bộ tham số cố định).

GIAI ĐOẠN 1 -- Search (Optuna):
  - Chạy N_TRIALS trial, mỗi trial 1 bộ hyperparameter khác nhau
    (learning_rate, lora_r, lora_dropout, weight_decay; alpha = 2*r tính
    động, không search riêng).
  - Mọi trial dùng CHUNG 1 subset cố định (fix seed) của train.jsonl --
    KHÔNG train trên toàn bộ Train ở giai đoạn này, để tiết kiệm thời gian.
  - Mỗi trial train LIÊN TỤC (không restart), cứ mỗi EVAL_STEPS thì eval
    trên val.jsonl và report(eval_loss, step) cho Optuna.
    SuccessiveHalvingPruner sẽ so sánh trial này với các trial khác tại
    CÙNG mốc step và cắt sớm nếu trial đang tệ hơn hẳn -- đây chính là bản
    chất "successive halving" (trial yếu bị loại dần khi ngân sách/step
    tăng lên), làm bằng cơ chế report/prune chuẩn của Optuna thay vì tự tay
    chia chunk + restart Trainer (cách đó dễ làm sai step cộng dồn / reset
    optimizer giữa các round).
  - Base model được load 1 lần rồi dùng lại cho MODEL_GROUP_SIZE trial liên
    tiếp (mỗi trial vẫn có adapter LoRA hoàn toàn riêng, độc lập) để đỡ tốn
    thời gian load lại nhiều lần, trước khi giải phóng và load model mới.

GIAI ĐOẠN 2 -- Final training:
  - Lấy bộ hyperparameter tốt nhất từ Optuna.
  - Train DUY NHẤT 1 LẦN trên TOÀN BỘ train.jsonl, eval theo step + early
    stopping + load_best_model_at_end. Đây CHÍNH LÀ Final Adapter -- không
    có bước train lại nào khác sau đó.

Cách chạy:
    python -m pipeline.hpo_train
"""

import json

import optuna
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    ProgressCallback,
    TrainerCallback,
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
    HPO_MAX_STEPS,
    HPO_SEED,
    HPO_STORAGE,
    HPO_STUDY_DIR,
    HPO_STUDY_NAME,
    HPO_SUBSET_RATIO,
    LORA_ALPHA_MULT,
    LORA_DROPOUT_MAX,
    LORA_DROPOUT_MIN,
    LORA_R_MAX,
    LORA_R_MIN,
    LR_MAX,
    LR_MIN,
    MAX_NEW_TOKENS_TRAIN_GEN,
    MODEL_GROUP_SIZE,
    N_TRIALS,
    OUTPUT_DIR,
    PREDICTIONS_CSV,
    SUMMARY_CSV,
    TEST_FILE,
    TRAIN_FILE,
    VAL_FILE,
    WARMUP_RATIO,
    WEIGHT_DECAY_MAX,
    WEIGHT_DECAY_MIN,
)
from src.evaluation import (
    append_summary_row,
    compute_perplexity,
    load_raw_test_samples,
    save_predictions_csv,
)
from src.utils import format_chat_dataset

USE_GPU = torch.cuda.is_available()
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


# ============================================================
# CALLBACK: report eval_loss theo step cho Optuna + cho phép Pruner cắt sớm
# ============================================================

class OptunaPruningCallback(TrainerCallback):
    """Sau mỗi lần Trainer eval, báo cáo eval_loss về Optuna (kèm step hiện
    tại) rồi hỏi Pruner xem trial này có nên bị cắt không."""

    def __init__(self, trial, monitor="eval_loss"):
        self.trial = trial
        self.monitor = monitor

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics or self.monitor not in metrics:
            return control
        self.trial.report(metrics[self.monitor], step=state.global_step)
        if self.trial.should_prune():
            control.should_training_stop = True
            raise optuna.TrialPruned()
        return control


# ============================================================
# LOAD BASE MODEL (dùng chung cho cả search lẫn final training)
# ============================================================

def load_base_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if USE_GPU:
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
            # Không để hàm này tự bật gradient checkpointing -- SFTTrainer sẽ
            # tự bật lại theo SFTConfig bên dưới, 2 bên bật không đồng bộ
            # (thiếu use_reentrant=False) là nguyên nhân gây đứt gradient.
            use_gradient_checkpointing=False,
        )
        # Vẫn cần dòng này để input embeddings (đang bị đóng băng/quantize)
        # cho phép gradient chảy qua, dù checkpointing được bật ở đâu.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.config.use_cache = False
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
        )

    return model, tokenizer


# ============================================================
# GIAI ĐOẠN 1: TRIAL RUNNER (quản lý việc load/reuse base model theo nhóm)
# ============================================================

class TrialRunner:
    def __init__(self, subset_dataset, val_dataset):
        self.subset_dataset = subset_dataset
        self.val_dataset = val_dataset
        self.base_model = None
        self.trials_in_current_group = 0

    def _ensure_base_model(self):
        if self.base_model is None or self.trials_in_current_group >= MODEL_GROUP_SIZE:
            if self.base_model is not None:
                del self.base_model
                if USE_GPU:
                    torch.cuda.empty_cache()
            print(f"[HPO] Load base model mới cho nhóm {MODEL_GROUP_SIZE} trial tiếp theo...")
            self.base_model, _ = load_base_model_and_tokenizer()
            self.trials_in_current_group = 0

    def __call__(self, trial):
        self._ensure_base_model()
        self.trials_in_current_group += 1

        # ---- Search space ----
        lr = trial.suggest_float("learning_rate", LR_MIN, LR_MAX, log=True)
        r = trial.suggest_int("lora_r", LORA_R_MIN, LORA_R_MAX, step=8)
        dropout = trial.suggest_float("lora_dropout", LORA_DROPOUT_MIN, LORA_DROPOUT_MAX)
        weight_decay = trial.suggest_float("weight_decay", WEIGHT_DECAY_MIN, WEIGHT_DECAY_MAX)
        alpha = LORA_ALPHA_MULT * r  # KHÔNG search alpha riêng -- tính động theo r

        print(f"\n[HPO] Trial {trial.number}: lr={lr:.2e} r={r} alpha={alpha} "
              f"dropout={dropout:.3f} weight_decay={weight_decay:.4f}")

        lora_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        )

        trial_dir = HPO_STUDY_DIR / f"trial_{trial.number}"
        args = SFTConfig(
            output_dir=str(trial_dir),
            max_steps=HPO_MAX_STEPS,        # trần step cho riêng giai đoạn search
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=lr,
            weight_decay=weight_decay,
            warmup_ratio=WARMUP_RATIO,
            logging_steps=20,
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_strategy="no",             # trial chỉ để tìm hyperparameter, không cần lưu checkpoint
            report_to="none",
            fp16=USE_GPU,       # bật loss scaling khi chạy fp16 thật trên GPU
            use_cpu=not USE_GPU,
            dataset_text_field="text",
            max_length=600,
            disable_tqdm=True,
        )

        trainer = SFTTrainer(
            model=self.base_model,      # model THÔ, chưa gắn LoRA
            args=args,
            train_dataset=self.subset_dataset,
            eval_dataset=self.val_dataset,
            peft_config=lora_config,    # để SFTTrainer tự gắn LoRA đúng thứ tự
            callbacks=[OptunaPruningCallback(trial)],
        )
        trainer.model.print_trainable_parameters()   # debug vẫn giữ được, chỉ đổi chỗ gọi

        try:
            trainer.train()
            eval_result = trainer.evaluate()
            eval_loss = eval_result["eval_loss"]
        except optuna.TrialPruned:
            eval_loss = None
            raise
        finally:
            # Gỡ adapter, trả base model "sạch" về cho trial kế tiếp trong nhóm
            self.base_model = trainer.model.unload()
            del trainer
            if USE_GPU:
                torch.cuda.empty_cache()

        return eval_loss


def run_search(subset_dataset, val_dataset):
    HPO_STUDY_DIR.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=HPO_STUDY_NAME,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=HPO_SEED),
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
        storage=HPO_STORAGE,
        load_if_exists=True,
    )
    runner = TrialRunner(subset_dataset, val_dataset)
    study.optimize(runner, n_trials=N_TRIALS)
    return study


def save_search_results(study):
    result = {
        "best_value_eval_loss": study.best_value,
        "best_params": study.best_params,
        "all_trials": [
            {
                "number": t.number,
                "state": str(t.state),
                "value": t.value,
                "params": t.params,
            }
            for t in study.trials
        ],
    }
    out_path = OUTPUT_DIR / "best_hyperparameters.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[HPO] Đã lưu kết quả search vào {out_path}")


# ============================================================
# GIAI ĐOẠN 2: FINAL TRAINING (1 lần duy nhất, trên toàn bộ Train)
# ============================================================

def train_final(tokenizer, train_dataset, val_dataset, best_params):
    print("\n[Final] Load base model cho final training...")
    model, _ = load_base_model_and_tokenizer()

    r = best_params["lora_r"]
    lora_config = LoraConfig(
        r=r,
        lora_alpha=LORA_ALPHA_MULT * r,
        lora_dropout=best_params["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )

    args = SFTConfig(
        output_dir=str(ADAPTER_DIR),
        num_train_epochs=FINAL_MAX_EPOCHS,   # trần cao, early stopping tự cắt nếu hội tụ sớm hơn
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        warmup_ratio=WARMUP_RATIO,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=USE_GPU,       # bật loss scaling khi chạy fp16 thật trên GPU (đồng bộ với TrialRunner)
        use_cpu=not USE_GPU,
        report_to="none",
        dataset_text_field="text",
        max_length=600,
        disable_tqdm=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,    # BUG CŨ: lora_config được tạo nhưng chưa từng gắn
                                     # vào model -- final training thực chất train model
                                     # KHÔNG có LoRA nào cả. Để SFTTrainer tự gắn đúng
                                     # thứ tự (đồng bộ với fp16), giống TrialRunner.
        callbacks=[
            ProgressCallback(),
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
        ],
    )
    trainer.model.print_trainable_parameters()   # debug: xác nhận LoRA có tham số trainable > 0

    print("[Final] Bắt đầu final training (đây là lần train DUY NHẤT trên toàn bộ Train)...")
    trainer.train()

    print(f"[Final] Xong. Best checkpoint đã tự load. Lưu Final Adapter vào {ADAPTER_DIR}")
    trainer.save_model(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))

    eval_result = trainer.evaluate()
    val_loss = eval_result.get("eval_loss")

    print("\n[Final] Đang tải tập test để tính ROUGE/BLEU...")
    test_dataset = load_raw_test_samples(TEST_FILE)
    if test_dataset is not None:
        print(f"[Final] Số câu hỏi test: {len(test_dataset)}")
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
                **best_params,
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
        print(f"[Final] {TEST_FILE} rỗng/không tồn tại -- bỏ qua bước ROUGE/BLEU.")


# ============================================================
# MAIN
# ============================================================

def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading train/val dataset...")
    train_dataset = format_chat_dataset(tokenizer, TRAIN_FILE, required=True)
    val_dataset = format_chat_dataset(tokenizer, VAL_FILE, required=True)
    if val_dataset is None:
        raise RuntimeError(
            "Cần val.jsonl (không rỗng) để chạy HPO -- dùng làm tín hiệu chọn "
            "hyperparameter và early stopping. Kiểm tra lại build_train_dataset.py."
        )
    print(f"Train: {len(train_dataset)} mẫu | Val: {len(val_dataset)} mẫu")

    n_subset = max(1, int(len(train_dataset) * HPO_SUBSET_RATIO))
    subset_dataset = train_dataset.shuffle(seed=HPO_SEED).select(range(n_subset))
    print(f"Subset cho giai đoạn search: {n_subset}/{len(train_dataset)} mẫu "
          f"({HPO_SUBSET_RATIO:.0%}), seed={HPO_SEED} (cố định cho mọi trial)")

    print("\n" + "=" * 60)
    print(f"GIAI ĐOẠN 1: Hyperparameter Search ({N_TRIALS} trial, Optuna)")
    print("=" * 60)
    study = run_search(subset_dataset, val_dataset)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\nSố trial hoàn thành: {len(completed)} | Số trial bị cắt sớm: {len(pruned)}")
    print(f"Best eval_loss: {study.best_value:.4f}")
    print(f"Best params   : {study.best_params}")
    save_search_results(study)

    print("\n" + "=" * 60)
    print("GIAI ĐOẠN 2: Final Training (toàn bộ Train set)")
    print("=" * 60)
    train_final(tokenizer, train_dataset, val_dataset, study.best_params)

    print("\n" + "=" * 60)
    print("HOÀN TẤT HPO + FINAL TRAINING")
    print(f"Final Adapter: {ADAPTER_DIR}")
    print(f"Best params  : {OUTPUT_DIR / 'best_hyperparameters.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()