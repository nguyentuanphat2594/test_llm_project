"""
config.py
---------
Cấu hình dùng chung cho toàn bộ pipeline (build data, HPO, train, chat, evaluate).
Gom hết đường dẫn + tên model vào 1 chỗ để tránh lệch cấu hình giữa các bước.

Đường dẫn được tính TƯƠNG ĐỐI theo vị trí file này (BASE_DIR = gốc project),
nên chạy được cả ở local, Colab lẫn Kaggle (miễn đã cd đúng vào thư mục
project / add vào sys.path).

Có thể override hầu hết giá trị bằng biến môi trường nếu cần.
"""

import os
from pathlib import Path

# Gốc project = thư mục cha của src/
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("MEDQUAD_DATA_DIR", BASE_DIR / "data"))
OUTPUT_DIR = Path(os.environ.get("MEDQUAD_OUTPUT_DIR", BASE_DIR / "output"))

# ---- Dữ liệu ----
INPUT_FILE = DATA_DIR / "medquad.json"
TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VAL_FILE = OUTPUT_DIR / "val.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"
TRAIN_SAMPLE_LIMIT = 11548  # số sample lấy từ medquad.json để build dataset

# Tỷ lệ chia train/val/test (phải cộng lại = 1.0)
# 80% train / 10% val (theo dõi eval_loss + chọn hyperparameter) /
# 10% test (chạm đúng 1 lần cuối cùng, xem pipeline/evaluate.py)
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# Seed cố định để chia tập lần nào cũng ra kết quả giống nhau (reproducible).
# QUAN TRỌNG: không chạy lại build_train_dataset.py với seed khác sau khi đã
# bắt đầu HPO/train, nếu không test set sẽ đổi mẫu -> phá nguyên tắc "test
# chỉ chạm 1 lần cuối cùng".
SPLIT_SEED = 42

# ---- RAG toggle ----
USE_RAG = os.environ.get("MEDQUAD_USE_RAG", "0") == "1"

# ---- Model ----
BASE_MODEL_NAME = os.environ.get("MEDQUAD_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = OUTPUT_DIR / "output_model"

# ============================================================
# ---- Hyperparameter Optimization (Optuna) ----
# ============================================================

# Giai đoạn 1 (search): chạy N_TRIALS trial trên 1 subset cố định của Train,
# dùng pruner để cắt sớm trial kém -> tiết kiệm GPU trước khi train full data.
N_TRIALS = int(os.environ.get("MEDQUAD_N_TRIALS", "8"))

# Tỉ lệ subset của Train dùng trong giai đoạn search (không phải toàn bộ Train)
HPO_SUBSET_RATIO = float(os.environ.get("MEDQUAD_HPO_SUBSET_RATIO", "0.35"))

# Trần số step cho MỖI trial ở giai đoạn search (chặn trial "tốt nhưng chạy
# mãi không dừng" ăn hết thời gian của các trial khác)
HPO_MAX_STEPS = int(os.environ.get("MEDQUAD_HPO_MAX_STEPS", "400"))

# Cứ load 1 base model thì dùng cho MODEL_GROUP_SIZE trial liên tiếp (mỗi
# trial vẫn có LoRA adapter RIÊNG, độc lập) trước khi giải phóng và load lại,
# để cân bằng giữa tiết kiệm thời gian load model và giữ các trial độc lập.
MODEL_GROUP_SIZE = int(os.environ.get("MEDQUAD_MODEL_GROUP_SIZE", "2"))

# Optuna study lưu vào SQLite -> nếu notebook bị ngắt giữa chừng, chạy lại
# vẫn tiếp tục được (load_if_exists=True trong pipeline/hpo_train.py)
HPO_STUDY_DIR = OUTPUT_DIR / "optuna"
HPO_STUDY_NAME = "medquad_lora_hpo"
HPO_STORAGE = f"sqlite:///{HPO_STUDY_DIR / 'study.db'}"

# Seed cố định để: (1) TPESampler tái lập được, (2) subset dùng chung cho
# MỌI trial giống hệt nhau -> eval_loss giữa các trial so sánh công bằng.
HPO_SEED = 42

# Search space -- Optuna tự chọn giá trị bên trong các khoảng này, KHÔNG
# hardcode 1 giá trị cố định.
LR_MIN, LR_MAX = 5e-5, 2e-4
LORA_R_MIN, LORA_R_MAX = 8, 32          # step=8 -> {8, 16, 24, 32}
LORA_DROPOUT_MIN, LORA_DROPOUT_MAX = 0.0, 0.1
WEIGHT_DECAY_MIN, WEIGHT_DECAY_MAX = 0.0, 0.05

# Alpha KHÔNG search riêng -- tính động theo rank để giữ tỉ lệ alpha/r ổn
# định (xem thảo luận: rank đổi mà alpha cố định sẽ làm lệch scale LoRA).
LORA_ALPHA_MULT = 2  # alpha = LORA_ALPHA_MULT * r

# Cố định để tiết kiệm 1 chiều search (đã thống nhất)
WARMUP_RATIO = 0.03
EVAL_STEPS = 100
EARLY_STOPPING_PATIENCE = 3

# Giai đoạn 2 (final): trần epoch cao, early stopping tự cắt sớm hơn nếu hội
# tụ trước đó -- KHÔNG có bước "Final Training" nào khác sau bước này.
FINAL_MAX_EPOCHS = int(os.environ.get("MEDQUAD_FINAL_MAX_EPOCHS", "3"))

# ---- Đánh giá (RAGAs) ----
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
JUDGE_MODEL_NAME = os.environ.get("MEDQUAD_JUDGE_MODEL", "prometheus-eval/prometheus-7b-v2.0")
JUDGE_LOAD_IN_4BIT = os.environ.get("MEDQUAD_JUDGE_4BIT", "1") != "0"

# ---- Sinh câu trả lời ----
MAX_NEW_TOKENS_TRAIN_GEN = 300
MAX_NEW_TOKENS_CHAT = 300