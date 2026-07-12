"""
config.py
---------
Cấu hình dùng chung cho toàn bộ pipeline (build data, train, chat, evaluate).
Gom hết đường dẫn + tên model vào 1 chỗ để tránh lệch cấu hình giữa các bước
(vd: train xong dùng model A nhưng chat.py lại load model B).

Đường dẫn được tính TƯƠNG ĐỐI theo vị trí file này (BASE_DIR = gốc project),
nên chạy được cả ở local, Colab (miễn đã mount + cd đúng vào thư mục project
trong Drive) lẫn Kaggle (miễn đã add project vào sys.path / Kaggle Dataset).

Có thể override bằng biến môi trường nếu cần (ví dụ đổi nơi lưu output khi
chạy trên Kaggle: MEDQUAD_OUTPUT_DIR=/kaggle/working/output).
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
TRAIN_SAMPLE_LIMIT = 1200  # số sample lấy từ medquad.json để build dataset (demo)

# Tỷ lệ chia train/val/test (phải cộng lại = 1.0)
TRAIN_RATIO = 0.75
VAL_RATIO = 0.2
TEST_RATIO = 0.05

# Seed cố định để chia tập lần nào cũng ra kết quả giống nhau (reproducible)
SPLIT_SEED = 42

# ---- RAG toggle ----
# Hiện tại CHƯA nối vector DB thật -> mặc định TẮT rag, train/test thuần
# Q&A (chỉ question -> answer, không có đoạn ngữ cảnh nào). Khi nối RAG thật
# vào (vector DB trả về chunks), bật lại bằng biến môi trường:
#   MEDQUAD_USE_RAG=1 python -m pipeline.build_train_dataset
# Bật/tắt được độc lập ở TỪNG bước (build_train_dataset / chat / evaluate)
# nếu cần so sánh có-RAG vs không-RAG, nhưng mặc định dùng chung 1 cờ này để
# đồng bộ giữa lúc train và lúc test.
USE_RAG = os.environ.get("MEDQUAD_USE_RAG", "0") == "1"

# ---- Model ----
# Model nhỏ, free, phổ biến cho fine-tune trên máy yếu / Colab-Kaggle free tier.
# Đổi ở ĐÂY DUY NHẤT nếu muốn dùng model khác — mọi script khác sẽ tự đồng bộ.
BASE_MODEL_NAME = os.environ.get("MEDQUAD_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
ADAPTER_DIR = OUTPUT_DIR / "output_model"

# ---- Đánh giá (RAGAs) ----
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # free, nhẹ

# Model GIÁM KHẢO — PHẢI KHÁC với model đang được đánh giá (BASE_MODEL_NAME +
# ADAPTER_DIR), nếu không sẽ bị lỗi "tự chấm bài mình" (self-preference bias):
# model vừa fine-tune có xu hướng tự thấy câu trả lời của chính nó hợp lý hơn
# thực tế khi được giao luôn vai giám khảo.
#
# Prometheus 2 (7B, Apache 2.0) là model được train CHUYÊN để chấm điểm LLM
# khác (không phải model chat thông thường), độ khớp với GPT-4/con người
# 72-85% theo benchmark của nhóm tác giả. Free, chạy local.
# https://huggingface.co/prometheus-eval/prometheus-7b-v2.0
JUDGE_MODEL_NAME = os.environ.get("MEDQUAD_JUDGE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# Prometheus 2 là model 7B -> cần ~16GB VRAM ở bf16. Load 4-bit để vừa GPU free
# tier (Colab T4 / Kaggle T4-P100, ~15-16GB VRAM), chỉ cần ~5-6GB.
JUDGE_LOAD_IN_4BIT = os.environ.get("MEDQUAD_JUDGE_4BIT", "1") != "0"

# ---- Sinh câu trả lời ----
MAX_NEW_TOKENS_TRAIN_GEN = 300   # dùng khi generate answer cho eval
MAX_NEW_TOKENS_CHAT = 300        # dùng khi chat/inference thật
