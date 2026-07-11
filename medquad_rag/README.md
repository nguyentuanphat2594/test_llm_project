# medquad_rag

Fine-tune một model nhỏ (Qwen2.5-0.5B-Instruct) theo hướng RAG cho chatbot tư
vấn y tế, dùng bộ dữ liệu MedQuAD.

## Cấu trúc

```
medquad_rag/
│
├── src/                        # code dùng chung, tái sử dụng
│   ├── __init__.py
│   ├── prompt_template.py      # build_prompt() — hỗ trợ cả 2 chế độ RAG / không-RAG
│   ├── config.py                # MỌI đường dẫn + tên model + tỷ lệ split — sửa 1 chỗ
│   └── utils.py                 # load_dataset / save_jsonl / validate_sample / split_dataset
│
├── pipeline/                    # script chạy từng bước
│   ├── __init__.py
│   ├── build_train_dataset.py   # medquad.json -> output/{train,val,test}.jsonl
│   ├── train.py                 # LoRA fine-tune (theo dõi eval loss bằng val.jsonl) -> output/output_model/
│   ├── chat.py                  # demo inference (base model + adapter)
│   └── evaluate.py              # đánh giá bằng RAGAs trên test.jsonl (LLM Judge = Prometheus 2, tách biệt)
│
├── data/
│   └── medquad.json             # (tự thêm vào, không có sẵn trong repo)
│
├── output/
│   ├── train.jsonl              # 80% — dùng để fine-tune
│   ├── val.jsonl                 # 10% — theo dõi eval loss lúc train, KHÔNG cập nhật trọng số
│   ├── test.jsonl                # 10% — GIỮ NGUYÊN lúc train, chỉ dùng để evaluate.py chấm điểm cuối
│   └── output_model/            # sinh ra bởi train.py
│
├── kaggle/
│   └── main_pipeline.ipynb      # orchestrator chạy cả pipeline trên Kaggle (có GPU free)
│
└── requirements.txt
```

## Train/val/test split

`pipeline/build_train_dataset.py` xáo trộn (seed cố định `SPLIT_SEED` trong
`src/config.py` — chia lần nào cũng ra kết quả giống nhau) rồi chia dữ liệu
theo tỷ lệ `TRAIN_RATIO` / `VAL_RATIO` / `TEST_RATIO` (mặc định 80/10/10):

- **train.jsonl** — fine-tune (`pipeline/train.py`)
- **val.jsonl** — theo dõi loss trong lúc train (`eval_strategy="epoch"` trong
  `SFTConfig`), giúp phát hiện overfitting, KHÔNG dùng để cập nhật trọng số
- **test.jsonl** — model **chưa từng thấy** lúc train, giữ riêng để
  `evaluate.py` chấm điểm cuối cùng bằng RAGAs — đo khả năng generalize thật,
  không phải học thuộc

`train.jsonl`/`val.jsonl` lưu dạng chat `messages` (sẵn sàng cho
`SFTTrainer`); `test.jsonl` lưu dạng thô `{question, contexts, ground_truth}`
vì `evaluate.py` cần tự sinh câu trả lời bằng model trước khi build prompt.

## RAG toggle

Dự án hiện **CHƯA nối vector DB thật**. Mặc định `USE_RAG=False` (xem
`src/config.py`): train/test thuần Q&A — `build_prompt()` không có phần
"NGỮ CẢNH", model học trả lời bằng kiến thức y tế đã fine-tune, system prompt
cũng đổi cho hợp lý (không ép "chỉ trả lời dựa trên ngữ cảnh" khi không có
ngữ cảnh nào).

Khi nối RAG thật (vector DB trả về chunks), bật lại bằng:

```bash
MEDQUAD_USE_RAG=1 python -m pipeline.build_train_dataset
```

Cờ này áp dụng đồng bộ cho cả `build_train_dataset.py` (chunks giả lập =
answer gốc) lẫn `evaluate.py` (chọn có dùng `context_precision`/
`context_recall`/`faithfulness` — các metric RAGAs cần context thật — hay chỉ
dùng `answer_relevancy` khi không có RAG).

`pipeline/chat.py` (demo inference) độc lập với cờ này — gọi
`build_prompt(chunks=..., question=...)` với `chunks=None` để test chế độ
không-RAG, hoặc truyền list chunks thật khi đã có vector DB.

## Cách chạy (local hoặc Colab)

```bash
pip install -r requirements.txt

# 1. Đặt medquad.json vào data/

# 2. Build train.jsonl
python -m pipeline.build_train_dataset

# 3. Train LoRA
python -m pipeline.train

# 4. Demo chat
python -m pipeline.chat

# 5. Đánh giá
python -m pipeline.evaluate
```

Chạy bằng `python -m pipeline.<tên_script>` (không phải `python pipeline/<tên_script>.py`)
để import `src.*` hoạt động đúng — lệnh `-m` sẽ tự thêm thư mục gốc project vào
`sys.path`.

## Đổi model / đường dẫn

Không sửa trực tiếp trong từng script. Sửa ở `src/config.py`, hoặc override
bằng biến môi trường lúc chạy:

```bash
MEDQUAD_BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct" python -m pipeline.train
MEDQUAD_OUTPUT_DIR="/kaggle/working/output" python -m pipeline.train
```

## Chạy trên Colab

Trong Colab, mount Drive rồi `%cd` vào đúng thư mục project đã upload lên Drive
(chứa `src/`, `pipeline/`, ...), sau đó chạy các lệnh ở trên bình thường —
không cần `sys.path.append(...)` thủ công như bản cũ nữa.

## Chạy trên Kaggle

Xem `kaggle/main_pipeline.ipynb`. Upload cả thư mục `medquad_rag/` làm 1
Kaggle Dataset (hoặc `!git clone` repo nếu đã đẩy lên GitHub), rồi chạy
notebook đó — nó sẽ tự cài `requirements.txt` và chạy tuần tự 4 bước ở trên.

## Lưu ý

- `evaluate.py` dùng **2 model tách biệt**: model bị đánh giá (base +
  adapter LoRA vừa train ở bước 3) chỉ để sinh câu trả lời, và
  `prometheus-eval/prometheus-7b-v2.0` (model được train chuyên để chấm điểm
  LLM khác — xem `src/config.py` → `JUDGE_MODEL_NAME`) làm giám khảo. Không
  dùng chung 1 model cho cả 2 vai để tránh model tự chấm cao câu trả lời của
  chính nó (self-preference bias). Prometheus 2 mặc định load 4-bit
  (`JUDGE_LOAD_IN_4BIT` trong config) để vừa VRAM GPU free tier (~5-6GB thay
  vì ~16GB); có thể override qua biến môi trường `MEDQUAD_JUDGE_MODEL` /
  `MEDQUAD_JUDGE_4BIT=0`. Kết quả vẫn chỉ mang tính tham khảo/demo (Prometheus
  2 nhỏ hơn nhiều so với GPT-4/Claude), không nên dùng làm số liệu báo cáo
  cuối cùng nếu cần độ tin cậy cao.
- `build_train_dataset.py` hiện dùng chính `answer` gốc trong MedQuAD làm
  "chunk" giả lập (chưa có vector DB thật) — khi có vector DB thật, thay phần
  gọi `build_prompt(chunks=[answer], ...)` bằng kết quả truy vấn thật.
