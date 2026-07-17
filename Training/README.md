# Medical Chatbot Fine-tuning Pipeline

Fine-tune LoRA cho chatbot hỏi-đáp y tế, base model `Qwen/Qwen2.5-0.5B-Instruct`.

---

## 📂 Train / Validation / Test split

`pipeline/build_train_dataset.py` đọc dữ liệu gốc từ `data/final_train_dataset.json`, xáo trộn (dùng `SPLIT_SEED` trong `src/config.py`
để mỗi lần chạy ra cùng kết quả), rồi chia theo:

- `TRAIN_RATIO = 0.7`
- `VAL_RATIO = 0.15`
- `TEST_RATIO = 0.15`

Các file được tạo trong `output/`:

- 📘 **train.jsonl**: fine-tune model (`pipeline/train.py`).
- 📗 **val.jsonl**: theo dõi validation loss trong lúc train (eval theo step,
  không phải theo epoch) + dùng cho early stopping. Không tham gia cập nhật
  trọng số.
- 📕 **test.jsonl**: tập model chưa từng thấy lúc train, giữ nguyên trường thô
  `{question, answer}` — dùng ở 2 bước đánh giá cuối (ROUGE/BLEU trong
  `train.py`, và LLM Judge trong `pipeline/evaluate.py`).

`train.jsonl` / `val.jsonl` lưu theo định dạng chat `messages`, sẵn sàng cho
`SFTTrainer`.

---

## 🔍 Bật / tắt RAG

Mặc định `USE_RAG=False` (chưa nối vector database thật). Bật bằng:

```bash
MEDQUAD_USE_RAG=1 python -m pipeline.build_train_dataset
```

---

## 🚀 Cách chạy

```bash
pip install -r requirements.txt

# 1. Đặt final_train_dataset.json vào thư mục data/

# 2. Tạo train/validation/test
python -m pipeline.build_train_dataset

# 3. Fine-tune LoRA (train.py TỰ ĐỘNG chạy luôn bước ROUGE/BLEU sau khi train xong)
python -m pipeline.train

# 4. Demo chatbot
python -m pipeline.chat

# 5. Đánh giá bằng LLM Judge -- KHÔNG chạy ở đây, xem mục
#    "🔀 Chạy tách rời: Kaggle / Colab" bên dưới (do Kaggle hay hết giờ session)
```

Chạy bằng `python -m pipeline.<script_name>` (không phải
`python pipeline/<script_name>.py`) để Python nhận đúng thư mục gốc project.

---

## 🧪 Quy trình đánh giá — 2 bước tách rời

Khác với các phiên bản trước, đánh giá được chia làm **2 bước độc lập**:

### Bước 1 — ROUGE/BLEU (chạy TỰ ĐỘNG ngay trong `pipeline/train.py`)

Sau khi train xong và lưu checkpoint tốt nhất, `train.py` gọi
`save_predictions_csv()` (trong `src/evaluation.py`):

- Lấy **100 mẫu đầu tiên** của `test.jsonl` (mặc định `max_samples=100`,
  không phải toàn bộ tập test — vì generate tuần tự từng mẫu khá chậm).
- Sinh câu trả lời bằng model vừa train, tính ROUGE-1/2/L + BLEU cho từng mẫu.
- Xuất ra `output/evaluation_results.csv` (cột: `question, reference,
  prediction, rouge1, rouge2, rougeL, bleu`).
- Ghi thêm 1 dòng tổng hợp vào `output/training_summary.csv` (model,
  rouge1/2/L, bleu, perplexity, train_loss, val_loss, mean_token_accuracy) —
  dễ so sánh giữa các lần train.

⚠️ BLEU tính theo geometric mean 1-4 gram nên rất dễ về gần 0 với câu trả lời
bị paraphrase (không trùng nguyên văn 4-gram liên tiếp) — không có nghĩa là
model kém, chỉ là BLEU quá khắt khe với kiểu dữ liệu này. ROUGE đáng tin hơn
trong trường hợp này.

⚠️ Vì chỉ chạy trên 100/1733 mẫu, con số này chỉ mang tính **ước lượng nhanh
(proxy)**, không đại diện đầy đủ cho toàn bộ tập test. Muốn số đáng tin hơn để
báo cáo chính thức, cần tăng `max_samples` trong lời gọi `save_predictions_csv`
(đánh đổi bằng thời gian chạy lâu hơn).

### Bước 2 — LLM Judge (`pipeline/evaluate.py`, chạy riêng, thủ công)

Đọc CSV từ bước 1, dùng LLM giám khảo **tách biệt** với model đang được đánh
giá để chấm điểm ngữ nghĩa (không chỉ so khớp từ vựng như ROUGE/BLEU):

- Ưu tiên gọi qua **API** (`MEDQUAD_JUDGE_API_KEY`, endpoint kiểu
  OpenAI-compatible, mặc định model `gpt-4o-mini`) — nhanh, không tốn VRAM.
- Nếu không có API key, fallback về Prometheus 2 (`prometheus-eval/prometheus-7b-v2.0`)
  chạy cục bộ, mặc định load 4-bit (`JUDGE_LOAD_IN_4BIT=1`) để vừa GPU free
  tier (Colab/Kaggle T4).

Tách 2 model (model bị đánh giá vs. model giám khảo) để tránh **self-preference
bias** — model có xu hướng tự chấm cao câu trả lời của chính mình nếu dùng
chung 1 model.

---

## ⚙️ Đổi model hoặc đường dẫn

Hầu hết thiết lập nằm trong `src/config.py`, có thể ghi đè bằng biến môi trường:

```bash
MEDQUAD_BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct" python -m pipeline.train
MEDQUAD_OUTPUT_DIR="/kaggle/working/output" python -m pipeline.train
MEDQUAD_JUDGE_API_KEY="sk-..." python -m pipeline.evaluate
```

---

## 🔀 Chạy tách rời: Kaggle (build data + train + chat) / Colab (evaluate)

Kaggle free tier giới hạn thời gian GPU/phiên khá ngắn, không đủ để chạy hết
bước LLM Judge (`pipeline/evaluate.py`) — mỗi câu tốn nhiều giây do rate limit
của API giám khảo, tổng thời gian cho vài nghìn câu test dễ vượt quá session
Kaggle cho phép. Vì vậy pipeline được tách làm 2 nơi chạy:

| Bước | Chạy ở đâu | Vì sao |
|---|---|---|
| `build_train_dataset.py` | Kaggle | Đọc `data/final_train_dataset.json`, không tốn nhiều thời gian |
| `train.py` (+ ROUGE/BLEU tự động) | Kaggle | Cần GPU để fine-tune LoRA |
| `chat.py` | Kaggle | Demo nhanh, dùng luôn GPU đang có sẵn |
| `evaluate.py` (LLM Judge) | **Colab** | Kaggle hết giờ session giữa chừng; evaluate.py chỉ gọi API giám khảo qua mạng, không cần GPU |

### Đưa `evaluation_results.csv` từ Kaggle sang Colab qua GitHub

`evaluate.py` không tự generate lại câu trả lời — nó chỉ đọc `PREDICTIONS_CSV`
(`output/evaluation_results.csv`) đã được `train.py` sinh sẵn ở Kaggle. Vì repo
project được clone về ở cả 2 nơi, cách đơn giản nhất để mang file này từ
Kaggle sang Colab là commit & push nó lên cùng repo:

1. Ở Kaggle, sau khi `train.py` chạy xong (đã có `output/evaluation_results.csv`),
   `git add output/evaluation_results.csv && git commit -m "..." && git push`
   từ trong notebook Kaggle. Lưu ý: nếu `.gitignore` đang chặn thư mục
   `output/`, cần thêm ngoại lệ cho riêng file CSV này (file nhỏ, khác với
   checkpoint model không nên đẩy lên git).
2. Ở Colab, `git clone` lại đúng repo đó — `evaluation_results.csv` đã có sẵn
   đúng vị trí tương đối (`output/`) mà `src/config.py` cần, không phải set
   thêm biến môi trường hay mount Drive để lấy file này.

### Chạy nhiều phiên (resume) trên Colab

`evaluate.py` chấm theo batch nhỏ (mặc định `BATCH_SIZE=1`, đặt qua
`MEDQUAD_EVAL_BATCH_SIZE`) và ghi CSV ngay sau mỗi batch vào Google Drive
(`/content/drive/MyDrive/medquad_eval/ragas_scores.csv`, tự nhận diện nếu
Drive đã mount) — tách biệt khỏi thư mục repo vừa clone, vì `/content` (kể cả
repo mới clone) sẽ mất sạch khi Colab runtime bị ngắt kết nối, còn Drive thì
không.

Vì vậy, mỗi lần vào Colab chạy tiếp: chỉ cần mount Drive + clone lại repo (thư
mục `medquad`) là đủ — **không cần giữ nguyên session cũ**. `evaluate.py` tự
đọc `ragas_scores.csv` đã có trên Drive từ lần trước, biết câu nào đã chấm rồi
và bỏ qua, tiếp tục đúng chỗ dừng dù thư mục repo vừa clone lại là mới hoàn
toàn.