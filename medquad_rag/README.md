## 📂 Train / Validation / Test split

`pipeline/build_train_dataset.py` sẽ xáo trộn dữ liệu (sử dụng `SPLIT_SEED`
trong `src/config.py` để mỗi lần chạy đều cho cùng một kết quả), sau đó chia
theo tỷ lệ **80% / 10% / 10%** (`TRAIN_RATIO`, `VAL_RATIO`, `TEST_RATIO`
trong `src/config.py`).

Các file được tạo gồm:

- 📘 **train.jsonl** (80%): dùng để fine-tune model.
- 📗 **val.jsonl** (10%): dùng để theo dõi validation loss trong lúc train
  (`eval_strategy="steps"`, eval mỗi `EVAL_STEPS` step — không đợi hết
  epoch mới eval). Tập này chỉ phục vụ đánh giá + chọn hyperparameter +
  early stopping, **không** tham gia cập nhật trọng số.
- 📕 **test.jsonl** (10%): tập dữ liệu model chưa từng thấy trong suốt quá
  trình train VÀ trong quá trình tìm hyperparameter. `pipeline/evaluate.py`
  chỉ chạm vào tập này **đúng một lần cuối cùng** để đánh giá bằng RAGAs,
  phản ánh khả năng tổng quát hóa của model thay vì khả năng ghi nhớ dữ liệu
  hay khả năng "được chọn vì hợp với val set".

`train.jsonl` và `val.jsonl` được lưu theo định dạng chat `messages`, phù hợp
để huấn luyện bằng `SFTTrainer`.

`test.jsonl` giữ nguyên các trường `{question, contexts, ground_truth}` vì
`evaluate.py` sẽ để model tự sinh câu trả lời trước khi tiến hành đánh giá.

⚠️ Không chạy lại `build_train_dataset.py` với `SPLIT_SEED` khác (hoặc data
gốc khác) sau khi đã bắt đầu HPO/train — test set sẽ đổi mẫu, phá nguyên tắc
"test chỉ chạm một lần cuối cùng".

---

## 🎯 Hyperparameter Optimization (`pipeline/hpo_train.py`)

Đây là **luồng train chính** của project (khác với `pipeline/train.py`, chỉ
dùng để chạy nhanh 1 bộ tham số cố định lúc debug — xem mục bên dưới).

**Giai đoạn 1 — Search (Optuna):**
- Chạy `N_TRIALS` trial (mặc định 8), mỗi trial thử một bộ hyperparameter
  khác nhau: `learning_rate`, `lora_r`, `lora_dropout`, `weight_decay`
  (`lora_alpha` không search riêng, tính động bằng `LORA_ALPHA_MULT * lora_r`
  để giữ tỉ lệ scale LoRA ổn định giữa các trial).
- Mọi trial dùng **chung một subset cố định** (`HPO_SUBSET_RATIO`, mặc định
  35% của `train.jsonl`, fix seed) — không train trên toàn bộ Train ở bước
  này, để tiết kiệm thời gian GPU.
- Mỗi trial eval trên `val.jsonl` mỗi `EVAL_STEPS` step và báo cáo
  `eval_loss` cho Optuna. `SuccessiveHalvingPruner` so sánh các trial tại
  cùng mốc step và cắt sớm trial đang kém hơn hẳn — tiết kiệm GPU cho các
  trial có triển vọng hơn.
- Base model được load một lần rồi dùng lại cho `MODEL_GROUP_SIZE` trial
  liên tiếp (mỗi trial vẫn có adapter LoRA hoàn toàn riêng) trước khi giải
  phóng và load lại, để cân bằng giữa tốc độ và tính độc lập giữa các trial.
- Kết quả (best params + lịch sử toàn bộ trial) được lưu vào
  `output/best_hyperparameters.json`.

**Giai đoạn 2 — Final training:**
- Lấy bộ hyperparameter tốt nhất từ Optuna, train **đúng một lần duy nhất**
  trên toàn bộ `train.jsonl`, eval theo step + early stopping +
  `load_best_model_at_end=True`.
- Đây chính là **Final Adapter**, lưu vào `src.config.ADAPTER_DIR`
  (mặc định `output/output_model/`) — không có bước train lại nào khác sau
  đó.

Chạy:

```bash
python -m pipeline.hpo_train
```

Các tham số điều chỉnh được (xem `src/config.py` hoặc biến môi trường):

| Biến | Ý nghĩa | Mặc định |
|---|---|---|
| `MEDQUAD_N_TRIALS` | Số trial Optuna | 8 |
| `MEDQUAD_HPO_SUBSET_RATIO` | Tỉ lệ subset dùng ở giai đoạn search | 0.35 |
| `MEDQUAD_HPO_MAX_STEPS` | Trần step cho mỗi trial ở giai đoạn search | 400 |
| `MEDQUAD_MODEL_GROUP_SIZE` | Số trial dùng chung 1 lần load base model | 2 |
| `MEDQUAD_FINAL_MAX_EPOCHS` | Trần epoch cho final training (early stopping có thể cắt sớm hơn) | 3 |

---

## ⚡ Chạy nhanh không HPO (`pipeline/train.py`)

Dùng khi chỉ muốn test nhanh pipeline hoặc thử 1 bộ tham số cố định (không
chạy Optuna). Bộ tham số mặc định khai báo ngay đầu file `train.py`
(`LR`, `LORA_R`, `LORA_DROPOUT`, `WEIGHT_DECAY`) — sửa trực tiếp ở đó nếu
muốn thử tay. Script này vẫn có eval theo step + early stopping +
`load_best_model_at_end`, chỉ là không tự động dò hyperparameter.

```bash
python -m pipeline.train
```

---

## 🔍 Bật / tắt RAG

Hiện tại project **chưa kết nối với vector database** nên mặc định
`USE_RAG=False` (xem trong `src/config.py`).

Ở chế độ này, model được huấn luyện như một chatbot hỏi đáp thông thường.
`build_prompt()` sẽ không chèn phần ngữ cảnh (context) và system prompt cũng
được điều chỉnh để phù hợp với trường hợp không có tài liệu tham chiếu.

Khi đã tích hợp vector database, có thể bật RAG bằng:

```bash
MEDQUAD_USE_RAG=1 python -m pipeline.build_train_dataset
```

Khi đó:

- ✅ `build_train_dataset.py` sẽ tạo prompt có context.
- ✅ `evaluate.py` sẽ sử dụng đầy đủ các metric của RAGAs như
  `faithfulness`, `context_precision` và `context_recall`.

Nếu không bật RAG, chỉ sử dụng `answer_relevancy`.

`pipeline/chat.py` hoạt động độc lập với cờ này. Có thể truyền
`chunks=None` để thử chế độ không dùng RAG hoặc truyền danh sách context thật
sau khi đã tích hợp vector database.

---

## 🚀 Cách chạy

```bash
pip install -r requirements.txt

# 1. Đặt medquad.json vào thư mục data/

# 2. Tạo train/validation/test
python -m pipeline.build_train_dataset

# 3. Tìm hyperparameter tốt nhất + train model cuối cùng
python -m pipeline.hpo_train

#    (hoặc chạy nhanh, không HPO, để debug/thử 1 bộ tham số cố định:)
#    python -m pipeline.train

# 4. Demo chatbot
python -m pipeline.chat

# 5. Đánh giá trên test set (chỉ chạy 1 lần, sau khi đã chốt model)
python -m pipeline.evaluate
```

Nên chạy bằng:

```bash
python -m pipeline.<script_name>
```

thay vì

```bash
python pipeline/<script_name>.py
```

để Python tự nhận đúng thư mục gốc của project.

---

## ⚙️ Đổi model hoặc đường dẫn

Hầu hết các thiết lập đều nằm trong `src/config.py`.

Ngoài ra có thể ghi đè bằng biến môi trường:

```bash
MEDQUAD_BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct" python -m pipeline.hpo_train

MEDQUAD_OUTPUT_DIR="/kaggle/working/output" python -m pipeline.hpo_train
```

---

## ☁️ Chạy trên Google Colab

- 📁 Mount Google Drive.
- 📂 Di chuyển vào thư mục project (`src/`, `pipeline/`, ...).
- ▶️ Chạy các lệnh như hướng dẫn ở trên.

Phiên bản hiện tại không cần thêm `sys.path.append(...)` thủ công.

---

## 🏆 Chạy trên Kaggle

Mở notebook `kaggle/main_pipeline.ipynb`.

Có thể:

- 📦 Upload toàn bộ project thành Kaggle Dataset.
- 🔗 Hoặc clone trực tiếp repository từ GitHub.

Notebook sẽ tự cài các thư viện trong `requirements.txt` (bao gồm `optuna`)
và chạy toàn bộ pipeline, bước train gọi `pipeline/hpo_train.py`.

---

## 📝 Lưu ý

- 🤖 `pipeline/evaluate.py` sử dụng **hai model riêng biệt**.

  - **Model được đánh giá**: base model + LoRA adapter (Final Adapter, sinh
    ra từ `pipeline/hpo_train.py`), chỉ dùng để sinh câu trả lời.
  - **Model giám khảo**: `prometheus-eval/prometheus-7b-v2.0`, chỉ dùng để
    chấm điểm.

  Cách làm này giúp giảm hiện tượng **self-preference bias** (model tự chấm
  cao câu trả lời của chính mình).

- 💾 Prometheus 2 mặc định được load ở chế độ **4-bit** (`JUDGE_LOAD_IN_4BIT`)
  để phù hợp với GPU miễn phí trên Colab hoặc Kaggle. Có thể thay đổi bằng
  `MEDQUAD_JUDGE_MODEL` hoặc `MEDQUAD_JUDGE_4BIT=0`.

- 📊 Điểm số từ Prometheus chỉ mang tính tham khảo. Nếu cần đánh giá có độ tin
  cậy cao hơn, nên sử dụng các LLM mạnh hơn như GPT-4 hoặc Claude.

- 🧩 Phiên bản hiện tại chưa tích hợp vector database. Vì vậy
  `build_train_dataset.py` đang dùng câu trả lời gốc của MedQuAD làm context
  giả lập. Khi hoàn thiện RAG, chỉ cần thay phần này bằng các chunks được truy
  xuất từ vector database.

- 📁 `pipeline/hpo_train.py` lưu Optuna study vào `output/optuna/study.db`
  (SQLite). Nếu notebook/phiên chạy bị ngắt giữa chừng, chạy lại
  `python -m pipeline.hpo_train` sẽ tự tiếp tục study cũ
  (`load_if_exists=True`) thay vì mất hết các trial đã chạy.
