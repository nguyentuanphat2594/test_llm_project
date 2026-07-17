"""
evaluate.py
-----------
Đánh giá chất lượng chatbot RAG bằng RAGAs.

QUAN TRỌNG - GIÁM KHẢO PHẢI TÁCH RỜI MODEL ĐANG ĐƯỢC ĐÁNH GIÁ:
Trước đây file này dùng chính model vừa fine-tune (base + adapter LoRA) làm
luôn LLM Judge -- giống 1 học sinh tự chấm bài thi của mình: model có xu
hướng tự đánh giá cao câu trả lời của chính nó (self-preference bias), kết
quả không đáng tin.

THAY ĐỔI SO VỚI BẢN CŨ:
  - KHÔNG tự load model bị đánh giá + tự generate câu trả lời nữa. Bước sinh
    câu trả lời (+ ROUGE/BLEU) đã làm ở CUỐI pipeline/train.py rồi, kết quả
    lưu sẵn ở src.config.PREDICTIONS_CSV (question, reference, prediction,
    rouge1/2/L, bleu). File này chỉ ĐỌC csv đó lên, tránh generate 2 lần.
  - Giám khảo ƯU TIÊN gọi qua API (nhanh, không tốn VRAM) thay vì load
    Prometheus 2 (7B) cục bộ. Đặt MEDQUAD_JUDGE_API_KEY để dùng API; nếu
    không set, tự động fallback về load Prometheus cục bộ (code cũ).

Cài đặt cần thiết:
    pip install -r requirements_ver1.txt
"""

import json
import logging
import os

import pandas as pd
import torch

from datasets import Dataset
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.callbacks.base import BaseCallbackHandler
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
    AnswerRelevancy,
)
answer_relevancy_fast = AnswerRelevancy(strictness=1)

from ragas.run_config import RunConfig

from src.config import (
    EMBEDDING_MODEL_NAME,
    JUDGE_API_BASE,
    JUDGE_API_KEY,
    JUDGE_API_MODEL,
    JUDGE_LOAD_IN_4BIT,
    JUDGE_MODEL_NAME,
    PREDICTIONS_CSV,
    USE_RAG,
)

# Logger riêng để debug giám khảo (finish_reason, token dùng, lỗi API cụ thể)
# -- bật lên để biết chính xác 1 lần gọi bị lỗi là do đâu, thay vì chỉ thấy
# "LLMDidNotFinishException" chung chung từ ragas.executor.
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
judge_logger = logging.getLogger("judge_debug")


class JudgeDebugCallback(BaseCallbackHandler):
    """
    Log lại mỗi lần gọi giám khảo để dễ chẩn đoán lỗi:
      - finish_reason != "stop"  -> bị cắt vì hết max_tokens (cần tăng max_tokens)
      - lỗi API (rate limit/timeout/...) -> in message gốc từ API
    """

    def on_llm_end(self, response, **kwargs):
        try:
            gen = response.generations[0][0]
            finish_reason = None
            usage = None
            info = getattr(gen, "generation_info", None) or {}
            finish_reason = info.get("finish_reason")

            msg = getattr(gen, "message", None)
            if msg is not None and getattr(msg, "response_metadata", None):
                finish_reason = finish_reason or msg.response_metadata.get("finish_reason")
                usage = msg.response_metadata.get("token_usage")

            if finish_reason and finish_reason != "stop":
                judge_logger.warning(
                    "Giám khảo trả lời KHÔNG hoàn chỉnh -> finish_reason=%s "
                    "(nếu là 'length' thì cần tăng max_tokens). token_usage=%s",
                    finish_reason, usage,
                )
            else:
                judge_logger.info("Giám khảo trả lời OK -> finish_reason=%s, token_usage=%s",
                                   finish_reason, usage)
        except Exception as e:
            judge_logger.info("Không đọc được chi tiết response giám khảo: %s", e)

    def on_llm_error(self, error, **kwargs):
        # In nguyên lỗi gốc từ API (rate limit 429, timeout, invalid request...)
        judge_logger.error("Lỗi gọi API giám khảo: %s: %s", type(error).__name__, error)



USE_GPU = torch.cuda.is_available()


# ============================================================
# CẤU HÌNH LƯU KẾT QUẢ THEO BATCH (để không mất hết nếu API hết token/lỗi giữa chừng
# ============================================================

# Nếu đang chạy trên Colab và đã mount Drive ở /content/drive, tự động lưu
# kết quả vào đó thay vì /content (sẽ mất khi runtime bị ngắt kết nối).
_DRIVE_DIR = "/content/drive/MyDrive/medquad_eval"
if os.path.isdir("/content/drive/MyDrive"):
    os.makedirs(_DRIVE_DIR, exist_ok=True)
    RESULTS_CSV = os.path.join(_DRIVE_DIR, "ragas_scores.csv")
else:
    # Không phải Colab hoặc chưa mount Drive -> lưu cạnh PREDICTIONS_CSV như cũ
    RESULTS_CSV = os.path.join(
        os.path.dirname(PREDICTIONS_CSV) or ".",
        "ragas_scores.csv",
    )

# Số câu chấm mỗi batch trước khi ghi CSV. Để nhỏ (3-5) nếu sợ hết token
# giữa chừng và muốn lưu sát sao; để lớn hơn nếu muốn ít overhead khởi tạo.
BATCH_SIZE = int(os.environ.get("MEDQUAD_EVAL_BATCH_SIZE", "5"))

print(f"Kết quả đánh giá sẽ được lưu (append theo batch) vào: {RESULTS_CSV}")


# ============================================================
# 1. LOAD MODEL GIÁM KHẢO
#    - Có JUDGE_API_KEY -> gọi qua API (ưu tiên, nhanh, không tốn VRAM)
#    - Không có -> fallback load Prometheus 2 cục bộ (code cũ)
#    Cả 2 đường đều KHÔNG liên quan gì tới model vừa fine-tune ở train.py.
# ============================================================

def load_judge_llm():
    if JUDGE_API_KEY:
        from langchain_core.rate_limiters import InMemoryRateLimiter
        from langchain_openai import ChatOpenAI

        # Groq free tier (llama-3.1-8b-instant) giới hạn 6000 token/phút.
        # Mỗi lần gọi giám khảo tốn ~1500-2000 token (prompt + completion) ->
        # giãn request ra để không vượt ngưỡng, thay vì bắn dồn dập rồi bị 429.
        rate_limiter = InMemoryRateLimiter(
            requests_per_second=1 / 8,  # ~1 request / 8 giây (an toàn dưới 6000 TPM)
            check_every_n_seconds=0.1,
            max_bucket_size=1,           # không cho dồn nhiều request cùng lúc
        )

        # THỬ bật JSON mode -- ép model trả JSON đúng cấu trúc thay vì text tự
        # do. Giúp giảm lỗi parse với model NHỎ/YẾU (như llama-3.1-8b-instant)
        # hay bị lỗi khi RAGAs cố ép output về prompt chuẩn.
        # CẢNH BÁO: không phải model/provider nào cũng hỗ trợ đúng cách RAGAs
        # cần (RAGAs không tự thêm "return JSON" vào prompt, chỉ dựa vào model
        # tự hiểu structure) -- nếu JSON mode làm mọi câu đều lỗi (model trả
        # JSON nhưng SAI schema RAGAs mong đợi), set biến môi trường
        # MEDQUAD_JUDGE_JSON_MODE=0 để tắt, quay lại text thường.
        use_json_mode = os.environ.get("MEDQUAD_JUDGE_JSON_MODE", "1") == "1"
        model_kwargs = {}
        if use_json_mode:
            print(
                "Bật JSON mode cho giám khảo (giảm lỗi parse với model nhỏ). "
                "Nếu thấy TOÀN BỘ câu bị lỗi/NaN sau khi bật, set "
                "MEDQUAD_JUDGE_JSON_MODE=0 và chạy lại để tắt JSON mode."
            )
            model_kwargs["response_format"] = {"type": "json_object"}

        print(f"Gọi model giám khảo qua API: {JUDGE_API_MODEL} ({JUDGE_API_BASE})")
        return ChatOpenAI(
            model=JUDGE_API_MODEL,
            base_url=JUDGE_API_BASE,
            api_key=JUDGE_API_KEY,
            temperature=0,
            max_tokens=512,
            callbacks=[JudgeDebugCallback()],
            rate_limiter=rate_limiter,
            max_retries=6,    
            model_kwargs=model_kwargs,
        )

    print(
        "Không có MEDQUAD_JUDGE_API_KEY -> fallback load Prometheus 2 cục bộ "
        f"({JUDGE_MODEL_NAME}). Đặt biến môi trường này để gọi giám khảo qua "
        "API thay thế (nhanh hơn nhiều, không cần load model 7B)."
    )
    from langchain_community.llms import HuggingFacePipeline
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        pipeline,
    )

    tokenizer = AutoTokenizer.from_pretrained(JUDGE_MODEL_NAME)

    if USE_GPU and JUDGE_LOAD_IN_4BIT:
        print(f"Load {JUDGE_MODEL_NAME} ở chế độ 4-bit (giám khảo, tách biệt model đang train)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
        )
    elif USE_GPU:
        model = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        print(
            "CẢNH BÁO: không có GPU -> chạy Prometheus 2 (7B) trên CPU sẽ RẤT chậm. "
            "Cân nhắc đặt MEDQUAD_JUDGE_API_KEY để gọi qua API thay vì load cục bộ."
        )
        model = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
        )

    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,  # tắt random để LLM Judge chấm điểm ổn định hơn
    )
    return HuggingFacePipeline(pipeline=gen_pipeline)


# ============================================================
# 2. ĐỌC DỰ ĐOÁN TỪ CSV (đã sinh sẵn ở pipeline/train.py)
# ============================================================

def load_predictions():
    """
    Đọc PREDICTIONS_CSV (sinh bởi save_predictions_csv() cuối train.py).
    Cột: question, reference, prediction, rouge1, rouge2, rougeL, bleu.
    """
    if not os.path.exists(PREDICTIONS_CSV):
        raise FileNotFoundError(
            f"Không tìm thấy {PREDICTIONS_CSV}. "
            f"Hãy chạy `python -m pipeline.train` trước để sinh CSV dự đoán "
            f"(bước cuối của train.py: inference + ROUGE/BLEU trên tập test)."
        )

    df = pd.read_csv(PREDICTIONS_CSV)
    has_contexts_col = "contexts" in df.columns

    samples = []
    for _, row in df.iterrows():
        contexts = []
        if has_contexts_col:
            raw = row.get("contexts", "[]")
            try:
                contexts = json.loads(raw) if isinstance(raw, str) else []
            except (json.JSONDecodeError, TypeError):
                contexts = []

        samples.append({
            "question": row["question"],
            "answer": row["prediction"],
            "ground_truth": row["reference"],
            "contexts": contexts,
        })

    has_real_contexts = has_contexts_col and any(s["contexts"] for s in samples)

    if has_real_contexts:
        n_with_context = sum(1 for s in samples if s["contexts"])
        print(
            f"Đã đọc cột 'contexts' từ CSV: {n_with_context}/{len(samples)} mẫu có "
            f"context thật (RAG) -> faithfulness/context_precision/context_recall "
            f"có thể dùng được."
        )
    elif USE_RAG:
        print(
            "[CẢNH BÁO] USE_RAG=True nhưng CSV không có contexts thật (cột thiếu "
            "hoặc toàn bộ rỗng) -- CSV này có thể được sinh bởi bản "
            "save_predictions_csv() CŨ (chưa hỗ trợ RAG), hoặc lần chạy đó bị lỗi "
            "retrieve context. Chạy lại `run_evaluation.main()` để sinh CSV mới."
        )

    return samples, has_real_contexts


# ============================================================
# 3. CHẠY RAGAs
# ============================================================

def select_metrics(has_real_contexts: bool):
    """
    context_precision/context_recall/faithfulness cần "contexts" thật (đo
    độ bám ngữ cảnh) -- vô nghĩa khi không có contexts nào.

    Args:
        has_real_contexts: True nếu CSV thực sự có ít nhất 1 mẫu với
            contexts khác rỗng (đã kiểm tra ở load_predictions()).
    """
    if USE_RAG and has_real_contexts:
        print(
            "USE_RAG=True và CSV có contexts thật -> chấm đủ 4 metrics: "
            "faithfulness, answer_relevancy, context_precision, context_recall."
        )
        return [faithfulness, answer_relevancy, context_precision, context_recall]

    if USE_RAG and not has_real_contexts:
        print(
            "[CẢNH BÁO] USE_RAG=True nhưng CSV không có contexts thật (cột "
            "'contexts' thiếu hoặc toàn bộ rỗng) -> chỉ chấm answer_relevancy_fast, "
            "bỏ qua faithfulness/context_precision/context_recall vì không có gì "
            "để so sánh."
        )
        return [answer_relevancy_fast]

    print(
        "USE_RAG=False -> bỏ qua faithfulness/context_precision/context_recall "
        "(không dùng RAG, không có contexts). Chỉ chấm answer_relevancy_fast."
    )
    return [answer_relevancy_fast]


def main():
    print("Đang đọc CSV dự đoán (đã sinh sẵn từ bước train, gồm ROUGE/BLEU)...")
    samples, has_real_contexts = load_predictions()
    print(f"Số mẫu: {len(samples)}")

    print("Đang khởi tạo model giám khảo (tách biệt model vừa train)...")
    llm_judge = load_judge_llm()

    print("Đang load embedding model (cho Answer Relevance)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    metrics = select_metrics(has_real_contexts)

    # Nếu RESULTS_CSV đã có từ lần chạy trước (bị đứt giữa chừng), đọc lại
    # để biết những câu nào đã chấm rồi -> chỉ chấm tiếp phần còn thiếu,
    # tránh gọi API tốn token chấm lại từ đầu.
    done_questions = set()
    if os.path.exists(RESULTS_CSV):
        try:
            done_df = pd.read_csv(RESULTS_CSV)
            # ragas >=0.2 đổi tên cột "question" -> "user_input" khi xuất ra to_pandas().
            # Dò cả 2 tên để tương thích ngược, tránh lặp lại lỗi resume-sai-cột.
            question_col = "user_input" if "user_input" in done_df.columns else "question"
            done_questions = set(done_df[question_col].astype(str).tolist())
            print(
                f"Tìm thấy {len(done_questions)} câu đã chấm từ lần chạy trước "
                f"trong {RESULTS_CSV} -> bỏ qua, chỉ chấm tiếp phần còn lại."
            )
        except Exception as e:
            judge_logger.warning("Không đọc được %s cũ (%s) -> coi như chưa chấm câu nào.",
                                  RESULTS_CSV, e)

    remaining = [s for s in samples if str(s["question"]) not in done_questions]
    print(f"Còn {len(remaining)}/{len(samples)} câu cần chấm (batch size = {BATCH_SIZE}).")

    if not remaining:
        print("Không còn câu nào cần chấm -> dùng luôn kết quả đã có.")
    else:
        for i in range(0, len(remaining), BATCH_SIZE):
            batch = remaining[i:i + BATCH_SIZE]
            batch_no = i // BATCH_SIZE + 1
            print(f"--- Batch {batch_no} ({len(batch)} câu, {i + len(batch)}/{len(remaining)}) ---")

            try:
                batch_dataset = Dataset.from_list(batch)
                result = evaluate(
                    batch_dataset,
                    metrics=metrics,
                    llm=llm_judge,
                    embeddings=embeddings,
                    run_config=RunConfig(
                        timeout=7200,
                        max_workers=1,
                    ),
                    # 1 câu lỗi (timeout/format sai/...) chỉ ra NaN cho câu đó,
                    # không làm crash cả batch -> vẫn lưu được các câu còn lại.
                    raise_exceptions=False,
                )
            except Exception as e:
                judge_logger.error(
                    "Batch %d lỗi nặng (có thể do hết quota API) -> dừng lại. "
                    "Các batch trước đã lưu an toàn ở %s. Chạy lại script để chấm "
                    "tiếp phần còn thiếu. Lỗi gốc: %s: %s",
                    batch_no, RESULTS_CSV, type(e).__name__, e,
                )
                break

            batch_df = result.to_pandas()
            header = not os.path.exists(RESULTS_CSV)
            batch_df.to_csv(RESULTS_CSV, mode="a", header=header, index=False)
            print(f"Đã lưu batch {batch_no} vào {RESULTS_CSV}")

    print("=" * 50)
    print("KẾT QUẢ ĐÁNH GIÁ (RAGAs + LLM Judge)")
    print("=" * 50)
    if os.path.exists(RESULTS_CSV):
        final_df = pd.read_csv(RESULTS_CSV)
        print(f"Tổng số câu đã chấm: {len(final_df)}")
        score_cols = [c for c in final_df.columns
                      if c not in ("question", "answer", "ground_truth", "contexts",
                                   "user_input", "response", "retrieved_contexts")]

        # ---- RÀO TRƯỚC: cảnh báo rõ nếu có câu bị NaN (không chấm được) ----
        # NaN thường do: model trả sai format (JSON mode không hợp), bị cắt
        # giữa chừng (max_tokens thấp), hoặc lỗi API tạm thời không retry nổi.
        for col in score_cols:
            if col not in final_df.columns:
                continue
            n_nan = final_df[col].isna().sum()
            n_total = len(final_df)
            if n_nan > 0:
                pct = 100 * n_nan / n_total
                print(
                    f"[CẢNH BÁO] Cột '{col}': {n_nan}/{n_total} câu ({pct:.1f}%) "
                    f"bị NaN -- không chấm điểm được."
                )
                if pct > 50:
                    print(
                        f"  -> Tỷ lệ lỗi RẤT CAO (>50%). Nhiều khả năng do JSON mode "
                        f"không hợp với model/provider này. Thử set biến môi trường "
                        f"MEDQUAD_JUDGE_JSON_MODE=0 rồi chạy lại (xem log "
                        f"'judge_debug' phía trên để biết finish_reason/lỗi cụ thể "
                        f"của từng lần gọi bị fail)."
                    )

        print(final_df[score_cols].mean(numeric_only=True))
        print(f"\nChi tiết đầy đủ: {RESULTS_CSV}")
    else:
        print("Chưa có kết quả nào được lưu.")


if __name__ == "__main__":
    main()