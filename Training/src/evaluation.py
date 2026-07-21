"""
evaluation.py — Đánh giá model: ROUGE, BLEU, Perplexity, và xuất CSV kết quả.

Sử dụng thư viện evaluate của HuggingFace cho các metrics chuẩn.
Hàm save_predictions_csv() xuất kết quả inference cho LLM-as-a-judge.

ĐÃ SỬA so với bản gốc:
  - Ghi CSV liên tục mỗi `save_every` mẫu (không đợi generate hết mới ghi
    1 lần) -- nếu Kaggle/session bị ngắt giữa chừng, tiến độ đã làm không
    bị mất.
  - Hỗ trợ resume=True: đọc CSV cũ (nếu có), bỏ qua các câu hỏi đã có sẵn
    prediction, chỉ generate tiếp phần còn thiếu.
    CẢNH BÁO: resume chỉ an toàn khi CSV cũ là của ĐÚNG model hiện tại (vd
    lần chạy trước bị ngắt giữa chừng). Nếu bạn vừa train lại model khác,
    PHẢI để resume=False (mặc định) để tránh CSV bị lẫn prediction của 2
    model khác nhau (model cũ ở các câu đầu, model mới ở các câu sau).
"""

import json
import math
import os
from typing import Dict

import evaluate
import nltk
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import PreTrainedTokenizer

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

_rouge_metric = evaluate.load("rouge")
_bleu_metric = evaluate.load("bleu")


def compute_perplexity(eval_loss: float) -> float:
    return math.exp(eval_loss) if eval_loss < 100 else float("inf")


def build_compute_metrics(tokenizer: PreTrainedTokenizer):
    def compute_metrics(eval_preds) -> Dict[str, float]:
        logits, labels = eval_preds

        if isinstance(logits, tuple):
            logits = logits[0]

        predictions = np.argmax(logits, axis=-1)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        predictions = np.where(labels != -100, predictions, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [label.strip() for label in decoded_labels]

        valid_pairs = [
            (p, l) for p, l in zip(decoded_preds, decoded_labels)
            if p and l
        ]
        if not valid_pairs:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "bleu": 0.0}

        valid_preds, valid_labels = zip(*valid_pairs)

        rouge_results = _rouge_metric.compute(
            predictions=list(valid_preds),
            references=list(valid_labels),
            use_stemmer=True,
        )

        bleu_preds = [nltk.word_tokenize(p) for p in valid_preds]
        bleu_refs = [[nltk.word_tokenize(r)] for r in valid_labels]

        try:
            bleu_result = _bleu_metric.compute(
                predictions=bleu_preds,
                references=bleu_refs,
            )
            bleu_score = bleu_result["bleu"]
        except (ZeroDivisionError, ValueError):
            bleu_score = 0.0

        return {
            "rouge1": rouge_results["rouge1"],
            "rouge2": rouge_results["rouge2"],
            "rougeL": rouge_results["rougeL"],
            "bleu": bleu_score,
        }

    return compute_metrics


def save_predictions_csv(
    model,
    tokenizer: PreTrainedTokenizer,
    dataset: Dataset,
    output_path: str,
    system_prompt: str,
    prompt_style: str = "alpaca",
    max_new_tokens: int = 256,
    max_samples: int = 1700,
    resume: bool = False,
    save_every: int = 5,
    retrieve_context_fn=None,
    rag_top_k: int = 3,
    rag_similarity_threshold: float = None,
    rag_relative_threshold: float = 0.5,
):
    """
    Chạy inference trên tập test và lưu kết quả ra CSV.

    File CSV: question, reference, prediction, rouge1, rouge2, rougeL, bleu
    (+ "contexts" nếu retrieve_context_fn được truyền vào).
    Sẵn sàng cho LLM-as-a-judge pipeline sau này.

    Args:
        model: Model đã train (PEFT wrapped)
        tokenizer: Tokenizer
        dataset: Tập test (HuggingFace Dataset)
        output_path: Đường dẫn file CSV đầu ra
        system_prompt: System prompt cho model
        prompt_style: "alpaca" hoặc "chatml"
        max_new_tokens: Số token tối đa sinh ra
        max_samples: Giới hạn số mẫu inference (tránh tốn thời gian)
        resume: True -> đọc CSV cũ tại output_path (nếu có), bỏ qua các câu
            hỏi đã có sẵn, chỉ generate tiếp phần còn thiếu. CHỈ dùng khi
            chắc chắn CSV cũ là của đúng model hiện tại (lần chạy trước bị
            ngắt giữa chừng). Mặc định False -> luôn ghi đè từ đầu.
        save_every: Ghi CSV ra đĩa sau mỗi bấy nhiêu mẫu mới (không đợi
            generate hết mới ghi 1 lần) -- giảm rủi ro mất tiến độ nếu bị
            ngắt giữa chừng (Kaggle hết giờ, mất kết nối, v.v.)
        retrieve_context_fn: hàm nhận (question, top_k, similarity_threshold)
            -> dict {"used_contexts", "raw_contexts", "scores",
            "similarity_pct", "rag_used", "retrieval_mode"} (xem
            src.rag_bridge.get_context_with_similarity). Nếu truyền vào
            (khi USE_RAG=True), mỗi câu hỏi sẽ được retrieve + lọc context
            theo độ tương đồng TRƯỚC khi build prompt -- chỉ context ĐỦ
            liên quan mới được đưa vào model, tránh model học/trả lời theo
            context sai chủ đề. CSV sẽ lưu thêm cột "rag_used",
            "rag_similarity_pct", "rag_raw_contexts" để biết câu nào có
            dùng RAG, độ tương đồng bao nhiêu, và context thô lấy được là
            gì (kể cả bị loại). Nếu để None (mặc định), giữ nguyên hành vi
            cũ: prompt Q&A thuần, không có ngữ cảnh.
        rag_top_k: số chunks lấy về mỗi câu hỏi khi retrieve_context_fn
            được dùng.
        rag_similarity_threshold: ngưỡng % tương đồng (0..1) để CHẤP NHẬN
            context -- CHỈ dùng khi RETRIEVAL_MODE="cosine". None -> dùng
            SIMILARITY_THRESHOLD mặc định của RAG project.
        rag_relative_threshold: ngưỡng % TƯƠNG ĐỐI (0..1, mặc định 0.5) --
            CHỈ dùng khi RETRIEVAL_MODE="bm25" hoặc "hybrid". Tự động so
            điểm mỗi context với điểm CAO NHẤT trong top-k của CHÍNH câu
            hỏi đó -- không cần tự đoán ngưỡng tuyệt đối.
    """
    model.eval()
    num_samples = min(len(dataset), max_samples)

    results = []
    done_questions = set()
    if resume and os.path.exists(output_path):
        old_df = pd.read_csv(output_path)
        results = old_df.to_dict("records")
        done_questions = set(old_df["question"].astype(str))
        print(
            f"[EVAL] Resume: đã có {len(results)} mẫu trong {output_path}, "
            f"sẽ bỏ qua các câu này và chỉ generate phần còn thiếu."
        )
    else:
        print(f"[EVAL] Bắt đầu MỚI -- sẽ ghi đè {output_path} nếu đã tồn tại.")

    print(f"[EVAL] Mục tiêu: {num_samples} mẫu.")

    new_count = 0
    for i in range(num_samples):
        example = dataset[i]
        question = example["question"]
        reference = example["answer"]

        if str(question) in done_questions:
            continue

        # ---- RAG: retrieve context THẬT + lọc theo % tương đồng ----
        used_contexts = []
        raw_contexts = []
        similarity_pct_list = []
        relative_pct_list = []
        rag_used = False
        if retrieve_context_fn is not None:
            try:
                rag_result = retrieve_context_fn(
                    question, top_k=rag_top_k,
                    similarity_threshold=rag_similarity_threshold,
                    relative_threshold=rag_relative_threshold,
                )
                used_contexts = rag_result["used_contexts"]
                raw_contexts = rag_result["raw_contexts"]
                similarity_pct_list = rag_result["similarity_pct"]
                relative_pct_list = rag_result["relative_pct"]
                rag_used = rag_result["rag_used"]
            except Exception as e:
                print(f"[CẢNH BÁO][RAG] Retrieve context lỗi cho câu '{question[:50]}...': {e} -> dùng prompt không context.")

        context_block = ""
        if used_contexts:
            joined_context = "\n\n".join(used_contexts)
            context_block = f"Context:\n{joined_context}\n\n"

        if prompt_style == "chatml":
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{context_block}{question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
        else:
            prompt = (
                f"### Instruction:\n{system_prompt}\n\n"
                f"### Input:\n{context_block}{question}\n\n"
                f"### Response:\n"
            )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        rouge_scores = _rouge_metric.compute(
            predictions=[prediction],
            references=[reference],
            use_stemmer=True,
        )

        pred_tokens = nltk.word_tokenize(prediction) if prediction else []
        ref_tokens = [nltk.word_tokenize(reference)]
        try:
            bleu_score = _bleu_metric.compute(
                predictions=[pred_tokens],
                references=[ref_tokens],
            )["bleu"]
        except (ZeroDivisionError, ValueError):
            bleu_score = 0.0

        results.append({
            "question": question,
            "reference": reference,
            "prediction": prediction,
            # Context THẬT SỰ được đưa vào prompt (đã lọc theo threshold).
            # Rỗng "[]" nếu không dùng RAG hoặc không có context nào đạt
            # threshold -- evaluate.py dùng cột này cho faithfulness/
            # context_precision/context_recall.
            "contexts": json.dumps(used_contexts, ensure_ascii=False),
            # True nếu câu này thực sự có dùng context (đạt threshold).
            # False -> model trả lời KHÔNG có ngữ cảnh (dù có retrieve
            # được gì đó, nhưng bị loại vì không đủ liên quan).
            "rag_used": rag_used,
            # % tương đồng của từng context retrieve được (song song với
            # rag_raw_contexts theo thứ tự) -- None nếu RETRIEVAL_MODE
            # không phải "cosine" (không quy đổi được thang đo).
            "rag_similarity_pct": json.dumps(similarity_pct_list),
            # % TƯƠNG ĐỐI (so với context tốt nhất trong CHÍNH câu hỏi này)
            # -- chỉ có giá trị khi RETRIEVAL_MODE="bm25"/"hybrid".
            "rag_relative_pct": json.dumps(relative_pct_list),
            # TOÀN BỘ context retrieve được (kể cả bị loại vì không đủ
            # tương đồng) -- để xem/debug tại sao 1 câu không dùng RAG.
            "rag_raw_contexts": json.dumps(raw_contexts, ensure_ascii=False),
            "rouge1": round(rouge_scores["rouge1"], 4),
            "rouge2": round(rouge_scores["rouge2"], 4),
            "rougeL": round(rouge_scores["rougeL"], 4),
            "bleu": round(bleu_score, 4),
        })
        done_questions.add(str(question))
        new_count += 1

        if new_count % save_every == 0:
            pd.DataFrame(results).to_csv(output_path, index=False, encoding="utf-8")
            print(f"[EVAL] Đã inference {len(results)}/{num_samples} mẫu... (đã lưu CSV)")

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(f"[KẾT QUẢ] Đã lưu {len(results)} mẫu → {output_path}")
    print(f"  ROUGE-1 (avg): {df['rouge1'].mean():.4f}")
    print(f"  ROUGE-2 (avg): {df['rouge2'].mean():.4f}")
    print(f"  ROUGE-L (avg): {df['rougeL'].mean():.4f}")
    print(f"  BLEU    (avg): {df['bleu'].mean():.4f}")
    if retrieve_context_fn is not None:
        n_used = int(df["rag_used"].sum())
        pct_used = 100 * n_used / len(df) if len(df) > 0 else 0
        print(f"  RAG được dùng: {n_used}/{len(df)} câu ({pct_used:.1f}%) đạt ngưỡng tương đồng.")
    print(f"{'=' * 60}")