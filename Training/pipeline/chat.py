"""
File dùng để CHAT THẬT với model đã train (sau khi có output_model/) --
người dùng nhập câu hỏi trực tiếp, model trả lời.

Nếu muốn dùng RAG, chunks liên quan sẽ được retrieve thật từ vector DB
(qua rag_bridge), tính % tương đồng, và chỉ chunks đủ liên quan mới được
đưa vào prompt cho model.

Điểm khác biệt quan trọng: việc bật/tắt RAG ở đây ĐỘC LẬP với cờ USE_RAG
dùng cho train/evaluate -- có thể tự chọn riêng cho từng câu hỏi ngay
trong lúc chat, không cần đổi biến môi trường hay restart.

Cách chạy:
    python -m pipeline.chat          # vòng lặp hỏi-đáp thật
    python -m pipeline.chat --demo   # demo nhanh 1 câu cố định

Trong lúc chat, gõ:
    /rag on   -> bật RAG cho các câu hỏi tiếp theo
    /rag off  -> tắt RAG (Q&A thuần)
    exit/quit -> thoát
"""

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.config import ADAPTER_DIR, BASE_MODEL_NAME, MAX_NEW_TOKENS_CHAT, USE_RAG
from src.prompt_template import build_prompt

USE_GPU = torch.cuda.is_available()


# ============================================================
# LOAD MODEL (base + adapter) - chỉ cần load 1 LẦN lúc khởi động chatbot,
# không load lại mỗi lần user hỏi (rất tốn thời gian nếu load lại)
# ============================================================

def load_chat_model():
    print("Đang load model...")

    tokenizer = AutoTokenizer.from_pretrained(str(ADAPTER_DIR))

    if USE_GPU:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
        )

    # Gắn adapter (kiến thức đã train) vào model gốc
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()

    print("Load model xong.")
    return model, tokenizer


# ============================================================
# LẤY CONTEXT (chunks liên quan) CHO CÂU HỎI -- CÓ LỌC THEO % TƯƠNG ĐỒNG
# ============================================================

def get_chunks_for_question(question: str, top_k: int = 3,
                             similarity_threshold: float = None,
                             verbose: bool = True,
                             use_rag: bool = None) -> list[str]:
    """
    use_rag: ghi đè cờ USE_RAG toàn cục CHỈ cho lần gọi này -- để chat có
        thể tự chọn dùng RAG hay không, ĐỘC LẬP với cờ USE_RAG chung dùng
        cho train/evaluate (không cần đổi biến môi trường/reload gì cả).
        - None (mặc định) -> dùng theo USE_RAG chung (src.config).
        - True  -> LUÔN thử RAG cho câu này, bất kể USE_RAG chung là gì.
        - False -> LUÔN bỏ qua RAG cho câu này, trả lời Q&A thuần.
    """
    effective_use_rag = USE_RAG if use_rag is None else use_rag
    if not effective_use_rag:
        return []

    from src.rag_bridge import get_context_with_similarity
    try:
        result = get_context_with_similarity(
            question, top_k=top_k, similarity_threshold=similarity_threshold,
        )
    except Exception as e:
        print(f"[CẢNH BÁO] Retrieve context thất bại ({e}) -> trả lời không có ngữ cảnh.")
        return []

    if verbose:
        print(f"\n--- RAG cho câu hỏi này (mode={result['retrieval_mode']}) ---")
        for i, (raw, pct) in enumerate(zip(result["raw_contexts"], result["similarity_pct"]), 1):
            pct_str = f"{pct:.1f}%" if pct is not None else "N/A (mode không quy đổi được % tương đồng)"
            preview = raw[:150].replace("\n", " ")
            print(f"  [{i}] Tương đồng: {pct_str} | {preview}...")

        if result["rag_used"]:
            print(f"  -> DÙNG RAG: {len(result['used_contexts'])} context đạt ngưỡng, đưa vào prompt.")
        else:
            print("  -> KHÔNG DÙNG RAG: không có context nào đạt ngưỡng tương đồng, trả lời Q&A thuần.")
        print("-" * 50)

    return result["used_contexts"]


# ============================================================
# SINH CÂU TRẢ LỜI
# ============================================================

def generate_answer(model, tokenizer, chunks: list[str], question: str,
                     max_new_tokens: int = MAX_NEW_TOKENS_CHAT) -> str:
    """
    Args:
        chunks: danh sách đoạn văn bản liên quan (đã lọc, sẵn sàng đưa vào prompt)
        question: câu hỏi của user

    Returns:
        Câu trả lời dạng string
    """
    messages = build_prompt(chunks, question)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Chỉ lấy phần token MỚI được sinh ra (bỏ phần prompt input đi)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return answer.strip()


# ============================================================
# HỎI 1 CÂU (dùng cho cả demo và vòng lặp chat thật)
# ============================================================

def ask(model, tokenizer, question: str, top_k: int = 3,
        similarity_threshold: float = None, verbose: bool = True,
        use_rag: bool = None) -> str:
    chunks = get_chunks_for_question(
        question, top_k=top_k,
        similarity_threshold=similarity_threshold, verbose=verbose,
        use_rag=use_rag,
    )
    return generate_answer(model, tokenizer, chunks, question)


# ============================================================
# VÒNG LẶP HỎI-ĐÁP THẬT (người dùng tự nhập câu hỏi)
# ============================================================

def chat_loop():
    model, tokenizer = load_chat_model()

    # use_rag=None -> mặc định theo USE_RAG chung (src.config, đồng bộ với
    # lúc train/evaluate). Người dùng có thể tự đổi bằng lệnh /rag on|off
    # ngay trong vòng lặp, ĐỘC LẬP với cờ chung, không cần restart/reload.
    session_use_rag = USE_RAG

    print("\n" + "=" * 50)
    print(f"CHATBOT SẴN SÀNG (mặc định USE_RAG={session_use_rag})")
    print("Lệnh: '/rag on' bật RAG | '/rag off' tắt RAG | 'exit'/'quit' thoát")
    print("=" * 50)

    while True:
        try:
            question = input("\nCâu hỏi của bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThoát chatbot.")
            break

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Thoát chatbot.")
            break
        if question.lower() == "/rag on":
            session_use_rag = True
            print("[OK] Đã BẬT RAG cho các câu hỏi tiếp theo.")
            continue
        if question.lower() == "/rag off":
            session_use_rag = False
            print("[OK] Đã TẮT RAG cho các câu hỏi tiếp theo (Q&A thuần).")
            continue

        answer = ask(model, tokenizer, question, use_rag=session_use_rag)
        print("\nTRẢ LỜI:", answer)


# ============================================================
# DEMO CHẠY NHANH (1 câu cố định, không cần nhập tay)
# ============================================================

def demo():
    model, tokenizer = load_chat_model()

    demo_question = "Triệu chứng của bệnh bạch cầu cấp ở người lớn là gì?"

    print("\n" + "=" * 50)
    print("CÂU HỎI:", demo_question)
    print(f"USE_RAG = {USE_RAG}")
    print("=" * 50)

    if USE_RAG:
        answer = ask(model, tokenizer, demo_question)
    else:
        # USE_RAG=False -> giữ demo_chunks giả lập để vẫn xem được ví dụ
        # có context hoạt động ra sao.
        demo_chunks = [
            "Signs and symptoms of adult ALL include fever, feeling tired, "
            "and easy bruising or bleeding. Check with your doctor if you have "
            "weakness, night sweats, easy bruising, petechiae, shortness of breath, "
            "weight loss, bone or stomach pain, or painless lumps."
        ]
        answer = generate_answer(model, tokenizer, demo_chunks, demo_question)

    print("TRẢ LỜI:", answer)
    print("=" * 50)


def main():
    if "--demo" in sys.argv:
        demo()
    else:
        chat_loop()


if __name__ == "__main__":
    main()