"""
chat.py
-------
File dùng lúc CHẠY THẬT (inference) - sau khi đã train xong (có output_model/).

Luồng:
  chunks (từ vector DB) + question
        -> build_prompt() (từ src/prompt_template.py)
        -> model đã train (base model + adapter LoRA)
        -> câu trả lời

Cách chạy thử (demo, chưa nối vector DB thật):
    python -m pipeline.chat
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.config import ADAPTER_DIR, BASE_MODEL_NAME, MAX_NEW_TOKENS_CHAT
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
            dtype=torch.float16,
            device_map="auto",
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            dtype=torch.float32,
            device_map={"": "cpu"},
        )

    # Gắn adapter (kiến thức đã train) vào model gốc
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()

    print("Load model xong.")
    return model, tokenizer


# ============================================================
# SINH CÂU TRẢ LỜI
# ============================================================

def generate_answer(model, tokenizer, chunks: list[str], question: str,
                     max_new_tokens: int = MAX_NEW_TOKENS_CHAT) -> str:
    """
    Args:
        chunks: danh sách đoạn văn bản liên quan (do vector DB trả về)
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
# DEMO CHẠY THỬ (giả lập chunks, vì chưa nối vector DB thật)
# ============================================================

def main():
    model, tokenizer = load_chat_model()

    # Đây là chunks GIẢ LẬP -- lúc chạy thật, phần này sẽ do vector DB
    # của đồng nghiệp bạn trả về, không phải viết tay như thế này.
    demo_chunks = [
        "Signs and symptoms of adult ALL include fever, feeling tired, "
        "and easy bruising or bleeding. Check with your doctor if you have "
        "weakness, night sweats, easy bruising, petechiae, shortness of breath, "
        "weight loss, bone or stomach pain, or painless lumps."
    ]
    demo_question = "Triệu chứng của bệnh bạch cầu cấp ở người lớn là gì?"

    print("\n" + "=" * 50)
    print("CÂU HỎI:", demo_question)
    print("=" * 50)

    answer = generate_answer(model, tokenizer, demo_chunks, demo_question)

    print("TRẢ LỜI:", answer)
    print("=" * 50)


if __name__ == "__main__":
    main()
