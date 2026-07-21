"""
src/rag_bridge.py
------------------
Cầu nối gọi retrieve_context() từ project RAG riêng (Embedding_RAG/ của
thành viên khác) mà KHÔNG bị đụng độ tên module.

LỌC CONTEXT THEO ĐỘ TƯƠNG ĐỒNG:
  RAG project trả về "score" khác Ý NGHĨA tuỳ RETRIEVAL_MODE:
    - "cosine": score là DISTANCE (Chroma), càng THẤP càng giống.
      -> similarity = 1 - distance (quy về khoảng ~0..1) -> có % chuẩn,
      dùng similarity_threshold (0..1, vd 0.70 = 70%).
    - "bm25": score là điểm BM25 thô, KHÔNG có thang chuẩn 0..1 -- KHÔNG
      quy đổi được thành "% tương đồng". Chỉ lọc được bằng NGƯỠNG ĐIỂM
      THÔ (raw_score_threshold) do người dùng tự chọn sau khi quan sát
      thực tế (không có ý nghĩa toán học rõ ràng như %).
    - "hybrid": score là điểm RRF (reciprocal rank fusion), cũng KHÔNG có
      thang chuẩn -- tương tự bm25, lọc bằng raw_score_threshold riêng.

  Vì vậy hàm get_context_with_similarity() nhận 2 tham số ngưỡng RIÊNG:
    - similarity_threshold: dùng khi mode="cosine" (0..1).
    - raw_score_threshold: dùng khi mode="bm25" hoặc "hybrid" (điểm thô,
      KHÔNG giới hạn 0..1 -- tự chọn số sau khi thử nghiệm quan sát điểm
      thực tế của vài chục câu hỏi mẫu). Nếu để None (mặc định), KHÔNG
      lọc gì cả (giữ hành vi cũ: dùng hết context lấy về).
"""

import os
import sys
import importlib
from pathlib import Path

from src.config import BASE_DIR

_DEFAULT_RAG_DIR = BASE_DIR.parent / "Embbeding_RAG" / "nomic-embed-text-v1.5"
RAG_PROJECT_DIR = Path(os.environ.get("MEDQUAD_RAG_DIR", str(_DEFAULT_RAG_DIR))).resolve()

_cached_retrieve_fn = None
_cached_rag_config = None
_warned_non_cosine = False


def _load_retrieve_context_fn():
    """Import retrieval.py + config.py từ RAG project 1 LẦN DUY NHẤT, cách
    ly sys.path an toàn (thêm rồi xoá ngay). Trả về (retrieve_context_fn,
    rag_config_module)."""
    global _cached_retrieve_fn, _cached_rag_config
    if _cached_retrieve_fn is not None:
        return _cached_retrieve_fn, _cached_rag_config

    test_vector_db_dir = RAG_PROJECT_DIR / "test_vector_db"
    if not test_vector_db_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {test_vector_db_dir}. "
            f"Kiểm tra lại biến môi trường MEDQUAD_RAG_DIR có trỏ đúng vào "
            f"thư mục model RAG (vd .../Embbeding_RAG/nomic-embed-text-v1.5) không."
        )

    paths_to_add = [str(test_vector_db_dir)]
    inserted = []
    try:
        for stale in ("config", "retrieval"):
            sys.modules.pop(stale, None)

        for p in paths_to_add:
            if p not in sys.path:
                sys.path.insert(0, p)
                inserted.append(p)

        # retrieval.py tự thêm parent dir (RAG_PROJECT_DIR) vào sys.path và
        # `import config` bên trong nó -- không cần mình làm thay.
        retrieval_module = importlib.import_module("retrieval")
        _cached_retrieve_fn = retrieval_module.retrieve_context
        # config.py của RAG project (đã bị retrieval.py import ở trên rồi,
        # nên giờ chỉ cần lấy lại từ sys.modules, KHÔNG import lại lần nữa).
        _cached_rag_config = sys.modules.get("config")

        print(
            f"[rag_bridge] Đã load retrieval module từ: {RAG_PROJECT_DIR}\n"
            f"[rag_bridge] RETRIEVAL_MODE = {getattr(_cached_rag_config, 'RETRIEVAL_MODE', '?')} | "
            f"SIMILARITY_THRESHOLD = {getattr(_cached_rag_config, 'SIMILARITY_THRESHOLD', '?')}"
        )
    finally:
        for p in inserted:
            if p in sys.path:
                sys.path.remove(p)

    return _cached_retrieve_fn, _cached_rag_config


def get_context_texts(question: str, top_k: int = 3) -> list[str]:
    """
    [GIỮ LẠI ĐỂ TƯƠNG THÍCH NGƯỢC -- không lọc theo similarity]
    Trả về danh sách text_content của top_k chunks, KHÔNG áp threshold.
    Dùng get_context_with_similarity() nếu muốn lọc context không liên quan.
    """
    retrieve_context, _ = _load_retrieve_context_fn()
    results = retrieve_context(question, top_k=top_k)
    return [r["text_content"] for r in results]


def get_context_with_similarity(
    question: str,
    top_k: int = 3,
    similarity_threshold: float = None,
    raw_score_threshold: float = None,
) -> dict:
    """
    Retrieve context + lọc theo ngưỡng, tuỳ RETRIEVAL_MODE dùng ngưỡng khác
    nhau (xem giải thích ở đầu file).

    Args:
        similarity_threshold: ngưỡng % (0..1) -- CHỈ dùng khi mode="cosine".
            None -> lấy SIMILARITY_THRESHOLD mặc định của RAG project.
        raw_score_threshold: ngưỡng điểm THÔ -- CHỈ dùng khi mode="bm25"
            hoặc "hybrid" (điểm càng cao càng liên quan ở cả 2 mode này).
            None (mặc định) -> KHÔNG lọc gì, dùng hết context lấy về (giữ
            hành vi cũ, an toàn ngược).

    Returns dict:
        {
            "used_contexts": List[str]   -- context ĐỦ liên quan, dùng để đưa
                                             vào prompt cho model (rỗng nếu
                                             không có context nào đạt ngưỡng)
            "raw_contexts": List[str]    -- TẤT CẢ context retrieve được (kể cả
                                             bị loại), để xem/debug
            "scores": List[float]        -- điểm thô tương ứng raw_contexts
            "similarity_pct": List[float or None] -- % tương đồng (chỉ có giá
                                             trị khi mode="cosine", None nếu
                                             mode khác không quy đổi được)
            "rag_used": bool             -- có context nào được dùng không
            "retrieval_mode": str
        }
    """
    global _warned_non_cosine

    retrieve_context, rag_config = _load_retrieve_context_fn()
    retrieval_mode = getattr(rag_config, "RETRIEVAL_MODE", "cosine")

    results = retrieve_context(question, top_k=top_k)
    raw_contexts = [r["text_content"] for r in results]
    scores = [r["score"] for r in results]

    if retrieval_mode == "cosine":
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else getattr(rag_config, "SIMILARITY_THRESHOLD", 0.70)
        )
        # Chroma cosine distance -> similarity = 1 - distance, clip về 0..1
        similarity_pct = [max(0.0, min(1.0, 1.0 - s)) * 100 for s in scores]
        used_contexts = [
            ctx for ctx, pct in zip(raw_contexts, similarity_pct)
            if pct >= threshold * 100
        ]
    else:
        # bm25/hybrid: score KHÔNG có thang chuẩn 0..1, không quy đổi được
        # thành % -- chỉ lọc được bằng NGƯỠNG ĐIỂM THÔ (raw_score_threshold)
        # do người dùng tự cung cấp, điểm càng CAO càng liên quan ở cả 2 mode.
        similarity_pct = [None] * len(raw_contexts)

        if raw_score_threshold is not None:
            used_contexts = [
                ctx for ctx, s in zip(raw_contexts, scores)
                if s >= raw_score_threshold
            ]
        else:
            if not _warned_non_cosine:
                print(
                    f"[rag_bridge][CẢNH BÁO] RETRIEVAL_MODE='{retrieval_mode}' -- "
                    f"score không quy đổi được thành % tương đồng chuẩn. Vì "
                    f"raw_score_threshold=None nên KHÔNG lọc gì, dùng hết context "
                    f"retrieve được. Muốn lọc, truyền raw_score_threshold=<số> "
                    f"(tự chọn sau khi quan sát điểm thực tế của vài chục câu "
                    f"hỏi mẫu), hoặc đổi RETRIEVAL_MODE='cosine' để dùng % chuẩn."
                )
                _warned_non_cosine = True
            used_contexts = list(raw_contexts)

    return {
        "used_contexts": used_contexts,
        "raw_contexts": raw_contexts,
        "scores": scores,
        "similarity_pct": similarity_pct,
        "rag_used": len(used_contexts) > 0,
        "retrieval_mode": retrieval_mode,
    }