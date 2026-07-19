"""Cầu nối giữa project chính (MedQuAD) và project RAG riêng (Embedding_RAG
của thành viên khác) để lấy context thật từ vector DB.

File này làm 2 việc:
1. Import an toàn retrieval.py từ RAG project mà không bị đụng độ tên
   module (cả 2 project đều có file config.py riêng).
2. Retrieve context cho 1 câu hỏi, tính % tương đồng, và LỌC BỎ context
   không đủ liên quan trước khi trả về -- tránh model bị "dắt mũi" bởi
   context sai chủ đề.

Cấu hình qua biến môi trường:
    MEDQUAD_RAG_DIR -- đường dẫn tới thư mục model RAG muốn dùng
        (vd .../Embbeding_RAG/nomic-embed-text-v1.5)

Hàm chính dùng ở nơi khác:
    get_context_with_similarity(question, top_k, similarity_threshold)
        -> dict gồm used_contexts (context đạt ngưỡng, dùng để đưa vào
        prompt), raw_contexts (tất cả context lấy được, kể cả bị loại),
        similarity_pct (% tương đồng), rag_used (có dùng RAG cho câu này
        không).

Lưu ý: % tương đồng chỉ tính đúng khi RAG project dùng RETRIEVAL_MODE=
"cosine". Với "bm25"/"hybrid", không lọc được theo threshold (điểm không
cùng thang đo), sẽ dùng hết context lấy được.
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
) -> dict:
    """
    Retrieve context + tính % tương đồng, LỌC BỎ context không đủ liên quan
    (chỉ áp dụng đúng ý nghĩa khi RETRIEVAL_MODE="cosine").

    Returns dict:
        {
            "used_contexts": List[str]   -- context ĐỦ liên quan, dùng để đưa
                                             vào prompt cho model (rỗng nếu
                                             không có context nào đạt threshold)
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
    threshold = (
        similarity_threshold
        if similarity_threshold is not None
        else getattr(rag_config, "SIMILARITY_THRESHOLD", 0.70)
    )

    results = retrieve_context(question, top_k=top_k)
    raw_contexts = [r["text_content"] for r in results]
    scores = [r["score"] for r in results]

    if retrieval_mode == "cosine":
        # Chroma cosine distance -> similarity = 1 - distance, clip về 0..1
        similarity_pct = [max(0.0, min(1.0, 1.0 - s)) * 100 for s in scores]
        used_contexts = [
            ctx for ctx, pct in zip(raw_contexts, similarity_pct)
            if pct >= threshold * 100
        ]
    else:
        # bm25/hybrid: score KHÔNG cùng thang đo % tương đồng -- không lọc
        # được công bằng theo threshold. Dùng tất cả context lấy về, để
        # nguyên "similarity_pct" là None để người dùng biết không so được.
        if not _warned_non_cosine:
            print(
                f"[rag_bridge][CẢNH BÁO] RETRIEVAL_MODE='{retrieval_mode}' -- "
                f"score không quy đổi được thành % tương đồng chuẩn, nên KHÔNG "
                f"lọc theo SIMILARITY_THRESHOLD. Toàn bộ context retrieve được "
                f"sẽ được dùng. Nếu muốn lọc theo threshold đúng nghĩa, đổi "
                f"RETRIEVAL_MODE='cosine' trong config.py của RAG project."
            )
            _warned_non_cosine = True
        similarity_pct = [None] * len(raw_contexts)
        used_contexts = list(raw_contexts)

    return {
        "used_contexts": used_contexts,
        "raw_contexts": raw_contexts,
        "scores": scores,
        "similarity_pct": similarity_pct,
        "rag_used": len(used_contexts) > 0,
        "retrieval_mode": retrieval_mode,
    }