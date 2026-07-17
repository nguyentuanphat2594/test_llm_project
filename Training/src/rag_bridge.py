"""
src/rag_bridge.py
------------------
Cầu nối gọi retrieve_context() từ project RAG riêng (Embedding_RAG/ của
thành viên khác) mà KHÔNG bị đụng độ tên module.

VẤN ĐỀ CẦN TRÁNH:
  Project RAG có file `config.py` VÀ project chính (src.config) cũng có
  khái niệm "config". Nếu import ẩu (thêm thẳng thư mục RAG vào sys.path
  toàn cục), Python có thể nhầm lẫn giữa 2 module "config" khác nhau khi
  bị import bằng tên trần `import config` (retrieval.py của RAG project
  tự làm việc này) -- dẫn tới lỗi khó debug (sai đường dẫn, sai model...).

CÁCH XỬ LÝ:
  - Chỉ thêm đường dẫn RAG project vào sys.path NGAY TRƯỚC khi import,
    xoá lại NGAY SAU khi import xong.
  - Cache lại hàm retrieve_context sau lần import đầu tiên (không import
    lại mỗi lần gọi -- vừa chậm vừa dễ dính lại vấn đề trên).
  - GIỚI HẠN: vì cache theo tiến trình (process), nếu muốn đổi sang model
    RAG khác (vd từ nomic sang bge) giữa chừng, phải RESTART kernel/session
    rồi set lại MEDQUAD_RAG_DIR, không thể đổi "nóng" trong cùng session.

CẤU HÌNH:
  Set biến môi trường MEDQUAD_RAG_DIR trỏ đúng vào thư mục model RAG muốn
  dùng, ví dụ:
      export MEDQUAD_RAG_DIR=/path/to/Embbeding_RAG/nomic-embed-text-v1.5
  Nếu không set, mặc định thử đường dẫn tương đối
  "../Embbeding_RAG/nomic-embed-text-v1.5" so với BASE_DIR của project chính.
"""

import os
import sys
import importlib
from pathlib import Path

from src.config import BASE_DIR

_DEFAULT_RAG_DIR = BASE_DIR.parent / "Embbeding_RAG" / "nomic-embed-text-v1.5"
RAG_PROJECT_DIR = Path(os.environ.get("MEDQUAD_RAG_DIR", str(_DEFAULT_RAG_DIR))).resolve()

_cached_retrieve_fn = None


def _load_retrieve_context_fn():
    """Import retrieval.py từ RAG project 1 LẦN DUY NHẤT, cách ly sys.path
    an toàn (thêm rồi xoá ngay), trả về hàm retrieve_context() của họ."""
    global _cached_retrieve_fn
    if _cached_retrieve_fn is not None:
        return _cached_retrieve_fn

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
        # Xoá module "config"/"retrieval" cũ (nếu lỡ có từ đâu đó) để đảm bảo
        # lần import này lấy ĐÚNG config.py/retrieval.py của RAG project,
        # không dính bản cache sai từ nơi khác.
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

        print(
            f"[rag_bridge] Đã load retrieval module từ: {RAG_PROJECT_DIR}\n"
            f"[rag_bridge] Retrieval mode / embedding model: xem "
            f"config.RETRIEVAL_MODE / config.EMBEDDING_MODEL_NAME trong "
            f"{RAG_PROJECT_DIR / 'config.py'}"
        )
    finally:
        # Dọn sys.path để không ảnh hưởng phần còn lại của chương trình.
        for p in inserted:
            if p in sys.path:
                sys.path.remove(p)

    return _cached_retrieve_fn


def get_context_texts(question: str, top_k: int = 3) -> list[str]:
    """
    Trả về danh sách text_content của top_k chunks liên quan nhất tới câu hỏi.
    Đây là hàm CHÍNH mà chat.py / evaluation.py sẽ gọi.
    """
    retrieve_context = _load_retrieve_context_fn()
    results = retrieve_context(question, top_k=top_k)
    return [r["text_content"] for r in results]