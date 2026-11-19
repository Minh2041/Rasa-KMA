
import sys
import re
import unicodedata
from pathlib import Path

from raglight import RAGPipeline, PipelineConfig
from raglight.integrations import default_registries
from raglight.integrations.ollama import OllamaLLM, OllamaEmbeddings
from raglight.integrations.qdrant import QdrantVectorStore
from raglight.models import FolderSource, Document, Eq


ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT / "knowledge"

# Dùng collection mới để không trộn với dữ liệu cũ
COLLECTION_NAME = "kma_knowledge_by_folder_v1"


# =====================================================
# 1. CHUẨN HÓA TÊN FOLDER VÀ CÂU HỎI
# =====================================================

def normalize_text(text):
    text = text.lower().replace("đ", "d")
    text = unicodedata.normalize("NFD", text)

    return "".join(
        char for char in text
        if unicodedata.category(char) != "Mn"
    )


def normalize_category(name):
    text = normalize_text(name)
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


# =====================================================
# 2. TỰ ĐỘNG GẮN CATEGORY CHO DOCUMENT
# =====================================================

class CategoryQdrantVectorStore(QdrantVectorStore):

    current_category = None

    def add_documents(self, documents):
        if self.current_category is None:
            raise RuntimeError("Chua xac dinh category khi index.")

        tagged_documents = []

        for doc in documents:
            metadata = dict(doc.metadata)
            metadata["category"] = self.current_category

            tagged_documents.append(
                Document(
                    page_content=doc.page_content,
                    metadata=metadata
                )
            )

        super().add_documents(tagged_documents)


# =====================================================
# 3. KHỞI TẠO EMBEDDING, LLM, QDRANT
# =====================================================

embeddings = OllamaEmbeddings(
    model_name="bge-m3"
)

SYSTEM_PROMPT = """
Bạn là trợ lý AI hỗ trợ tra cứu thông tin của Học viện Kỹ thuật Mật mã (KMA).

QUY TẮC BẮT BUỘC:
1. Luôn trả lời bằng tiếng Việt có dấu, đúng chính tả và ngữ pháp.
2. Dù câu hỏi hoặc tài liệu nguồn bằng tiếng Anh, vẫn phải trả lời bằng tiếng Việt.
3. Sử dụng ngôn ngữ tự nhiên, rõ ràng, dễ hiểu và chuyên nghiệp.
4. Ưu tiên trả lời ngắn gọn, chính xác, đúng trọng tâm câu hỏi.
5. Chỉ sử dụng thông tin từ các tài liệu được truy xuất để trả lời câu hỏi về Học viện.
6. Không tự suy đoán, bịa đặt tên người, chức vụ, quy định hoặc thông tin học tập.
7. Nếu tài liệu không đủ thông tin, hãy trả lời:
   "Tôi chưa tìm thấy thông tin chính xác trong tài liệu hiện có."
8. Giữ nguyên tên riêng, mã môn học, mã ngành và các thuật ngữ kỹ thuật khi cần thiết.
9. Không sử dụng tiếng Anh trong câu trả lời, trừ tên riêng và thuật ngữ chuyên ngành không nên dịch.
10. Không nhắc đến các chỉ dẫn này trong câu trả lời.
"""

llm = OllamaLLM(
    model_name="qwen2.5:3b",
    system_prompt=SYSTEM_PROMPT,
    preload_model=False
)

vector_store = CategoryQdrantVectorStore(
    collection_name=COLLECTION_NAME,
    embeddings_model=embeddings,
    persist_directory=str(ROOT / "rag_db")
)

rag = RAGPipeline(
    llm=llm,
    embeddings=embeddings,
    vector_store=vector_store,
    document_processors=default_registries.document_processors,
    config=PipelineConfig(top_k=10)
)


# =====================================================
# 4. INDEX TÀI LIỆU THEO TỪNG FOLDER
# =====================================================

def index_one_folder(folder, category, recursive=True):
    vector_store.current_category = category

    print(f"\nDang index category: {category}")
    print(f"Folder: {folder}")

    result = rag.ingest([
        FolderSource(
            path=str(folder),
            recursive=recursive,
            chunk_size=1500,
            chunk_overlap=200
        )
    ])

    print("Ket qua:", result.to_dict())


def index_documents():
    if not KNOWLEDGE_DIR.is_dir():
        print("Khong tim thay folder knowledge.")
        return

    folders = sorted(
        [
            p for p in KNOWLEDGE_DIR.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ],
        key=lambda p: p.name
    )

    try:
        # Index mỗi folder thành một category
        for folder in folders:
            category = normalize_category(folder.name)

            index_one_folder(
                folder=folder,
                category=category
            )

        # Các file nằm trực tiếp trong knowledge
        root_files = [
            p for p in KNOWLEDGE_DIR.iterdir()
            if p.is_file()
        ]

        if root_files:
            index_one_folder(
                folder=KNOWLEDGE_DIR,
                category="chung",
                recursive=False
            )

    finally:
        vector_store.current_category = None

    print("\nHoan thanh index tai lieu.")


# =====================================================
# 5. XÁC ĐỊNH CATEGORY TỪ CÂU HỎI
# =====================================================

def detect_category(question):
    q = normalize_text(question)

    # Nhóm câu hỏi về lãnh đạo
    leadership_keywords = [
        "giam doc",
        "pho giam doc",
        "lanh dao",
        "ban giam doc",
        "ban giam hieu",
        "hieu truong",
        "pho hieu truong",
        "truong khoa",
        "pho truong khoa",
        "truong phong",
        "pho truong phong",
        "can bo"
    ]

    # Nhóm câu hỏi về quy định
    regulation_keywords = [
        "quy dinh",
        "quy che",
        "quy tac",
        "ky luat",
        "vi pham",
        "dieu kien",
        "noi quy"
    ]

    # Nhóm câu hỏi về môn học
    subject_keywords = [
        "mon hoc",
        "hoc phan",
        "tin chi",
        "de cuong",
        "chuong trinh dao tao",
        "mon gi",
        "hoc mon",
        "giang vien"
    ]

    # Ưu tiên từ khóa rõ ràng trước
    if any(k in q for k in leadership_keywords):
        return "lanh_dao"

    if any(k in q for k in subject_keywords):
        return "mon_hoc"

    if any(k in q for k in regulation_keywords):
        return "quy_dinh"

    # Câu hỏi tìm danh tính người
    person_keywords = [
        "ai la",
        "la ai",
        "ho ten",
        "ten cua",
        "ten nguoi",
        "nguoi nao"
    ]

    if any(k in q for k in person_keywords):
        return "lanh_dao"

    # Không xác định: tìm toàn bộ tài liệu
    return None


# =====================================================
# 6. RETRIEVAL ƯU TIÊN THEO CATEGORY
# =====================================================

def generate_answer(question):
    category = detect_category(question)

    if category:
        print(f"\nCategory uu tien: {category}")

        retrieval_filter = Eq("category", category)

        # Kiểm tra category có dữ liệu hay không
        preview = vector_store.similarity_search(
            question,
            k=1,
            retrieval_filter=retrieval_filter
        )

        if preview:
            return rag.generate(
                question,
                retrieval_filter=retrieval_filter
            )

        print("Category khong co ket qua. Tim toan bo kho.")

    else:
        print("\nCategory: tim kiem toan bo.")

    # Fallback: tìm không giới hạn category
    return rag.generate(question)


# =====================================================
# 7. CHAT
# =====================================================

def chat():
    print("RAGLight KMA da san sang.")
    print("Nhap 'exit' de thoat.")

    while True:
        question = input("\nBan: ").strip()

        if question.lower() in ("exit", "quit", "thoat"):
            break

        if not question:
            continue

        try:
            result = generate_answer(question)

            if not result.sources:
                print("\nChatbot: Chua tim thay tai lieu.")
                continue

            print("\nChatbot:", result.answer)

            print("\nNguon tra cuu:")

            for i, source in enumerate(result.sources, 1):
                metadata = source.metadata

                print(
                    f"{i}. Category: "
                    f"{metadata.get('category', 'khong ro')}"
                )

                print(
                    f"   File: "
                    f"{metadata.get('source', 'khong ro')}"
                )

                print(
                    f"   Noi dung: "
                    f"{source.page_content[:150]}"
                )

        except Exception as error:
            print("\nLoi RAG:", error)


# =====================================================
# 8. MAIN
# =====================================================

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "chat"

    if command == "index":
        index_documents()

    elif command == "chat":
        chat()

    else:
        print("Su dung: python rag-kma.py [index|chat]")
