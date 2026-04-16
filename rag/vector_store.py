from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from chat.model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.retrievers.bm25 import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
import os, pickle
from utils.file_handler import (
    pdf_loader,
    txt_loader,
    listdir_with_allowed_type,
    get_file_md5_hex,
)
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embedding_model,
            persist_directory=chroma_conf["persist_directory"],
        )
        # chunk_size 单位是字符数（非 token 数）
        # 中文场景下 1 字符 ≈ 1.5–2 token，chunk_size=200 约对应 300–400 token
        # 远小于 reranker 的 max_length=512，保持安全边距
        self.parent_spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["parent_chunk_size"],
            chunk_overlap=chroma_conf["parent_chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,  # 按字符数切分，非 token 数
        )

        self.child_spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["child_chunk_size"],
            chunk_overlap=chroma_conf["child_chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,  # 按字符数切分，非 token 数
        )

        self.docstore = InMemoryStore()
        self.docstore_path = get_abs_path(
            chroma_conf.get("docstore_path", "data/docstore.pkl")
        )
        self._load_docstore()

        self.all_documents: list[Document] = []
        self._load_existing_documents()
        self.vector_retriever = self.get_vector_retriever()

    def _load_docstore(self):
        """重启时从磁盘恢复 docstore"""
        if os.path.exists(self.docstore_path):
            with open(self.docstore_path, "rb") as f:
                data = pickle.load(f)
                self.docstore.mset(list(data.items()))
            logger.info(f"[Docstore] Restored {len(data)} parent docs from disk.")

    def _save_docstore(self):
        """持久化 docstore 到磁盘"""
        all_keys = list(self.docstore.yield_keys())
        data = dict(zip(all_keys, self.docstore.mget(all_keys)))
        os.makedirs(os.path.dirname(self.docstore_path), exist_ok=True)
        with open(self.docstore_path, "wb") as f:
            pickle.dump(data, f)

    def _load_existing_documents(self):
        try:
            all_keys = list(self.docstore.yield_keys())
            if all_keys:
                parent_docs = self.docstore.mget(all_keys)
                self.all_documents = [
                    doc for doc in parent_docs if isinstance(doc, Document)
                ]
                logger.info(
                    f"[Load existing documents] Loaded {len(self.all_documents)} parent documents from docstore for BM25."
                )
            else:
                self.all_documents = []
        except Exception as e:
            logger.warning(
                f"[Load existing documents] Failed to load parent documents: {str(e)}"
            )

    def get_vector_retriever(self):
        k = chroma_conf["k"]
        # vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        vector_retriever = ParentDocumentRetriever(
            vectorstore=self.vector_store,
            docstore=self.docstore,
            child_splitter=self.child_spliter,
            parent_splitter=self.parent_spliter,
            search_kwargs={"k": k},
        )
        return vector_retriever

    def get_retriever(self):
        k = chroma_conf["k"]
        if self.all_documents:
            bm25_retriever = BM25Retriever.from_documents(self.all_documents, k=k)
            return EnsembleRetriever(
                retrievers=[self.vector_retriever, bm25_retriever],
                weights=[0.5, 0.5],
            )

        # 文档尚未加载时降级为纯向量检索
        logger.warning(
            "[get_retriever] No documents available for BM25, falling back to vector-only retrieval."
        )
        return self.vector_retriever

    def load_document(self):
        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(
                    get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8"
                ).close()
                return False

            with open(
                get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8"
            ) as f:
                for line in f.readlines():
                    if line.strip() == md5_for_check:
                        return True
            return False

        def save_md5_hex(md5_to_save: str):
            with open(
                get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8"
            ) as f:
                f.write(md5_to_save + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)
            elif read_path.endswith(".txt"):
                return txt_loader(read_path)
            else:
                return []

        allowed_files_path = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        # 一致性检查：docstore 为空但 md5 记录存在，说明 docstore.pkl 丢失但
        # Chroma 中已有 child chunk。此时直接用 md5 跳过会导致 ParentDocumentRetriever
        # 从空 docstore 查不到 parent，返回空列表。需要清除 md5 和 Chroma 重建索引。
        md5_path = get_abs_path(chroma_conf["md5_hex_store"])
        docstore_is_empty = not list(self.docstore.yield_keys())
        md5_has_records = os.path.exists(md5_path) and os.path.getsize(md5_path) > 0
        if docstore_is_empty and md5_has_records:
            logger.warning(
                "[Load documents] docstore is empty but md5 records exist — "
                "docstore.pkl was likely lost. Resetting md5 and Chroma to force re-indexing."
            )
            open(md5_path, "w").close()
            self.vector_store.reset_collection()

        loaded_any = False
        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"[Load documents]{path} has been loaded before, skip it.")
                continue
            try:
                documents: list[Document] = get_file_documents(path)
                if not documents:
                    logger.warning(f"[Load documents]No documents loaded from {path}")
                    continue

                # 关键改动：用 retriever.add_documents 替代原来的手动切分
                # 它会自动完成：parent切分 → child切分 → child存Chroma → parent存docstore
                self.vector_retriever.add_documents(documents)

                save_md5_hex(md5_hex)
                loaded_any = True
                logger.info(f"[Load documents]Loaded documents from {path}")
            except Exception as e:
                logger.error(
                    f"[Load documents]Error loading documents from {path}: {str(e)}",
                    exc_info=True,
                )

        # 有新文档加入时，持久化 docstore
        if loaded_any:
            self._save_docstore()
            self._load_existing_documents()


if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)
