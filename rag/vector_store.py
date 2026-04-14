from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from chat.model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_classic.retrievers.bm25 import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers import ParentDocumentRetriever
import os
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
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,  # 按字符数切分，非 token 数
        )
        self.all_documents: list[Document] = []
        self._load_existing_documents()

    def _load_existing_documents(self):
        """从 Chroma 中读取已持久化的文档，用于重启后重建 BM25 索引"""
        try:
            existing = self.vector_store.get()
            if existing and existing.get("documents"):
                self.all_documents = [
                    Document(page_content=doc, metadata=meta)
                    for doc, meta in zip(existing["documents"], existing["metadatas"])
                ]
                logger.info(
                    f"[Load existing documents] Loaded {len(self.all_documents)} documents from Chroma for BM25."
                )
        except Exception as e:
            logger.warning(
                f"[Load existing documents] Failed to load existing documents: {str(e)}"
            )

    def get_retriever(self):
        k = chroma_conf["k"]
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": k})

        if self.all_documents:
            bm25_retriever = BM25Retriever.from_documents(self.all_documents, k=k)
            return EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.5, 0.5],
            )

        # 文档尚未加载时降级为纯向量检索
        logger.warning(
            "[get_retriever] No documents available for BM25, falling back to vector-only retrieval."
        )
        return vector_retriever

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

        for path in allowed_files_path:
            # Get th md5 hex of the file
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[Load documents]{path} has been loaded before, skip it.")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(
                        f"[Load documents]No documents loaded from {path}, maybe the file type is not supported."
                    )
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(
                        f"[Load documents]No documents after splitting from {path}, maybe the file content is empty."
                    )
                    continue

                self.vector_store.add_documents(split_document)
                self.all_documents.extend(
                    split_document
                )  # 同步更新内存副本供 BM25 使用
                save_md5_hex(md5_hex)

                logger.info(
                    f"[Load documents]Loaded and added documents from {path} to vector store."
                )

            except Exception as e:
                logger.error(
                    f"[Load documents]Error loading documents from {path}: {str(e)}",
                    exc_info=True,
                )


if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)
