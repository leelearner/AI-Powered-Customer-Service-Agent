from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from model.factory import embedding_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
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
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

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
