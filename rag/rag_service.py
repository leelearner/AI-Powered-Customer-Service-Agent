from functools import lru_cache

from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompt
from langchain_core.prompts import PromptTemplate
from chat.model.factory import rag_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from utils.logger_handler import logger


def log_prompt(prompt):
    """Trace the fully rendered RAG prompt without writing to stdout."""
    logger.debug("RAG prompt:\n%s", prompt.to_string())
    return prompt


class RagSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.prompt_text = load_rag_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = rag_model
        self.chain = self.__init__chain()

    def __init__chain(self):
        chain = self.prompt_template | log_prompt | self.model | StrOutputParser()
        return chain

    def ingest(self):
        """Load new knowledge documents into the vector store.

        Kept out of __init__ so constructing the service has no disk-scanning side
        effect. Call it once at startup, not per query.
        """
        self.vector_store.load_document()

    def retriever_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    def bge_rerank(self, query: str, docs: list[Document]):
        if not docs:
            return []
        pairs = [(query, doc.page_content) for doc in docs]
        with torch.no_grad():
            inputs = self.reranker_tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to("cuda")
            scores = self.reranker_model(**inputs).logits.squeeze(-1).tolist()
        # sort the docs by the reranker scores
        ranked_docs = [
            doc
            for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        ]
        return ranked_docs[:3]

    def rag_summarize(self, query: str) -> str:
        # Get relevent docs from Chroma
        content_docs = self.retriever_docs(query)

        reranked_docs = self.bge_rerank(query, content_docs)

        # Concatenate all documents together to form the prompt
        context = ""
        cnt = 0
        for doc in reranked_docs:
            cnt += 1
            context += f"[Document {cnt}]: document: {doc.page_content} | metadata: {doc.metadata}\n"

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )


@lru_cache(maxsize=1)
def get_rag_service() -> RagSummarizeService:
    """Process-wide singleton.

    Building the service opens the vector store and compiles the chain; doing that
    per query also re-scanned data/ and re-hashed every file.
    """
    return RagSummarizeService()


if __name__ == "__main__":
    rag_service = get_rag_service()
    rag_service.ingest()
    print(rag_service.rag_summarize("小户型适合哪些扫地机器人？"))
