from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompt
from langchain_core.prompts import PromptTemplate
from chat.model.factory import chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


def print_prompt(prompt):
    print("========== Prompt Start ==========")
    print(prompt.to_string())
    print("========== Prompt End ==========")
    return prompt


class RagSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()
        self.prompt_text = load_rag_prompt()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self.__init__chain()
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-reranker-v2-m3"
        )
        self.reranker_model = (
            AutoModelForSequenceClassification.from_pretrained(
                "BAAI/bge-reranker-v2-m3"
            )
            .half()
            .cuda()
            .eval()
        )

        # 先加载文档（更新 all_documents），再构建包含 BM25 的混合检索器
        self.vector_store.load_document()
        self.retriever = self.vector_store.get_retriever()

    def __init__chain(self):
        # chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        chain = self.prompt_template
        return chain

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


if __name__ == "__main__":
    rag_service = RagSummarizeService()
    print(rag_service.rag_summarize("小户型适合哪些扫地机器人？"))
