from abc import ABC, abstractmethod
from typing import Optional
from chromadb import Embeddings
from langchain.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from chat.utils.config_handler import rag_conf
from langchain_openai import OpenAIEmbeddings


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return ChatAnthropic(model=rag_conf["chat_model_name"], temperature=0.7)


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return OpenAIEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingsFactory().generator()
