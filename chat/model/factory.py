from abc import ABC, abstractmethod
from typing import Optional
from chromadb import Embeddings
from langchain.chat_models import BaseChatModel
from langchain_anthropic import ChatAnthropic
from utils.config_handler import models_conf
from langchain_openai import OpenAIEmbeddings


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    """Builds the chat model for one role, as configured in config/models.yml.

    Every LLM in the project comes from here — no module constructs a client with
    an inline model name.
    """

    def __init__(self, role: str):
        self.role = role

    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        conf = models_conf[self.role]
        kwargs = {
            "model": conf["name"],
            "timeout": models_conf.get("request_timeout_seconds", 120),
            "max_tokens": models_conf.get("max_tokens", 4096),
            "stop": None,
        }
        # Sampling params are model-gated: the current Opus/Sonnet generation rejects
        # temperature with a 400, while Haiku still accepts it. So it is only sent
        # when the role's config declares it — see the note in config/models.yml.
        if "temperature" in conf:
            kwargs["temperature"] = conf["temperature"]
        return ChatAnthropic(**kwargs)


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return OpenAIEmbeddings(model=models_conf["embedding"]["name"])


synthesis_model = ChatModelFactory("synthesis").generator()
classification_model = ChatModelFactory("classification").generator()
rag_model = ChatModelFactory("rag").generator()
embedding_model = EmbeddingsFactory().generator()
