from langchain_groq.chat_models import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

import os

from src.core.logging import get_logger
from src.core.config import settings

from functools import lru_cache

log = get_logger(__name__)

def _ensure_groq_key() -> None:
    if settings.groq_api_key and not os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = settings.groq_api_key


@lru_cache
def get_llm() -> ChatGroq:
    _ensure_groq_key()
    log.info(f"Initalising the llm model: {settings.llm_model}")
    return ChatGroq(model=settings.llm_model, temperature=settings.llm_temperature, streaming=settings.Streaming)

@lru_cache
def get_grader() -> ChatGroq:
    # grader_model_name, not grader_model: the raw field defaults to "" and
    # ChatGroq(model="") fails at call time, not construction time -- so the
    # fallback property existed but was never actually reached.
    _ensure_groq_key()
    log.info(f"Initalising the llm grader: {settings.grader_model_name}")
    return ChatGroq(
        model=settings.grader_model_name,
        temperature=settings.grader_temperature,
    )

@lru_cache()
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )