from pathlib import Path

import chainlit as cl

from config import settings
from utils.core import DocLLM, FileManager, VectorStoreManager


def get_data_dir(user_id: str, thread_id: str) -> Path:
    data_dir = settings.data_dir / user_id / thread_id
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


async def init_session(thread_id: str | None = None) -> None:
    user = cl.user_session.get("user")
    if user is None:
        raise RuntimeError("Authenticated user is required to start a session.")

    user_id = user.identifier
    thread_id = thread_id or cl.context.session.thread_id
    data_dir = get_data_dir(user_id, thread_id)

    file_manager = FileManager(data_dir)
    vector_store = VectorStoreManager(
        persist_dir=data_dir / "chroma",
        embedding_model=settings.embedding_model,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        retriever_top_k=settings.retriever_top_k,
        supported_extensions=settings.supported_extensions,
    )
    llm = DocLLM(
        model_name=settings.ollama_model,
        temperature=settings.temperature,
        prompt_path=str(settings.prompt_path),
        summarize_prompt_path=str(settings.summarize_prompt_path),
        summarize_map_prompt_path=str(settings.summarize_map_prompt_path),
        summarize_batch_chars=settings.summarize_batch_chars,
        summarize_max_batches=settings.summarize_max_batches,
        summarize_parallel_requests=settings.summarize_parallel_requests,
        num_predict=settings.ollama_num_predict,
    )

    cl.user_session.set("thread_id", thread_id)
    cl.user_session.set("file_manager", file_manager)
    cl.user_session.set("vector_store", vector_store)
    cl.user_session.set("llm", llm)

    vector_store.init_retriever()
    if vector_store.has_documents():
        llm.init_chain(vector_store.retriever, session_id=thread_id)


def get_services() -> tuple[FileManager, VectorStoreManager, DocLLM, str]:
    file_manager = cl.user_session.get("file_manager")
    vector_store = cl.user_session.get("vector_store")
    llm = cl.user_session.get("llm")
    thread_id = cl.user_session.get("thread_id")

    if not all([file_manager, vector_store, llm, thread_id]):
        raise RuntimeError("Session is not initialized.")

    return file_manager, vector_store, llm, thread_id
