import logging

import chainlit as cl
from chainlit.input_widget import Select, Slider

from config import settings
from utils.core import DocLLM, VectorStoreManager

logger = logging.getLogger(__name__)

LARGE_MODEL_BYTES = 10_000_000_000
PRIMARY_MODEL = settings.ollama_model


def _format_model_label(name: str, size: int) -> str:
    lowered = name.lower()
    if "cloud" in lowered:
        return f"{name} · cloud"
    if size <= 0:
        return name
    if size >= LARGE_MODEL_BYTES:
        return f"{name} · {size / 1_000_000_000:.0f} GB · slow"
    if size >= 1_000_000_000:
        return f"{name} · {size / 1_000_000_000:.1f} GB"
    return f"{name} · {size / 1_000_000:.0f} MB"


def get_model_select_items(llm: DocLLM) -> dict[str, str]:
    try:
        catalog = llm.get_model_catalog()
    except Exception:
        logger.warning("Could not list Ollama models", exc_info=True)
        catalog = []

    if not catalog:
        return {_format_model_label(PRIMARY_MODEL, 0): PRIMARY_MODEL}

    def sort_key(entry: dict[str, int | str]) -> tuple[int, int, str]:
        name = str(entry["name"])
        if name == PRIMARY_MODEL:
            return (0, 0, name)
        if "cloud" in name.lower():
            return (1, int(entry["size"]), name)
        return (2, int(entry["size"]), name)

    catalog = sorted(catalog, key=sort_key)

    items = {
        _format_model_label(str(entry["name"]), int(entry["size"])): str(entry["name"])
        for entry in catalog
    }

    if llm.model_name not in items.values():
        items[_format_model_label(llm.model_name, 0)] = llm.model_name

    return items


def build_chat_settings(
    llm: DocLLM, vector_store: VectorStoreManager
) -> cl.ChatSettings:
    model_items = get_model_select_items(llm)

    return cl.ChatSettings(
        [
            Select(
                id="model",
                label="Chat model",
                items=model_items,
                initial_value=llm.model_name,
                description=(
                    f"Default: {PRIMARY_MODEL}. "
                    "Cloud models run on Ollama's servers and are faster than large local models."
                ),
            ),
            Slider(
                id="temperature",
                label="Temperature",
                initial=llm.temperature,
                min=0.0,
                max=1.0,
                step=0.05,
                description="Lower values are more focused; higher values are more creative.",
            ),
            Slider(
                id="max_tokens",
                label="Max response tokens",
                initial=llm.num_predict,
                min=256,
                max=4096,
                step=128,
                description="Lower values finish sooner. Increase for longer summaries.",
            ),
            Slider(
                id="retriever_top_k",
                label="Context chunks",
                initial=vector_store.retriever_top_k,
                min=1,
                max=10,
                step=1,
                description="Number of document chunks retrieved for each question.",
            ),
            Slider(
                id="summarize_max_batches",
                label="Summary sections",
                initial=llm.summarize_max_batches,
                min=2,
                max=8,
                step=1,
                description="Fewer sections is faster; more sections captures more detail.",
            ),
            Slider(
                id="summarize_parallel_requests",
                label="Parallel summary jobs",
                initial=llm.summarize_parallel_requests,
                min=1,
                max=4,
                step=1,
                description="How many summary sections Ollama processes at once.",
            ),
        ]
    )


async def send_chat_settings(llm: DocLLM, vector_store: VectorStoreManager) -> None:
    await build_chat_settings(llm, vector_store).send()


def apply_chat_settings(
    new_settings: dict,
    llm: DocLLM,
    vector_store: VectorStoreManager,
    thread_id: str,
) -> None:
    model = str(new_settings.get("model", llm.model_name))
    temperature = float(new_settings.get("temperature", llm.temperature))
    num_predict = int(new_settings.get("max_tokens", llm.num_predict))
    retriever_top_k = int(new_settings.get("retriever_top_k", vector_store.retriever_top_k))
    summarize_max_batches = int(
        new_settings.get("summarize_max_batches", llm.summarize_max_batches)
    )
    summarize_parallel_requests = int(
        new_settings.get(
            "summarize_parallel_requests", llm.summarize_parallel_requests
        )
    )

    vector_store.apply_retriever_settings(retriever_top_k)
    llm.apply_summarize_settings(
        summarize_max_batches=summarize_max_batches,
        summarize_parallel_requests=summarize_parallel_requests,
    )

    retriever = vector_store.retriever if vector_store.has_documents() else None
    llm.apply_model_settings(
        model_name=model,
        temperature=temperature,
        retriever=retriever,
        session_id=thread_id,
        num_predict=num_predict,
    )

    logger.info(
        "Updated settings: model=%s temperature=%s max_tokens=%s top_k=%s "
        "summary_batches=%s parallel=%s",
        model,
        temperature,
        num_predict,
        retriever_top_k,
        summarize_max_batches,
        summarize_parallel_requests,
    )
