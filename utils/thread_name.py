import logging

import chainlit as cl
from chainlit.data import get_data_layer

logger = logging.getLogger(__name__)

DEFAULT_THREAD_NAMES = {"", "untitled conversation", "untitled"}


def _normalize_title(text: str) -> str:
    return " ".join(text.split()).strip()[:255]


async def set_thread_name_from_first_document(
    document_title: str,
    thread_id: str,
    *,
    is_first_document: bool,
) -> None:
    if not is_first_document:
        return

    data_layer = get_data_layer()
    if not data_layer:
        return

    thread_name = _normalize_title(document_title)
    if not thread_name:
        return

    try:
        thread = await data_layer.get_thread(thread_id=thread_id)
        existing_name = (thread.get("name") or "").strip() if thread else ""
        if existing_name.lower() not in DEFAULT_THREAD_NAMES:
            return

        await data_layer.update_thread(thread_id=thread_id, name=thread_name)
        await cl.context.emitter.emit(
            "first_interaction",
            {"interaction": thread_name, "thread_id": thread_id},
        )
        logger.info("Renamed thread %s to %s", thread_id, thread_name)
    except Exception:
        logger.exception("Failed to rename thread %s", thread_id)
