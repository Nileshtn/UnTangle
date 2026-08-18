import hmac
import logging
import os

import chainlit as cl
from chainlit.data.chainlit_data_layer import ChainlitDataLayer
from chainlit.types import ThreadDict
from dotenv import load_dotenv

from config import settings
from utils.session import get_services, init_session
from utils.settings_ui import apply_chat_settings, send_chat_settings
from utils.storage import build_storage_client
from utils.thread_name import set_thread_name_from_first_document

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@cl.data_layer
def get_data_layer():
    return ChainlitDataLayer(
        database_url=os.environ["DATABASE_URL"],
        storage_client=build_storage_client(),
    )


SUMMARIZE_TRIGGERS = {
    "summarize",
    "summary",
    "summarize document",
    "summarize the document",
}


def parse_summarize_request(content: str) -> tuple[bool, str | None]:
    text = content.strip()
    lowered = text.lower()

    if lowered.startswith("/summarize"):
        parts = text.split(maxsplit=1)
        return True, parts[1].strip() if len(parts) > 1 else None

    if lowered in SUMMARIZE_TRIGGERS:
        return True, None

    return False, None


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if hmac.compare_digest(username, settings.auth_username) and hmac.compare_digest(
        password, settings.auth_password
    ):
        return cl.User(
            identifier=username,
            metadata={"role": "user", "provider": "credentials"},
        )
    return None


@cl.on_chat_start
async def on_chat_start():
    await init_session()
    _, vector_store, llm, _ = get_services()
    await send_chat_settings(llm, vector_store)
    await cl.Message(
        content=(
            "Welcome to UnTangle. Upload a PDF or text file, then ask questions "
            "about your document or type **/summarize** to get a summary.\n\n"
            "Use the **settings panel** (gear icon) to change the model and options."
        )
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    await init_session(thread_id=thread["id"])
    _, vector_store, llm, _ = get_services()
    await send_chat_settings(llm, vector_store)
    logger.info("Resumed thread %s", thread["id"])


@cl.on_settings_update
async def on_settings_update(new_settings: dict):
    _, vector_store, llm, thread_id = get_services()
    apply_chat_settings(new_settings, llm, vector_store, thread_id)


async def run_summarize(source_name: str | None = None) -> None:
    file_manager, vector_store, llm, _thread_id = get_services()

    if not file_manager.has_documents() and not vector_store.has_documents():
        await cl.Message(
            content="Please upload a PDF or `.txt` file before requesting a summary."
        ).send()
        return

    title = f"**{source_name}**" if source_name else "your documents"
    doc_count = len(file_manager.get_db_identifiers())
    if not source_name and doc_count == 1:
        title = "your document"
    elif not source_name and doc_count > 1:
        title = f"**{doc_count} documents**"
    response = cl.Message(content=f"_Summarizing {title}..._\n\n")
    await response.send()

    try:
        await llm.summarize(vector_store, response, source_name=source_name)
    except Exception as exc:
        logger.exception("Failed to generate summary")
        error_text = str(exc).lower()
        if "unauthorized" in error_text or "401" in error_text:
            response.content = (
                f"Could not reach the chat model **{llm.model_name}**. "
                "Choose a pulled local model in the settings panel "
                "or set `OLLAMA_MODEL` in `.env`, then restart the app."
            )
        else:
            response.content = (
                "Sorry, something went wrong while generating the summary. Please try again."
            )
        await response.update()


async def process_uploads(elements: list[cl.File]) -> None:
    file_manager, vector_store, llm, thread_id = get_services()

    file_manager.get_files(elements)
    if not file_manager.available_files:
        return

    is_first_document = not file_manager.has_documents()
    results = await vector_store.add_documents(file_manager.available_files)
    renamed_thread = False
    for result, element in zip(results, file_manager.available_files):
        if result["success"]:
            file_manager.add_db_identifier(element.name)
            if is_first_document and not renamed_thread:
                await set_thread_name_from_first_document(
                    result.get("document_title") or element.name,
                    thread_id,
                    is_first_document=True,
                )
                renamed_thread = True
            await cl.Message(
                content=f"Added **{element.name}**.",
                actions=[
                    cl.Action(
                        name="summarize",
                        payload={"filename": element.name},
                        label="Summarize",
                    )
                ],
            ).send()
        else:
            await cl.Message(
                content=f"Could not process **{element.name}**: {result['error']}"
            ).send()

    file_manager.clean_available()
    vector_store.init_retriever()
    if vector_store.retriever:
        llm.init_chain(vector_store.retriever, session_id=thread_id)


@cl.action_callback("summarize")
async def on_summarize(action: cl.Action):
    filename = action.payload.get("filename")
    await run_summarize(source_name=filename)


@cl.on_message
async def on_message(message: cl.Message):
    file_manager, vector_store, llm, _thread_id = get_services()

    if message.elements:
        await process_uploads(message.elements)

    if not message.content:
        return

    is_summarize, source_name = parse_summarize_request(message.content)
    if is_summarize:
        await run_summarize(source_name=source_name)
        return

    if not file_manager.has_documents() and not vector_store.has_documents():
        await cl.Message(
            content="Please upload a PDF or `.txt` file before asking questions."
        ).send()
        return

    if llm.chain is None:
        vector_store.init_retriever()
        if vector_store.retriever:
            llm.init_chain(vector_store.retriever, session_id=_thread_id)

    if llm.chain is None:
        await cl.Message(
            content="Your documents are still being processed. Try again in a moment."
        ).send()
        return

    response = cl.Message(content="")
    await response.send()
    try:
        await llm.chat(message.content, response)
    except Exception:
        logger.exception("Failed to generate response")
        await response.stream_token(
            "Sorry, something went wrong while generating a response. Please try again."
        )
        await response.update()
