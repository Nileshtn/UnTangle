import hmac
import logging

import chainlit as cl
from chainlit.types import ThreadDict
from dotenv import load_dotenv

from config import settings
from utils.session import get_services, init_session

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


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
    await cl.Message(
        content=(
            "Welcome to UnTangle. Upload a PDF or text file, then ask questions "
            "about your document."
        )
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    await init_session(thread_id=thread["id"])
    logger.info("Resumed thread %s", thread["id"])


async def process_uploads(elements: list[cl.File]) -> None:
    file_manager, vector_store, llm, thread_id = get_services()

    file_manager.get_files(elements)
    if not file_manager.available_files:
        return

    results = await vector_store.add_documents(file_manager.available_files)
    for result, element in zip(results, file_manager.available_files):
        if result["success"]:
            file_manager.add_db_identifier(element.name)
            await cl.Message(content=f"Added **{element.name}**.").send()
        else:
            await cl.Message(
                content=f"Could not process **{element.name}**: {result['error']}"
            ).send()

    file_manager.clean_available()
    vector_store.init_retriever()
    if vector_store.retriever:
        llm.init_chain(vector_store.retriever, session_id=thread_id)


@cl.on_message
async def on_message(message: cl.Message):
    file_manager, vector_store, llm, _thread_id = get_services()

    if message.elements:
        await process_uploads(message.elements)

    if not message.content:
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
