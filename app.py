from typing import Optional
import chainlit as cl
from chainlit.input_widget import Select, Slider
import asyncio
from utils import *


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == ("user", "user"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None

@cl.on_chat_start
async def main():
    global llm
    global file_manager
    global vector_store
    user = cl.user_session.get("user")

    llm = DocLLM()
    file_manager = FileManager()
    vector_store = VectorStoreManager(f"{user}")

    await cl.Message(content="Hello! How can I help you today?").send()




async def check_element(elements : list[cl.File]):
    file_manager.get_files(elements)

    pass_list = await vector_store.add_documents(file_manager.available_files)
    for tag, element in zip(pass_list, file_manager.available_files):
        if tag:
            file_manager.add_db_identifier(element.name)
            await cl.Message(f"{element.name} added").send()
        else:
            await cl.Message(f"{element.name} already exisit").send()
    file_manager.clean_available()
    vector_store.init_retriever()

    if vector_store.retriever:
        llm.init_chain(vector_store.retriever)


# @cl.on_chat_change


@cl.on_message
async def main(message: cl.Message):
    if len(message.elements) == 0 and len(file_manager.db_identifiers['docs'])==0:
        await cl.Message(f"attach file to continue chat").send()
        return

    await check_element(message.elements)

    response = cl.Message(content="")
    if message.content and llm.chain:
        await llm.chat(message.content, response)



@cl.on_chat_resume
def on_chat_resume():
    pass