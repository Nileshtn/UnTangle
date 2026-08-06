import chainlit as cl
import asyncio
from utils import *

llm = DocLLM("gemma4:31b-cloud")
file_manager = FileManager()
vector_store = VectorStoreManager("chainlit_db")

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

@cl.on_message
async def main(message: cl.Message):
    if len(message.elements) == 0 and len(file_manager.db_identifiers['docs'])==0:
        await cl.Message(f"attach file to continue chat").send()
        return

    await check_element(message.elements)

    response = cl.Message(content="")
    if message.content and llm.chain:
        await llm.chat(message.content, response)

    