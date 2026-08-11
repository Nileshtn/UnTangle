import os
import tempfile
import yaml
from pathlib import Path

import ollama

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import load_prompt
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import chainlit as cl

class FileManager:
    def __init__(self, db_identifier_path='db_identifier.yaml'):
        self.db_identifier_path = db_identifier_path
        self.available_files:list[cl.File] = []
        self._load_db_identifiers()

    def clean_available(self):
        self.available_files = []

    def get_files(self, elements):
        self.available_files = [element for element in elements if not self.is_identifier_exists(element.name)]

    def _load_db_identifiers(self):
        with open(self.db_identifier_path, 'r') as file:
            self.db_identifiers = yaml.safe_load(file)
        if self.db_identifiers is None:
            self.db_identifiers = {'docs': []}
            self.save_db_identifiers()
            
    def get_db_identifiers(self):
        return self.db_identifiers.get('docs', [])

    def is_identifier_exists(self, identifier):
        return True if identifier in self.db_identifiers.get('docs', []) else False

    def add_db_identifier(self, identifier):
        if identifier not in self.db_identifiers['docs']:
            self.db_identifiers['docs'].append(identifier)
            self.save_db_identifiers()

    def save_db_identifiers(self):
        with open(self.db_identifier_path, 'w') as file:
            yaml.dump(self.db_identifiers, file)

    def get_file_paths(self):
        return [os.path.join(self.doc_store_path, file) for file in self.available_files]

class VectorStoreManager:
    def __init__(self, db_identifier):
        self.db_identifier = db_identifier
        self.vectorstore_path = f'chroma_db/{db_identifier}'
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vectorstore = Chroma(persist_directory=self.vectorstore_path, embedding_function=self.embeddings)
        self.retriever = None

    def extra_content(self, element):
        ext = Path(element.name).suffix.lower()

        if ext == ".pdf":
            loader = PyPDFLoader(element.path)
        elif ext == ".txt":
            loader = TextLoader(element.path)
        else:
            return ext, None
        
        documents = loader.load()
        return ext , self.text_splitter.split_documents(documents)

    async def add_documents(self, elements:list[cl.File]) -> list:
        pass_list = []
        for element in elements:
            ext, content = await cl.make_async(self.extra_content)(element)
            if content:
                self.vectorstore.add_documents(content)
                pass_list.append(True)
            else:
                await cl.Message(f"{ext} not supported we are working on that!")
                pass_list.append(False)
        return pass_list
            
    def init_retriever(self, search_type="mmr",  top_k: int = 3):
        self.retriever = self.vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": top_k})        

    def get_vectorstore(self):
        return self.vectorstore

class DocLLM:
    """
    manages the llm call and chain
    """
    def __init__(self, model_name="gemma4:31b-cloud", temperature=0.2, prompt_path="prompts/rag_prompt.yaml"):
        self.model_name = model_name
        self.temperature = temperature
        self.chain = None
        self.prompt = load_prompt(prompt_path)
        self.chat_model = ChatOllama(model=self.model_name, temperature=self.temperature)
        self.store = {}


    def get_avalable_models(self) ->list:
        response = ollama.list()

        chat_models = []
        for model in response.models:
            name = model.model.lower()
            family = getattr(model.details, 'family', '').lower()
            
            if not any(keyword in name or keyword in family for keyword in ['embed', 'bge', 'minilm', 'bert']):
                chat_models.append(model.model)

        return chat_models



    def get_session_history(self, session_id: str):
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def format_docs(self,docs):
            return "\n\n".join(doc.page_content for doc in docs) 
    
    def init_chain(self, retriever):
        self.chain = ({"context": (lambda x: x["query"]) | retriever | self.format_docs, 
                      "query" : lambda x: x['query'],
                      "chat_history" : lambda x: x['chat_history']
                      }
                    | self.prompt 
                    | self.chat_model)

        self.chain = RunnableWithMessageHistory(self.chain, 
                                                get_session_history= self.get_session_history, 
                                                input_messages_key= 'query', 
                                                history_messages_key= "chat_history")

    async def chat(self, query:str, message_placeholder: cl.Message):
        if self.chain is None:
            print("need to init chain first call llm.init_chain(retriever) first")
            return
        async for chunk in self.chain.astream({"query" : query}, {'configurable': {'session_id': 'hello'}}):
            await message_placeholder.stream_token(chunk.content)
        await message_placeholder.update()