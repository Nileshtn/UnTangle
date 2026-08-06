import os
import tempfile
import yaml
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
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
    def __init__(self, model_name="ollama", temperature=0.2):
        self.model_name = model_name
        self.temperature = temperature
        self.chain = None
        self.mesg = ChatPromptTemplate.from_messages([
            ("system", """
                        You are a precise data extraction assistant. 

                        OUTPUT FORMAT RULES:
                        - Start your response directly with the first bullet point.
                        - Do NOT include introductory phrases, conversational filler, or restatements of the query (e.g., "Based on the text...", "Here is...", "Skip connections are used...").
                        - Output ONLY a markdown bulleted list.
                        """),
            ("human", "{context}\n\nQuestion: {query}\nAnswer:")
        ])
        self.chat_model = ChatOllama(model=self.model_name, temperature=self.temperature)

    def format_docs(self,docs):
            return "\n\n".join(doc.page_content for doc in docs) 
    
    def init_chain(self, retriever):
        self.chain = {"context": retriever | self.format_docs, "query" : RunnablePassthrough()} | self.mesg | self.chat_model

    async def chat(self, query):
        if self.chain is None:
            print("need to init chain first call llm.init_chain(retriever) first")
            return
        response = self.chain.invoke(query)
        return response.content