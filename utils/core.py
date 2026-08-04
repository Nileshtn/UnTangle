import os
import tempfile
import yaml
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

import streamlit as st



class FileManager:
    def __init__(self, db_identifier_path='db_identifier.yaml'):
        self.db_identifier_path = db_identifier_path
        self.doc_store_path = []
        self.available_files = []
        self._load_db_identifiers()

    def get_files(self, files_path):
        self.available_files = [file.name for file in files_path if not self.is_identifier_exists(file.name)]
        print(self.available_files)

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
        for id in identifier:
            if identifier.name not in self.db_identifiers['docs']:
                self.db_identifiers['docs'].append(identifier.name)
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

    def add_documents(self, documents_path_list):
        # for document in documents_path_list
        for document_path in documents_path_list:
            st.write(f"adding {document_path.name}")
            if document_path.type == "application/pdf":
                documents = self.pdf_parser(document_path)
            elif document_path.type == "text/plain":
                documents = self.text_parser(document_path)
            else:
                raise ValueError(f"Unsupported file type: {document_path}")
            self.vectorstore.add_documents(documents)
            st.write(f"completed {document_path.name}")
            
        return True

    def init_retriever(self, search_type="mmr",  top_k: int = 3):
        self.retriever = self.vectorstore.as_retriever(search_type=search_type, search_kwargs={"k": top_k})

    def pdf_parser(self, uploaded_file):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        try:
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            return self.text_splitter.split_documents(documents)
            
        finally:
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def text_parser(self, text_path):
        loader = TextLoader(text_path)
        documents = loader.load()
        documents = self.text_splitter.split_documents(documents)
        return documents

    def get_vectorstore(self):
        return self.vectorstore


class DocLLM:
    def __init__(self, model_name="ollama", temperature=0.2):
        self.model_name = model_name
        self.temperature = temperature
        self.chain = None
        self.mesg = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that answers questions based on the provided context only. else say 'I don't know'."),
            ("human", "{context}\n\nQuestion: {query}\nAnswer:")
        ])
        self.chat_model = ChatOllama(model=self.model_name, temperature=self.temperature)

    def format_docs(self,docs):
            return "\n\n".join(doc.page_content for doc in docs) 
    
    def init_chain(self, retriever):
        self.chain = {"context": retriever | self.format_docs, "query" : RunnablePassthrough()} | self.mesg | self.chat_model

    def chat(self, query):
        if self.chain is None:
            print("need to init chain first call llm.init_chain(retriever) first")
        response = self.chain.invoke(query)
        return response.content