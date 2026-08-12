import logging
from pathlib import Path

import ollama
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.prompts import load_prompt
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chainlit as cl

logger = logging.getLogger(__name__)


class FileManager:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_identifier_path = self.data_dir / "db_identifier.yaml"
        self.available_files: list[cl.File] = []
        self._load_db_identifiers()

    def clean_available(self) -> None:
        self.available_files = []

    def get_files(self, elements: list[cl.File]) -> None:
        self.available_files = [
            element
            for element in elements
            if not self.is_identifier_exists(element.name)
        ]

    def _load_db_identifiers(self) -> None:
        if self.db_identifier_path.exists():
            import yaml

            with open(self.db_identifier_path, "r", encoding="utf-8") as file:
                self.db_identifiers = yaml.safe_load(file) or {"docs": []}
        else:
            self.db_identifiers = {"docs": []}
            self.save_db_identifiers()

    def get_db_identifiers(self) -> list[str]:
        return self.db_identifiers.get("docs", [])

    def is_identifier_exists(self, identifier: str) -> bool:
        return identifier in self.db_identifiers.get("docs", [])

    def add_db_identifier(self, identifier: str) -> None:
        if identifier not in self.db_identifiers["docs"]:
            self.db_identifiers["docs"].append(identifier)
            self.save_db_identifiers()

    def save_db_identifiers(self) -> None:
        import yaml

        with open(self.db_identifier_path, "w", encoding="utf-8") as file:
            yaml.dump(self.db_identifiers, file)

    def has_documents(self) -> bool:
        return bool(self.get_db_identifiers())


class VectorStoreManager:
    def __init__(
        self,
        persist_dir: Path,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
        retriever_top_k: int,
        supported_extensions: tuple[str, ...],
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.supported_extensions = supported_extensions
        self.retriever_top_k = retriever_top_k
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings,
        )
        self.retriever = None

    def extract_content(self, element: cl.File) -> tuple[str, list | None]:
        ext = Path(element.name).suffix.lower()

        if ext not in self.supported_extensions:
            return ext, None

        if ext == ".pdf":
            loader = PyPDFLoader(element.path)
        else:
            loader = TextLoader(element.path, encoding="utf-8")

        documents = loader.load()
        return ext, self.text_splitter.split_documents(documents)

    async def add_documents(self, elements: list[cl.File]) -> list[dict]:
        results: list[dict] = []
        for element in elements:
            ext, content = await cl.make_async(self.extract_content)(element)
            if content:
                self.vectorstore.add_documents(content)
                results.append({"name": element.name, "success": True})
                logger.info("Indexed document %s", element.name)
            else:
                results.append(
                    {
                        "name": element.name,
                        "success": False,
                        "error": f"Unsupported file type: {ext}",
                    }
                )
        return results

    def init_retriever(self, search_type: str = "mmr") -> None:
        self.retriever = self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": self.retriever_top_k},
        )

    def has_documents(self) -> bool:
        try:
            return self.vectorstore._collection.count() > 0
        except Exception:
            return False


class DocLLM:
    """Manages the LLM call and RAG chain."""

    def __init__(
        self,
        model_name: str,
        temperature: float,
        prompt_path: str,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.chain = None
        self.session_id = "default"
        self.prompt = load_prompt(prompt_path)
        self.chat_model = ChatOllama(model=self.model_name, temperature=self.temperature)
        self.store: dict[str, ChatMessageHistory] = {}

    def get_available_models(self) -> list[str]:
        response = ollama.list()
        chat_models = []
        for model in response.models:
            name = model.model.lower()
            family = getattr(model.details, "family", "").lower()
            if not any(
                keyword in name or keyword in family
                for keyword in ["embed", "bge", "minilm", "bert"]
            ):
                chat_models.append(model.model)
        return chat_models

    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def format_docs(self, docs) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def init_chain(self, retriever, session_id: str) -> None:
        self.session_id = session_id
        chain = (
            {
                "context": (lambda x: x["query"]) | retriever | self.format_docs,
                "query": lambda x: x["query"],
                "chat_history": lambda x: x["chat_history"],
            }
            | self.prompt
            | self.chat_model
        )
        self.chain = RunnableWithMessageHistory(
            chain,
            get_session_history=self.get_session_history,
            input_messages_key="query",
            history_messages_key="chat_history",
        )

    async def chat(self, query: str, message_placeholder: cl.Message) -> None:
        if self.chain is None:
            raise RuntimeError("RAG chain is not initialized. Upload a document first.")

        async for chunk in self.chain.astream(
            {"query": query},
            {"configurable": {"session_id": self.session_id}},
        ):
            await message_placeholder.stream_token(chunk.content)
        await message_placeholder.update()
