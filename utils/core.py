import asyncio
import logging
from pathlib import Path

import ollama
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
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
        splits = self.text_splitter.split_documents(documents)
        for document in splits:
            document.metadata["filename"] = element.name
        return ext, splits

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

    def apply_retriever_settings(self, retriever_top_k: int) -> None:
        self.retriever_top_k = retriever_top_k
        if self.has_documents():
            self.init_retriever()

    def has_documents(self) -> bool:
        try:
            return self.vectorstore._collection.count() > 0
        except Exception:
            return False

    def _matches_source(self, metadata: dict, source_name: str) -> bool:
        if metadata.get("filename") == source_name:
            return True

        source = metadata.get("source", "")
        return source_name in source or Path(source).name == source_name

    def get_all_documents(self, source_name: str | None = None) -> list[Document]:
        collection = self.vectorstore._collection
        result = collection.get(include=["documents", "metadatas"])
        documents: list[Document] = []

        for content, metadata in zip(
            result.get("documents") or [],
            result.get("metadatas") or [],
        ):
            metadata = metadata or {}
            documents.append(Document(page_content=content, metadata=metadata))

        if not source_name:
            return documents

        filtered = [
            document
            for document in documents
            if self._matches_source(document.metadata, source_name)
        ]
        if filtered:
            return filtered

        # Chunks indexed before filename metadata was added only store a temp path.
        sources = {document.metadata.get("source") for document in documents}
        if documents and len(sources) == 1:
            return documents

        return []

    def get_documents_grouped(
        self, source_name: str | None = None
    ) -> dict[str, list[Document]]:
        grouped: dict[str, list[Document]] = {}

        for document in self.get_all_documents(source_name=source_name):
            filename = document.metadata.get("filename")
            if not filename:
                source = document.metadata.get("source", "")
                filename = Path(source).name if source else "unknown"

            grouped.setdefault(filename, []).append(document)

        for filename in grouped:
            grouped[filename].sort(
                key=lambda doc: (
                    doc.metadata.get("page", 0),
                    str(doc.metadata.get("page_label", "")),
                )
            )

        return grouped


class DocLLM:
    """Manages the LLM call and RAG chain."""

    def __init__(
        self,
        model_name: str,
        temperature: float,
        prompt_path: str,
        summarize_prompt_path: str,
        summarize_map_prompt_path: str,
        summarize_batch_chars: int,
        summarize_max_batches: int,
        summarize_parallel_requests: int,
        num_predict: int,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.num_predict = num_predict
        self.chain = None
        self.session_id = "default"
        self.summarize_batch_chars = summarize_batch_chars
        self.summarize_max_batches = summarize_max_batches
        self.summarize_parallel_requests = summarize_parallel_requests
        self.prompt = load_prompt(prompt_path)
        self.summarize_prompt = load_prompt(summarize_prompt_path)
        self.summarize_map_prompt = load_prompt(summarize_map_prompt_path)
        self.chat_model = self._build_chat_model()
        self.store: dict[str, ChatMessageHistory] = {}

    def _build_chat_model(self) -> ChatOllama:
        return ChatOllama(
            model=self.model_name,
            temperature=self.temperature,
            num_predict=self.num_predict,
        )

    def get_model_catalog(self) -> list[dict[str, int | str]]:
        response = ollama.list()
        catalog: list[dict[str, int | str]] = []

        for model in response.models:
            name = model.model
            family = getattr(model.details, "family", "").lower()
            if any(
                keyword in name.lower() or keyword in family
                for keyword in ["embed", "bge", "minilm", "bert"]
            ):
                continue

            catalog.append({"name": name, "size": model.size or 0})

        return catalog

    def get_available_models(self) -> list[str]:
        return [str(item["name"]) for item in self.get_model_catalog()]

    def apply_model_settings(
        self,
        model_name: str,
        temperature: float,
        retriever=None,
        session_id: str | None = None,
        num_predict: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        if num_predict is not None:
            self.num_predict = num_predict
        self.chat_model = self._build_chat_model()

        if session_id is not None:
            self.session_id = session_id

        if retriever is not None:
            self.init_chain(retriever, session_id=self.session_id)
        else:
            self.chain = None

    def apply_summarize_settings(
        self,
        summarize_max_batches: int,
        summarize_parallel_requests: int,
    ) -> None:
        self.summarize_max_batches = summarize_max_batches
        self.summarize_parallel_requests = summarize_parallel_requests

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

    def _batch_texts(self, documents: list[Document]) -> list[str]:
        batches: list[str] = []
        current_parts: list[str] = []
        current_length = 0

        for document in documents:
            content = document.page_content
            if (
                current_parts
                and current_length + len(content) > self.summarize_batch_chars
            ):
                batches.append("\n\n".join(current_parts))
                current_parts = [content]
                current_length = len(content)
            else:
                current_parts.append(content)
                current_length += len(content)

        if current_parts:
            batches.append("\n\n".join(current_parts))

        return self._limit_batches(batches)

    def _limit_batches(self, batches: list[str]) -> list[str]:
        if len(batches) <= self.summarize_max_batches:
            return batches

        merged: list[str] = []
        group_size = len(batches) / self.summarize_max_batches
        for index in range(self.summarize_max_batches):
            start = int(index * group_size)
            end = (
                len(batches)
                if index == self.summarize_max_batches - 1
                else int((index + 1) * group_size)
            )
            merged.append("\n\n".join(batches[start:end]))
        return merged

    async def _summarize_text(self, text: str, *, map_phase: bool = False) -> str:
        prompt = self.summarize_map_prompt if map_phase else self.summarize_prompt
        chain = prompt | self.chat_model
        response = await chain.ainvoke({"context": text})
        return response.content

    async def _summarize_batches(self, batches: list[str]) -> list[str]:
        semaphore = asyncio.Semaphore(self.summarize_parallel_requests)

        async def summarize_one(batch: str) -> str:
            async with semaphore:
                return await self._summarize_text(batch, map_phase=True)

        return list(await asyncio.gather(*(summarize_one(batch) for batch in batches)))

    async def _summarize_document_chunks(self, documents: list[Document]) -> str:
        batches = self._batch_texts(documents)

        if len(batches) == 1:
            return await self._summarize_text(batches[0])

        section_summaries = await self._summarize_batches(batches)
        combined = (
            "The following are summaries of different sections of the same "
            "document. Combine them into one coherent summary:\n\n"
            + "\n\n---\n\n".join(section_summaries)
        )
        return await self._summarize_text(combined)

    async def _stream_summary(self, text: str, message_placeholder: cl.Message) -> None:
        chain = self.summarize_prompt | self.chat_model
        async for chunk in chain.astream({"context": text}):
            await message_placeholder.stream_token(chunk.content)

    async def summarize(
        self,
        vector_store: "VectorStoreManager",
        message_placeholder: cl.Message,
        source_name: str | None = None,
    ) -> None:
        groups = vector_store.get_documents_grouped(source_name=source_name)
        if not groups:
            label = source_name or "the uploaded documents"
            raise ValueError(f"No content found for {label}.")

        if len(groups) == 1:
            documents = next(iter(groups.values()))
            batches = self._batch_texts(documents)
            logger.info(
                "Summarizing %s chunks in %s batch(es)",
                len(documents),
                len(batches),
            )

            if len(batches) == 1:
                await self._stream_summary(batches[0], message_placeholder)
            else:
                summary = await self._summarize_document_chunks(documents)
                await message_placeholder.stream_token(summary)

            await message_placeholder.update()
            return

        logger.info("Summarizing %s documents separately", len(groups))
        for index, (filename, documents) in enumerate(sorted(groups.items())):
            if index > 0:
                await message_placeholder.stream_token("\n\n")
            await message_placeholder.stream_token(f"## {filename}\n\n")
            summary = await self._summarize_document_chunks(documents)
            await message_placeholder.stream_token(summary)

        await message_placeholder.update()
