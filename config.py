import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    auth_username: str
    auth_password: str
    ollama_model: str
    embedding_model: str
    temperature: float
    chunk_size: int
    chunk_overlap: int
    retriever_top_k: int
    data_dir: Path
    prompt_path: Path
    supported_extensions: tuple[str, ...]

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            auth_username=os.getenv("AUTH_USERNAME", "admin"),
            auth_password=os.getenv("AUTH_PASSWORD", "changeme"),
            ollama_model=os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud"),
            embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            retriever_top_k=int(os.getenv("RETRIEVER_TOP_K", "3")),
            data_dir=Path(os.getenv("DATA_DIR", "data")),
            prompt_path=Path(os.getenv("PROMPT_PATH", "prompts/rag_prompt.yaml")),
            supported_extensions=(".pdf", ".txt"),
        )


settings = Settings.from_env()
