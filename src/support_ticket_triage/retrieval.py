"""Versioned Markdown references indexed in persistent Chroma through LangChain."""

import asyncio
import hashlib
import math
import os
import re
from pathlib import Path
from typing import Literal

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from support_ticket_triage.schemas import KnowledgeBaseArticle

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_PATH = REPOSITORY_ROOT / "data" / "knowledge"
CHROMA_PATH = REPOSITORY_ROOT / ".rag-cache" / "chroma"
DocumentType = Literal["support_guide", "handbook"]
NON_TOPICAL_HEADINGS = {
    "introduction", "overview", "references", "sources", "research sources",
    "research and provenance",
}


def load_handbook(source_path: Path | None = None) -> str:
    """Load mandatory policy directly, independently of retrieval or credentials."""
    path = (source_path or KNOWLEDGE_PATH) / "agent_handbook.md"
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("The mandatory agent handbook is empty.")
    return text


class KnowledgeBaseRetriever:
    """Own reference ingestion and retrieval; matching does not authorize a reply."""

    def __init__(
        self,
        embeddings: Embeddings,
        model: str,
        source_path: Path | None = None,
        index_path: Path | None = None,
        collection_name: str = "support_knowledge",
    ) -> None:
        self.embeddings = embeddings
        self.model = model
        self.source_path = (source_path or KNOWLEDGE_PATH).resolve()
        self.index_path = (index_path or CHROMA_PATH).resolve()
        self.collection_name = collection_name

    def _source_files(self) -> list[Path]:
        paths = sorted(self.source_path.glob("*.md"))
        if not paths:
            raise ValueError("Add Markdown references to the knowledge source directory.")
        return paths

    def _fingerprint(self) -> str:
        digest = hashlib.sha256()
        for path in self._source_files():
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    def load_documents(self) -> list[Document]:
        """Make one searchable document per topical second-level heading."""
        documents = []
        ids = set()
        for path in self._source_files():
            text = path.read_text(encoding="utf-8")
            version_match = re.search(r"(?mi)^Version:\s*(\S+)", text)
            version = version_match.group(1) if version_match else "1.0"
            doc_type = "handbook" if path.name == "agent_handbook.md" else "support_guide"
            try:
                source = path.relative_to(REPOSITORY_ROOT).as_posix()
            except ValueError:
                source = path.name
            for section in re.split(r"(?m)^##[ \t]+", text)[1:]:
                title, _, body = section.partition("\n")
                title, body = title.strip(), body.strip()
                if not body or title.casefold() in NON_TOPICAL_HEADINGS:
                    continue
                slug = re.sub(r"[^a-z0-9]+", "-", title.casefold()).strip("-")
                article_id = f"{path.stem}:{slug}"
                if not slug or article_id in ids:
                    raise ValueError("Knowledge sections need unique, descriptive headings.")
                ids.add(article_id)
                documents.append(Document(
                    page_content=f"{title}\n\n{body}",
                    metadata={
                        "article_id": article_id, "title": title,
                        "source": f"{source}#{slug}", "version": version,
                        "doc_type": doc_type, "audience": "internal", "mock": True,
                        "embedding_model": self.model,
                    },
                ))
        if not documents:
            raise ValueError("Knowledge references have no topical second-level sections.")
        return documents

    def _open_index(
        self, *, create: bool = False, rebuild: bool = False
    ) -> tuple[Chroma, Collection]:
        if not create and not (self.index_path / "chroma.sqlite3").is_file():
            raise FileNotFoundError("Build the Chroma knowledge index before searching.")
        client = chromadb.PersistentClient(
            path=str(self.index_path), settings=Settings(anonymized_telemetry=False)
        )
        names = {item.name for item in client.list_collections()}
        if rebuild and self.collection_name in names:
            client.delete_collection(self.collection_name)
        if create:
            collection = client.get_or_create_collection(
                self.collection_name, embedding_function=None,
                metadata={"embedding_model": self.model},
                configuration={"hnsw": {"space": "cosine"}},
            )
        else:
            if self.collection_name not in names:
                raise FileNotFoundError("Build the Chroma knowledge collection before searching.")
            collection = client.get_collection(self.collection_name, embedding_function=None)
        if (collection.metadata or {}).get("embedding_model") != self.model:
            raise ValueError("The embedding model changed; rebuild the knowledge index explicitly.")
        store = Chroma(
            client=client, collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )
        return store, collection

    async def build_index(self, *, rebuild: bool = False) -> int:
        """Upsert changed sections and remove deleted ones; unchanged builds are free."""
        documents = self.load_documents()
        fingerprint = self._fingerprint()
        store, collection = await asyncio.to_thread(
            self._open_index, create=True, rebuild=rebuild
        )
        existing = await asyncio.to_thread(collection.get, include=["metadatas"])
        desired = {doc.metadata["article_id"]: doc for doc in documents}
        previous = dict(zip(existing["ids"], existing["metadatas"]))
        changed = []
        for article_id, document in desired.items():
            section_hash = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()
            document.metadata["content_hash"] = section_hash
            if previous.get(article_id) != document.metadata:
                changed.append(document)
        # Invalidate before writes so an interrupted build cannot appear current.
        metadata = dict(collection.metadata or {})
        await asyncio.to_thread(collection.modify, metadata={**metadata, "corpus_hash": "building"})
        if changed:
            await store.aadd_documents(changed, ids=[doc.metadata["article_id"] for doc in changed])
        deleted = sorted(set(previous) - set(desired))
        if deleted:
            await asyncio.to_thread(collection.delete, ids=deleted)
        await asyncio.to_thread(collection.modify, metadata={**metadata, "corpus_hash": fingerprint})
        return len(documents)

    async def search(
        self, query: str, *, top_k: int = 3,
        minimum_similarity: float | None = None,
        doc_type: DocumentType = "support_guide",
    ) -> list[KnowledgeBaseArticle]:
        """Search an existing index; support procedures are the default evidence."""
        if not query.strip() or top_k < 1:
            raise ValueError("Search needs a non-empty query and a positive top_k.")
        if doc_type not in {"support_guide", "handbook"}:
            raise ValueError("Choose support_guide or handbook document type.")
        if minimum_similarity is not None and (
            not math.isfinite(minimum_similarity) or not -1 <= minimum_similarity <= 1
        ):
            raise ValueError("The cosine similarity cutoff must be finite and between -1 and 1.")
        store, collection = await asyncio.to_thread(self._open_index)
        if (collection.metadata or {}).get("corpus_hash") != self._fingerprint():
            raise ValueError("The knowledge sources changed; rebuild the index before searching.")
        matches = await store.asimilarity_search_with_score(
            query, k=top_k, filter={"doc_type": doc_type}
        )
        return [KnowledgeBaseArticle(
            article_id=doc.metadata["article_id"], title=doc.metadata["title"],
            content=doc.page_content, source_url=doc.metadata["source"],
            doc_type=doc.metadata["doc_type"], version=doc.metadata["version"],
            audience=doc.metadata["audience"], is_mock=doc.metadata["mock"],
        ) for doc, distance in matches
            if minimum_similarity is None or 1 - distance >= minimum_similarity]


def configured_retriever(
    *, source_path: Path | None = None, index_path: Path | None = None,
    model: str | None = None,
) -> KnowledgeBaseRetriever:
    """Create an embedding client only when explicitly preparing or querying knowledge."""
    load_dotenv(REPOSITORY_ROOT / ".env")
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY in the project's .env file.")
    model = model or os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    embeddings = OpenAIEmbeddings(model=model, api_key=key, max_retries=1, request_timeout=30)
    return KnowledgeBaseRetriever(embeddings, model, source_path, index_path)
