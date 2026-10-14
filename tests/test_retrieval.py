"""Exercise the persistent retrieval boundary without an API key or network."""

import asyncio
import re
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from support_ticket_triage.retrieval import KnowledgeBaseRetriever


class FakeEmbeddings(Embeddings):
    """Provide deterministic topic vectors and count document embedding work."""

    def __init__(self) -> None:
        self.document_batches: list[list[str]] = []
        self.queries: list[str] = []

    @staticmethod
    def _vector(text: str) -> list[float]:
        words = set(re.findall(r"[a-z]+", text.lower()))
        topics = (
            {"billing", "payment", "charges", "refund", "holds"},
            {"outage", "error", "access", "availability"},
            {"theme", "appearance", "dark", "light"},
        )
        # A small common component avoids zero-length vectors for other text.
        return [float(len(words & topic)) for topic in topics] + [0.01]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(text)


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    source = tmp_path / "knowledge"
    source.mkdir()
    (source / "billing.md").write_text(
        "# Mock billing guide\n\n"
        "Version: 1.0\n\n"
        "INTRODUCTION_NOT_RETRIEVAL_EVIDENCE\n\n"
        "## Introduction\n\n"
        "INTRODUCTORY_SECTION_NOT_RETRIEVAL_EVIDENCE\n\n"
        "## Overview\n\n"
        "OVERVIEW_NOT_RETRIEVAL_EVIDENCE\n\n"
        "## Payment holds\n\n"
        "Billing payment holds can appear as pending charges. "
        "Ask support to investigate; a refund is not confirmed.\n\n"
        "## References\n\n"
        "https://example.test/reference-not-product-policy\n\n"
        "## Research sources\n\n"
        "https://example.test/research-not-product-policy\n",
        encoding="utf-8",
    )
    (source / "appearance.md").write_text(
        "# Mock appearance guide\n\n"
        "## Theme settings\n\n"
        "Open appearance settings to select a light or dark theme.\n",
        encoding="utf-8",
    )
    (source / "agent_handbook.md").write_text(
        "# Internal handbook\n\n"
        "Version: 1.0\n\n"
        "## Billing investigation\n\n"
        "Internal billing payment investigation requires a specialist.\n",
        encoding="utf-8",
    )
    return source


def make_retriever(
    corpus: Path,
    index_path: Path,
    embeddings: FakeEmbeddings,
    *,
    model: str = "offline-test-embedding",
) -> KnowledgeBaseRetriever:
    return KnowledgeBaseRetriever(
        embeddings=embeddings,
        model=model,
        source_path=corpus,
        index_path=index_path,
    )


def source_filename(source: str) -> str:
    return Path(source.split("#", 1)[0]).name


def test_sections_retain_source_document_type_and_version(corpus: Path, tmp_path: Path):
    retriever = make_retriever(corpus, tmp_path / "index", FakeEmbeddings())

    documents = retriever.load_documents()

    assert len(documents) == 3
    assert len({doc.metadata["article_id"] for doc in documents}) == 3
    by_source = {source_filename(doc.metadata["source"]): doc for doc in documents}
    assert "Payment holds" in by_source["billing.md"].metadata["title"]
    assert "pending charges" in by_source["billing.md"].page_content
    assert by_source["billing.md"].metadata["doc_type"] == "support_guide"
    assert by_source["agent_handbook.md"].metadata["doc_type"] == "handbook"
    assert by_source["agent_handbook.md"].metadata["audience"] == "internal"
    for document in documents:
        assert document.metadata["audience"] == "internal"
        assert document.metadata["version"] == "1.0"
        assert document.metadata["mock"] is True
        assert document.metadata["embedding_model"] == "offline-test-embedding"


def test_persisted_index_can_be_reopened_without_embedding_documents(
    corpus: Path, tmp_path: Path
):
    index_path = tmp_path / "index"
    embeddings = FakeEmbeddings()
    original = make_retriever(corpus, index_path, embeddings)
    chunk_count = asyncio.run(original.build_index())

    assert chunk_count == 3
    assert embeddings.document_batches
    assert any(index_path.rglob("*"))

    reopened_embeddings = FakeEmbeddings()
    reopened = make_retriever(corpus, index_path, reopened_embeddings)
    matches = asyncio.run(reopened.search("billing payment holds", top_k=1))

    assert len(matches) == 1
    assert "pending charges" in matches[0].content
    assert matches[0].article_id
    assert source_filename(matches[0].source_url) == "billing.md"
    assert "#" in matches[0].source_url
    assert matches[0].audience == "internal"
    assert matches[0].doc_type == "support_guide"
    assert matches[0].version == "1.0"
    assert matches[0].is_mock is True
    assert reopened_embeddings.document_batches == []
    assert reopened_embeddings.queries == ["billing payment holds"]


def test_unchanged_rebuild_is_idempotent(corpus: Path, tmp_path: Path):
    embeddings = FakeEmbeddings()
    index_path = tmp_path / "index"
    retriever = make_retriever(corpus, index_path, embeddings)

    assert asyncio.run(retriever.build_index()) == 3
    batches_after_build = len(embeddings.document_batches)
    assert asyncio.run(retriever.build_index()) == 3
    assert len(embeddings.document_batches) == batches_after_build

    reopened = make_retriever(corpus, index_path, embeddings)
    assert asyncio.run(reopened.build_index()) == 3
    assert len(embeddings.document_batches) == batches_after_build


def test_changed_corpus_requires_rebuild_and_removes_deleted_articles(
    corpus: Path, tmp_path: Path
):
    embeddings = FakeEmbeddings()
    retriever = make_retriever(corpus, tmp_path / "index", embeddings)
    initial_ids = {
        source_filename(document.metadata["source"]): document.metadata["article_id"]
        for document in retriever.load_documents()
    }
    asyncio.run(retriever.build_index())

    (corpus / "billing.md").write_text(
        "# Mock billing guide\n\n"
        "## Payment holds\n\n"
        "Updated billing payment policy: settled charges require investigation.\n",
        encoding="utf-8",
    )
    (corpus / "appearance.md").unlink()
    with pytest.raises(ValueError):
        asyncio.run(retriever.search("billing payment"))
    assert embeddings.queries == []

    assert asyncio.run(retriever.build_index()) == 2
    matches = asyncio.run(retriever.search("billing payment", top_k=10))
    assert len(matches) == 1
    assert matches[0].article_id == initial_ids["billing.md"]
    assert "settled charges" in matches[0].content
    assert "pending charges" not in matches[0].content
    assert all(match.article_id != initial_ids["appearance.md"] for match in matches)


def test_handbook_requires_explicit_document_type(corpus: Path, tmp_path: Path):
    retriever = make_retriever(corpus, tmp_path / "index", FakeEmbeddings())
    asyncio.run(retriever.build_index())

    guide_matches = asyncio.run(retriever.search("billing payment", top_k=10))
    assert len(guide_matches) == 2
    assert all(match.audience == "internal" for match in guide_matches)
    assert all(match.doc_type == "support_guide" for match in guide_matches)
    assert all(
        source_filename(match.source_url) != "agent_handbook.md"
        for match in guide_matches
    )

    handbook_matches = asyncio.run(
        retriever.search("billing payment", top_k=10, doc_type="handbook")
    )
    assert len(handbook_matches) == 1
    assert source_filename(handbook_matches[0].source_url) == "agent_handbook.md"
    assert handbook_matches[0].audience == "internal"
    assert handbook_matches[0].doc_type == "handbook"
    assert handbook_matches[0].is_mock is True


def test_embedding_model_mismatch_is_rejected_before_query_embedding(
    corpus: Path, tmp_path: Path
):
    index_path = tmp_path / "index"
    asyncio.run(make_retriever(corpus, index_path, FakeEmbeddings()).build_index())
    other_embeddings = FakeEmbeddings()
    other_model = make_retriever(
        corpus, index_path, other_embeddings, model="another-embedding-model"
    )

    with pytest.raises(ValueError):
        asyncio.run(other_model.search("billing payment"))
    with pytest.raises(ValueError):
        asyncio.run(other_model.build_index())
    assert other_embeddings.queries == []
    assert other_embeddings.document_batches == []

    assert asyncio.run(other_model.build_index(rebuild=True)) == 3
    assert other_embeddings.document_batches
    matches = asyncio.run(other_model.search("billing payment", top_k=1))
    assert source_filename(matches[0].source_url) == "billing.md"


def test_explicit_model_rebuild_preserves_other_collections(corpus: Path, tmp_path: Path):
    index_path = tmp_path / "index"
    original = make_retriever(corpus, index_path, FakeEmbeddings())
    unrelated_embeddings = FakeEmbeddings()
    unrelated = KnowledgeBaseRetriever(
        embeddings=unrelated_embeddings,
        model="offline-test-embedding",
        source_path=corpus,
        index_path=index_path,
        collection_name="unrelated_knowledge",
    )
    asyncio.run(original.build_index())
    asyncio.run(unrelated.build_index())
    unrelated_batches = len(unrelated_embeddings.document_batches)

    replacement = make_retriever(
        corpus, index_path, FakeEmbeddings(), model="replacement-embedding-model"
    )
    assert asyncio.run(replacement.build_index(rebuild=True)) == 3

    matches = asyncio.run(unrelated.search("billing payment", top_k=1))
    assert source_filename(matches[0].source_url) == "billing.md"
    assert asyncio.run(unrelated.build_index()) == 3
    assert len(unrelated_embeddings.document_batches) == unrelated_batches


def test_introductions_and_references_are_not_embedded(corpus: Path, tmp_path: Path):
    embeddings = FakeEmbeddings()
    retriever = make_retriever(corpus, tmp_path / "index", embeddings)
    asyncio.run(retriever.build_index())

    embedded = "\n".join(text for batch in embeddings.document_batches for text in batch)
    assert "pending charges" in embedded
    assert "INTRODUCTION_NOT_RETRIEVAL_EVIDENCE" not in embedded
    assert "INTRODUCTORY_SECTION_NOT_RETRIEVAL_EVIDENCE" not in embedded
    assert "OVERVIEW_NOT_RETRIEVAL_EVIDENCE" not in embedded
    assert "reference-not-product-policy" not in embedded
    assert "research-not-product-policy" not in embedded


def test_similarity_cutoff_can_return_no_useful_evidence(corpus: Path, tmp_path: Path):
    retriever = make_retriever(corpus, tmp_path / "index", FakeEmbeddings())
    asyncio.run(retriever.build_index())

    matches = asyncio.run(
        retriever.search("billing payment", top_k=10, minimum_similarity=0.8)
    )
    assert len(matches) == 1
    assert source_filename(matches[0].source_url) == "billing.md"
    unrelated = asyncio.run(
        retriever.search("outage error availability", minimum_similarity=0.8)
    )
    assert unrelated == []


def test_missing_index_reports_preparation_step(corpus: Path, tmp_path: Path):
    embeddings = FakeEmbeddings()
    retriever = make_retriever(corpus, tmp_path / "missing-index", embeddings)

    with pytest.raises(FileNotFoundError):
        asyncio.run(retriever.search("billing payment"))
    assert embeddings.queries == []
    assert embeddings.document_batches == []


def test_invalid_search_input_does_not_make_embedding_requests(
    corpus: Path, tmp_path: Path
):
    embeddings = FakeEmbeddings()
    retriever = make_retriever(corpus, tmp_path / "index", embeddings)
    asyncio.run(retriever.build_index())

    for query, top_k in [("", 3), ("   ", 3), ("billing", 0), ("billing", -1)]:
        with pytest.raises(ValueError):
            asyncio.run(retriever.search(query, top_k=top_k))
    for minimum_similarity in [-1.1, 1.1, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            asyncio.run(retriever.search("billing", minimum_similarity=minimum_similarity))
    with pytest.raises(ValueError):
        asyncio.run(retriever.search("billing", doc_type="all"))
    assert embeddings.queries == []
