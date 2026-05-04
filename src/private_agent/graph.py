"""LangGraph workflow for retrieval-augmented enterprise Q&A."""

from __future__ import annotations

import logging
from typing import TypedDict

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from private_agent.config import Settings, load_settings
from private_agent.modeling import create_chat_model, create_embeddings

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Shared graph state."""

    question: str
    context: list[Document]
    answer: str


_SYSTEM_PROMPT = (
    "You are a private enterprise AI assistant.\n"
    "Rules:\n"
    "1) Use only provided context and do not invent facts.\n"
    "2) If context is insufficient, answer exactly: "
    '"I do not have enough internal knowledge to answer this."\n'
    "3) For HR, legal, financial, security, or other high-risk decisions, "
    "recommend human review.\n"
    "4) Keep answers concise, professional, and include relevant source filenames."
)


def _format_context(docs: list[Document]) -> str:
    """Format retrieved docs into a prompt-ready context block."""
    parts: list[str] = []

    for doc in docs:
        source_name = str(doc.metadata.get("source", "unknown"))
        page_value = doc.metadata.get("page")
        page_suffix = f" (page {page_value})" if page_value is not None else ""
        parts.append(f"Source: {source_name}{page_suffix}\n{doc.page_content}")

    return "\n\n".join(parts)


def _build_retriever(settings: Settings):
    """Create a retriever backed by persisted Chroma vectors."""
    vector_store = Chroma(
        collection_name=settings.collection_name,
        embedding_function=create_embeddings(settings),
        persist_directory=str(settings.vector_db_dir),
    )
    return vector_store.as_retriever(search_kwargs={"k": settings.retriever_k})


def build_graph(settings: Settings | None = None):
    """Build and compile the retrieval + generation workflow graph."""
    runtime_settings = settings or load_settings()

    retriever = _build_retriever(runtime_settings)
    llm = create_chat_model(runtime_settings)

    def retrieve(state: AgentState) -> AgentState:
        """Retrieve top-K relevant chunks for the user question."""
        question = state["question"].strip()
        docs = retriever.invoke(question) if question else []

        logger.info("Retrieved %d documents for question.", len(docs))

        return {
            **state,
            "context": docs,
        }

    def generate_answer(state: AgentState) -> AgentState:
        """Generate final answer constrained by retrieved context."""
        docs = state["context"]
        if not docs:
            return {
                **state,
                "answer": "I do not have enough internal knowledge to answer this.",
            }

        user_prompt = (
            "Question:\n"
            f"{state['question']}\n\n"
            "Context:\n"
            f"{_format_context(docs)}\n\n"
            "Respond using the policy above."
        )

        response = llm.invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]
        )

        return {
            **state,
            "answer": str(response.content).strip(),
        }

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", generate_answer)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()
