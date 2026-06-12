from __future__ import annotations

import os
from dotenv import load_dotenv

from agents.state import GoalState

load_dotenv()


def _mock_enabled() -> bool:
    return os.getenv("AGENTFORGE_MOCK_MODE", "true").lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
        "mock",
    }


def _choose_storage_type(state: GoalState) -> str:
    goal = str(state.get("goal", "")).lower()
    tools = state.get("tool_configurations") or []
    tool_text = " ".join(str(tool).lower() for tool in tools)

    semantic_keywords = [
        "search",
        "retrieve",
        "knowledge",
        "document",
        "embedding",
        "rag",
        "research",
    ]

    if any(keyword in goal or keyword in tool_text for keyword in semantic_keywords):
        return "pgvector"

    return "postgresql"


def _mock_memory_config(state: GoalState) -> dict:
    storage_type = _choose_storage_type(state)

    return {
        "short_term": [
            {
                "key": "current_user_request",
                "description": "Stores the current user request during one execution of the agent.",
                "retention_days": 1,
            },
            {
                "key": "intermediate_results",
                "description": "Stores temporary outputs from workflow steps before final response generation.",
                "retention_days": 1,
            },
        ],
        "long_term": [
            {
                "key": "successful_workflows",
                "description": "Stores successful workflow patterns for improving future agent runs.",
                "retention_days": 30,
            },
            {
                "key": "user_output_preferences",
                "description": "Stores stable user preferences about response format and output style.",
                "retention_days": 90,
            },
        ],
        "storage_type": storage_type,
    }


def configure_memory(state: GoalState) -> GoalState:
    if _mock_enabled():
        return {
            "memory_config": _mock_memory_config(state)
        }

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Either add an OpenAI API key or run with "
            "AGENTFORGE_MOCK_MODE=true."
        )

    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel

    class MemoryItem(BaseModel):
        key: str
        description: str
        retention_days: int

    class MemoryConfiguration(BaseModel):
        short_term: list[MemoryItem]
        long_term: list[MemoryItem]
        storage_type: str

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(MemoryConfiguration)

    workflow_steps = state.get("workflow_steps") or []
    steps_summary = "\n".join(
        f"- Step {step.get('step_number', '?')}: {step.get('name', 'Unnamed step')}"
        for step in workflow_steps
    )

    prompt = f"""
Design a memory configuration for this AI agent.

Goal: {state.get("goal")}
Frequency: {state.get("frequency")}

Workflow steps:
{steps_summary}

Decide:
- short_term: what to remember within a single run, cleared after each execution
- long_term: what to remember across multiple runs, persisted in database
- storage_type: where to store long-term memory

Use:
- "pgvector" for semantic search or RAG-style memory
- "postgresql" for structured memory

For each memory item provide:
- key
- description
- retention_days
"""

    result = structured_llm.invoke([HumanMessage(content=prompt)])

    return {
        "memory_config": result.model_dump()
    }
