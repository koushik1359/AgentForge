from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from agents.state import GoalState

class MemoryItem(BaseModel):
    key: str
    description: str
    retention_days: int

class MemoryConfiguration(BaseModel):
    short_term: list[MemoryItem]
    long_term: list[MemoryItem]
    storage_type: str

def configure_memory(state: GoalState) -> GoalState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(MemoryConfiguration)

    steps_summary = "\n".join(
        f"- Step {s['step_number']}: {s['name']}" for s in state["workflow_steps"]
    )

    prompt = f"""
    Design a memory configuration for this AI agent.

    Goal: {state["goal"]}
    Frequency: {state["frequency"]}
    Workflow steps:
    {steps_summary}

    Decide:
    - short_term: what to remember within a single run (cleared after each execution)
    - long_term: what to remember across multiple runs (persisted in database)
    - storage_type: where to store long-term memory (use "pgvector" for semantic search, "postgresql" for structured data)

    For each memory item provide a key, description, and how many days to retain it.
    """

    result = structured_llm.invoke([HumanMessage(content=prompt)])

    return {
        "memory_config": result.model_dump()
    }
