from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from agents.state import GoalState

# Define the exact shape we want from the LLM
class WorkflowStep(BaseModel):
    step_number: int
    name: str
    description: str
    tool: str

class WorkflowDesign(BaseModel):
    steps: list[WorkflowStep]

def design_workflow(state: GoalState) -> GoalState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # with_structured_output enforces the schema — no manual JSON parsing needed
    structured_llm = llm.with_structured_output(WorkflowDesign)

    tools_summary = "\n".join(
        f"- {t['name']}: {t['purpose']}" for t in state["tool_configurations"]
    )

    prompt = f"""
    Design a step-by-step workflow for this AI agent.

    Goal: {state["goal"]}
    Frequency: {state["frequency"]}
    Output: {state["output_type"]}
    Available tools:
    {tools_summary}

    Create a logical sequence of steps the agent should follow to complete its task.
    Each step should use one of the available tools.
    """

    result = structured_llm.invoke([HumanMessage(content=prompt)])

    return {
        "workflow_steps": [step.model_dump() for step in result.steps]
    }