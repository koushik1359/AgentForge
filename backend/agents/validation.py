from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from agents.state import GoalState

class ValidationResult(BaseModel):
    passed: bool
    score: int
    issues: list[str]
    recommendations: list[str]

def validate_agent(state: GoalState) -> GoalState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(ValidationResult)

    steps_summary = "\n".join(
        f"- Step {s['step_number']}: {s['name']} (tool: {s['tool']})"
        for s in state["workflow_steps"]
    )

    prompt = f"""
    Validate this AI agent blueprint and check if it is ready for deployment.

    Goal: {state["goal"]}
    Frequency: {state["frequency"]}
    Output: {state["output_type"]}
    Tools configured: {[t['name'] for t in state["tool_configurations"]]}
    Workflow steps:
    {steps_summary}
    System prompt length: {len(state["system_prompt"])} characters
    Memory storage: {state["memory_config"].get("storage_type", "none")}

    Check for:
    - Are the tools sufficient to achieve the goal?
    - Does the workflow cover all necessary steps?
    - Is there anything missing or potentially broken?
    - Are there any security concerns?

    Give a score out of 100. Passed if score >= 70.
    List specific issues if any, and recommendations to improve.
    """

    result = structured_llm.invoke([HumanMessage(content=prompt)])
    return {"validation_result": result.model_dump()}
