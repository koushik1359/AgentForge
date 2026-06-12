import os

from pydantic import BaseModel, Field

from agents.state import GoalState


class ValidationResult(BaseModel):
    passed: bool = Field(description="Whether the generated agent blueprint passed validation")
    score: int = Field(description="Validation score from 0 to 100")
    issues: list[str] = Field(description="Problems found in the generated blueprint")
    recommendations: list[str] = Field(description="Suggestions to improve the generated agent")


def _dump_model(model: BaseModel) -> dict:
    """
    Supports both Pydantic v1 and v2.
    """

    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


def _mock_validation(state: GoalState) -> GoalState:
    issues = []
    recommendations = []

    goal = state.get("goal")
    frequency = state.get("frequency")
    output_type = state.get("output_type")
    tool_configurations = state.get("tool_configurations", [])
    system_prompt = state.get("system_prompt", "")
    workflow_steps = state.get("workflow_steps", [])
    memory_config = state.get("memory_config", {})

    score = 100

    if not goal:
        issues.append("Missing agent goal.")
        recommendations.append("Add a clear goal for the generated agent.")
        score -= 20

    if not frequency:
        issues.append("Missing execution frequency.")
        recommendations.append("Define whether the agent runs on-demand, daily, weekly, etc.")
        score -= 10

    if not output_type:
        issues.append("Missing output type.")
        recommendations.append("Define the expected output format.")
        score -= 10

    if not tool_configurations:
        issues.append("No tools configured.")
        recommendations.append("Add at least one tool required for the generated agent.")
        score -= 20

    if not system_prompt or len(system_prompt) < 50:
        issues.append("System prompt is missing or too short.")
        recommendations.append("Generate a stronger system prompt with role, goal, tool usage, and output rules.")
        score -= 15

    if not workflow_steps:
        issues.append("Workflow steps are missing.")
        recommendations.append("Create a step-by-step workflow for the generated agent.")
        score -= 20

    if not memory_config:
        issues.append("Memory configuration is missing.")
        recommendations.append("Add short-term and long-term memory configuration.")
        score -= 10

    score = max(score, 0)

    passed = score >= 70

    if passed and not issues:
        recommendations.append("Generated agent blueprint is ready for MVP execution.")

    return {
        "validation_result": {
            "passed": passed,
            "score": score,
            "issues": issues,
            "recommendations": recommendations,
        }
    }


def validate_agent(state: GoalState) -> GoalState:
    """
    Validation Agent.

    Uses OpenAI when OPENAI_API_KEY is available.
    Uses deterministic mock mode when no API key is available.
    """

    if not os.getenv("OPENAI_API_KEY"):
        return _mock_validation(state)

    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(ValidationResult)

    prompt = f"""
You are the Validation Agent inside AgentForge.

Validate the generated AI agent blueprint.

Generated agent blueprint:
Goal: {state.get("goal")}
Frequency: {state.get("frequency")}
Output type: {state.get("output_type")}
Tools: {state.get("tool_configurations")}
System prompt: {state.get("system_prompt")}
Workflow steps: {state.get("workflow_steps")}
Memory config: {state.get("memory_config")}

Validation criteria:
1. Goal must be clear.
2. Tools must match the goal.
3. Workflow must be logical and executable.
4. System prompt must be useful and safe.
5. Memory configuration must be relevant.
6. No secrets, passwords, or API keys should be included.
7. The generated agent should be good enough for an MVP demo.

Return:
- passed
- score from 0 to 100
- issues
- recommendations
"""

    result = structured_llm.invoke([HumanMessage(content=prompt)])

    return {
        "validation_result": _dump_model(result)
    }