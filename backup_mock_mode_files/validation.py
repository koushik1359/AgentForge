from __future__ import annotations

import os
from dotenv import load_dotenv
from pydantic import BaseModel

from agents.state import GoalState

load_dotenv()


class ValidationResult(BaseModel):
    passed: bool
    score: int
    issues: list[str]
    recommendations: list[str]


def _mock_enabled() -> bool:
    return os.getenv("AGENTFORGE_MOCK_MODE", "true").lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
        "mock",
    }


def _mock_validate_agent(state: GoalState) -> dict:
    issues = []
    recommendations = []

    if not state.get("goal"):
        issues.append("Agent goal is missing.")

    if not state.get("tools_needed"):
        recommendations.append("Add at least one tool if the agent needs external actions.")

    if not state.get("workflow_steps"):
        issues.append("Workflow steps are missing.")

    if not state.get("system_prompt"):
        issues.append("System prompt is missing.")

    if not state.get("memory_config"):
        recommendations.append("Add a memory configuration for better long-term agent behavior.")

    score = 90

    if issues:
        score -= 20

    if len(recommendations) > 1:
        score -= 5

    score = max(score, 0)

    return {
        "passed": score >= 70,
        "score": score,
        "issues": issues,
        "recommendations": recommendations
        if recommendations
        else ["Mock validation passed. Real LLM validation can be enabled later with OPENAI_API_KEY."],
    }


def validate_agent(state: GoalState) -> GoalState:
    if _mock_enabled():
        return {
            "validation_result": _mock_validate_agent(state)
        }

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Either add an OpenAI API key or run with "
            "AGENTFORGE_MOCK_MODE=true."
        )

    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

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

Tools configured:
{[t["name"] for t in state["tool_configurations"]]}

Workflow steps:
{steps_summary}

System prompt length:
{len(state["system_prompt"])} characters

Memory storage:
{state["memory_config"].get("storage_type", "none")}

Check:
- Are the tools sufficient to achieve the goal?
- Does the workflow cover all necessary steps?
- Is anything missing or potentially broken?
- Are there any security concerns?

Give a score out of 100.
Passed should be true if score is 70 or above.
List specific issues and recommendations.
"""

    result = structured_llm.invoke([HumanMessage(content=prompt)])

    return {
        "validation_result": result.model_dump()
    }
