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

def _order_tools(tools: list[dict]) -> list[dict]:
    priority = {
        "input_analyzer": 1,
        "web_search": 2,
        "linkedin_monitor": 2,
        "job_search": 3,
        "news_filter": 3,
        "ai_news_monitor": 4,
        "document_loader": 4,
        "research_paper_summarizer": 5,
        "summarizer": 6,
        "report_generator": 7,
        "notification_sender": 8,
        "notification": 8,
        "response_generator": 9,
    }

    return sorted(
        tools,
        key=lambda tool: priority.get(tool.get("name", ""), 50)
    )


def _mock_workflow_steps(state: GoalState) -> list[dict]:
    goal = state.get("goal", "complete the requested task")
    tools = _order_tools(state.get("tool_configurations") or [])

    steps = [
        {
            "step_number": 1,
            "name": "Understand User Objective",
            "description": f"Read the user request and identify the main goal: {goal}",
            "tool": "internal_reasoning",
        }
    ]

    step_number = 2

    if tools:
        for tool in tools:
            tool_name = tool.get("name", "unknown_tool")
            tool_purpose = tool.get("purpose", "support the task")

            steps.append(
                {
                    "step_number": step_number,
                    "name": f"Use {tool_name}",
                    "description": f"Use {tool_name} to {tool_purpose}",
                    "tool": tool_name,
                }
            )
            step_number += 1
    else:
        steps.append(
            {
                "step_number": step_number,
                "name": "Plan Task Internally",
                "description": "Create a simple internal plan because no external tools are configured.",
                "tool": "internal_reasoning",
            }
        )
        step_number += 1

    steps.append(
        {
            "step_number": step_number,
            "name": "Generate Final Output",
            "description": "Prepare the final response in the required output format.",
            "tool": "response_generator",
        }
    )

    return steps


def design_workflow(state: GoalState) -> GoalState:
    if _mock_enabled():
        return {
            "workflow_steps": _mock_workflow_steps(state)
        }

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Either add an OpenAI API key or run with "
            "AGENTFORGE_MOCK_MODE=true."
        )

    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
    from pydantic import BaseModel

    class WorkflowStep(BaseModel):
        step_number: int
        name: str
        description: str
        tool: str

    class WorkflowDesign(BaseModel):
        steps: list[WorkflowStep]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(WorkflowDesign)

    tools = _order_tools(state.get("tool_configurations") or [])
    tools_summary = "\n".join(
        f"- {tool.get('name', 'unknown_tool')}: {tool.get('purpose', 'No purpose provided')}"
        for tool in tools
    )

    prompt = f"""
Design a step-by-step workflow for this AI agent.

Goal: {state.get("goal")}
Frequency: {state.get("frequency")}
Output: {state.get("output_type")}

Available tools:
{tools_summary}

Create a logical sequence of steps the agent should follow to complete its task.
Each step should use one of the available tools when relevant.
"""

    result = structured_llm.invoke([HumanMessage(content=prompt)])

    return {
        "workflow_steps": [step.model_dump() for step in result.steps]
    }
