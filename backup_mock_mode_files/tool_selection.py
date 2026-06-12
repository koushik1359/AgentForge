from __future__ import annotations

import json
import os
from typing import Any

from agents.state import GoalState


def _mock_enabled() -> bool:
    return os.getenv("AGENTFORGE_MOCK_MODE", "true").lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
        "mock",
    }


def _clean_json_response(content: str) -> str:
    content = content.strip()

    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1].strip()

    if content.lower().startswith("json"):
        content = content[4:].strip()

    return content


def _safe_tool_name(tool: str) -> str:
    return (
        tool.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def _mock_tool_configurations(state: GoalState) -> list[dict[str, Any]]:
    tools_needed = state.get("tools_needed") or []

    if not tools_needed:
        tools_needed = [
            "input_analyzer",
            "task_planner",
            "response_generator",
        ]

    tool_configurations = []

    for tool in tools_needed:
        tool_name = _safe_tool_name(str(tool))

        tool_configurations.append(
            {
                "name": tool_name,
                "purpose": f"handle {tool_name.replace('_', ' ')} related operations for the agent goal.",
                "config": {
                    "mode": "mock",
                    "requires_api_key": False,
                    "timeout_seconds": 30,
                    "retry_count": 2,
                },
            }
        )

    return tool_configurations


def select_tools(state: GoalState) -> GoalState:
    if _mock_enabled():
        return {
            "tool_configurations": _mock_tool_configurations(state)
        }

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Either add an OpenAI API key or run with "
            "AGENTFORGE_MOCK_MODE=true."
        )

    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
Based on this agent's requirements, provide specific tool configurations.

Goal: {state.get("goal")}
Frequency: {state.get("frequency")}
Output: {state.get("output_type")}
Tools needed: {state.get("tools_needed")}

For each tool specify:
- name: tool name
- purpose: what it does in this agent
- config: key settings needed to use it

Respond ONLY with a valid JSON array, no markdown, no code blocks:

[
  {{
    "name": "...",
    "purpose": "...",
    "config": {{}}
  }}
]
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    content = _clean_json_response(response.content)
    result = json.loads(content)

    return {
        "tool_configurations": result
    }