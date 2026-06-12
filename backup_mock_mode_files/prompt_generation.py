from __future__ import annotations

import os

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


def _build_tools_summary(state: GoalState) -> str:
    tools = state.get("tool_configurations") or []

    if not tools:
        return "- No external tools configured. Use internal reasoning and structured output."

    return "\n".join(
        f"- {tool.get('name', 'unknown_tool')}: {tool.get('purpose', 'No purpose provided')}"
        for tool in tools
    )


def _mock_system_prompt(state: GoalState) -> str:
    goal = state.get("goal", "Complete the user's requested task")
    frequency = state.get("frequency", "on demand")
    output_type = state.get("output_type", "structured response")
    tools_summary = _build_tools_summary(state)

    return f"""You are a specialized AI agent created by AgentForge.

Your goal:
{goal}

Execution frequency:
{frequency}

Expected output type:
{output_type}

Available tools:
{tools_summary}

Operating instructions:
1. Understand the user's request clearly before acting.
2. Use the available tools only when they are relevant to the goal.
3. Produce a clear, useful, and well-structured final output.
4. Keep the response professional, concise, and practical.
5. If required information is missing, state the assumption clearly instead of failing silently.

This is running in mock mode, so tool behavior is simulated and no external API key is required."""


def generate_prompt(state: GoalState) -> GoalState:
    if _mock_enabled():
        return {
            "system_prompt": _mock_system_prompt(state)
        }

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Either add an OpenAI API key or run with "
            "AGENTFORGE_MOCK_MODE=true."
        )

    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    tools_summary = _build_tools_summary(state)

    prompt = f"""
Write a system prompt for an AI agent with these specifications:

Goal: {state.get("goal")}
Frequency: {state.get("frequency")}
Output: {state.get("output_type")}

Available tools:
{tools_summary}

The system prompt should:
- Define the agent's role clearly
- Tell it what tools to use and when
- Specify the output format
- Set the tone as professional and concise

Respond with ONLY the system prompt text, nothing else.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "system_prompt": response.content.strip()
    }