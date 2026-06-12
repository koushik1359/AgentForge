from __future__ import annotations

import json
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


def _clean_json_response(content: str) -> str:
    content = content.strip()

    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            content = parts[1].strip()

    if content.lower().startswith("json"):
        content = content[4:].strip()

    return content


def _mock_analyze_goal(state: GoalState) -> dict:
    user_request = state.get("user_request", "")
    request_lower = user_request.lower()

    if "daily" in request_lower:
        frequency = "daily"
    elif "weekly" in request_lower:
        frequency = "weekly"
    elif "hourly" in request_lower:
        frequency = "hourly"
    elif "monthly" in request_lower:
        frequency = "monthly"
    else:
        frequency = "on-demand"

    if "email" in request_lower:
        output_type = "email summary"
    elif "report" in request_lower:
        output_type = "report"
    elif "alert" in request_lower or "notify" in request_lower:
        output_type = "alert"
    elif "summary" in request_lower or "summaries" in request_lower or "summarize" in request_lower:
        output_type = "summary"
    elif "dashboard" in request_lower:
        output_type = "dashboard"
    else:
        output_type = "structured response"

    tools_needed = []

    if "news" in request_lower:
        tools_needed.append("web_search")
        tools_needed.append("news_filter")

    if "ai" in request_lower and "news" in request_lower:
        tools_needed.append("ai_news_monitor")

    if "linkedin" in request_lower:
        tools_needed.append("linkedin_monitor")

    if "job" in request_lower or "jobs" in request_lower:
        tools_needed.append("job_search")

    if "research" in request_lower or "paper" in request_lower or "papers" in request_lower:
        tools_needed.append("document_loader")
        tools_needed.append("research_paper_summarizer")

    if "web" in request_lower or "search" in request_lower:
        tools_needed.append("web_search")

    if "email" in request_lower or "send" in request_lower or "notify" in request_lower:
        tools_needed.append("notification_sender")

    if (
        "summary" in request_lower
        or "summarize" in request_lower
        or "summaries" in request_lower
        or "news" in request_lower
    ):
        tools_needed.append("summarizer")

    if "report" in request_lower or "markdown" in request_lower:
        tools_needed.append("report_generator")

    if not tools_needed:
        tools_needed = [
            "input_analyzer",
            "task_planner",
            "response_generator",
        ]

    tools_needed = list(dict.fromkeys(tools_needed))

    return {
        "goal": user_request if user_request else "Create a useful AI agent based on the user request",
        "frequency": frequency,
        "output_type": output_type,
        "tools_needed": tools_needed,
    }


def analyze_goal(state: GoalState) -> GoalState:
    if _mock_enabled():
        return _mock_analyze_goal(state)

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Either add an OpenAI API key or run with "
            "AGENTFORGE_MOCK_MODE=true."
        )

    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
A user wants to create an AI agent.

Analyze their request and extract:
- goal: what the agent should achieve
- frequency: how often it runs, for example daily, hourly, weekly, monthly, or on-demand
- output_type: what it produces, for example email summary, report, alert, dashboard, or structured response
- tools_needed: list of tools required

User request:
{state["user_request"]}

Respond ONLY with valid JSON:

{{
  "goal": "...",
  "frequency": "...",
  "output_type": "...",
  "tools_needed": ["...", "..."]
}}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    content = _clean_json_response(response.content)
    result = json.loads(content)

    return {
        "goal": result["goal"],
        "frequency": result["frequency"],
        "output_type": result["output_type"],
        "tools_needed": result["tools_needed"],
    }
