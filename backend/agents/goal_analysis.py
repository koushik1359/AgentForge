from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agents.state import GoalState
import json

def analyze_goal(state: GoalState) -> GoalState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
    A user wants to create an AI agent. Analyze their request and extract:
    - goal: what the agent should achieve
    - frequency: how often it runs (e.g. daily, hourly, on-demand)
    - output_type: what it produces (e.g. email summary, report, alert)
    - tools_needed: list of tools required (e.g. web search, gmail, database)

    User request: {state["user_request"]}

    Respond ONLY with valid JSON:
    {{
        "goal": "...",
        "frequency": "...",
        "output_type": "...",
        "tools_needed": ["...", "..."]
    }}
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    result = json.loads(response.content)

    return {
        "goal": result["goal"],
        "frequency": result["frequency"],
        "output_type": result["output_type"],
        "tools_needed": result["tools_needed"]
    }
