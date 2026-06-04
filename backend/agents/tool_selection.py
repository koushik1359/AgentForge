from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agents.state import GoalState
import json

def select_tools(state: GoalState) -> GoalState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""
    Based on this agent's requirements, provide specific tool configurations.

    Goal: {state["goal"]}
    Frequency: {state["frequency"]}
    Output: {state["output_type"]}
    Tools needed: {state["tools_needed"]}

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

    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    content = content.strip()

    result = json.loads(content)
    return {"tool_configurations": result}
