from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from agents.state import GoalState

def generate_prompt(state: GoalState) -> GoalState:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    tools_summary = "\n".join(
        f"- {t['name']}: {t['purpose']}" for t in state["tool_configurations"]
    )

    prompt = f"""
    Write a system prompt for an AI agent with these specifications:

    Goal: {state["goal"]}
    Frequency: {state["frequency"]}
    Output: {state["output_type"]}
    Available tools:
    {tools_summary}

    The system prompt should:
    - Define the agent's role clearly
    - Tell it what tools to use and when
    - Specify the output format
    - Set the tone (professional, concise)

    Respond with ONLY the system prompt text, nothing else.
    """

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"system_prompt": response.content.strip()}
