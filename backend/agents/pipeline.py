from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from agents.state import GoalState
from agents.goal_analysis import analyze_goal
from agents.tool_selection import select_tools
from agents.prompt_generation import generate_prompt

load_dotenv()

def build_graph():
    graph = StateGraph(GoalState)

    graph.add_node("analyze_goal", analyze_goal)
    graph.add_node("select_tools", select_tools)
    graph.add_node("generate_prompt", generate_prompt)

    graph.set_entry_point("analyze_goal")
    graph.add_edge("analyze_goal", "select_tools")
    graph.add_edge("select_tools", "generate_prompt")
    graph.add_edge("generate_prompt", END)

    return graph.compile()

if __name__ == "__main__":
    agent = build_graph()

    result = agent.invoke({
        "user_request": "Create an agent that monitors LinkedIn for Data Engineer jobs and sends me daily summaries",
        "goal": "",
        "frequency": "",
        "output_type": "",
        "tools_needed": [],
        "tool_configurations": [],
        "system_prompt": ""
    })

    print("\n--- Goal Analysis ---")
    print(f"Goal:       {result['goal']}")
    print(f"Frequency:  {result['frequency']}")
    print(f"Output:     {result['output_type']}")
    print(f"Tools:      {result['tools_needed']}")

    print("\n--- Tool Configurations ---")
    for tool in result["tool_configurations"]:
        print(f"\n  Tool:    {tool['name']}")
        print(f"  Purpose: {tool['purpose']}")
        print(f"  Config:  {tool['config']}")

    print("\n--- Generated System Prompt ---")
    print(result["system_prompt"])
