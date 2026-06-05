from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from agents.state import GoalState
from agents.goal_analysis import analyze_goal
from agents.tool_selection import select_tools
from agents.prompt_generation import generate_prompt
from agents.workflow_design import design_workflow
from agents.memory_config import configure_memory
from agents.validation import validate_agent

load_dotenv()

def route_after_validation(state: GoalState) -> str:
    if state["validation_result"]["passed"]:
        return "approved"
    else:
        return "rejected"

def build_graph():
    graph = StateGraph(GoalState)

    graph.add_node("analyze_goal", analyze_goal)
    graph.add_node("select_tools", select_tools)
    graph.add_node("generate_prompt", generate_prompt)
    graph.add_node("design_workflow", design_workflow)
    graph.add_node("configure_memory", configure_memory)
    graph.add_node("validate_agent", validate_agent)

    graph.set_entry_point("analyze_goal")
    graph.add_edge("analyze_goal", "select_tools")
    graph.add_edge("select_tools", "generate_prompt")
    graph.add_edge("generate_prompt", "design_workflow")
    graph.add_edge("design_workflow", "configure_memory")
    graph.add_edge("configure_memory", "validate_agent")
    graph.add_conditional_edges(
        "validate_agent",
        route_after_validation,
        {
            "approved": END,
            "rejected": END
        }
    )

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
        "system_prompt": "",
        "workflow_steps": [],
        "memory_config": {},
        "validation_result": {}
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

    print("\n--- Workflow Design ---")
    for step in result["workflow_steps"]:
        print(f"\n  Step {step['step_number']}: {step['name']}")
        print(f"  What:  {step['description']}")
        print(f"  Tool:  {step['tool']}")
    

    print("\n--- Memory Configuration ---")
    print(f"\n  Storage: {result['memory_config']['storage_type']}")
    print("\n  Short-term memory:")
    for item in result["memory_config"]["short_term"]:
        print(f"    - {item['key']}: {item['description']} ({item['retention_days']} days)")
    print("\n  Long-term memory:")
    for item in result["memory_config"]["long_term"]:
        print(f"    - {item['key']}: {item['description']} ({item['retention_days']} days)")


    print("\n--- Validation ---")
    v = result["validation_result"]
    status = "PASSED" if v["passed"] else "FAILED"
    print(f"\n  Status: {status}")
    print(f"  Score:  {v['score']}/100")
    if v["issues"]:
        print("\n  Issues:")
        for issue in v["issues"]:
            print(f"    - {issue}")
    if v["recommendations"]:
        print("\n  Recommendations:")
        for rec in v["recommendations"]:
            print(f"    - {rec}")
