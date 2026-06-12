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
    return "rejected"


def build_agent():
    workflow = StateGraph(GoalState)

    workflow.add_node("analyze_goal", analyze_goal)
    workflow.add_node("select_tools", select_tools)
    workflow.add_node("generate_prompt", generate_prompt)
    workflow.add_node("design_workflow", design_workflow)
    workflow.add_node("configure_memory", configure_memory)
    workflow.add_node("validate_agent", validate_agent)

    workflow.set_entry_point("analyze_goal")

    workflow.add_edge("analyze_goal", "select_tools")
    workflow.add_edge("select_tools", "generate_prompt")
    workflow.add_edge("generate_prompt", "design_workflow")
    workflow.add_edge("design_workflow", "configure_memory")
    workflow.add_edge("configure_memory", "validate_agent")

    workflow.add_conditional_edges(
        "validate_agent",
        route_after_validation,
        {
            "approved": END,
            "rejected": END,
        },
    )

    return workflow.compile()


def print_result(result: GoalState) -> None:
    print("\n--- Goal Analysis ---")
    print(f"Goal:       {result.get('goal')}")
    print(f"Frequency:  {result.get('frequency')}")
    print(f"Output:     {result.get('output_type')}")
    print(f"Tools:      {result.get('tools_needed')}")

    print("\n--- Tool Configurations ---")
    for tool in result.get("tool_configurations", []):
        print(f"\n  Tool:    {tool.get('name')}")
        print(f"  Purpose: {tool.get('purpose')}")
        print(f"  Config:  {tool.get('config')}")

    print("\n--- Generated System Prompt ---")
    print(result.get("system_prompt"))

    print("\n--- Workflow Design ---")
    for step in result.get("workflow_steps", []):
        print(f"\n  Step {step.get('step_number')}: {step.get('name')}")
        print(f"  What:  {step.get('description')}")
        print(f"  Tool:  {step.get('tool')}")

    memory = result.get("memory_config", {})

    print("\n--- Memory Configuration ---")
    print(f"\n  Storage: {memory.get('storage_type')}")

    print("\n  Short-term memory:")
    for item in memory.get("short_term", []):
        print(
            f"    - {item.get('key')}: {item.get('description')} "
            f"({item.get('retention_days')} days)"
        )

    print("\n  Long-term memory:")
    for item in memory.get("long_term", []):
        print(
            f"    - {item.get('key')}: {item.get('description')} "
            f"({item.get('retention_days')} days)"
        )

    validation = result.get("validation_result", {})

    print("\n--- Validation ---")
    print(f"\n  Status: {'PASSED' if validation.get('passed') else 'FAILED'}")
    print(f"  Score:  {validation.get('score')}/100")

    if validation.get("issues"):
        print("\n  Issues:")
        for issue in validation.get("issues", []):
            print(f"    - {issue}")

    if validation.get("recommendations"):
        print("\n  Recommendations:")
        for recommendation in validation.get("recommendations", []):
            print(f"    - {recommendation}")


if __name__ == "__main__":
    agent = build_agent()

    user_request = input("Enter your agent idea: ").strip()

    if not user_request:
        user_request = "Create an agent that monitors LinkedIn for Data Engineer jobs and sends me daily summaries"

    result = agent.invoke(
        {
            "user_request": user_request,
            "goal": "",
            "frequency": "",
            "output_type": "",
            "tools_needed": [],
            "tool_configurations": [],
            "system_prompt": "",
            "workflow_steps": [],
            "memory_config": {},
            "validation_result": {},
        }
    )

    print_result(result)
