from typing import TypedDict

class GoalState(TypedDict):
    user_request: str
    goal: str
    frequency: str
    output_type: str
    tools_needed: list[str]
    tool_configurations: list[dict]
    system_prompt: str
