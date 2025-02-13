from typing import Annotated

from pydantic import BaseModel, Field
from langchain_core.messages.tool import ToolCall
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class Task(BaseModel):
    """Task describes the individual tasks to be executed in order to complete the plan."""

    title: str = Field(
        description="The short and succinct description of the task.")
    description: str = Field(
        description="The detailed description of the task.")
    tool_name: str = Field(
        description="The tool_name to be used to execute the task.")
    tool_args: dict = Field(
        default={}, description="The arguments to be passed to the tool.")


class Plan(BaseModel):
    """Plan consists of tasks to complete a given user input."""

    revised_user_input: str = Field(
        description="The revised user input after the planning process.")
    category: str = Field(description="The category of the user input.")
    overview: str = Field(
        description="The brief description of the overall plan.")
    tasks: list[Task] = Field(
        description="The list of tasks to be executed to complete the plan.")

    def __str__(self) -> str:
        return "\n".join([f"{i+1}. {task.title}: {task.description}" for i, task in enumerate(self.tasks)])


class ResearchState(TypedDict):
    """
    messages: list of chat messages, for conversation history
    user_input: the user input
    plan: the plan to complete the user input. `structured_planner` will fills it.
    remaining_tasks: remaning task queue. `structured_planner` will fills it.
    tool_execution: last exectuted tool call. `task_solver` will fills it.
    sources: searched url links, `task_solver.web_search` will fills it.
    """

    messages: Annotated[list, add_messages]
    user_input: str
    plan: Plan
    remaining_tasks: list[Task]
    tool_execution: ToolCall
    sources: list
