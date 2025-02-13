from langchain_aws import ChatBedrockConverse
from langgraph.graph import StateGraph, END
from langchain_core.tools import StructuredTool

from .state import ResearchState
from .node.semantic_router import SemanticRouter
from .node.structured_planner import StructuredPlanner
from .node.task_solver import TaskSolver
from .tool.web_search import tool as web_search_tool


class ResearchFlow:
    state_graph: StateGraph

    def __init__(
        self,
        model: ChatBedrockConverse,
    ) -> None:
        tools: list[StructuredTool] = [
            web_search_tool,
        ]

        state_graph = StateGraph(ResearchState)

        state_graph.add_node("semantic_router", SemanticRouter(model))
        state_graph.add_node("structured_planner",
                             StructuredPlanner(model, tools))
        state_graph.add_node("task_solver", TaskSolver(model, tools))

        state_graph.set_entry_point("semantic_router")
        state_graph.add_conditional_edges(
            "semantic_router",
            self._pre_guardrail,
            {True: "structured_planner", False: END},
        )

        state_graph.add_conditional_edges(
            "structured_planner",
            self._has_tasks,
            {True: "task_solver", False: END},
        )
        state_graph.add_conditional_edges(
            "task_solver",
            self._has_remaining_tasks,
            {True: "task_solver", False: END},
        )

        self.state_graph = state_graph

    def _pre_guardrail(self, state: ResearchState) -> bool:
        """Check if the category is Compliant."""
        return state["category"] != "NonCompliant"

    def _has_tasks(self, state: ResearchState) -> bool:
        """Check if a plan has tasks."""
        return len(state["plan"].tasks) > 0

    def _has_remaining_tasks(self, state: ResearchState) -> bool:
        """Check if there are remaining tasks to complete."""
        return bool(state["remaining_tasks"])
