import json
from typing import cast
from datetime import datetime, timezone

from langchain_aws import ChatBedrockConverse
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.tools import StructuredTool

from ..state import ResearchState, Task
from ...logger import get_logger

logger = get_logger("task_solver")


SYSTEM_PROMPT = """
You are an AI assistant tasked with completing a specific task using a given tool.
Your goal is to understand the tool's capabilities and use it effectively to accomplish the assigned task.

## Tools
First, review the available tools will be provided within the <available-tools> tags.

### Tool Selection
Remember below guidelines while selecting the tool:
- Only use the capabilities of the tool provided. Do not assume any additional functionalities.
- If the tool has any limitations or specific usage instructions, adhere to them strictly.
- If you need any clarification about the tool or the task, ask for it before attempting to use the tool.

## Take Action with The Tool
Now, follow these steps, to complete the task with the selected tool:
1. Carefully read and understand the capabilities of the provided tool.
2. Analyze the task and determine how the tool can be used to accomplish it.
3. If the tool is sufficient to complete the task, proceed to use it as needed.
4. If the tool is not sufficient or if you're unsure how to use it for the task, do not proceed. \
And returns empty response with reason why the tool is not sufficient.

Begin your response by analyzing the tool and the task, then proceed with your attempt to complete the task.
""".strip()

INSTRUCTION = """
<current-datetime>{datetime}</current-datetime>

Here is the task to complete:
<task>{task}</task>

Here are the available tools to use:
<available-tools>
{tool_desc}
<tool>
 <name>WrapUp</name>
 <description>Summarize the observations and generate final response to user.</description>
</tool>
</available-tools>

Please select the appropriate tool from the list of available tools and take action to complete the task.
""".strip()


class TaskSolver:
    def __init__(self, model: ChatBedrockConverse, tools: list[StructuredTool]) -> None:
        self.model = model.bind_tools(tools)
        self.system_prompt = SYSTEM_PROMPT
        self.instruction = INSTRUCTION
        self.tools = tools
        self.tool_dict = {tool.name.lower(): tool for tool in tools}

    # generate tool description for each tool at tools
    def _generate_tool_desc(self) -> str:
        tool_descs = []
        for tool in self.tools:
            tool_descs.append("<tool>")
            tool_descs.append(f" <name>{tool.name}</name>")
            tool_descs.append(
                f" <description>{tool.description}</<description>")
            tool_descs.append("</tool>")
        return "\n".join(tool_descs)

    def _build_messages(self, task: Task) -> PromptValue:
        return ChatPromptTemplate(
            [
                ("system", self.system_prompt),
                ("human", self.instruction),
            ]
        ).invoke(
            {
                "tool_desc": self._generate_tool_desc(),
                "datetime": datetime.now(timezone.utc).isoformat(),
                "task": task,
            }
        )

    def __call__(self, state: ResearchState) -> ResearchState:
        """If remaining tasks exists, take proper action to complete the task."""
        task = state["remaining_tasks"].pop(0)
        result = cast(AIMessage, self.model.invoke(self._build_messages(task)))

        tool_executions = []
        sources = []
        task_results = {}
        for i, tool_call in enumerate(result.tool_calls):
            tool_name = tool_call["name"].lower()
            tool_args = tool_call["args"]
            if tool_name not in self.tool_dict:
                logger.error(f"Tool {tool_name} not found in available tools.")
                continue
            selected_tool = self.tool_dict[tool_name]
            msg = selected_tool.invoke(tool_call)
            task_results[task.title] = msg.content
            tool_executions.append({
                "id": i+1,
                "name": tool_name,
                "args": tool_args,
                "result": msg.content,
            })

            if tool_name == "web_search":
                try:
                    sources.extend(json.loads(msg.content))
                except Exception:
                    logger.error(
                        f"Error in deserializing web search result: {msg.content}")
            else:
                logger.error(f"Tool {tool_name} not supported.")
        return {
            **state,
            "tool_execution": tool_executions[-1] if tool_executions else None,
            "sources": state.get("sources", []) + sources,
            "task_results": task_results,
        }
