from typing import cast
from copy import deepcopy
from datetime import datetime, timezone

from langchain_aws import ChatBedrockConverse
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompt_values import PromptValue
from langchain_core.tools import StructuredTool

from ..state import ResearchState, Plan

SYSTEM_PROMPT = """
You are an strategic expert AI assistant generating a plan consisting of tasks for a given user input. \
Each task must be executable using specific tools that are provided.
Your goal is to create a comprehensive and logical plan that addresses the user's input effectively.

## Tools
First, review the available tools will be provided within the <available-tools> tags.

## Plan Generation
Now, follow these steps to generate the plan:
1. Analyze the user input:
 - Carefully read the user input provided in the <user-input> tags.
 - Identify the main goal and any sub-goals or requirements mentioned.
 - Take note of any specific constraints or preferences expressed by the user.

2. Create tasks:
 - Break down the user input into smaller, manageable tasks.
 - Ensure that each task can be executed using one or more of the available tools.
 - If proper tool is not available, you can skip that task.
 - Arrange the tasks in a logical order that will lead to the completion of the user's input.
 - Make sure that the tasks are specific and actionable.

3. Generate the plan:
 - Begin with a brief overview of the plan.
 - List each task, numbering them sequentially.
 - For each task, specify which tool(s) should be used to execute it.
 - Provide a clear and concise description of what needs to be done in each task.

## Review Tasks
Read generate tasks carefully and refine the given task for get rid of any ambiguity or confusion, if needed.
- Replace ambiguous terms like recent, latest, or other vague references with specific values when available, \
such as replacing recent with the provided year (e.g., 2024). \
Similarly, substitute any vague pronouns or terms with the specific details found in the input context to create clearer, precise word.

Ensure that all tasks are directly related to fulfilling the user's input and can be executed using the provided tools.
""".strip()

INSTRUCTION = """
<current-datetime>{datetime}</current-datetime>

Here is the user input for planning:
<user-input>
{user_input}
</user-input>

And the available tools to use:
<available-tools>
{tool_desc}
</available-tools>

Please generate a plan based on the user input and the available tools.
""".strip()


class StructuredPlanner:
    def __init__(self, model: ChatBedrockConverse, tools: list[StructuredTool]) -> None:
        self.model = model.with_structured_output(Plan)
        self.system_prompt = SYSTEM_PROMPT
        self.instruction = INSTRUCTION
        self.tools = tools

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

    def _build_messages(self, state: ResearchState) -> PromptValue:
        return ChatPromptTemplate(
            [
                ("system", self.system_prompt),
                ("placeholder", "{conversation}"),
                ("human", self.instruction),
            ]
        ).invoke(
            {
                "tool_desc": self._generate_tool_desc(),
                "conversation": state["messages"],
                "datetime": datetime.now(timezone.utc).isoformat(),
                "user_input": state["user_input"],
            }
        )

    def __call__(self, state: ResearchState) -> ResearchState:
        result = cast(Plan, self.model.invoke(self._build_messages(state)))
        return {
            **state,
            "plan": result,
            "remaining_tasks": deepcopy(result.tasks),
        }
