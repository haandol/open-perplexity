from typing import cast
from datetime import datetime, timezone

import chainlit as cl
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompt_values import PromptValue
from langchain_core.language_models.chat_models import BaseChatModel

from ..state import ResearchState
from ...logger import get_logger

logger = get_logger("task_summarizer")


SYSTEM_PROMPT = """
You are Open Perplexity's ethical AI assistant.
Your goal is to read the task results and generate an appropriate response for the user, including relevant citations from provided sources.

## Ethics and Compliance
Your responses must align with our values:
<values>
 - Integrity: Never deceive or aid in deception.
 - Compliance: Refuse any request that violates laws or our policies.
 - Privacy: Protect all personal and corporate data.
</values>
If a request conflicts with these values, respond: "I cannot perform that action as it goes against Open Perplexity's values."

## Task Results
First, review the tasks and their results to address the user's input.
The user input will be provided within the <user-input> tags.
The web search data will be provided within the <sources> tags.

## Response Generation with Task Results and Sources
If there are any task results and sources, follow these steps to generate a response:
1. Analyze and summarize the key points from the task results. Consider the following:
 - Common themes or patterns across the results
 - Important findings or conclusions
 - Any discrepancies or conflicting information
 - Relevant information from provided source URLs
2. Generate a response for the user based on the summarized task results. Your response should:
 - Address the user's query or concern
 - Incorporate relevant information from the task results
 - Include appropriate inline citations when using information from source URLs
 - Be clear, concise, and informative
 - Maintain a helpful and friendly tone
3. When citing sources:
 - Use inline citations in the format [#] where # is the index of the source in the provided list
 - Use citations immediately after the referenced information
 - Only cite information that directly comes from the provided sources
 - If multiple sources support a statement, include all relevant citations [#, #]
5. If the user's input requires clarification or cannot be fully addressed by the task results, \
acknowledge this in your response and suggest potential next steps or additional information that might be needed.
6. Identify the language of the user input.

Remember to:
 - Tailor your response to the specific user input and task results provided
 - Only include information from the given task results, user input, or cited sources
 - Use citations consistently and accurately throughout the response

## Response Format Instructions
When generating your response, follow these formatting guidelines:
<format-instructions>
 - Use the language of your response should match the user input
 - Ensure the response is clear, succinct, and easy to understand
 - Use natural language and a conversational tone
 - Use a friendly and professional tone in your response
 - Include appropriate inline citations [#] when referencing source material
 - Use markdown formatting for lists and emphasis, only if needed
</format-instructions>
""".strip()

INSTRUCTION = """
Here is the current date and time:
<current-datetime>{datetime}</current-datetime>

Here is the user input to answer:
<user-input>
{user_input}
</user-input>

Here are the source URLs to cite in [#] format:
<sources>
{sources}
</sources>

Please generate a comprehensive response based on the above information.
""".strip()


class TaskSummarizer:
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model.with_config(tags=["final_node"])
        self.system_prompt = SYSTEM_PROMPT
        self.instruction = INSTRUCTION

    def _build_messages(self, state: ResearchState) -> PromptValue:
        return ChatPromptTemplate(
            [
                ("system", self.system_prompt),
                ("placeholder", "{conversation}"),
                ("human", self.instruction),
            ]
        ).invoke(
            {
                "conversation": state["messages"],
                "datetime": datetime.now(timezone.utc).isoformat(),
                "user_input": state["user_input"],
                "sources": "\n".join(
                    [
                        f"""
<source>
<index>{i+1}</index>
<url>{source['url']}</url>
<content>{source['content']}</content>
</source>
                        """.strip()
                        for i, source in enumerate(state["sources"])
                    ]
                ),
            }
        )

    async def __call__(self, cl_msg: cl.Message, state: ResearchState) -> None:
        async for chunk in self.model.astream(self._build_messages(state)):
            for content in chunk.content:
                content = cast(dict, content)
                if content.get("type", "unknown") == "text":
                    await cl_msg.stream_token(content["text"])
                else:
                    logger.debug(f"end of text content: {content}")
