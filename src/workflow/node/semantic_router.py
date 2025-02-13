from typing import cast

from pydantic import BaseModel, Field
from langchain_aws import ChatBedrockConverse
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompt_values import PromptValue

from ..state import ResearchState


class Category(BaseModel):
    """Category classifies the user input into one of the given categories."""

    name: str = Field(description="The name of category of the user input.")
    user_input: str = Field(description="The original user input.")
    revised_user_input: str = Field(
        description="The revised user input after correction.")
    reason: str = Field(description="The reason for the classification.")


SYSTEM_PROMPT = """
You are a highly accurate topic classifier for the user input.
Your goal is to classify the user input into one of the categories and generate the reason.

## Ethics and Compliance
Your responses must align with our values:
<values>
- Integrity: Never deceive or aid in deception.
- Compliance: Refuse any request that violates laws or our policies.
- Privacy: Protect all personal and corporate data.
</values>
If a request conflicts with these values, respond: “I cannot perform that action as it goes against Open Perplexity's values.”

## Categories
First, review the categories that will be provided within the <categories> tags.

## Input Analysis and Revise
Before classification, analyze the user input for revision:
1. Check if the input contains incomplete sentences or context-dependent references
2. If incomplete, review the chat history context provided
3. Reconstruct a complete statement by:
 - Resolving pronouns and references using context
 - Adding implied subjects or objects
 - Expanding abbreviated or partial phrases
4. Present both the original and completed input:
 - Original: [original incomplete input]
 - Completed: [reconstructed complete statement]

## Classification
Now, classify the completed input into one of the following categories.
If the input does not fit into any of the categories, classify it as "Unknown".
""".strip()

INSTRUCTION = """
Here is the user input to classify:
<user-input>
{user_input}
</user-input>

Here are the categories:
<categories>
{categories}
</categories>

Please generate a comprehensive response based on the above information.
""".strip()


class SemanticRouter:
    """
    SemanticRouter classifies the user input into one of the categories and generates the reason.

    To do this, following steps are needed:
    1. Correct the user input if necessary.
    2. Classify the user input into one of the categories.
    3. Generate the reason for the classification.

    The result will be provided in the JSON format.
    """

    def __init__(self, model: ChatBedrockConverse) -> None:
        self.model = model.with_structured_output(Category)
        self.system_prompt = SYSTEM_PROMPT
        self.instruction = INSTRUCTION
        self.categories = self._build_category_tags(
            [
                {
                    "name": "NonCompliant",
                    "description": "Input is related to unethical or illegal activities.",
                },
                {
                    "name": "Unknown",
                    "description": "Input is not related to any of the above categories.",
                },
                {
                    "name": "Game",
                    "description": "Input is related to mobile and PC games.",
                },
            ]
        )

    def _build_category_tags(self, categories: list[dict]) -> str:
        result = []
        for c in categories:
            result.append("<category>")
            for k, v in c.items():
                result.append(f"<{k}>{v}</{k}>")
            result.append("</category>")
        return "\n".join(result)

    def _build_messages(self, state: ResearchState) -> PromptValue:
        return ChatPromptTemplate(
            [
                ("system", self.system_prompt),
                ("placeholder", "{conversation}"),
                ("human", self.instruction),
            ]
        ).invoke(
            {
                "user_input": state["user_input"],
                "conversation": state["messages"],
                "categories": self.categories,
            }
        )

    def __call__(self, state: ResearchState) -> ResearchState:
        result = cast(Category, self.model.invoke(self._build_messages(state)))
        return {
            **state,
            "user_input": result.revised_user_input or result.user_input,
            "category": result.name,
        }
