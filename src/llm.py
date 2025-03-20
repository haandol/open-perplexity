import os
from typing import Optional

from langchain_aws.chat_models import ChatBedrockConverse
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register


class BedrockLLM(object):
    def __init__(
        self,
        model: str,
        aws_profile_name: Optional[str] = None,
        aws_region: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024 * 2,
        phoenix_project_name: Optional[str] = None,
        phoenix_endpoint: Optional[str] = None,
    ):
        if os.getenv("ENABLE_TRACING", "false").lower() == "true" and phoenix_endpoint:
            # initialize Phoenix tracer
            tracer_provider = register(
                project_name=phoenix_project_name,
                endpoint=phoenix_endpoint,
            )
            LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

        self.model = ChatBedrockConverse(
            model=model,
            credentials_profile_name=aws_profile_name,
            region_name=aws_region,
            temperature=temperature,
            max_tokens=max_tokens,
        )
