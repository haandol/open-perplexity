import os
import json
from typing import cast

import chainlit as cl
from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import AIMessage, HumanMessage
from chainlit.user_session import UserSession

from src.reranker import Reranker
from src.llm import BedrockLLM
from src.workflow.node.quick_responder import QuickResponder
from src.workflow.node.task_summarizer import TaskSummarizer
from src.workflow.graph import ResearchFlow
from src.logger import get_logger

load_dotenv()


logger = get_logger("app")


# load environment variables
# for LLM
MODEL_ID = os.environ.get(
    "MODEL_ID", "us.anthropic.claude-3-5-haiku-20241022-v1:0")
logger.info(f"MODEL_ID: {MODEL_ID}")
AWS_PROFILE_NAME = os.environ.get("AWS_PROFILE_NAME", None)
logger.info(f"AWS_PROFILE_NAME: {AWS_PROFILE_NAME}")
AWS_REGION = os.environ.get("AWS_REGION", None)
logger.info(f"AWS_REGION: {AWS_REGION}")

# for observability using phoenix tracer
PHOENIX_PROJECT_NAME = os.environ.get("PHOENIX_PROJECT_NAME", "default")
PHOENIX_ENDPOINT = os.environ.get("PHOENIX_ENDPOINT", "")


def _deduplicate_source(sources: list[dict]) -> list[str]:
    """
    We are going to use rerank with JSON string input, because url data should be stick to the content in order.

    1. deduplicate sources based on the URL
    2. leave title, url and content field only
    3. stringify each content
    """
    D = {source["url"]: source for source in sources}
    return [
        json.dumps(
            {"title": source["title"], "url": source["url"], "content": source["content"]})
        for source in D.values()
    ]


async def rerank(user_input: str, sources: list[dict]) -> list[dict]:
    reranker = Reranker(
        aws_profile_name=AWS_PROFILE_NAME,
        aws_region=AWS_REGION,
    )
    new_sources = await reranker.rerank(
        query=user_input,
        docs=_deduplicate_source(sources),
        k=5,
    )
    return [json.loads(ns) for ns in new_sources]


@cl.on_chat_start
async def on_chat_start():
    """Callback for when the chat starts."""

    # setup the agent graph
    model = BedrockLLM(
        model=MODEL_ID,
        aws_profile_name=AWS_PROFILE_NAME,
        aws_region=AWS_REGION,
        phoenix_project_name=PHOENIX_PROJECT_NAME,
        phoenix_endpoint=PHOENIX_ENDPOINT,
    )
    task_summarizer = TaskSummarizer(model.model)
    cl.user_session.set("task-summarizer", task_summarizer)
    quick_responder = QuickResponder(model.model)
    cl.user_session.set("quick_responder", quick_responder)

    state_graph = ResearchFlow(model.model).state_graph.compile()
    cl.user_session.set("state-graph", state_graph)

    # setup the history cache
    cl.user_session.set("history-cache", [])


def restore_session(user_session: UserSession) -> tuple[CompiledStateGraph, list]:
    """Restore the session from the user session."""

    state_graph = cast(CompiledStateGraph, user_session.get("state-graph"))
    history_cache = cast(list, user_session.get("history-cache"))
    return (state_graph, history_cache)


@cl.on_message
async def on_message(message: cl.Message):
    """Callback for when a message is received."""

    # restore session
    (
        state_graph,
        history_cache,
    ) = restore_session(cl.user_session)
    history_cache.append(HumanMessage(content=message.content))

    # set empty ai response message
    ai_msg = cl.Message(content="")

    state = None
    async with cl.Step(name="Reasoning"):
        # process the message
        async for event in state_graph.astream(
            {
                "user_input": message.content,
                "messages": history_cache,
            },
            stream_mode="updates",
        ):
            if "structured_planner" in event:
                logger.info(f"Plan: {event['structured_planner']}")
                planner_event = event["structured_planner"]
                async with cl.Step(name="Planner") as step:
                    plan = planner_event["plan"]
                    step.output = f"**Revised User Input:** {plan.revised_user_input}\n**Category:** {plan.category}"
                    if plan.tasks:
                        plan_message = "\n".join(
                            [f"- {task.title}({task.description})" for task in plan.tasks])
                        step.output += f"\n**Tasks:**\n{plan_message}"
                state = event["structured_planner"]
            elif "task_solver" in event:
                logger.info(f"Task Solver: {event['task_solver']}")
                async with cl.Step(name="Task Solver") as step:
                    tool_call = event["task_solver"]["tool_execution"]
                    step.input = tool_call["args"]
                    step.output = tool_call["result"]
                state = event["task_solver"]

        if state and state["plan"].tasks:
            # display sources after rerank
            state["sources"] = await rerank(state["user_input"], state["sources"])
            sources = "\n".join(
                [f"[{i+1}] {source['url']}" for i, source in enumerate(state["sources"])])
            if sources:
                async with cl.Step(name="Web Search Results", show_input=False) as step:
                    step.output = str(sources)

    if state and state["plan"].tasks:
        logger.info("Invoke Task Summarizer")
        task_summarizer = cast(
            TaskSummarizer, cl.user_session.get("task-summarizer"))
        await task_summarizer(ai_msg, state)
        await ai_msg.send()
        history_cache.append(AIMessage(content=ai_msg.content))
    elif state and not state["plan"].tasks:
        logger.info("Invoke Quick Responder")
        quick_responder = cast(
            QuickResponder, cl.user_session.get("quick_responder"))
        await quick_responder(ai_msg, state)
        await ai_msg.send()
        history_cache.append(AIMessage(content=ai_msg.content))
    else:
        logger.error("state should not be None")

    # update history
    cl.user_session.set("history-cache", history_cache)
