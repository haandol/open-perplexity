# Open Perplexity

This repository is about simplified Perplexity clone based Amazon Bedrock, LangGraph and Chainlit to illustrate how the Perplexity works.

## Key Features

- Natural Language Processing using Amazon Bedrock Claude 3.5 Haiku model
- Rerank web search result using Amazon Bedrock Cohere Rerank 3.5 model
- State-based conversation flow management with LangGraph
- Interactive web interface using Chainlit
- Real-time web search functionality through Tavily API

## System Requirements

- Python 3.13 or higher
- AWS account and Bedrock access
- Tavily API key

### Enable Required Models on AWS Console

1. Visit [AWS Bedrock Web Console](https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/modelaccess)

2. Enable Claude 3.5 Haiku

3. Enable Cohere Rerank 3.5

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/open-perplexity.git
cd open-perplexity
```

2. Install `uv`:

```bash
pip install uv
```

2. Install dependencies:

```bash
uv sync
```

3. Set up environment variables:

```bash
cp env/dev.env .env
```

Required environment variables:

- `MODEL_ID`: Bedrock model ID (default: `us.anthropic.claude-3-5-haiku-20241022-v1:0`)
- `AWS_PROFILE_NAME`: AWS profile name
- `AWS_REGION`: AWS region
- `TAVILY_API_KEY`: Tavily API key

## Running the Application

```bash
uv run -- chainlit run app.py -h
```

### Screenshot

![screenshot](/docs/screenshot.jpg)

## Architecture

> Blue boxes represent LLM requests

### How Perplexity Works


![illustrate how perplexity works](/docs/perplexity.jpg)

### How Open Perplexity Works

- Separated Planner into Semantic Router and uses LLM for generating task execution parameters
- Utilizes Reranker to process dozens of search results cost-effectively while minimizing performance degradation

![illustrate how open-perplexity works](/docs/open-perplexity.jpg)

### LangGraph State Machine

![langgraph compiled](/docs/graph.png)

## License

Apache License 2.0