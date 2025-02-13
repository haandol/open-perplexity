import json
from typing import Optional

import boto3


class Reranker:
    def __init__(
        self,
        model: str = "cohere.rerank-v3-5:0",
        aws_profile_name: Optional[str] = None,
        aws_region: Optional[str] = None,
    ):
        self.model = model
        session = boto3.Session(profile_name=aws_profile_name)
        self.client = session.client("bedrock-runtime", region_name=aws_region)

    async def rerank(self, docs: list[str], query: str, k: int = 5) -> list[str]:
        if not docs:
            return docs

        top_n = min(k, len(docs))
        request_body = {
            "query": query,
            "documents": docs,
            "top_n": top_n,
        }
        if "cohere" in self.model:
            request_body["api_version"] = 2

        response = self.client.invoke_model(
            modelId=self.model,
            body=json.dumps(request_body),
            contentType="application/json",
        )
        results = json.loads(response["body"].read())["results"]
        # ignore the order of the result because it is too small number
        indices = [r["index"] for r in results]
        return [s for i, s in enumerate(docs) if i in indices]
