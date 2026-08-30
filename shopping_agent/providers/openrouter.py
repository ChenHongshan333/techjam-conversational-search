from __future__ import annotations

import json
import http.client
import os
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


class OpenRouterError(RuntimeError):
    pass


@dataclass
class OpenRouterResponse:
    payload: dict
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OpenRouterClient:
    def __init__(self, api_key: str, timeout_seconds: float = 45.0) -> None:
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        cafile = os.environ.get("SSL_CERT_FILE")
        if not cafile and os.path.exists("/etc/ssl/cert.pem"):
            cafile = "/etc/ssl/cert.pem"
        self.ssl_context = ssl.create_default_context(cafile=cafile)

    def _post(self, path: str, payload: dict, retries: int = 2) -> OpenRouterResponse:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"https://openrouter.ai/api/v1/{path.lstrip('/')}",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/techjam-conversational-search",
                "X-OpenRouter-Title": "TechJam Conversational Search",
            },
            method="POST",
        )
        for attempt in range(retries + 1):
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout_seconds,
                    context=self.ssl_context,
                ) as response:
                    result = json.loads(response.read().decode("utf-8"))
                usage = result.get("usage") if isinstance(result, dict) else {}
                usage = usage if isinstance(usage, dict) else {}
                return OpenRouterResponse(
                    payload=result,
                    prompt_tokens=int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if attempt < retries and exc.code in {408, 429, 500, 502, 503, 504, 529}:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise OpenRouterError(f"OpenRouter returned HTTP {exc.code}: {detail}") from exc
            except (
                urllib.error.URLError,
                TimeoutError,
                ConnectionError,
                http.client.RemoteDisconnected,
                ssl.SSLError,
            ) as exc:
                if attempt < retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                raise OpenRouterError(f"OpenRouter request failed: {exc}") from exc
        raise OpenRouterError("OpenRouter request failed")

    def rewrite_query(self, model: str, category: str, constraints: list[str]) -> OpenRouterResponse:
        schema = {
            "name": "shopping_query_rewrite",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "semantic_query": {"type": "string", "maxLength": 400},
                    "used_constraint_indexes": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
                "required": ["semantic_query", "used_constraint_indexes"],
                "additionalProperties": False,
            },
        }
        indexed = "\n".join(f"[{index}] {value}" for index, value in enumerate(constraints))
        prompt = (
            "Rewrite the disclosed shopping need as one compact standalone product-search query. "
            "Use only the supplied category and constraints. Preserve negation, numbers, variants, "
            "and uncertainty. Do not invent a brand, audience, use case, style, material, colour, "
            "feature, or product type. Treat metadata clauses separated by semicolons as possible "
            "variants rather than simultaneous requirements.\n\n"
            f"Category: {category or 'unspecified'}\nConstraints:\n{indexed or '(none)'}"
        )
        return self._post("chat/completions", {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise e-commerce query normalizer."},
                {"role": "user", "content": prompt},
            ],
            "reasoning": {"effort": "none"},
            "response_format": {"type": "json_schema", "json_schema": schema},
            "temperature": 0,
            "max_tokens": 180,
        })

    def classify_shopping_intent(self, model: str, message: str) -> OpenRouterResponse:
        schema = {
            "name": "shopping_intent",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "enum": ["buying", "browsing", "uncertain"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "reason": {"type": "string", "maxLength": 160},
                },
                "required": ["intent", "confidence", "reason"],
                "additionalProperties": False,
            },
        }
        prompt = (
            "Classify the customer's current shopping mode. Buying means they have a concrete "
            "product goal, requirement, deadline, or purchase use case. Browsing means they are "
            "open-endedly exploring. Use uncertain when the message does not support either. "
            "Do not infer intent from demographics or invent missing needs.\n\n"
            f"Customer message: {message[:1200]}"
        )
        return self._post("chat/completions", {
            "model": model,
            "messages": [
                {"role": "system", "content": "You classify shopping intent conservatively."},
                {"role": "user", "content": prompt},
            ],
            "reasoning": {"effort": "none"},
            "response_format": {"type": "json_schema", "json_schema": schema},
            "temperature": 0,
            "max_tokens": 120,
        })

    def embeddings(
        self,
        model: str,
        texts: list[str],
        dimensions: int,
        input_type: str,
    ) -> OpenRouterResponse:
        return self._post("embeddings", {
            "model": model,
            "input": texts,
            "dimensions": dimensions,
            "input_type": input_type,
            "encoding_format": "float",
        })

    def rerank(self, model: str, query: str, documents: list[str], top_n: int) -> OpenRouterResponse:
        return self._post("rerank", {
            "model": model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
        })
