"""Claude as an analyst, through the official Anthropic SDK.

One request per forecast, with structured output so the answer is always
the forecast JSON. With web search on, a short news check runs first and
its findings go into the dossier (DeepSeek gets them too).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import anthropic

from .forecast import FORECAST_SCHEMA, NEWS_PROMPT, SYSTEM_PROMPT, AnalystError, Reply, parse_forecast

DEFAULT_MODEL = "claude-opus-5-5"
# US dollars per million tokens: input, output, cache read (a cache write costs 1.25x input). Claude Opus 5
# and Opus 4.8 are where a refusal fallback ("default") can send a request, and they cost more than Opus 5.5
PRICES = {
    "claude-fable-5-1": (10.0, 50.0, 0.25),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-opus-4-8": (5.0, 25.0, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
# A model missing above is priced at the dearest known rates, so spending is never under-counted
UNKNOWN_PRICE = tuple(max(price[i] for price in PRICES.values()) for i in range(3))
WEB_SEARCH_PRICE = 0.01  # per search
# Models that accept server-side refusal fallbacks ("default" picks the substitute by refusal reason)
FALLBACK_MODELS = {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def usage_cost(requested: str, usage: Any, served: Optional[str] = None) -> Tuple[float, int, int]:
    """(estimated dollars, input tokens, output tokens) of one response, from its usage.

    With a refusal fallback, ``usage.iterations`` lists every attempt (a request declined mid-answer is
    billed too) while the top-level counts cover only the attempt that answered. So each attempt is priced
    at the rates of the model that ran it: the entry's model, else the requested one. Without iterations
    the top-level counts are priced at the model that served the answer.
    """
    iterations = list(getattr(usage, "iterations", None) or [])
    parts = [(getattr(entry, "model", None) or requested, entry) for entry in iterations]
    parts = parts or [(served or requested, usage)]
    dollars, tokens_in, tokens_out = 0.0, 0, 0
    for model, part in parts:
        price_in, price_out, price_read = PRICES.get(str(model), UNKNOWN_PRICE)
        fresh = getattr(part, "input_tokens", 0) or 0
        written = getattr(part, "cache_creation_input_tokens", 0) or 0
        read = getattr(part, "cache_read_input_tokens", 0) or 0
        out = getattr(part, "output_tokens", 0) or 0
        dollars += (fresh * price_in + written * price_in * 1.25 + read * price_read + out * price_out) / 1e6
        tokens_in += fresh + written + read
        tokens_out += out
    return dollars, tokens_in, tokens_out


class ClaudeAnalyst:
    key = "claude"
    label = "Claude"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, effort: str = "medium",
                 web_search: bool = False, client: Optional[Any] = None) -> None:
        self.model = model
        self.effort = effort
        self.web_search = web_search
        # A forecast usually takes under a minute; one retry on a 429 or 5xx keeps "Ask now" from hanging long
        self.client = client or anthropic.Anthropic(api_key=api_key, timeout=180.0, max_retries=1)

    def _create(self, **request: Any) -> Any:
        extra: Dict[str, Any] = {"betas": [FALLBACK_BETA], "fallbacks": "default"} if self.model in FALLBACK_MODELS else {}
        try:
            return self.client.beta.messages.create(model=self.model, **extra, **request)
        except anthropic.AuthenticationError:
            raise AnalystError("Claude rejected the API key (check ANTHROPIC_API_KEY)") from None
        except anthropic.PermissionDeniedError:
            raise AnalystError(f"This API key can't use {self.model}") from None
        except anthropic.NotFoundError:
            raise AnalystError(f"Unknown Claude model {self.model} (check ANTHROPIC_MODEL)") from None
        except anthropic.RateLimitError:
            raise AnalystError("Claude is rate-limiting this API key: try again later") from None
        except anthropic.BadRequestError as exc:
            raise AnalystError(f"Claude didn't accept the request: {str(exc.message)[:200]}") from None
        except anthropic.APIStatusError as exc:
            raise AnalystError(f"Claude's API answered {exc.status_code}: try again later") from None
        except anthropic.APIConnectionError:
            raise AnalystError("Couldn't reach Claude's API") from None

    def _cost(self, usage: Any, served: Optional[str] = None) -> Tuple[float, int, int]:
        """(estimated dollars, input tokens, output tokens) of one response, refusal fallbacks included."""
        dollars, tokens_in, tokens_out = usage_cost(self.model, usage, served)
        tool_use = getattr(usage, "server_tool_use", None)
        searches = (getattr(tool_use, "web_search_requests", 0) or 0) if tool_use else 0
        return dollars + searches * WEB_SEARCH_PRICE, tokens_in, tokens_out

    def check_news(self, request: str) -> Tuple[str, List[Dict[str, str]], float]:
        """Late injury and lineup news for one game: (plain-text note, sources, cost)."""
        tool = {"type": "web_search_20250305" if "haiku" in self.model else "web_search_20260209",
                "name": "web_search", "max_uses": 3}
        messages: List[Dict[str, Any]] = [{"role": "user", "content": request}]
        cost = 0.0
        for _ in range(4):
            response = self._create(
                max_tokens=8000,
                system=[{"type": "text", "text": NEWS_PROMPT, "cache_control": {"type": "ephemeral"}}],
                messages=messages,
                tools=[tool],
                output_config={"effort": "low"},
            )
            cost += self._cost(response.usage, getattr(response, "model", None))[0]
            if response.stop_reason != "pause_turn":
                break
            # A long search turn pauses; sending it back lets the server carry on where it stopped
            messages = messages + [{"role": "assistant", "content": response.content}]
        if response.stop_reason in ("refusal", "pause_turn"):
            raise AnalystError("Claude couldn't finish the news check", cost)
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        sources: List[Dict[str, str]] = []
        for block in response.content:
            for citation in getattr(block, "citations", None) or []:
                url = getattr(citation, "url", None)
                if url and all(source["url"] != url for source in sources):
                    sources.append({"url": url, "title": str(getattr(citation, "title", None) or url)[:120]})
        return text[:1200], sources[:5], cost

    def analyze(self, dossier: str) -> Reply:
        response = self._create(
            max_tokens=16000,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": dossier}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": FORECAST_SCHEMA}},
        )
        cost, tokens_in, tokens_out = self._cost(response.usage, getattr(response, "model", None))
        if response.stop_reason == "refusal":
            raise AnalystError("Claude declined to forecast this game", cost)
        if response.stop_reason == "max_tokens":
            raise AnalystError("Claude's answer was cut off", cost)
        text = "".join(block.text for block in response.content if block.type == "text")
        try:
            forecast = parse_forecast(json.loads(text))
        except ValueError:
            raise AnalystError("Claude's answer wasn't valid JSON", cost) from None
        except AnalystError as exc:
            raise AnalystError(str(exc), cost) from None
        return Reply(forecast, response.model, cost, tokens_in, tokens_out)
