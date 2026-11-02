"""DeepSeek as an analyst, through its chat completions API (api.deepseek.com).

JSON output mode makes it answer with the forecast JSON; it reasons first
(thinking mode is on by default). DeepSeek has no web search, so it uses the
news Claude found for the same game when there is any.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import requests

from .forecast import SYSTEM_PROMPT, AnalystError, Reply, parse_forecast

URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-v4-pro"
# US dollars per million tokens at peak hours (off-peak is half): cached input, input, output
PRICES = {"deepseek-v4-pro": (0.044, 1.32, 3.96), "deepseek-flash": (0.006, 0.30, 1.20)}


class DeepSeekAnalyst:
    key = "deepseek"
    label = "DeepSeek"
    web_search = False

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL, session: Optional[Any] = None,
                 timeout: float = 300.0) -> None:
        self.api_key = api_key
        self.model = model
        self.session = session or requests.Session()
        self.timeout = timeout

    def _post(self, body: Dict[str, Any], cost: float) -> Dict[str, Any]:
        try:
            response = self.session.post(URL, json=body, timeout=self.timeout,
                                         headers={"Authorization": f"Bearer {self.api_key}"})
        except requests.RequestException as exc:
            raise AnalystError(f"Couldn't reach DeepSeek's API ({type(exc).__name__})", cost) from None
        if response.status_code == 401:
            raise AnalystError("DeepSeek rejected the API key (check DEEPSEEK_API_KEY)", cost)
        if response.status_code == 402:
            raise AnalystError("The DeepSeek account is out of credit", cost)
        if response.status_code == 429:
            raise AnalystError("DeepSeek is rate-limiting this API key: try again later", cost)
        if response.status_code >= 400:
            raise AnalystError(f"DeepSeek's API answered {response.status_code}: try again later", cost)
        try:
            return response.json()
        except ValueError:
            raise AnalystError("DeepSeek's answer wasn't JSON", cost) from None

    def _cost(self, usage: Dict[str, Any]) -> Tuple[float, int, int]:
        """(estimated dollars, input tokens, output tokens) of one response."""
        price_hit, price_miss, price_out = PRICES.get(self.model, PRICES[DEFAULT_MODEL])
        hit = usage.get("prompt_cache_hit_tokens") or 0
        miss = usage.get("prompt_cache_miss_tokens")
        miss = (usage.get("prompt_tokens") or 0) - hit if miss is None else miss
        out = usage.get("completion_tokens") or 0
        return (hit * price_hit + miss * price_miss + out * price_out) / 1e6, hit + miss, out

    def analyze(self, dossier: str) -> Reply:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": dossier}],
            "response_format": {"type": "json_object"},
            "max_tokens": 16000,
        }
        cost, tokens_in, tokens_out, text, payload = 0.0, 0, 0, "", {}
        for _ in range(2):
            payload = self._post(body, cost)
            spent, used_in, used_out = self._cost(payload.get("usage") or {})
            cost, tokens_in, tokens_out = cost + spent, tokens_in + used_in, tokens_out + used_out
            choice = (payload.get("choices") or [{}])[0]
            text = ((choice.get("message") or {}).get("content") or "").strip()
            if text and choice.get("finish_reason") != "length":
                break
            # JSON mode sometimes comes back empty (a known DeepSeek issue): ask once more without thinking
            body = {**body, "thinking": {"type": "disabled"}}
            text = ""
        if not text:
            raise AnalystError("DeepSeek returned an empty answer", cost)
        try:
            forecast = parse_forecast(json.loads(text))
        except ValueError:
            raise AnalystError("DeepSeek's answer wasn't valid JSON", cost) from None
        except AnalystError as exc:
            raise AnalystError(str(exc), cost) from None
        return Reply(forecast, payload.get("model") or self.model, cost, tokens_in, tokens_out)
