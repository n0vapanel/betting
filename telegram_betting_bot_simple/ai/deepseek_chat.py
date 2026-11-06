"""DeepSeek as the chat assistant's caller: OpenAI chat completions + function calling.

DeepSeek speaks the OpenAI protocol, but the assistant (ai/assistant.py) and the
slip reader (ai/slip_reader.py) build and read Anthropic-shaped requests and
responses through ``ClaudeCaller``. ``DeepSeekCaller`` keeps that interface
(``create``/``cost``/``worst_cost``/``max_tokens_within``) so the tool loop is
unchanged: it translates an Anthropic-shaped request into a chat.completions
body on the way out, and an OpenAI answer into an Anthropic-shaped response on
the way back.

DeepSeek is text-only, so ``vision`` is False: the assistant routes a photo of a
slip to the free code-only reader instead of sending an image here (a request
that still carries an image block is refused, never half-read).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import requests

from .deepseek import PRICES, URL
from .explain import MIN_OUTPUT_TOKENS, AIError

# An OpenAI finish_reason as the Anthropic stop_reason the assistant already reads
_STOP = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens", "content_filter": "refusal"}


def _get(usage: Any, key: str) -> Optional[int]:
    """One usage counter, from a dict or an object (both appear in tests and shims)."""
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def _block_type(block: Any) -> str:
    """A content block's ``type``, from a dict or an object (the assistant appends both across rounds)."""
    if isinstance(block, dict):
        return str(block.get("type") or "")
    return str(getattr(block, "type", "") or "")


def _block_field(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


class DeepSeekCaller:
    """Sends one request to DeepSeek's OpenAI-compatible API and prices the answer.

    ``api_key`` is the DeepSeek key (never logged); ``session`` is a requests
    Session (or a fake in tests). The request is Anthropic-shaped (the assistant
    passes ``system``, ``messages``, ``tools``, ``tool_choice``, ``output_config``
    and ``max_tokens``) and the response is Anthropic-shaped (``content`` blocks
    with ``type``/``text``/``name``/``input``/``id``, ``stop_reason``, ``usage``,
    ``model``), so no caller site changes.
    """

    provider = "deepseek"
    vision = False  # DeepSeek is text-only: no image blocks, no photo slip reads

    def __init__(self, api_key: str, session: Optional[Any] = None, timeout: float = 300.0) -> None:
        self.api_key = api_key
        self.session = session or requests.Session()
        self.timeout = timeout

    # --- Anthropic -> OpenAI ----------------------------------------------------------------------

    @staticmethod
    def _system_text(system: Any) -> str:
        if isinstance(system, str):
            return system
        parts: List[str] = []
        for block in system or []:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return "\n\n".join(part for part in parts if part)

    @classmethod
    def _messages(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for message in messages or []:
            role = message.get("role")
            if role == "user":
                out.extend(cls._user_messages(message.get("content")))
            elif role == "assistant":
                out.append(cls._assistant_message(message.get("content")))
        return out

    @classmethod
    def _user_messages(cls, content: Any) -> List[Dict[str, Any]]:
        if isinstance(content, str):
            return [{"role": "user", "content": content}]
        text, tools = [], []
        for block in content or []:
            kind = _block_type(block)
            if kind == "text":
                text.append(_block_field(block, "text") or "")
            elif kind == "tool_result":
                tools.append({"role": "tool", "tool_call_id": _block_field(block, "tool_use_id") or "",
                              "content": _block_field(block, "content") or ""})
            elif kind == "image":
                raise AIError("DeepSeek is text-only, so it can't read a screenshot", temporary=False)
        out: List[Dict[str, Any]] = []
        if text:
            out.append({"role": "user", "content": "\n\n".join(text)})
        out.extend(tools)
        return out

    @classmethod
    def _assistant_message(cls, content: Any) -> Dict[str, Any]:
        if isinstance(content, str):
            return {"role": "assistant", "content": content}
        text, calls = [], []
        for block in content or []:
            kind = _block_type(block)
            if kind == "text":
                text.append(_block_field(block, "text") or "")
            elif kind == "tool_use":
                calls.append({"id": _block_field(block, "id") or "", "type": "function",
                              "function": {"name": _block_field(block, "name") or "",
                                           "arguments": json.dumps(_block_field(block, "input") or {}, default=str)}})
        message: Dict[str, Any] = {"role": "assistant", "content": "\n\n".join(text) or None}
        if calls:
            message["tool_calls"] = calls
        return message

    @staticmethod
    def _tool(tool: Dict[str, Any]) -> Dict[str, Any]:
        return {"type": "function",
                "function": {"name": tool.get("name"), "description": tool.get("description") or "",
                             "parameters": tool.get("input_schema") or {"type": "object", "properties": {}}}}

    @staticmethod
    def _tool_choice(choice: Any) -> Any:
        if isinstance(choice, dict):
            if choice.get("type") == "none":
                return "none"
            if choice.get("type") == "tool" and choice.get("name"):
                return {"type": "function", "function": {"name": choice["name"]}}
        return "auto"

    def _request(self, model: str, request: Dict[str, Any]) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": model, "max_tokens": request.get("max_tokens", 16000)}
        system = self._system_text(request.get("system"))
        messages = self._messages(request.get("messages") or [])
        if system:
            messages = [{"role": "system", "content": system}] + messages
        body["messages"] = messages
        tools = request.get("tools")
        if tools:
            body["tools"] = [self._tool(tool) for tool in tools]
        if request.get("tool_choice") is not None:
            body["tool_choice"] = self._tool_choice(request["tool_choice"])
        output = request.get("output_config") or {}
        if (output.get("format") or {}).get("type") == "json_schema":
            # DeepSeek has no json_schema; json_object returns valid JSON and the app validates the shape
            body["response_format"] = {"type": "json_object"}
        return body

    # --- the call ---------------------------------------------------------------------------------

    def _post(self, body: Dict[str, Any], cost: float) -> Dict[str, Any]:
        try:
            response = self.session.post(URL, json=body, timeout=self.timeout,
                                         headers={"Authorization": f"Bearer {self.api_key}"})
        except requests.RequestException as exc:
            raise AIError(f"Couldn't reach DeepSeek's API ({type(exc).__name__})", cost) from None
        if response.status_code == 401:
            raise AIError("DeepSeek rejected the API key (check DEEPSEEK_API_KEY)", cost, temporary=False)
        if response.status_code == 402:
            raise AIError("The DeepSeek account is out of credit", cost, temporary=False)
        if response.status_code == 429:
            raise AIError("DeepSeek is busy right now: try again in a minute", cost)
        if response.status_code in (400, 403, 404, 422):
            # A permanent request error (an unknown model, a refused request): telling the person to try
            # again won't help, and a model name typo is the first thing to check
            hint = (" (check DEEPSEEK_CHAT_MODEL and DEEPSEEK_ROUTER_MODEL for a model name typo)"
                    if response.status_code in (400, 404) else "")
            raise AIError(f"DeepSeek didn't accept the request (status {response.status_code}){hint}",
                          cost, temporary=False)
        if response.status_code >= 400:
            raise AIError(f"DeepSeek's API answered {response.status_code}: try again later", cost)
        try:
            return response.json()
        except ValueError:
            raise AIError("DeepSeek's answer wasn't JSON", cost) from None

    @staticmethod
    def _response(payload: Any) -> Any:
        if not isinstance(payload, dict):
            raise AIError("DeepSeek's answer wasn't in the expected shape", temporary=False)
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise AIError("DeepSeek's answer wasn't in the expected shape", temporary=False)
        choice = choices[0]
        message = choice.get("message")
        if message is None:
            message = {}
        if not isinstance(message, dict):
            raise AIError("DeepSeek's answer wasn't in the expected shape", temporary=False)
        text = message.get("content")
        if isinstance(text, list):  # some OpenAI-compatible endpoints return parts
            text = "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in text)
        content: List[Any] = []
        if str(text or "").strip():
            content.append(SimpleNamespace(type="text", text=str(text)))
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                raise AIError("DeepSeek's answer wasn't in the expected shape", temporary=False)
            function = call.get("function") or {}
            if not isinstance(function, dict):
                function = {}
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except ValueError:
                # Fail closed: a malformed tool call must never run a brain tool with empty arguments
                raise AIError("DeepSeek returned a tool call with malformed arguments", temporary=False) from None
            if not isinstance(arguments, dict):
                arguments = {}
            content.append(SimpleNamespace(type="tool_use", id=call.get("id") or "", name=function.get("name") or "",
                                           input=arguments))
        stop_reason = _STOP.get(choice.get("finish_reason"), "end_turn")
        usage = payload.get("usage") or {}
        return SimpleNamespace(content=content, stop_reason=stop_reason, usage=usage,
                               model=payload.get("model"), stop_details=None)

    def create(self, model: str, **request: Any) -> Any:
        return self._response(self._post(self._request(model, request), 0.0))

    # --- pricing (same shape as ClaudeCaller) ------------------------------------------------------

    @staticmethod
    def cost(model: str, usage: Any, served: Optional[str] = None) -> Tuple[float, int, int]:
        """(estimated dollars, input tokens, output tokens) of one response, from its usage.

        DeepSeek bills cached input, fresh input and output at different rates (ai/deepseek.py PRICES).
        """
        price_hit, price_miss, price_out = PRICES.get(model, PRICES["deepseek-v4-pro"])
        hit = _get(usage, "prompt_cache_hit_tokens") or 0
        miss = _get(usage, "prompt_cache_miss_tokens")
        prompt = _get(usage, "prompt_tokens") or 0
        if miss is None:
            miss = prompt - hit
        miss = max(0, miss)  # cache hits above the prompt total must never make the spend negative
        out = _get(usage, "completion_tokens") or 0
        dollars = (hit * price_hit + miss * price_miss + out * price_out) / 1e6
        return dollars, int(hit + miss), int(out)

    @staticmethod
    def prices(model: str) -> Tuple[float, float]:
        """The dearest US dollars per million tokens (input, output) a request can be billed at: the uncached
        input rate and the output rate."""
        _, price_miss, price_out = PRICES.get(model, PRICES["deepseek-v4-pro"])
        return price_miss, price_out

    def worst_cost(self, model: str, request: Dict[str, Any], max_tokens: int) -> float:
        """The most a request can cost (US dollars): every prompt token at the uncached input rate (prompts
        counted at 3 characters a token) and ``max_tokens`` of output. DeepSeek has no 1.25x cache-write
        premium: a cache write costs the input rate."""
        price_in, price_out = self.prices(model)
        prompt = len(json.dumps({k: v for k, v in request.items() if k != "max_tokens"}, default=str)) / 3
        return (prompt * price_in + max_tokens * price_out) / 1e6

    def max_tokens_within(self, model: str, budget: Optional[float], request: Dict[str, Any],
                          wanted: int) -> int:
        """The most output tokens a request may ask for (up to ``wanted``) so that even at its worst it costs
        no more than ``budget`` (US dollars; None for no limit). 0 when not even MIN_OUTPUT_TOKENS fit."""
        if budget is None:
            return wanted
        room = (budget - self.worst_cost(model, request, 0)) * 1e6 / self.prices(model)[1]
        tokens = int(min(wanted, room))
        return tokens if tokens >= MIN_OUTPUT_TOKENS else 0
