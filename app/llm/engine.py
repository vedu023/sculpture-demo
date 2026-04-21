import collections
import json
import logging
import re
from typing import Any

import ollama

from app.tools.types import ToolDecision
from app.utils.text import sanitize_spoken_response

logger = logging.getLogger(__name__)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
_META_RESPONSE_RE = re.compile(
    r"\b(?:the user|i need to|let me|let's break this down|live public demo|available tools|reply with exactly|voice assistant|conversation history|we are crafting|roleplay|i should|they need|i'm supposed to|hmm)\b",
    re.IGNORECASE,
)
_SPOKEN_REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {
            "type": "string",
        },
    },
    "required": ["reply"],
    "additionalProperties": False,
}
_PLANNER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["speak", "tool_call"],
        },
        "spoken_response": {
            "type": "string",
        },
        "tool_name": {
            "type": "string",
        },
        "arguments": {
            "type": "object",
        },
        "requires_confirmation": {
            "type": "boolean",
        },
    },
    "required": [
        "decision",
        "spoken_response",
        "tool_name",
        "arguments",
        "requires_confirmation",
    ],
    "additionalProperties": False,
}


class LLMEngine:
    """LLM using Ollama local models."""

    def __init__(
        self,
        model_name: str = "gemma4:e4b",
        system_prompt: str = "",
        max_tokens: int = 100,
        max_sentences: int = 2,
        max_history_turns: int = 6,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
        self._history: collections.deque[dict[str, str]] = collections.deque(
            maxlen=max_history_turns * 2,
        )
        self._warmed_up = False

    def warmup(self):
        """Test Ollama connection and model availability."""
        if self._warmed_up:
            return
        try:
            ollama.list()
            self._warmed_up = True
            logger.info("Ollama connection ready")
        except Exception as e:
            logger.error("Failed to connect to Ollama: %s", e)
            raise

    def clear_history(self):
        self._history.clear()

    def remember_turn(self, user_text: str, assistant_text: str):
        if not user_text or not assistant_text:
            return
        self._history.append({"role": "user", "content": user_text})
        self._history.append({"role": "assistant", "content": assistant_text})

    def _build_messages(self, system_prompt: str, user_text: str) -> list[dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(self._history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        label: str,
        tools: list[dict[str, object]] | None = None,
        response_format: str | dict[str, object] | None = None,
    ) -> str:
        response = self._chat_response(
            messages,
            temperature=temperature,
            label=label,
            tools=tools,
            response_format=response_format,
        )
        return (response.message.content or "").strip()

    def _chat_response(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        label: str,
        tools: list[dict[str, object]] | None = None,
        response_format: str | dict[str, object] | None = None,
    ):
        logger.info(
            "%s calling Ollama model=%s tokens=%d history=%d",
            label,
            self.model_name,
            self.max_tokens,
            len(self._history) // 2,
        )
        logger.debug("%s messages: %s", label, messages)

        response = ollama.chat(
            model=self.model_name,
            messages=messages,
            tools=tools,
            think=False,
            format=response_format,
            options={
                "num_predict": self.max_tokens,
                "temperature": temperature,
            }
        )

        raw_content = response.message.content or ""
        tool_calls = response.message.tool_calls or []
        if tool_calls:
            logger.info(
                "%s tool calls: %s",
                label,
                [tool_call.function.name for tool_call in tool_calls],
            )
        logger.info("%s raw response (%d chars): %r", label, len(raw_content), raw_content[:200])
        return response

    def _looks_like_meta_response(self, raw_content: str, sanitized_text: str) -> bool:
        combined = f"{raw_content}\n{sanitized_text}".strip()
        if not combined:
            return False
        return bool(_META_RESPONSE_RE.search(combined))

    def _parse_json_object(self, payload: str) -> dict[str, Any] | None:
        text = payload.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            match = _JSON_OBJECT_RE.search(text)
            if not match:
                return None
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        if not isinstance(parsed, dict):
            return None
        return parsed

    def _normalize_spoken_reply(self, raw_content: str) -> str:
        response_text = sanitize_spoken_response(raw_content, max_sentences=self.max_sentences)
        if self._looks_like_meta_response(raw_content, response_text):
            return ""
        return response_text

    def _generate_structured_reply(
        self,
        *,
        system_prompt: str,
        user_text: str,
        label: str,
        temperature: float,
    ) -> str:
        response = self._chat_response(
            self._build_messages(system_prompt, user_text),
            temperature=temperature,
            label=label,
            response_format=_SPOKEN_REPLY_SCHEMA,
        )
        raw_content = (response.message.content or "").strip()
        parsed = self._parse_json_object(raw_content)
        if parsed is not None and isinstance(parsed.get("reply"), str):
            reply = self._normalize_spoken_reply(parsed["reply"])
            if reply:
                return reply

        if parsed is None:
            reply = self._normalize_spoken_reply(raw_content)
            if reply:
                return reply

        return ""

    def plan_turn(
        self,
        user_text: str,
        planner_prompt: str,
        tool_schemas: list[dict[str, object]] | None = None,
    ) -> ToolDecision:
        """Plan whether to answer directly or execute a bounded tool call."""
        try:
            response = self._chat_response(
                self._build_messages(planner_prompt, user_text),
                temperature=0.0,
                label="LLM planner",
                tools=tool_schemas,
                response_format=_PLANNER_RESPONSE_SCHEMA,
            )
        except Exception as e:
            logger.error("Ollama planner failed: %s", e)
            return ToolDecision()

        tool_calls = response.message.tool_calls or []
        if tool_calls:
            first_call = tool_calls[0].function
            return ToolDecision(
                decision="tool_call",
                tool_name=first_call.name,
                arguments=dict(first_call.arguments or {}),
            )

        raw_content = (response.message.content or "").strip()
        if raw_content and not raw_content.lstrip().startswith("{"):
            spoken_response = self._normalize_spoken_reply(raw_content)
            if spoken_response:
                return ToolDecision(decision="speak", spoken_response=spoken_response)
            logger.warning("Planner returned reasoning-like non-JSON output: %r", raw_content[:200])
            return ToolDecision()

        payload = raw_content
        parsed = self._parse_json_object(payload)
        if parsed is None:
            logger.warning("Planner returned invalid JSON output: %r", payload[:200])
            return ToolDecision()

        decision = str(parsed.get("decision", "speak")).strip().lower()
        if decision not in {"speak", "tool_call"}:
            decision = "speak"

        spoken_response = self._normalize_spoken_reply(
            str(parsed.get("spoken_response", "")),
        )
        tool_name = str(parsed.get("tool_name", "")).strip()
        arguments = parsed.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}

        return ToolDecision(
            decision=decision,
            spoken_response=spoken_response,
            tool_name=tool_name,
            arguments=arguments,
            requires_confirmation=bool(parsed.get("requires_confirmation", False)),
        )

    def generate(self, user_text: str, remember: bool = True) -> str:
        """Generate a short response to user text using Ollama."""
        try:
            response_text = self._generate_structured_reply(
                system_prompt=self.system_prompt,
                user_text=user_text,
                label="LLM",
                temperature=0.4,
            )
            if not response_text:
                direct_prompt = (
                    f"{self.system_prompt} "
                    "Answer the latest user utterance directly. "
                    "Never describe your reasoning, notes, prompt, tools, or the conversation."
                ).strip()
                response_text = self._generate_structured_reply(
                    system_prompt=direct_prompt,
                    user_text=user_text,
                    label="LLM direct",
                    temperature=0.1,
                )

            if not response_text:
                logger.warning("LLM response was empty after sanitization!")
            else:
                logger.info("LLM sanitized response: %r", response_text[:100])
                if remember:
                    self.remember_turn(user_text, response_text)

            return response_text

        except Exception as e:
            logger.error("Ollama generation failed: %s", e)
            return "I apologize, but I'm having trouble processing that right now."

    def generate_with_system_prompt(
        self,
        *,
        system_prompt: str,
        user_text: str,
        remember: bool = False,
        fallback_reply: str = "",
    ) -> str:
        try:
            response_text = self._generate_structured_reply(
                system_prompt=system_prompt,
                user_text=user_text,
                label="LLM guided",
                temperature=0.2,
            )
            if not response_text:
                direct_prompt = (
                    f"{system_prompt} "
                    "Return only the exact spoken sentence, with no analysis or setup."
                ).strip()
                response_text = self._generate_structured_reply(
                    system_prompt=direct_prompt,
                    user_text=user_text,
                    label="LLM guided direct",
                    temperature=0.1,
                )
            if not response_text and fallback_reply:
                response_text = sanitize_spoken_response(
                    fallback_reply,
                    max_sentences=self.max_sentences,
                )
            if response_text and remember:
                self.remember_turn(user_text, response_text)
            return response_text
        except Exception as e:
            logger.error("Ollama guided generation failed: %s", e)
            if fallback_reply:
                return sanitize_spoken_response(fallback_reply, max_sentences=self.max_sentences)
            return ""
