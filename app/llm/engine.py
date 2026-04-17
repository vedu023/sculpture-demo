import collections
import json
import logging
import re

import ollama

from app.tools.types import ToolDecision
from app.utils.text import sanitize_spoken_response

logger = logging.getLogger(__name__)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMEngine:
    """LLM using Ollama local models."""

    def __init__(
        self,
        model_name: str = "qwen3:14b",
        system_prompt: str = "",
        max_tokens: int = 128,
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
    ) -> str:
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
            think=False,
            options={
                "num_predict": self.max_tokens,
                "temperature": temperature,
            }
        )

        raw_content = response["message"]["content"]
        logger.info("%s raw response (%d chars): %r", label, len(raw_content), raw_content[:200])
        return raw_content.strip()

    def plan_turn(self, user_text: str, planner_prompt: str) -> ToolDecision:
        """Plan whether to answer directly or execute a bounded tool call."""
        try:
            raw_content = self._chat(
                self._build_messages(planner_prompt, user_text),
                temperature=0.1,
                label="LLM planner",
            )
        except Exception as e:
            logger.error("Ollama planner failed: %s", e)
            return ToolDecision()

        payload = raw_content
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            match = _JSON_OBJECT_RE.search(payload)
            if not match:
                logger.warning("Planner returned non-JSON output: %r", payload[:200])
                return ToolDecision()
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning("Planner returned invalid JSON output: %r", payload[:200])
                return ToolDecision()

        if not isinstance(parsed, dict):
            logger.warning("Planner output was not a JSON object: %r", parsed)
            return ToolDecision()

        decision = str(parsed.get("decision", "speak")).strip().lower()
        if decision not in {"speak", "tool_call"}:
            decision = "speak"

        spoken_response = sanitize_spoken_response(
            str(parsed.get("spoken_response", "")),
            max_sentences=self.max_sentences,
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
            raw_content = self._chat(
                self._build_messages(self.system_prompt, user_text),
                temperature=0.7,
                label="LLM",
            )
            response_text = sanitize_spoken_response(raw_content, max_sentences=self.max_sentences)

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
