import logging

import ollama

from app.utils.text import sanitize_spoken_response

logger = logging.getLogger(__name__)


class LLMEngine:
    """LLM using Ollama local models."""

    def __init__(
        self,
        model_name: str = "gemma4:e4b",
        system_prompt: str = "",
        max_tokens: int = 48,
        max_sentences: int = 2,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_sentences = max_sentences
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

    def generate(self, user_text: str) -> str:
        """Generate a short response to user text using Ollama."""
        try:
            messages = []
            if self.system_prompt:
                # Bake instructions into the user turn — some models (e.g.
                # gemma4) ignore the system role and return empty responses.
                user_text = f"[Instructions: {self.system_prompt}]\n\n{user_text}"
            messages.append({"role": "user", "content": user_text})

            logger.info("LLM calling Ollama model=%s tokens=%d", self.model_name, self.max_tokens)
            logger.debug("LLM messages: %s", messages)

            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                think=False,
                options={
                    "num_predict": self.max_tokens,
                    "temperature": 0.7,
                }
            )

            raw_content = response['message']['content']
            logger.info("LLM raw response (%d chars): %r", len(raw_content), raw_content[:200])

            response_text = raw_content.strip()
            response_text = sanitize_spoken_response(response_text, max_sentences=self.max_sentences)

            if not response_text:
                logger.warning("LLM response was empty after sanitization!")
            else:
                logger.info("LLM sanitized response: %r", response_text[:100])

            return response_text

        except Exception as e:
            logger.error("Ollama generation failed: %s", e)
            return "I apologize, but I'm having trouble processing that right now."
