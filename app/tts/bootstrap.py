from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def bootstrap_runtime_models(
    config,
    *,
    asr_engine,
    llm_engine,
    tts_engine,
    force: bool = False,
):
    """Warm runtime models for the active backend."""
    logger.info("Prefetching models for %s...", config.llm.assistant_name)
    
    # Warm up engines
    asr_engine.warmup()
    llm_engine.warmup()
    tts_engine.warmup()
    
    logger.info("Model prefetch complete. tts=%s", tts_engine.describe())
