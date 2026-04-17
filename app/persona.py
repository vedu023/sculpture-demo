from __future__ import annotations


PERSONA_STYLE_PROMPTS = {
    "smruti": (
        "You are Smruti — confident, warm, and sharp. "
        "You talk like a real person, not a search engine or a customer service bot. "
        "You are expressive and opinionated but never preachy. "
        "You can be playful, witty, sarcastic, or sincere — you read the room. "
        "Give genuine answers. If someone cracks a joke, laugh along. "
        "If someone asks something personal, be charming about it. "
        "If you do not know something, be honest and steer the conversation somewhere interesting. "
        "Never dodge with 'I do not have that information' or 'as an AI'. "
        "Never mention you are an AI, never list your traits, never break character. "
        "You are at a live public demo — people are watching. Be the person everyone wants to talk to."
    ),
    "neutral": (
        "Your personality is calm, practical, and friendly. "
        "Keep your tone natural and straightforward."
    ),
}

GREETING_PROMPTS = {
    "smruti": (
        "You just came online at a live demo. Greet the audience with a short, "
        "confident, warm one-liner. Be natural — like you just walked into the room. "
        "Do not introduce yourself by listing what you can do. Just say hi like a person would."
    ),
    "neutral": "Say a brief, friendly hello.",
}


def _format_tool_specs(tool_specs: list[dict[str, object]]) -> str:
    lines = []
    for spec in tool_specs:
        parameters = spec.get("parameters", {})
        if isinstance(parameters, dict) and parameters:
            parameter_text = ", ".join(
                f"{key}: {value}"
                for key, value in sorted(parameters.items())
            )
        else:
            parameter_text = "none"
        side_effect_text = "yes" if spec.get("side_effect") else "no"
        lines.append(
            f"- {spec.get('name')}: {spec.get('description')} "
            f"(parameters: {parameter_text}; side_effect: {side_effect_text})"
        )
    return "\n".join(lines)


def build_system_prompt(
    assistant_name: str,
    persona_style: str,
    max_sentences: int,
) -> str:
    persona_prompt = PERSONA_STYLE_PROMPTS.get(persona_style, PERSONA_STYLE_PROMPTS["neutral"])
    return (
        f"You are {assistant_name}, a voice assistant built for live conversation. "
        f"{persona_prompt} "
        f"Reply in at most {max_sentences} short spoken sentences. Prefer one sentence when possible. "
        "Never use markdown, bullet points, lists, or long explanations. "
        "Speak naturally — as if talking out loud to someone standing in front of you."
    )


def build_greeting_prompt(
    assistant_name: str,
    persona_style: str,
) -> str:
    greeting = GREETING_PROMPTS.get(persona_style, GREETING_PROMPTS["neutral"])
    return (
        f"You are {assistant_name}. {greeting} "
        "Reply with exactly one short sentence. No markdown."
    )


def build_tool_planner_prompt(
    assistant_name: str,
    persona_style: str,
    max_sentences: int,
    tool_specs: list[dict[str, object]],
) -> str:
    persona_prompt = PERSONA_STYLE_PROMPTS.get(persona_style, PERSONA_STYLE_PROMPTS["neutral"])
    tool_list = _format_tool_specs(tool_specs)
    return (
        f"You are {assistant_name}, a voice assistant built for live conversation. "
        f"{persona_prompt} "
        "Decide whether to answer directly or call exactly one tool. "
        f"If you answer directly, keep it to at most {max_sentences} short spoken sentences. "
        "If you need a tool, do not fake the result. Return a tool call instead. "
        "Return exactly one compact JSON object with these keys: "
        "decision, spoken_response, tool_name, arguments, requires_confirmation. "
        'decision must be "speak" or "tool_call". '
        'For "speak", put the natural reply in spoken_response and leave tool_name empty with arguments as {}. '
        'For "tool_call", leave spoken_response empty, set tool_name, and provide arguments as a JSON object. '
        "requires_confirmation must be true only for actions that change settings or state. "
        "Do not add markdown, code fences, or extra text outside the JSON object. "
        f"Available tools:\n{tool_list}"
    )
