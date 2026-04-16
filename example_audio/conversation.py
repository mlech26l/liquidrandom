"""Conversation generation logic: seeding, prompting, parsing."""

from __future__ import annotations

import json
import random
from typing import Any

import liquidrandom
from openai import AsyncOpenAI

from llm import llm_call


def _prepare_tools(
    tool_groups: list[liquidrandom.ToolGroup],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Prepare tool definitions from tool groups.

    Returns (original_order_tools, shuffled_tools) in OpenAI function format.
    """
    tools: list[dict[str, Any]] = []

    for group in tool_groups:
        # Pick one random variation index per group — variations at the same
        # index are guaranteed interface-compatible across tools within a group.
        var_idx = random.randint(0, 7)
        for tool_fn in group.tools:
            if not tool_fn.variations:
                continue
            idx = var_idx if var_idx < len(tool_fn.variations) else 0
            var = tool_fn.variations[idx]
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": var.name,
                        "description": var.description,
                        "parameters": var.parameters,
                    },
                }
            )

    if not tools:
        return [], []

    # Target 1-15 tools per sample. Drop from the pool to hit a random target.
    target = random.randint(1, 15)
    if len(tools) > target:
        drop_indices = random.sample(range(len(tools)), len(tools) - target)
        tools = [t for i, t in enumerate(tools) if i not in drop_indices]

    original_order = list(tools)
    shuffled = list(tools)
    random.shuffle(shuffled)

    return original_order, shuffled


def _build_generation_prompt(
    persona: liquidrandom.Persona,
    shuffled_tools: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build the prompt for conversation generation."""
    tools_json = json.dumps(shuffled_tools, indent=2)
    num_turns = random.randint(2, 5)

    system = f"""You are a conversation generator. Generate a realistic multi-turn conversation between a user and an AI assistant that has access to tools.

The user has this persona:
{persona}

The assistant has access to these tools:
{tools_json}

Rules:
1. Generate exactly {num_turns} user-assistant exchange rounds (a round = user turn, then assistant turn, then if assistant called a tool: a tool response turn followed by a second assistant turn that uses the tool result).
2. USER turns ONLY will be converted to speech audio later, so for USER turns:
   - Use natural spoken language with filler words sprinkled THROUGHOUT (not just at the start!)
   - Filler words include: um, uh, well, so, like, you know, I mean, right, hmm, let me think, okay so, actually, basically
   - Place fillers at the start, in the middle of sentences, and between clauses — just like real speech
   - Occasionally add a double space "  " to represent a brief pause in speech (use sparingly, maybe 1-2 per conversation)
   - Keep sentences short and conversational
   - No special characters, URLs, code snippets, or complex formatting
   - No overly long turns (max ~2-3 sentences per user turn)
   ASSISTANT turns must be CLEAN, professional, and friendly — like a real AI assistant:
   - NO filler words (no "um", "uh", "like", "you know", etc.)
   - NO double spaces or simulated speech pauses
   - Clear, concise, helpful language
   - Normal punctuation and grammar
3. Not every round needs a tool call. Some rounds should be regular conversation.
4. When the assistant uses a tool call, the arguments MUST:
   - Only use parameters defined in the tool's parameter schema
   - Only contain values that were explicitly mentioned by the user, present in the system prompt, or established in prior conversation
   - Never hallucinate or assume parameter values not provided by the user
5. After each tool call, include a "tool" role turn with a realistic JSON response, then a follow-up assistant turn that interprets the tool result for the user.
6. The conversation should feel natural and coherent, matching the user's persona.

Respond with a JSON array of turns. Each turn is an object with:
- "role": "user", "assistant", or "tool"
- "content": the text content (for tool role, this is a JSON string of the tool's return value)
- "tool_calls": (optional, assistant only) array of objects with "function_name" and "arguments" (dict)

Example format:
[
  {{"role": "user", "content": "Hey, um, can you check the weather for me  in New York?"}},
  {{"role": "assistant", "content": "Sure, let me check that for you.", "tool_calls": [{{"function_name": "get_weather", "arguments": {{"city": "New York"}}}}]}},
  {{"role": "tool", "content": "{{\\"temperature\\": 72, \\"condition\\": \\"partly cloudy\\", \\"humidity\\": 45}}"}},
  {{"role": "assistant", "content": "It's 72 degrees and partly cloudy in New York right now, with 45 percent humidity."}},
  {{"role": "user", "content": "Nice, thanks! So like, I was thinking  what should I, you know, wear today?"}},
  {{"role": "assistant", "content": "I'd say a light jacket would be perfect for that kind of weather."}}
]"""

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": "Generate the conversation now. Return only the JSON array.",
        },
    ]


def _build_qa_prompt(
    persona: liquidrandom.Persona,
    seed: Any,
) -> list[dict[str, str]]:
    """Build prompt for generating a non-tool Q&A pair."""
    system = f"""Generate a single casual question-answer pair for a conversation between a user and an AI assistant.

The user has this persona:
{persona}

Use the following ONLY as a broad thematic hint (do NOT reference specific details from it):
{seed}

Rules:
- The question must be self-contained and make sense on its own, as if it's the start of a new topic in a conversation. It must NOT reference anything from prior context (no "those", "that thing", "the one you mentioned", etc.).
- The question should be SIMPLE and general. Use the seed theme only to pick a broad topic area.
- The USER question should be natural spoken language (will be converted to audio):
  - Sprinkle filler words THROUGHOUT, not just at the start: um, uh, well, so, like, you know, I mean, right, hmm, actually, basically
  - Place fillers at the start, mid-sentence, and between clauses
  - Optionally use a double space "  " for a brief speech pause (sparingly)
  - No special characters, URLs, or code
- The ASSISTANT answer must be CLEAN and professional:
  - NO filler words, NO double spaces, NO simulated speech pauses
  - Clear, concise, helpful language with normal punctuation
- Keep it short: 1-2 sentences for the question, 1-3 sentences for the answer
- No tool calls, just a regular conversation exchange

Examples of GOOD Q&A pairs (notice fillers throughout and self-contained questions):
- {{"user": "Hey so um, what's the deal with  like black holes? Are they actually, you know, holes?", "assistant": "Not really holes in the traditional sense. They're regions where gravity is so strong that nothing can escape, not even light. Think of them more like cosmic vacuum cleaners."}}
- {{"user": "So like, do you know if it's  I mean is it better to learn piano or guitar first?", "assistant": "It depends on what you're going for. Piano is great for understanding music theory since everything is laid out visually. Guitar is more portable and you can start playing songs pretty quickly."}}
- {{"user": "I was wondering actually, um, what makes sourdough bread  like sour exactly?", "assistant": "It's the wild yeast and bacteria in the starter. They produce lactic and acetic acids during fermentation, which give it that tangy flavor."}}

Examples of BAD questions (these reference prior context and feel out of place):
- "So what happens when those lipid droplets break down?" (references something not in the conversation)
- "Can you tell me more about that algorithm you mentioned?" (references prior context)

Return a JSON object with exactly two fields:
{{"user": "the user's question", "assistant": "the assistant's response"}}"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "Generate the Q&A pair. Return only JSON."},
    ]


async def _generate_qa_pairs(
    client: AsyncOpenAI,
    persona: liquidrandom.Persona,
    count: int,
) -> list[list[dict[str, str]]]:
    """Generate non-tool Q&A turn pairs. Returns list of pairs (each pair = [user_turn, assistant_turn])."""
    pairs: list[list[dict[str, str]]] = []
    # Pick one seed type for all pairs in this sample
    seed_funcs = [
        liquidrandom.scenario,
        liquidrandom.domain,
        liquidrandom.science_topic,
        liquidrandom.writing_style,
    ]
    for _ in range(count):
        seed = random.choice(seed_funcs)()
        messages = _build_qa_prompt(persona, seed)
        try:
            result = await llm_call(client, messages)
            if isinstance(result, dict) and "user" in result and "assistant" in result:
                pairs.append([
                    {"role": "user", "content": result["user"]},
                    {"role": "assistant", "content": result["assistant"]},
                ])
        except Exception:
            continue
    return pairs


def _inject_qa_pairs(
    conversation: list[dict[str, Any]],
    qa_pairs: list[list[dict[str, str]]],
    allow_first: bool = True,
) -> list[dict[str, Any]]:
    """Insert Q&A pairs at valid positions in the conversation.

    Valid positions are boundaries between exchange rounds (i.e., spots where
    a user turn starts a new round). We never place two user turns back-to-back.
    If allow_first is True, position 0 (before the first turn) is also valid.
    """
    if not qa_pairs:
        return conversation

    # Find valid insertion points: indices where a user turn begins a new round.
    # These are positions right before a user turn that follows an assistant turn
    # (or the very start of the conversation).
    insert_points: list[int] = []
    if allow_first:
        insert_points.append(0)
    for i in range(1, len(conversation)):
        if conversation[i]["role"] == "user" and conversation[i - 1]["role"] in ("assistant",):
            insert_points.append(i)

    if not insert_points:
        return conversation

    result = list(conversation)
    offset = 0
    for pair in qa_pairs:
        if not insert_points:
            break
        pos = random.choice(insert_points)
        actual_pos = pos + offset
        for j, turn in enumerate(pair):
            result.insert(actual_pos + j, turn)
        offset += len(pair)
        # Shift all insertion points after this one
        insert_points = [p for p in insert_points if p != pos]

    return result


async def generate_sample(
    client: AsyncOpenAI,
    sample_id: int,
) -> dict[str, Any] | None:
    """Generate a single conversation sample.

    Returns None if generation fails after retries.
    """
    from validator import validate_conversation

    # 1. Seed
    persona = liquidrandom.persona()
    num_groups = random.randint(1, 5)
    tool_groups = [liquidrandom.physical_tool_group() for _ in range(num_groups)]
    original_tools, shuffled_tools = _prepare_tools(tool_groups)

    if not original_tools:
        return None

    # 2. Generate conversation (with retries via validation)
    for attempt in range(3):
        # Generate main conversation
        messages = _build_generation_prompt(persona, shuffled_tools)
        try:
            conversation = await llm_call(client, messages)
        except Exception:
            continue

        if not isinstance(conversation, list):
            continue

        # Filter out malformed turns (LLM may omit "role")
        conversation = [t for t in conversation if isinstance(t, dict) and "role" in t]
        if not conversation:
            continue

        # 3. Generate and inject non-tool Q&A pairs (1-3)
        num_qa = random.randint(1, 3)
        qa_pairs = await _generate_qa_pairs(client, persona, num_qa)
        if qa_pairs:
            # allow_first=True so sometimes the conversation starts with an unrelated turn
            conversation = _inject_qa_pairs(conversation, qa_pairs, allow_first=True)

        # 4. Build sample
        num_tool_calls = sum(
            len(turn.get("tool_calls") or [])
            for turn in conversation
            if turn.get("role") == "assistant"
        )

        sample = {
            "sample_id": str(sample_id),
            "persona": str(persona),
            "tools_json": json.dumps(original_tools),
            "conversation": json.dumps(conversation),
            "num_turns": len(conversation),
            "num_tool_calls": num_tool_calls,
        }

        # 5. Validate
        try:
            is_valid, reason = await validate_conversation(
                client, sample, original_tools
            )
        except Exception:
            continue

        if is_valid:
            return sample

    return None
