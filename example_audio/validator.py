"""Quality validation for generated conversations."""

from __future__ import annotations

import json
from typing import Any

from openai import AsyncOpenAI

from llm import llm_call


async def validate_conversation(
    client: AsyncOpenAI,
    sample: dict[str, Any],
    tools: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Validate a generated conversation for quality.

    Checks:
    - Tool calls use only defined tools
    - Arguments match parameter schemas (no hallucinated args)
    - Argument values come from user request, system prompt, or prior context
    - Every tool call is followed by a tool response turn and then an assistant turn
    - Tool responses are valid JSON and plausible for the tool
    - User turns are audio-appropriate

    Returns (is_valid, reason).
    """
    conversation = json.loads(sample["conversation"])
    tools_json = json.dumps(tools, indent=2)
    conv_json = json.dumps(conversation, indent=2)

    system = f"""You are a quality validator for synthetic conversations. Analyze the conversation below and check for issues.

Available tool definitions:
{tools_json}

Conversation to validate:
{conv_json}

Check ALL of the following:

1. TOOL CALL CORRECTNESS:
   - Every tool call must use a function name that exists in the tool definitions above
   - Every argument in a tool call must be a valid parameter defined in that tool's parameter schema
   - No extra/hallucinated arguments that aren't in the schema

2. ARGUMENT VALUE SOURCES:
   - Every argument value must come from one of:
     a) The user's explicit request in the current or prior turns
     b) Information established earlier in the conversation
     c) Reasonable defaults that the tool schema specifies
   - Flag any argument values that appear "made up" or assumed without basis
   - Example of hallucinated value: user says "set a reminder for tomorrow" but tool call includes timezone="PST" when user never mentioned timezone

3. TOOL RESPONSE TURNS:
   - Every assistant turn with tool_calls must be immediately followed by a "tool" role turn
   - The tool response content must be valid JSON
   - The tool response must be plausible given the tool's description and the arguments passed
   - After the tool response, there must be an assistant turn that interprets/uses the result

4. AUDIO APPROPRIATENESS of user turns:
   - No URLs, email addresses, or code snippets
   - No special characters or complex formatting
   - Reasonable length (not overly long)
   - Natural spoken language

5. CONVERSATION COHERENCE:
   - Responses are relevant to the questions
   - The conversation makes sense as a whole
   - No two user turns appear back-to-back without an assistant turn between them

Return a JSON object:
{{"valid": true/false, "reason": "explanation of any issues found, or 'all checks passed'"}}"""

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Validate this conversation. Return only JSON."},
    ]

    result = await llm_call(client, messages)

    if isinstance(result, dict):
        is_valid = result.get("valid", False)
        reason = result.get("reason", "unknown")
        return bool(is_valid), str(reason)

    return False, "Unexpected validation response format"
