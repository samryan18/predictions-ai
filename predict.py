#!/usr/bin/env python3
"""Prediction harness for AI models with extended thinking and web search."""

import json
import csv
import re
import argparse
import time
from datetime import datetime
import anthropic
import openai
from secrets import ANTHROPIC_API_KEY, OPENAI_API_KEY

SYSTEM_PROMPT = """Today is January 1, 2026. You are participating in a predictions contest for 2026. You will answer questions by providing a probability between 0 and 1 representing the likelihood that the event resolves to TRUE.

## Instructions
- Answer with a single number between 0.000001 and 0.999999
- A scoring committee (Sam, John, Arbel, Evan) will resolve any ambiguities
- Reason from information available as of January 1, 2026

## Thinking Process
Before answering, consider:
1. What is the historical base rate or reference class for this type of event?
2. What would have to be true for this to happen? What would have to be true for it NOT to happen?
3. Adjust from the base rate based on current specifics
4. Use web search to do research as needed. Please spend time thinking about these.
5. Take your time and try really really hard to win this contest by submitting very high quality predictions.

## Scoring
You will be scored with binary cross-entropy loss (log loss). Confidently correct predictions are rewarded; confidently incorrect predictions are heavily penalized.

**Calibration warning:** Probabilities below 0.02 or above 0.98 should be relatively rare. Surprising things happen — reserve extreme confidence for near-certainties like "the sun rises tomorrow."

## Notes
- "In the year 2026" is implied unless otherwise stated
- "The group" in Personal category questions refers to everyone who submits a prediction slip

## Response Format
Provide your probability, then a brief justification (2-4 sentences) showing your reasoning. Use exactly the example format below.

# Question
Will the US unemployment rate exceed 5% at any point in 2026?


# Answer
0.28

Justification: The unemployment rate has stayed below 5% since late 2021, and current rate is approximately 4.2%. However, economic conditions can shift quickly — in 6 of the last 20 years, unemployment exceeded 5%. I'm adjusting upward slightly from a pure base rate given elevated recession indicators, but staying below 0.5 since no imminent downturn is forecasted."""


def parse_prediction(answer: str) -> tuple[str, str]:
    """Extract probability and justification from answer text."""
    lines = answer.strip().split("\n")
    probability = ""
    justification = ""

    for i, line in enumerate(lines):
        # Look for a line that's just a number (the probability)
        match = re.match(r"^(0\.\d+)$", line.strip())
        if match:
            probability = match.group(1)
            # Everything after is justification
            rest = "\n".join(lines[i+1:]).strip()
            if rest.startswith("Justification:"):
                justification = rest[14:].strip()
            else:
                justification = rest
            break

    return probability, justification


def predict_claude(question: str) -> dict:
    """Get prediction from Claude Opus 4.5 with extended thinking and web search."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    start_time = time.time()
    response = client.messages.create(
        model="claude-opus-4-5-20250514",
        max_tokens=16000,
        thinking={
            "type": "enabled",
            "budget_tokens": 10000
        },
        tools=[{"type": "web_search", "name": "web_search", "max_uses": 10}],
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}]
    )
    elapsed = time.time() - start_time

    thinking = ""
    answer = ""
    thinking_tokens = 0
    for block in response.content:
        if block.type == "thinking":
            thinking = block.thinking
            thinking_tokens = len(block.thinking.split())  # Approximate
        elif block.type == "text":
            answer = block.text

    probability, justification = parse_prediction(answer)

    return {
        "thinking": thinking,
        "answer": answer,
        "probability": probability,
        "justification": justification,
        "model_id": "claude-opus-4-5-20250514",
        "model_settings": "thinking_budget=10000, web_search_max_uses=10",
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "thinking_tokens_approx": thinking_tokens,
        "elapsed_seconds": round(elapsed, 2),
    }


def predict_openai(question: str) -> dict:
    """Get prediction from GPT-5.2-pro with reasoning and web search."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    start_time = time.time()
    response = client.chat.completions.create(
        model="gpt-5.2-pro-2025-12-11",
        max_completion_tokens=16000,
        reasoning_effort="high",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        tools=[{"type": "function", "function": {"name": "web_search", "description": "Search the web", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}],
    )
    elapsed = time.time() - start_time

    answer = response.choices[0].message.content or ""
    reasoning_tokens = getattr(response.usage, "completion_tokens_details", None)
    reasoning_tokens = getattr(reasoning_tokens, "reasoning_tokens", 0) if reasoning_tokens else 0

    probability, justification = parse_prediction(answer)

    return {
        "thinking": "",  # OpenAI doesn't expose reasoning trace
        "answer": answer,
        "probability": probability,
        "justification": justification,
        "model_id": "gpt-5.2-pro-2025-12-11",
        "model_settings": "reasoning_effort=high",
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "thinking_tokens_approx": reasoning_tokens,
        "elapsed_seconds": round(elapsed, 2),
    }


MODELS = {
    "claude": predict_claude,
    "openai": predict_openai,
}


def main():
    parser = argparse.ArgumentParser(description="AI Prediction Harness")
    parser.add_argument("questions_file", help="JSON file with questions")
    parser.add_argument("--model", choices=["claude", "openai"], required=True)
    parser.add_argument("--output", default="predictions.csv", help="Output CSV file")
    args = parser.parse_args()

    with open(args.questions_file) as f:
        questions = json.load(f)

    predict_fn = MODELS[args.model]

    fieldnames = [
        "question_id", "category", "question", "probability", "justification",
        "model_id", "model_settings", "input_tokens", "output_tokens",
        "thinking_tokens_approx", "elapsed_seconds", "timestamp",
        "raw_answer", "thinking"
    ]

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for q in questions:
            qid = q.get("id", "")
            category = q.get("category", "")
            question_text = q["question"]
            context = q.get("context", "")

            # Build full prompt with context if available
            full_prompt = f"# Question\n{question_text}"
            if context:
                full_prompt += f"\n\nContext: {context}"

            print(f"[{qid}] {question_text[:50]}...")

            result = predict_fn(full_prompt)

            writer.writerow({
                "question_id": qid,
                "category": category,
                "question": question_text,
                "probability": result["probability"],
                "justification": result["justification"],
                "model_id": result["model_id"],
                "model_settings": result["model_settings"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "thinking_tokens_approx": result["thinking_tokens_approx"],
                "elapsed_seconds": result["elapsed_seconds"],
                "timestamp": datetime.now().isoformat(),
                "raw_answer": result["answer"],
                "thinking": result["thinking"],
            })
            f.flush()
            print(f"  Done. Probability: {result['probability']}")

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
