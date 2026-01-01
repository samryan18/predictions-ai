#!/usr/bin/env python3
"""Prediction harness for AI models with extended thinking and web search."""

import json
import csv
import re
import argparse
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
You MUST respond with EXACTLY this format - no other text before the answer:

# Answer
[probability as a decimal, e.g. 0.28]

Justification: [2-4 sentences explaining your reasoning]

IMPORTANT: Start your response with "# Answer" immediately. Do not include any preamble or discussion before giving your answer.

Example:

# Answer
0.28

Justification: The unemployment rate has stayed below 5% since late 2021, and current rate is approximately 4.2%. However, economic conditions can shift quickly — in 6 of the last 20 years, unemployment exceeded 5%. I'm adjusting upward slightly from a pure base rate given elevated recession indicators, but staying below 0.5 since no imminent downturn is forecasted."""


def parse_prediction(answer: str) -> tuple[str, str]:
    """Extract probability and justification from answer text."""
    probability = ""
    justification = ""

    # Try multiple patterns to find the probability
    # Pattern 1: After "# Answer" header (with possible whitespace/newlines)
    answer_match = re.search(r"#\s*Answer\s*\n+\s*(0\.\d+)", answer)
    if answer_match:
        probability = answer_match.group(1)
    else:
        # Pattern 2: Line starting with 0.XX (probability on its own line)
        line_match = re.search(r"^\s*(0\.\d{2,6})\s*$", answer, re.MULTILINE)
        if line_match:
            probability = line_match.group(1)
        else:
            # Pattern 3: Any 0.XX probability-like number in text
            prob_match = re.search(r"\b(0\.\d{2,6})\b", answer)
            if prob_match:
                probability = prob_match.group(1)

    # Extract justification - get text after "Justification:" until end or next section
    just_match = re.search(r"Justification:\s*(.+?)(?:\n\n|\n#|$)", answer, re.DOTALL)
    if just_match:
        justification = just_match.group(1).strip()

    return probability, justification


def predict_claude(question: str) -> dict:
    """Get prediction from Claude Opus 4.5 with extended thinking and web search."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    start_time = time.time()

    # Use streaming for long requests (required for >10 min operations)
    thinking = ""
    answer = ""
    input_tokens = 0
    output_tokens = 0

    with client.messages.stream(
        model="claude-opus-4-5-20251101",
        max_tokens=64000,
        thinking={
            "type": "enabled",
            "budget_tokens": 32000
        },
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 20}],
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}]
    ) as stream:
        response = stream.get_final_message()

    elapsed = time.time() - start_time

    thinking_tokens = 0
    text_parts = []
    for block in response.content:
        if block.type == "thinking":
            thinking = block.thinking
            thinking_tokens = len(block.thinking.split())
        elif block.type == "text":
            text_parts.append(block.text)
    answer = "\n".join(text_parts)

    probability, justification = parse_prediction(answer)

    return {
        "thinking": thinking,
        "answer": answer,
        "probability": probability,
        "justification": justification,
        "model_id": "claude-opus-4-5-20251101",
        "model_settings": "thinking_budget=32000, web_search_max_uses=20",
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "thinking_tokens_approx": thinking_tokens,
        "elapsed_seconds": round(elapsed, 2),
    }


def predict_openai(question: str) -> dict:
    """Get prediction from GPT-5.2 pro with reasoning and web search."""
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    start_time = time.time()
    response = client.responses.create(
        model="gpt-5.2-pro-2025-12-11",
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ],
        reasoning={"effort": "high"},
        tools=[{"type": "web_search_preview"}],
    )
    elapsed = time.time() - start_time

    answer = response.output_text or ""
    reasoning_tokens = 0
    if response.usage and hasattr(response.usage, "output_tokens_details"):
        reasoning_tokens = getattr(response.usage.output_tokens_details, "reasoning_tokens", 0)

    probability, justification = parse_prediction(answer)

    return {
        "thinking": "",  # GPT-5.2 doesn't expose reasoning trace
        "answer": answer,
        "probability": probability,
        "justification": justification,
        "model_id": "gpt-5.2-pro-2025-12-11",
        "model_settings": "reasoning_effort=high, web_search=true",
        "input_tokens": response.usage.input_tokens if response.usage else 0,
        "output_tokens": response.usage.output_tokens if response.usage else 0,
        "thinking_tokens_approx": reasoning_tokens,
        "elapsed_seconds": round(elapsed, 2),
    }


MODELS = {
    "claude": predict_claude,
    "openai": predict_openai,
}


def process_question(q: dict, predict_fn, print_lock: threading.Lock):
    """Process a single question and return result."""
    qid = q.get("id", "")
    category = q.get("category", "")
    question_text = q["question"]
    context = q.get("context", "")

    # Build full prompt with context if available
    full_prompt = f"# Question\n{question_text}"
    if context:
        full_prompt += f"\n\nContext: {context}"

    with print_lock:
        print(f"[{qid}] Starting: {question_text[:50]}...")

    try:
        result = predict_fn(full_prompt)

        row = {
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
        }

        with print_lock:
            print(f"[{qid}] Done. Probability: {result['probability']} ({result['elapsed_seconds']}s)")

        return qid, True, None, row

    except Exception as e:
        with print_lock:
            print(f"[{qid}] ERROR: {e}")
        return qid, False, str(e), None


def main():
    parser = argparse.ArgumentParser(description="AI Prediction Harness")
    parser.add_argument("questions_file", help="JSON file with questions")
    parser.add_argument("--model", choices=["claude", "openai"], required=True)
    parser.add_argument("--output", default="predictions.csv", help="Output CSV file")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
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

    print_lock = threading.Lock()
    results = []
    errors = []

    if args.workers == 1:
        # Sequential processing
        for q in questions:
            qid, success, error, row = process_question(q, predict_fn, print_lock)
            if success:
                results.append(row)
            else:
                errors.append((qid, error))
    else:
        # Parallel processing
        print(f"Running with {args.workers} parallel workers...")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_question, q, predict_fn, print_lock): q
                for q in questions
            }

            completed = 0
            for future in as_completed(futures):
                qid, success, error, row = future.result()
                completed += 1
                if success:
                    results.append(row)
                else:
                    errors.append((qid, error))
                with print_lock:
                    print(f"Progress: {completed}/{len(questions)}")

    # Sort results by question_id and write to CSV
    results.sort(key=lambda r: r["question_id"])

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    if errors:
        print(f"\n{len(errors)} errors occurred:")
        for qid, error in errors:
            print(f"  [{qid}] {error}")

    print(f"\nResults saved to {args.output} ({len(results)} predictions)")


if __name__ == "__main__":
    main()
