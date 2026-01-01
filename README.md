# AI Predictions Harness

Collects probability predictions from AI models for a forecasting contest. Uses extended thinking/reasoning and web search for maximum accuracy.

## Models

| Model | Model ID | Thinking | Web Search |
|-------|----------|----------|------------|
| Claude Opus 4.5 | `claude-opus-4-5-20251101` | 32k token budget | Up to 20 searches |
| GPT-5.2 Pro | `gpt-5.2-pro-2025-12-11` | reasoning_effort=high | Enabled |

## Setup

```bash
pip install anthropic openai

# Copy and fill in your API keys
cp secrets_example.py secrets.py
```

## Usage

```bash
# Run with Claude
python predict.py questions.json --model claude --output claude_predictions.csv

# Run with OpenAI
python predict.py questions.json --model openai --output openai_predictions.csv
```

## Questions Format

See `questions_example.json`. Copy to `questions.json` and add your questions:

```json
[
  {
    "id": 1,
    "category": "Sports",
    "question": "49ers win the Super Bowl",
    "context": "Optional background info or reference links"
  }
]
```

Fields:
- `id` (required): Question number
- `question` (required): The prediction question
- `category` (optional): For organization
- `context` (optional): Background info passed to the model

## Output CSV

| Column | Description |
|--------|-------------|
| question_id | Question number |
| category | Question category |
| question | The question text |
| probability | Extracted probability (0-1) |
| justification | Model's reasoning summary |
| model_id | Exact model version |
| model_settings | Thinking/reasoning config |
| input_tokens | Prompt tokens |
| output_tokens | Response tokens |
| thinking_tokens_approx | Thinking/reasoning tokens |
| elapsed_seconds | API call duration |
| timestamp | When prediction was made |
| raw_answer | Full model response |
| thinking | Full thinking trace (Claude only) |
