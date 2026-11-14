# Sufficiency Judge

## Code locations
- `query.py:20` imports `judge_sufficiency`; `query.py:254-273` calls it after generating the RAG answer and passes retrieved passages as evidence.
- `sufficiency_judge/llm_judge.py` defines `judge_sufficiency()`, which calls Gemini and validates the result.

## Folder files
- `llm_judge.py` — Gets and validates the sufficiency judgement from Gemini.
- `prompt_builder.py` — Builds the judgement prompt from the question, answer, examples, and evidence.
- `dataset_loader.py` — Loads and validates reference examples, excluding the current question.
- `dataset_A.csv` — Reference question-and-answer examples with sufficiency labels.

## Requirements (judge only)
- Python packages: `google-genai`, `python-dotenv`, `pandas` (already listed in `requirements.txt`).
- Install with: `pip install google-genai python-dotenv pandas`

## `.env` values (judge only)
```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_supported_gemini_model
```

## How the sufficiency judge works
### 1. The RAG system prepares the inputs
In the interactive query flow, `query.py` searches the knowledge base, displays the retrieved results, and calls `generate_rag_response()` to create an answer. After printing that answer, it builds `knowledge_context` from the retrieved passages. Each passage is paired with its source name where available.

`query.py` then calls `judge_sufficiency(question, answer, knowledge_context=knowledge_context)`. The judge evaluates the generated answer; it does not generate or revise that answer.

### 2. The reference dataset is loaded and checked
If a dataset is not explicitly passed to `judge_sufficiency()`, `llm_judge.py` loads `dataset_A.csv`. The default path is resolved relative to `dataset_loader.py`, so it does not depend on the directory from which the program was launched.

The CSV must include `question`, `answer`, `sufficiency_label`, and `source_file` columns. The loader removes rows without a question, answer, or label; checks that labels are numeric and are either `0` or `1`; and trims the text fields. Invalid columns or labels raise an error instead of silently being accepted.

Before using the data as examples, the judge excludes rows whose question matches the current question after trimming whitespace and ignoring letter case. This helps prevent the judge from seeing the reference answer for the exact question it is evaluating. The remaining records are examples for calibrating the label distinction, not evidence for the current answer.

### 3. The prompt combines examples and retrieved evidence
`prompt_builder.py` builds one prompt containing:
- the current question and the RAG-generated answer;
- the retrieved knowledge-base passages, including source names where available; and
- up to eight reference examples for each label (`0` and `1`).

When there are more than eight examples for a label, the builder selects a repeatable sample using a fixed random seed. Reference examples are included to show what the labels mean; the retrieved passages are the evidence used to assess the current answer.

The prompt asks Gemini to make an independent binary decision:
- **1 — SUFFICIENT:** the answer addresses the user's actual question with enough relevant information and is supported by the supplied retrieved passages.
- **0 — INSUFFICIENT:** the answer is missing important information, is only loosely related, or does not answer the substantive question.

The judge is instructed not to treat reference answers as evidence, invent facts, or fill gaps with outside knowledge. If the user asks for information and the answer only says it is unavailable in the passages, that is insufficient. A response about information being unavailable can be sufficient when the user explicitly asks whether the supplied passages contain that information and the answer accurately addresses that question. Responsible caution and answer sufficiency are evaluated separately.

### 4. Gemini evaluates the prompt
`llm_judge.py` reads `GEMINI_API_KEY` and `GEMINI_MODEL` from the environment, creates a Gemini client, and sends the prompt to that model. The client is reused within the process after it is created. The request asks Gemini to return JSON only, with these fields:
- `sufficiency_label`: `1` or `0`;
- `judgement`: `SUFFICIENT` or `INSUFFICIENT`; and
- `reason`: a brief explanation.

The request uses a JSON response format and temperature `0` to encourage consistent, structured evaluations.

### 5. The result is checked and displayed
Before returning a result, `llm_judge.py` checks that Gemini returned text that can be parsed as JSON, that all three required fields are present, that the label is either `0` or `1`, and that the judgement text agrees with the label. An inconsistent or invalid response raises an error rather than being treated as a successful evaluation.

The function returns `gemini_label`, `gemini_judgement`, and `gemini_reason`. `query.py` prints the judgement and reason after the RAG answer. If the API key or model is missing, the Gemini SDK is unavailable, the response is empty or invalid, or another judging error occurs, the query loop catches the error and prints `Judge unavailable` with the error message.