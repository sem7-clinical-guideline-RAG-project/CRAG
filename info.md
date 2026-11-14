# Sufficiency Judge

## Purpose

The sufficiency judge evaluates whether the RAG-generated answer sufficiently answers the user's question and is supported by the retrieved knowledge-base evidence. It does not generate, revise, or repair the answer. The interactive application uses Gemini for this evaluation; Groq generates the RAG answer.

## Information the judge receives

- The user's question.
- The RAG-generated answer.
- The top-1 retrieved passage, including passage text and source name. Retrieval similarity scores are not passed.
- Up to eight reference examples for each label (`0` and `1`), excluding examples with the same question. These examples calibrate the labels; retrieved passages provide evidence for judging the current answer.

## Code locations

- `query.py` — Searches the knowledge base, generates the RAG answer, builds the evidence input, and calls the judge.
- `sufficiency_judge/llm_judge.py` — Loads the reference dataset, sends the prompt to Gemini, and validates the returned judgement.
- `sufficiency_judge/prompt_builder.py` — Formats the reference examples and retrieved passages and defines the judging instructions.
- `sufficiency_judge/dataset_loader.py` — Loads and validates reference examples and excludes the current question.
- `sufficiency_judge/dataset_A.csv` — Reference question-and-answer examples with binary sufficiency labels.

## Requirements and configuration

The judge uses `google-genai`, `python-dotenv`, and `pandas`, listed in `requirements.txt`. Install them with:

```sh
python -m pip install google-genai python-dotenv pandas
```

Set these values in `.env` (or the process environment):

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=your_supported_gemini_model
```

The answer-generation step separately uses `GROQ_API_KEY`; it is not a judge setting.

## How the judge works

### 1. Retrieval and answer generation

In the interactive flow, `query.py` calls `search(question, top_k=3)`. `search()` asks ChromaDB for up to three passages from the `research_abstracts` collection, in retrieval rank order. All returned results are displayed, but only the first (top-1) passage is passed with the question to `generate_rag_response()` for answer generation.

Afterward, `query.py` builds `knowledge_context` from that same top-1 passage and its `source` metadata. It passes this evidence, along with the question and generated answer, to `judge_sufficiency()`. If retrieval returns no passages, answer generation returns a not-found response and the judge receives an empty evidence list. The judge receives passage text and source name; retrieval distances/similarity scores are not passed.

### 2. Reference dataset

If no dataset is explicitly provided to `judge_sufficiency()`, `llm_judge.py` loads `dataset_A.csv`. Its path is resolved relative to `dataset_loader.py`, so loading does not depend on the program's working directory.

The CSV must contain `question`, `answer`, `sufficiency_label`, and `source_file`. The loader drops rows missing a question, answer, or label; requires labels to be numeric `0` or `1`; trims text fields; and removes rows with blank questions or answers. Missing columns, nonnumeric labels, or labels outside `0` and `1` raise an error.

Before building the prompt, the loader excludes examples whose question matches the current question after trimming whitespace and ignoring case. Remaining examples calibrate the label distinction; their answers are not evidence for the current question.

### 3. Prompt and evidence

`prompt_builder.py` places the current question, answer, retrieved passages, and reference examples in the Gemini prompt. For each label, it includes up to eight examples. If a label has more than eight examples, it selects a repeatable sample using random seed `0`.

Retrieved passages are provided as JSON records with a passage number, source, and content. They are the evidence for judging whether the current answer is relevant, complete enough, and grounded. Reference examples are for calibration only. The prompt instructs the judge not to use outside knowledge, invent facts, or treat reference answers as evidence.

If `knowledge_context` is omitted when calling `judge_sufficiency()` directly, the prompt receives an empty retrieved-passages list. The interactive `query.py` flow supplies the retrieved passages.

The binary labels mean:

- **1 — SUFFICIENT:** The answer addresses the user's actual question with enough relevant information and is supported by the supplied evidence.
- **0 — INSUFFICIENT:** The answer misses important information, is only loosely related, does not answer the substantive question, or is not adequately supported by the evidence.

The prompt treats a refusal as insufficient when the user asked for information, even if the evidence genuinely lacks it. A statement that information is absent can be sufficient when the user specifically asked whether the passages contain that information and the answer accurately addresses that question. It also asks the judge to assess answer sufficiency separately from whether the RAG system behaved responsibly.

### 4. Gemini request and response

`llm_judge.py` reads `GEMINI_API_KEY` and `GEMINI_MODEL`, creates a Gemini client, and reuses that client for later judgements in the same Python process. It requests JSON output with `response_mime_type="application/json"` and `temperature=0`.

The response must contain:

- `sufficiency_label`: `0` or `1`;
- `judgement`: `INSUFFICIENT` or `SUFFICIENT`, consistent with the label; and
- `reason`: a brief explanation.

The function returns these as `gemini_label`, `gemini_judgement`, and `gemini_reason`. The label is a binary decision; the code does not calculate a confidence score.

### 5. Display and errors

`query.py` prints the judgement and reason after the answer. It catches errors from the judge and prints `Judge unavailable` with the error message. Errors include missing Gemini configuration, a missing Gemini SDK, empty or invalid JSON output, missing response fields, an invalid label, or disagreement between the label and judgement text. The judge validates the returned structure and label consistency; it does not independently verify the truth of Gemini's reason.

## Judge input and evaluation steps

After the RAG answer is generated, the judge processes it as follows:

1. **Receive the answer and evidence.** `query.py` passes the user's question, Groq's generated answer, and the top-1 retrieved passage with its source name to `judge_sufficiency()`. Retrieval similarity scores are not passed. If no passage was retrieved, the judge receives an empty evidence list.
2. **Load reference examples.** `judge_sufficiency()` loads `dataset_A.csv`, excludes examples whose question matches the current question (ignoring case and surrounding whitespace), then selects up to eight examples for each label. These examples calibrate the labels; they are not evidence for the current question.
3. **Build the evaluation prompt.** `prompt_builder.py` combines the question, answer, top-1 evidence, and sampled examples. The instructions tell Gemini to assess whether the answer addresses the question with enough relevant information and is supported by the retrieved evidence. Gemini is instructed not to use outside knowledge, invent facts, or treat reference answers as evidence for the current question.
4. **Return a judgement.** Gemini returns JSON with `sufficiency_label`, `judgement`, and `reason`. Label `1` means `SUFFICIENT`; label `0` means `INSUFFICIENT`.
5. **Validate and display the result.** `llm_judge.py` checks that the response has the required fields and that the label agrees with the judgement text. `query.py` prints the judgement and reason. If configuration, the SDK, the API call, or response validation fails, it prints `Judge unavailable` and continues the interactive loop.

## Direct use

The judge can be called from Python with an explicit evidence list:

```python
from sufficiency_judge.llm_judge import judge_sufficiency

result = judge_sufficiency(
    question="What does the evidence report?",
    rag_answer="The answer based on the retrieved passages.",
    knowledge_context=[
        {"source": "example.txt", "content": "Retrieved passage text."}
    ],
)
```
An optional `dataset` argument can supply an already loaded dataframe instead of reading the default CSV. The module's `__main__` example omits `knowledge_context`, so that standalone example evaluates without retrieved evidence; callers who want evidence-grounded judgement should pass it explicitly.
