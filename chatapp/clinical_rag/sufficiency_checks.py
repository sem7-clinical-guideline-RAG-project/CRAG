"""
Sufficiency checks for the RAG pipeline.

Each check evaluates a specific aspect of evidence sufficiency and returns
a standardized result. Checks are designed to be pluggable and independent.
"""

import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import config

logger = logging.getLogger(__name__)


def _get_formatter():
    """Lazy load the formatter to avoid torch import chain at module load time."""
    import sys
    sys.path.insert(0, str(config.PROJECT_ROOT))
    from classification.common.data_loader import (
        format_dataset_b_input,
        format_multi_passage_input,
        LABEL_MAP,
    )
    return format_multi_passage_input, LABEL_MAP


def _get_torch_and_transformers():
    """Lazy load torch and transformers to avoid Windows DLL issues at import time."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    return torch, AutoModelForSequenceClassification, AutoTokenizer


# ============================================================
# Exceptions
# ============================================================

class SufficiencyError(Exception):
    """Base exception for sufficiency check failures."""
    pass


class ModelCCheckpointError(SufficiencyError):
    """Model C checkpoint or tokenizer unavailable."""
    pass


class ModelCInferenceError(SufficiencyError):
    """Model C inference failed."""
    pass


class ModelCOutputError(SufficiencyError):
    """Model C produced invalid output."""
    pass


class CosineSimilarityError(SufficiencyError):
    """Cosine similarity unavailable or invalid."""
    pass


# ============================================================
# Result Types
# ============================================================

@dataclass
class CheckResult:
    """Result from a single sufficiency check."""
    name: str
    score: float
    details: dict[str, Any]

    def __post_init__(self):
        if not isinstance(self.score, (int, float)):
            raise ValueError(f"Score must be numeric, got {type(self.score)}")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"Score must be in [0, 1], got {self.score}")


# ============================================================
# Model C Check
# ============================================================

class _ModelCSingleton:
    """Thread-safe lazy singleton for Model C model and tokenizer."""
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._load_model()
                    self._initialized = True

    def _load_model(self):
        checkpoint = config.MODEL_C_CHECKPOINT
        if not checkpoint.exists():
            raise ModelCCheckpointError(
                f"Model C checkpoint not found at {checkpoint}. "
                f"Train Model C first (python classification/model_c/train.py)."
            )

        logger.info(f"Loading Model C from {checkpoint}...")

        torch, AutoModelForSequenceClassification, AutoTokenizer = _get_torch_and_transformers()

        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        self.max_length = config.MODEL_C_MAX_LENGTH

        logger.info(f"  Model C loaded on {self.device}")
        logger.info(f"  Max length: {self.max_length}")
        _, LABEL_MAP = _get_formatter()
        logger.info(f"  Label mapping: {LABEL_MAP}")

    def predict_proba(self, question: str, passages: list[str] | str) -> float:
        """
        Return P(sufficient) for the given question and retrieved passages.

        Uses format_multi_passage_input matching Model C V2's training data.
        Accepts a list of 1 to 3 passage strings, or a single passage string.
        """
        if isinstance(passages, str):
            passages = [passages]

        # Filter out empty or whitespace-only passages while preserving retrieval order
        valid_passages = [p.strip() for p in passages if p and p.strip()]

        # Lazy load formatter
        format_multi_passage_input, LABEL_MAP = _get_formatter()

        # Format input matching Model C V2 training
        formatted_text = format_multi_passage_input(question, valid_passages, max_length=self.max_length)

        inputs = self.tokenizer(
            formatted_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        torch, _, _ = _get_torch_and_transformers()

        with torch.inference_mode():
            outputs = self.model(**inputs)

        logits = outputs.logits.cpu()
        probs = torch.softmax(logits, dim=-1)[0].numpy()

        # Label 1 = sufficient (per LABEL_MAP)
        prob_sufficient = float(probs[1])

        if not (0.0 <= prob_sufficient <= 1.0):
            raise ModelCOutputError(f"Invalid probability: {prob_sufficient}")

        return prob_sufficient


def get_model_c() -> _ModelCSingleton:
    """Get the Model C singleton (loads on first call)."""
    return _ModelCSingleton()


class ModelCCheck:
    """Evidence sufficiency check using Model C classifier across retrieved passages."""

    name = "model_c"

    def evaluate(self, question: str, hits: list[dict]) -> CheckResult:
        """
        Evaluate evidence sufficiency using Model C across retrieved passages.

        Args:
            question: The user's question.
            hits: Retrieved passages from ChromaDB (must have at least 1 hit).

        Returns:
            CheckResult with Model C's P(sufficient) as score.
        """
        if not hits:
            raise ModelCInferenceError("No passages retrieved; cannot evaluate Model C")

        # Extract contents of up to 3 retrieved passages in original retrieval order
        passages = [h.get("content", "").strip() for h in hits[:3]]
        valid_passages = [p for p in passages if p]
        if not valid_passages:
            raise ModelCInferenceError("All retrieved passages are empty")

        try:
            model_c = get_model_c()
            prob_sufficient = model_c.predict_proba(question, valid_passages)
        except ModelCCheckpointError:
            raise
        except ModelCOutputError:
            raise
        except Exception as e:
            logger.exception("Model C inference failed")
            raise ModelCInferenceError(f"Model C inference failed: {e}") from e

        # Validate probability before creating CheckResult (fail-closed)
        if not (0.0 <= prob_sufficient <= 1.0):
            raise ModelCOutputError(f"Invalid probability from Model C: {prob_sufficient}")

        return CheckResult(
            name=self.name,
            score=prob_sufficient,
            details={
                "probability_sufficient": round(prob_sufficient, 4),
                "probability_insufficient": round(1.0 - prob_sufficient, 4),
                "passages_used": len(valid_passages),
                "passage_used": len(valid_passages),  # Kept for backward compatibility
                "max_length": config.MODEL_C_MAX_LENGTH,
            }
        )


# ============================================================
# Cosine Similarity Check (Rank-Weighted)
# ============================================================

class CosineCheck:
    """Evidence sufficiency check using rank-weighted cosine similarity across retrieved passages."""

    name = "cosine"
    RANK_WEIGHTS = [1.0, 0.5, 0.25]

    def evaluate(self, question: str, hits: list[dict]) -> CheckResult:
        """
        Evaluate evidence sufficiency using rank-weighted cosine similarity of retrieved passages.

        Formula:
            C = sum(w_i * P_i) / sum(w_i)
        where:
            w = [1.0, 0.5, 0.25] for top-3 hits
            P_i = clamped cosine similarity (1 - distance) in [0.0, 1.0]

        Args:
            question: The user's question (unused, kept for interface consistency).
            hits: Retrieved passages from ChromaDB (must have at least 1 hit).

        Returns:
            CheckResult with weighted cosine similarity as score.
        """
        if not hits:
            raise CosineSimilarityError("No passages retrieved; cannot evaluate cosine similarity")

        active_hits = hits[:3]
        raw_similarities = []
        clamped_sims = []

        for idx, hit in enumerate(active_hits, start=1):
            raw_sim = hit.get("similarity")
            if raw_sim is None and "distance" in hit:
                # Fallback to computing 1 - distance if similarity field missing
                try:
                    raw_sim = 1.0 - float(hit["distance"])
                except (ValueError, TypeError):
                    raise CosineSimilarityError(f"Passage {idx} distance must be numeric")

            if raw_sim is None:
                raise CosineSimilarityError(f"Passage {idx} missing 'similarity' field")

            if not isinstance(raw_sim, (int, float)):
                raise CosineSimilarityError(f"Similarity must be numeric, got {type(raw_sim)}")

            clamped = max(0.0, min(1.0, float(raw_sim)))
            raw_similarities.append(float(raw_sim))
            clamped_sims.append(clamped)

        weights = self.RANK_WEIGHTS[:len(clamped_sims)]
        weight_sum = sum(weights)

        if weight_sum <= 0:
            weighted_similarity = 0.0
        else:
            weighted_similarity = sum(s * w for s, w in zip(clamped_sims, weights)) / weight_sum

        weighted_similarity = max(0.0, min(1.0, weighted_similarity))
        final_score = round(weighted_similarity, 4)

        return CheckResult(
            name=self.name,
            score=final_score,
            details={
                "similarity": final_score,
                "raw_similarity": round(raw_similarities[0], 4),  # Passage 1 raw similarity
                "passage_similarities": [round(s, 4) for s in clamped_sims],
                "raw_similarities": [round(s, 4) for s in raw_similarities],
                "weights": weights,
                "weight_sum": round(weight_sum, 4),
                "passages_used": len(clamped_sims),
                "passage_used": len(clamped_sims),  # Kept for backward compatibility
                "formula": f"sum(w_i * P_i) / {round(weight_sum, 4)}",
            }
        )


# ============================================================
# LLM Judge Check (Gemini and Groq backends)
# ============================================================


class LLMJudgeError(SufficiencyError):
    """LLM Judge check failed."""
    pass


def _build_judge_prompt(question: str, passages: list[str] | str) -> str:
    """
    Build prompt for LLM evidence sufficiency evaluation across retrieved passages.

    Accepts 1 to 3 passages in their original retrieval order and instructs the model
    to evaluate whether the retrieved passages collectively contain the specific
    information required to answer the user's question.
    """
    if isinstance(passages, str):
        passages = [passages]

    # Filter out empty or whitespace-only passages while preserving retrieval order
    valid_passages = [p.strip() for p in passages if p and p.strip()]

    if not valid_passages:
        raise LLMJudgeError("No non-empty passages provided for LLM judge evaluation")

    passages_text_blocks = []
    for idx, p in enumerate(valid_passages, start=1):
        passages_text_blocks.append(f"--- Retrieved Passage {idx} ---\n{p}")
    passages_formatted = "\n\n".join(passages_text_blocks)

    return f"""You are an evidence sufficiency evaluator for a pediatric diabetes clinical RAG system.

TASK:
Evaluate whether the retrieved evidence passages COLLECTIVELY contain the specific information required to answer the user's question. Rate sufficiency on a scale from 0.0 to 1.0.

QUESTION:
{question}

RETRIEVED EVIDENCE PASSAGES:
{passages_formatted}

EVALUATION PRINCIPLES:
1. SPECIFIC INFORMATION OVER GENERAL TOPIC:
   - Determine whether the passages contain the exact clinical facts, values, or recommendations requested.
   - Do NOT give a high score merely because passages discuss the correct medical condition (e.g. diabetes, DKA, or insulin) if they do not provide the specific answer asked for.
2. COLLECTIVE & COMPLEMENTARY EVIDENCE:
   - Evaluate all passages together as a combined body of evidence.
   - Separate passages may provide complementary parts of the complete answer (e.g., Passage 1 defines criteria while Passage 2 specifies pediatric dosing or thresholds). If the pieces together answer the question, rate sufficiency high.
3. IRRELEVANT PASSAGES:
   - If one or two passages are irrelevant or off-topic, ignore them as long as the remaining passage(s) contain the necessary evidence. Do NOT penalize merely because an unhelpful passage was retrieved alongside sufficient ones.
4. DO NOT ASSUME OR GUESS:
   - Judge answerability strictly from the provided passages, NOT from your external medical knowledge.
   - If answering would require guessing, extrapolation, or external medical facts, rate sufficiency low.
5. CONTRADICTIONS:
   - If passages present conflicting clinical values, diagnostic thresholds, or contradictory treatment recommendations that prevent a clear, safe answer, lower the sufficiency score and note the conflict in the reason.
6. NO AUTOMATIC SUFFICIENCY:
   - Do NOT assume that retrieving three topical passages automatically means evidence is sufficient. Focus on whether the specific question is genuinely answered.

SCORING CRITERIA:
- 1.0: Passages collectively contain all specific facts needed to answer the question directly and completely without guessing.
- 0.7-0.9: Passages answer the core clinical question, but minor secondary details are missing.
- 0.4-0.6: Passages are partially relevant but miss crucial information needed for a complete answer.
- 0.1-0.3: Passages discuss the general medical topic but lack the specific information requested.
- 0.0: Passages are completely irrelevant, entirely lack the requested information, or contain irreconcilable contradictions.

Use the full range [0.0 to 1.0]. Do not default to 0 or 1 unless clearly warranted.

Return JSON only:
{{
    "sufficiency_score": <number between 0.0 and 1.0>,
    "reason": "<Brief explanation referencing the specific evidence or gaps across passages>"
}}"""

def _parse_judge_response(text: str) -> tuple[float, str]:
    if not text:
        raise LLMJudgeError("Judge returned empty response")

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        result = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        raise LLMJudgeError(f"Judge response is not valid JSON: {e}") from e

    if not isinstance(result, dict):
        raise LLMJudgeError("Judge response is not a JSON object")

    score = result.get("sufficiency_score")
    reason = result.get("reason", "")

    if score is None:
        raise LLMJudgeError("Missing 'sufficiency_score' in response")

    try:
        score = float(score)
    except (ValueError, TypeError):
        raise LLMJudgeError(f"Invalid score type: {type(score)}")

    if not (0.0 <= score <= 1.0):
        raise LLMJudgeError(f"Score must be in [0, 1], got: {score}")

    return score, reason

class _GeminiJudge:
    """Gemini-backed judge implementation."""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model = config.LLM_JUDGE_MODEL
        self.timeout = config.LLM_JUDGE_TIMEOUT_SECONDS
        self._client = None
        self._client_lock = threading.Lock()

    def _get_client(self):
        """Lazy-load Gemini client."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    from google import genai
                    if not self.api_key:
                        raise LLMJudgeError("GEMINI_API_KEY not set")
                    self._client = genai.Client(api_key=self.api_key)
        return self._client

    def evaluate(self, question: str, passages: list[str] | str) -> tuple[float, str]:
        """Call Gemini API and parse response."""
        from google.genai import types

        client = self._get_client()
        prompt = _build_judge_prompt(question, passages)

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0,
            ),
        )

        if not response.text:
            raise LLMJudgeError("Gemini returned empty response")

        return _parse_judge_response(response.text)


class _GroqJudge:
    """Groq-backed judge implementation."""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = config.LLM_JUDGE_GROQ_MODEL
        self.url = config.GROQ_URL
        self.timeout = config.LLM_JUDGE_TIMEOUT_SECONDS
        self._client = None
        self._client_lock = threading.Lock()

    def _get_client(self):
        """Lazy-load Groq client."""
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    from groq import Groq
                    if not self.api_key:
                        raise LLMJudgeError("GROQ_API_KEY not set")
                    self._client = Groq(api_key=self.api_key, timeout=self.timeout)
        return self._client

    def evaluate(self, question: str, passages: list[str] | str) -> tuple[float, str]:
        """Call Groq API and parse response."""
        import httpx

        client = self._get_client()

        prompt = _build_judge_prompt(question, passages)

        # Use the Groq client (OpenAI-compatible)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        if not response.choices or not response.choices[0].message.content:
            raise LLMJudgeError("Groq returned empty response")

        return _parse_judge_response(response.choices[0].message.content)


class LLMJudgeCheck:
    """Evidence sufficiency check using LLM (Gemini or Groq) as judge on question + retrieved passages."""

    name = "llm_judge"

    def __init__(self):
        self.provider = config.LLM_JUDGE_PROVIDER
        if self.provider == "gemini":
            self._backend = _GeminiJudge()
        elif self.provider == "groq":
            self._backend = _GroqJudge()
        else:
            raise LLMJudgeError(f"Unknown LLM_JUDGE_PROVIDER: {self.provider}")

    def evaluate(self, question: str, hits: list[dict]) -> CheckResult:
        """
        Evaluate evidence sufficiency across retrieved passages using LLM judge.

        Args:
            question: The user's question.
            hits: Retrieved passages from ChromaDB (must have at least 1 hit).

        Returns:
            CheckResult with sufficiency score (0.0 to 1.0) and details.
        """
        if not hits:
            raise LLMJudgeError("No passages retrieved; cannot evaluate LLM judge")

        # Extract contents of up to 3 retrieved passages in original order
        passages = [h.get("content", "").strip() for h in hits[:3]]
        valid_passages = [p for p in passages if p]
        if not valid_passages:
            raise LLMJudgeError("All retrieved passages are empty")

        try:
            score, reason = self._backend.evaluate(question, valid_passages)
            judge_status = "done"
        except LLMJudgeError:
            raise
        except Exception as e:
            # Check for 429 quota exceeded
            error_text = str(e)
            if "429" in error_text and ("quota" in error_text.lower() or "resource_exhausted" in error_text.lower() or "rate limit" in error_text.lower()):
                raise LLMJudgeError(f"quota_exceeded: {error_text}") from e
            # Re-raise with the actual error message for better debugging
            raise LLMJudgeError(f"{type(e).__name__}: {error_text}") from e

        score = float(score)

        return CheckResult(
            name=self.name,
            score=score,
            details={
                "label": 1 if score >= config.COMBINED_THRESHOLD else 0,
                "sufficiency_score": score,
                "reason": reason,
                "model": self._backend.model,
                "provider": self.provider,
                "passages_used": len(valid_passages),
                "passage_used": len(valid_passages),  # Kept for backward compatibility
                "status": judge_status,
            }
        )