"""The whole RAG flow for one chat question, as a stream of small events."""

import logging

from . import config, generation, retrieval, scope
from .sufficiency import (
    NoChecksConfiguredError,
    PartialSufficiencyResult,
    SufficiencyEngine,
    SufficiencyResult,
    calculate_combined_score,
    calculate_fallback_score,
)
from .sufficiency_checks import (
    CosineSimilarityError,
    ModelCCheckpointError,
    ModelCInferenceError,
)

logger = logging.getLogger(__name__)


def build_sources(hits):
    """What the chat shows under an answer: which abstract and chunk each passage came from."""
    sources = []
    for hit in hits:
        meta = hit["metadata"]
        filename = meta.get("source", "Unknown")
        stem = filename[:-4] if filename.lower().endswith(".txt") else filename
        chunk = meta.get("chunk_index")

        parts = []
        if isinstance(chunk, int):
            parts.append(f"Chunk {chunk + 1}")
        parts.append(f"similarity {hit['similarity']:.2f}")

        sources.append({
            "source": filename,
            "title": stem,
            "chunk": chunk,
            "meta": " · ".join(parts),
            "score": round(hit["similarity"], 3),
        })
    return sources


def _build_sufficiency_details(
    model_c_prob: float,
    cosine_sim: float,
    llm_judge_score: float | None,
    model_c_details: dict,
    cosine_details: dict,
    llm_judge_details: dict,
    judge_status: str,
    sufficient: bool,
    extra_check_details: dict | None = None,
) -> dict:
    """Build the sufficiency details dict for events."""
    if llm_judge_score is not None:
        combined_score = calculate_combined_score(model_c_prob, cosine_sim, llm_judge_score)
        weights = {
            "model_c": config.MODEL_C_WEIGHT,
            "cosine": config.COSINE_WEIGHT,
            "llm_judge": config.LLM_JUDGE_WEIGHT,
        }
        contributions = {
            "model_c": round(config.MODEL_C_WEIGHT * model_c_prob, 4),
            "cosine": round(config.COSINE_WEIGHT * cosine_sim, 4),
            "llm_judge": round(config.LLM_JUDGE_WEIGHT * llm_judge_score, 4),
        }
    else:
        combined_score = calculate_fallback_score(model_c_prob, cosine_sim)
        fallback_mc_weight = config.MODEL_C_WEIGHT / (config.MODEL_C_WEIGHT + config.COSINE_WEIGHT)
        fallback_cos_weight = config.COSINE_WEIGHT / (config.MODEL_C_WEIGHT + config.COSINE_WEIGHT)
        weights = {
            "model_c": fallback_mc_weight,
            "cosine": fallback_cos_weight,
            "llm_judge": 0.0,
        }
        contributions = {
            "model_c": round(fallback_mc_weight * model_c_prob, 4),
            "cosine": round(fallback_cos_weight * cosine_sim, 4),
            "llm_judge": 0.0,
        }

    check_details = {
        "model_c": model_c_details,
        "cosine": cosine_details,
        "llm_judge": llm_judge_details,
        "weights": weights,
        "contributions": contributions,
        "judge_status": judge_status,
    }
    if extra_check_details:
        check_details.update(extra_check_details)

    return {
        "combined_score": combined_score,
        "threshold": config.COMBINED_THRESHOLD,
        "model_c_probability": model_c_prob,
        "cosine_similarity": cosine_sim,
        "llm_judge_score": llm_judge_score,
        "check_details": check_details,
        "weights": weights,
        "contributions": contributions,
        "sufficient": sufficient,
    }


def stream_rag_answer(question, history=None):
    """
    Generator of events:
      {"type": "scope", "decision": "...", "label_id": int, "probabilities": {...}, "message": "..."}
      {"type": "query", "text": "..."}                         the question that was searched
      {"type": "sources", "sources": [...], "confidence": f}   the retrieved abstracts
      {"type": "sufficiency", "details": {...}}                evaluation results (emitted twice: partial then final)
      {"type": "token", "text": "..."}                         (many) the answer
      {"type": "abstain", "reason": "...", "details": {...}}   evidence insufficient
      {"type": "error", "message": "..."}                      operational error
    """
    # Step 1: Scope detection on the RAW user question (before condensation)
    scope_response = scope.classify_scope(question)

    # Emit scope decision event
    pred = scope_response.scope_prediction
    scope_event = {
        "type": "scope",
        "decision": scope_response.decision,
        "label_id": pred.label_id if pred else None,
        "probabilities": pred.probabilities if pred else {},
        "message": scope_response.message,
    }
    yield scope_event

    # Step 2: Route based on scope decision
    if scope_response.decision == scope.ScopeDecision.OUT_OF_SCOPE:
        # Out of scope: emit abstain with formal message and stop
        if scope_response.message:
            yield {
                "type": "abstain",
                "reason": scope_response.message,
                "details": {
                    "scope_decision": scope_response.decision,
                    "scope_probabilities": pred.probabilities if pred else {},
                },
            }
        return

    if scope_response.decision == scope.ScopeDecision.NEEDS_CLARIFICATION:
        # Needs clarification: emit abstain with formal message and stop
        if scope_response.message:
            yield {
                "type": "abstain",
                "reason": scope_response.message,
                "details": {
                    "scope_decision": scope_response.decision,
                    "scope_probabilities": pred.probabilities if pred else {},
                },
            }
        return

    if scope_response.decision != scope.ScopeDecision.IN_SCOPE:
        # Unexpected decision or error: fail-safe abstain
        error_msg = scope_response.message or scope.SCOPE_DETECTION_ERROR_MESSAGE
        yield {
            "type": "abstain",
            "reason": error_msg,
            "details": {
                "scope_decision": scope_response.decision,
                "scope_probabilities": pred.probabilities if pred else {},
                "scope_error": scope_response.error,
            },
        }
        return

    # Step 3: IN_SCOPE - continue with existing pipeline
    search_question = generation.condense_question(question, history or [])
    yield {"type": "query", "text": search_question}

    hits = retrieval.search(search_question, top_k=config.TOP_K)

    if not hits:
        yield {"type": "sources", "sources": [], "confidence": 0.0}
        # Emit sufficiency event for empty retrieval
        yield {
            "type": "sufficiency",
            "details": {
                "combined_score": 0.0,
                "threshold": config.COMBINED_THRESHOLD,
                "model_c_probability": 0.0,
                "cosine_similarity": 0.0,
                "llm_judge_score": None,
                "check_details": {"empty_retrieval": True},
                "weights": {"model_c": config.MODEL_C_WEIGHT, "cosine": config.COSINE_WEIGHT, "llm_judge": config.LLM_JUDGE_WEIGHT},
                "contributions": {"model_c": 0.0, "cosine": 0.0, "llm_judge": 0.0},
                "sufficient": False,
            },
        }
        # No passages retrieved: abstain with explanation
        yield {
            "type": "abstain",
            "reason": "No relevant passages found in the knowledge base.",
            "details": {
                "combined_score": 0.0,
                "threshold": config.COMBINED_THRESHOLD,
                "model_c_probability": 0.0,
                "cosine_similarity": 0.0,
                "llm_judge_score": None,
                "check_details": {"empty_retrieval": True},
            },
        }
        return

    # Run Model C + Cosine first, start LLM judge in background
    try:
        engine = SufficiencyEngine()
        partial: PartialSufficiencyResult = engine.evaluate_partial(search_question, hits)
    except (ModelCInferenceError, ModelCCheckpointError, CosineSimilarityError, NoChecksConfiguredError) as e:
        # Fail-closed: operational errors yield error event and stop
        logger.warning(f"Sufficiency check failed: {e}")
        yield {
            "type": "error",
            "message": f"Sufficiency evaluation failed: {e}",
        }
        return
    except Exception as e:
        logger.exception("Sufficiency evaluation failed")
        yield {
            "type": "error",
            "message": f"Sufficiency evaluation failed: {e}",
        }
        return

    model_c_prob = partial.model_c_probability
    cosine_sim = partial.cosine_similarity

    # Emit sources
    yield {
        "type": "sources",
        "sources": build_sources(hits),
        "confidence": max(hit["similarity"] for hit in hits),
    }

    # If gate is disabled, emit single sufficiency event and proceed
    if not partial.llm_judge_enabled:
        final_details = _build_sufficiency_details(
            model_c_prob=model_c_prob,
            cosine_sim=cosine_sim,
            llm_judge_score=1.0,
            model_c_details=partial.model_c_details,
            cosine_details=partial.cosine_details,
            llm_judge_details={},
            judge_status="disabled",
            sufficient=True,
            extra_check_details={"disabled": True},
        )
        yield {"type": "sufficiency", "details": final_details}

        # Set variables for the sufficient path below
        sufficient = True
        combined_score = final_details["combined_score"]
        llm_judge_score = final_details["llm_judge_score"]
        judge_status = "disabled"

    else:
        # Emit PARTIAL sufficiency event immediately (Model C + Cosine only, judge pending)
        partial_details = _build_sufficiency_details(
            model_c_prob=model_c_prob,
            cosine_sim=cosine_sim,
            llm_judge_score=None,
            model_c_details=partial.model_c_details,
            cosine_details=partial.cosine_details,
            llm_judge_details={"status": "pending", "model": partial.judge_model},
            judge_status="pending",
            sufficient=False,  # Not final yet
        )
        yield {"type": "sufficiency", "details": partial_details}

        # Wait for LLM judge to complete and get final result
        result: SufficiencyResult = engine.finalize(partial)

        llm_judge_score = result.llm_judge_score
        judge_status = result.check_details.get("judge_status", "disabled")
        combined_score = result.combined_score
        sufficient = result.sufficient

        # Emit FINAL sufficiency event
        final_details = _build_sufficiency_details(
            model_c_prob=model_c_prob,
            cosine_sim=cosine_sim,
            llm_judge_score=llm_judge_score,
            model_c_details=partial.model_c_details,
            cosine_details=partial.cosine_details,
            llm_judge_details=result.check_details.get("llm_judge", {}),
            judge_status=judge_status,
            sufficient=sufficient,
        )
        yield {"type": "sufficiency", "details": final_details}

    if not sufficient:
        # Evidence insufficient: yield abstain event and stop
        reason = (
            "The retrieved passages were not considered sufficient to answer the question. "
            "The combined sufficiency score is below the required threshold."
        )
        if judge_status in ("timeout", "error"):
            reason += f" (LLM Judge: {judge_status})"

        yield {
            "type": "abstain",
            "reason": reason,
            "details": {
                "combined_score": combined_score,
                "threshold": config.COMBINED_THRESHOLD,
                "model_c_probability": model_c_prob,
                "cosine_similarity": cosine_sim,
                "llm_judge_score": llm_judge_score,
                "check_details": final_details["check_details"],
            },
        }
        return

    # Evidence sufficient: proceed with generation
    for text in generation.stream_answer(search_question, hits):
        yield {"type": "token", "text": text}