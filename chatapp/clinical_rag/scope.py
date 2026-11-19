"""
Django-specific scope detection integration.

This module reuses the existing trained PubMedBERT ScopeDetector from
classification/scope_detection/train_scope.py and provides a thread-safe
singleton wrapper for use in the Django RAG pipeline.
"""

import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from . import config

logger = logging.getLogger(__name__)


# ============================================================
# Data Classes (mirroring train_scope.py ScopeDetector output)
# ============================================================

class ScopeDecision:
    """Scope classification outcomes."""
    IN_SCOPE = "IN_SCOPE"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"

    _VALUES = {IN_SCOPE, OUT_OF_SCOPE, NEEDS_CLARIFICATION}

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value in cls._VALUES


@dataclass
class ScopePrediction:
    """Result of scope detection prediction (matches train_scope.py output)."""
    label_id: int
    label_name: str
    probabilities: dict[str, float]
    routing: str  # From ScopeDetector._get_routing()


@dataclass
class ScopeQueryResponse:
    """
    Unified response from the scope detection gate.

    For IN_SCOPE: decision=IN_SCOPE, message=None
    For OUT_OF_SCOPE/NEEDS_CLARIFICATION: decision=..., message=formal response
    For ERROR: decision=None, message=error details
    """
    decision: Optional[str]
    message: Optional[str] = None
    scope_prediction: Optional[ScopePrediction] = None
    error: Optional[str] = None


# ============================================================
# Formal Messages (reused from scope_query.py)
# ============================================================

OUT_OF_SCOPE_MESSAGE = (
    "I cannot process your question because it falls outside the scope of this "
    "clinical research question-answering system.\n\n"
    "This system is specifically designed to assist with questions related to "
    "clinical and biomedical research literature, including topics such as "
    "diabetes in children and adolescents, pediatric endocrinology, clinical "
    "trial methodology, and related medical research domains.\n\n"
    "Your question appears to be unrelated to these areas. Please consider "
    "reformulating your question to focus on a clinical or biomedical research "
    "topic, or submit a question that falls within the system's area of expertise.\n\n"
    "If you believe this classification is in error, you may try rephrasing "
    "your question with more specific clinical or research context."
)

NEEDS_CLARIFICATION_MESSAGE = (
    "I cannot reliably process your question because it is ambiguous, incomplete, "
    "or insufficiently specific for clinical research retrieval.\n\n"
    "To provide accurate answers from the biomedical literature, the system requires "
    "questions that are clearly formulated with sufficient context. Questions that are "
    "too vague, lack a defined population/intervention/outcome, or combine multiple "
    "distinct inquiries cannot be reliably matched to relevant research passages.\n\n"
    "Please reformulate and resubmit your question with the following guidance:\n"
    "  • Specify the clinical population of interest (e.g., 'adolescents with type 1 diabetes')\n"
    "  • Define the intervention, exposure, or comparison being investigated\n"
    "  • State the outcome or research objective clearly\n"
    "  • Avoid overly broad or multi-part questions; submit one focused question at a time\n"
    "  • Include relevant clinical context (e.g., treatment setting, disease stage, age range)\n\n"
    "Example of a well-formulated question:\n"
    "  'What is the effect of continuous glucose monitoring on HbA1c levels in adolescents "
    "aged 13–18 with type 1 diabetes compared to self-monitoring of blood glucose?'\n\n"
    "Once you provide a more specific and complete question, the system will be able to "
    "search the clinical literature and retrieve relevant evidence."
)

SCOPE_DETECTION_ERROR_MESSAGE = (
    "An error occurred during scope detection. The system cannot safely classify "
    "your question, so it must abstain from processing.\n\n"
    "This is a protective measure to ensure that questions are properly routed. "
    "Please try again, or contact the system administrator if the problem persists."
)


# ============================================================
# ScopeDetector Wrapper (thread-safe singleton reusing train_scope.py)
# ============================================================

class _ScopeDetectorSingleton:
    """Thread-safe lazy singleton for PubMedBERT scope detection model.

    Reuses the existing ScopeDetector from classification.scope_detection.train_scope
    """

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
        """Load the ScopeDetector from the existing implementation."""
        model_path = config.SCOPE_MODEL_PATH
        if not model_path.exists():
            raise RuntimeError(
                f"Scope detection model not found at {model_path}. "
                f"Ensure the model has been trained and saved to this location."
            )

        logger.info(f"[ScopeDetector] Loading model from {model_path}...")

        # Import the existing ScopeDetector from train_scope.py
        import sys
        # Ensure classification/scope_detection is on path
        scope_detection_path = config.PROJECT_ROOT / "classification" / "scope_detection"
        sys.path.insert(0, str(scope_detection_path))

        from train_scope import ScopeDetector as TrainScopeDetector

        self.detector = TrainScopeDetector(
            model_path=str(model_path),
            max_length=config.SCOPE_MAX_LENGTH,
        )

        logger.info(f"[ScopeDetector] Model loaded successfully")
        logger.info(f"[ScopeDetector] Max length: {config.SCOPE_MAX_LENGTH}")
        logger.info(f"[ScopeDetector] Label map: {self.detector.predict('test')['probabilities'].keys()}")

    def predict(self, question: str) -> ScopePrediction:
        """
        Predict scope class for a single question using the existing ScopeDetector.

        Returns:
            ScopePrediction matching train_scope.py output structure
        """
        result = self.detector.predict(question)

        return ScopePrediction(
            label_id=result["label_id"],
            label_name=result["label_name"],
            probabilities=result["probabilities"],
            routing=result["routing"],
        )


def get_scope_detector() -> _ScopeDetectorSingleton:
    """Get the ScopeDetector singleton (loads on first call)."""
    return _ScopeDetectorSingleton()


# ============================================================
# Main Pipeline Function
# ============================================================

def classify_scope(question: str) -> ScopeQueryResponse:
    """
    Classify a user question using the scope detection model.

    This is the main entry point for the Django pipeline.

    Args:
        question: The raw user question (before condensation).

    Returns:
        ScopeQueryResponse with decision, message (for non-IN_SCOPE),
        and scope_prediction details.
    """
    if not question or not question.strip():
        return ScopeQueryResponse(
            decision=None,
            error="Empty question provided.",
        )

    question = question.strip()

    if not config.SCOPE_DETECTION_ENABLED:
        logger.warning("Scope detection disabled; failing closed")
        return ScopeQueryResponse(
            decision=None,
            message=SCOPE_DETECTION_ERROR_MESSAGE,
            scope_prediction=None,
            error="Scope detection gate is disabled",
        )

    # Step 1: Scope Detection
    try:
        detector = get_scope_detector()
        scope_prediction = detector.predict(question)
    except Exception as e:
        # Fail-safe: do NOT treat detection failure as IN_SCOPE
        logger.exception("Scope detection failed")
        return ScopeQueryResponse(
            decision=None,
            scope_prediction=None,
            message=SCOPE_DETECTION_ERROR_MESSAGE,
            error=f"Scope detection failed: {e}",
        )

    decision = scope_prediction.label_name

    # Step 2: Route based on decision
    if decision == ScopeDecision.OUT_OF_SCOPE:
        return ScopeQueryResponse(
            decision=decision,
            message=OUT_OF_SCOPE_MESSAGE,
            scope_prediction=scope_prediction,
        )

    elif decision == ScopeDecision.NEEDS_CLARIFICATION:
        return ScopeQueryResponse(
            decision=decision,
            message=NEEDS_CLARIFICATION_MESSAGE,
            scope_prediction=scope_prediction,
        )

    elif decision == ScopeDecision.IN_SCOPE:
        return ScopeQueryResponse(
            decision=decision,
            message=None,
            scope_prediction=scope_prediction,
        )

    else:
        # Unexpected label - fail safely
        logger.error(f"Unexpected scope label: {decision}")
        return ScopeQueryResponse(
            decision=None,
            scope_prediction=scope_prediction,
            message=SCOPE_DETECTION_ERROR_MESSAGE,
            error=f"Unexpected scope label: {decision}",
        )