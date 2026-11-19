"""
Backend tests for scope detection integration.

Tests the complete integration of PubMedBERT scope detection as the first gate
in the Django RAG pipeline.

Run with: python -m pytest chatapp/clinical_rag/test_scope_integration.py -v
"""

import json
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import dataclass

from django.test import TestCase, override_settings
from django.contrib.auth import get_user_model

from chatapp.clinical_rag import scope, pipeline
from chatapp.clinical_rag.scope import (
    ScopeDecision,
    ScopePrediction,
    ScopeQueryResponse,
    OUT_OF_SCOPE_MESSAGE,
    NEEDS_CLARIFICATION_MESSAGE,
    SCOPE_DETECTION_ERROR_MESSAGE,
)

User = get_user_model()


# ============================================================
# Test Helpers
# ============================================================

@dataclass
class MockScopePrediction:
    label_id: int
    label_name: str
    probabilities: dict[str, float]


def create_mock_prediction(label_id: int, label_name: str, probs: dict = None):
    """Create a mock ScopePrediction."""
    if probs is None:
        probs = {"OUT_OF_SCOPE": 0.1, "IN_SCOPE": 0.8, "NEEDS_CLARIFICATION": 0.1}
    return MockScopePrediction(label_id=label_id, label_name=label_name, probabilities=probs)


# ============================================================
# Scope Module Tests
# ============================================================

class ScopeModuleTests(TestCase):
    """Test the scope.py module directly."""

    def setUp(self):
        # Reset singleton for each test
        scope._ScopeDetectorSingleton._instance = None
        scope._ScopeDetectorSingleton._initialized = False

    @patch('chatapp.clinical_rag.scope.config.SCOPE_DETECTION_ENABLED', True)
    @patch('chatapp.clinical_rag.scope.get_scope_detector')
    def test_classify_scope_in_scope(self, mock_get_detector):
        """Test IN_SCOPE classification returns correct response."""
        mock_detector = MagicMock()
        mock_detector.predict.return_value = create_mock_prediction(1, "IN_SCOPE")
        mock_get_detector.return_value = mock_detector

        response = scope.classify_scope("What is HbA1c in type 1 diabetes?")

        self.assertEqual(response.decision, ScopeDecision.IN_SCOPE)
        self.assertIsNone(response.message)
        self.assertIsNotNone(response.scope_prediction)
        self.assertEqual(response.scope_prediction.label_name, "IN_SCOPE")
        mock_detector.predict.assert_called_once_with("What is HbA1c in type 1 diabetes?")

    @patch('chatapp.clinical_rag.scope.config.SCOPE_DETECTION_ENABLED', True)
    @patch('chatapp.clinical_rag.scope.get_scope_detector')
    def test_classify_scope_out_of_scope(self, mock_get_detector):
        """Test OUT_OF_SCOPE classification returns formal message."""
        mock_detector = MagicMock()
        mock_detector.predict.return_value = create_mock_prediction(0, "OUT_OF_SCOPE",
            {"OUT_OF_SCOPE": 0.9, "IN_SCOPE": 0.05, "NEEDS_CLARIFICATION": 0.05})
        mock_get_detector.return_value = mock_detector

        response = scope.classify_scope("What is the capital of France?")

        self.assertEqual(response.decision, ScopeDecision.OUT_OF_SCOPE)
        self.assertEqual(response.message, OUT_OF_SCOPE_MESSAGE)
        self.assertIsNotNone(response.scope_prediction)

    @patch('chatapp.clinical_rag.scope.config.SCOPE_DETECTION_ENABLED', True)
    @patch('chatapp.clinical_rag.scope.get_scope_detector')
    def test_classify_scope_needs_clarification(self, mock_get_detector):
        """Test NEEDS_CLARIFICATION classification returns formal message."""
        mock_detector = MagicMock()
        mock_detector.predict.return_value = create_mock_prediction(2, "NEEDS_CLARIFICATION",
            {"OUT_OF_SCOPE": 0.1, "IN_SCOPE": 0.2, "NEEDS_CLARIFICATION": 0.7})
        mock_get_detector.return_value = mock_detector

        response = scope.classify_scope("diabetes?")

        self.assertEqual(response.decision, ScopeDecision.NEEDS_CLARIFICATION)
        self.assertEqual(response.message, NEEDS_CLARIFICATION_MESSAGE)
        self.assertIsNotNone(response.scope_prediction)

    @patch('chatapp.clinical_rag.scope.config.SCOPE_DETECTION_ENABLED', True)
    @patch('chatapp.clinical_rag.scope.get_scope_detector')
    def test_classify_scope_model_failure(self, mock_get_detector):
        """Test fail-safe behavior when model inference fails."""
        mock_detector = MagicMock()
        mock_detector.predict.side_effect = RuntimeError("CUDA out of memory")
        mock_get_detector.return_value = mock_detector

        response = scope.classify_scope("What is HbA1c?")

        self.assertIsNone(response.decision)
        self.assertEqual(response.message, SCOPE_DETECTION_ERROR_MESSAGE)
        self.assertIsNotNone(response.error)
        self.assertIn("CUDA out of memory", response.error)

    @patch('chatapp.clinical_rag.scope.config.SCOPE_DETECTION_ENABLED', False)
    def test_classify_scope_disabled(self):
        """Test behavior when scope detection is disabled - should fail closed."""
        response = scope.classify_scope("Any question")

        # Fail-closed: disabled detection returns error, not IN_SCOPE
        self.assertIsNone(response.decision)
        self.assertEqual(response.message, SCOPE_DETECTION_ERROR_MESSAGE)
        self.assertIsNotNone(response.error)
        self.assertIn("Scope detection gate is disabled", response.error)
        self.assertIsNone(response.scope_prediction)

    def test_classify_scope_empty_question(self):
        """Test empty question handling."""
        response = scope.classify_scope("")
        self.assertIsNone(response.decision)
        self.assertEqual(response.error, "Empty question provided.")

    def test_classify_scope_whitespace_only(self):
        """Test whitespace-only question handling."""
        response = scope.classify_scope("   ")
        self.assertIsNone(response.decision)
        self.assertEqual(response.error, "Empty question provided.")


# ============================================================
# Pipeline Integration Tests
# ============================================================

class PipelineIntegrationTests(TestCase):
    """Test the full pipeline integration with scope detection."""

    def setUp(self):
        # Reset singleton for each test
        scope._ScopeDetectorSingleton._instance = None
        scope._ScopeDetectorSingleton._initialized = False

    def _collect_events(self, question, history=None):
        """Collect all events from stream_rag_answer."""
        events = []
        for event in pipeline.stream_rag_answer(question, history=history):
            events.append(event)
        return events

    def _get_event_types(self, events):
        """Extract event types from event list."""
        return [e["type"] for e in events]

    @patch('chatapp.clinical_rag.pipeline.generation.condense_question')
    @patch('chatapp.clinical_rag.pipeline.retrieval.search')
    @patch('chatapp.clinical_rag.pipeline.SufficiencyEngine')
    @patch('chatapp.clinical_rag.pipeline.scope.classify_scope')
    def test_in_scope_full_pipeline(self, mock_classify, mock_engine_cls, mock_search, mock_condense):
        """Test A: IN_SCOPE - full pipeline executes."""
        # Setup scope detection
        mock_classify.return_value = ScopeQueryResponse(
            decision=ScopeDecision.IN_SCOPE,
            message=None,
            scope_prediction=create_mock_prediction(1, "IN_SCOPE"),
        )

        # Setup condensation
        mock_condense.return_value = "What is HbA1c in type 1 diabetes?"

        # Setup retrieval
        mock_search.return_value = [
            {"content": "HbA1c is a marker...", "metadata": {"source": "paper1.txt"}, "similarity": 0.85, "distance": 0.15}
        ]

        # Setup sufficiency engine
        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_partial = MagicMock()
        mock_partial.model_c_probability = 0.9
        mock_partial.cosine_similarity = 0.85
        mock_partial.model_c_details = {}
        mock_partial.cosine_details = {}
        mock_partial.llm_judge_future = None
        mock_partial.llm_judge_enabled = False
        mock_partial.judge_model = "test"
        mock_partial.gate_disabled = False
        mock_engine.evaluate_partial.return_value = mock_partial

        mock_result = MagicMock()
        mock_result.sufficient = True
        mock_result.combined_score = 0.85
        mock_result.llm_judge_score = 0.8
        mock_result.model_c_probability = 0.9
        mock_result.cosine_similarity = 0.85
        mock_result.threshold = 0.7
        mock_result.check_details = {
            "model_c": {}, "cosine": {}, "llm_judge": {"status": "disabled"},
            "weights": {}, "contributions": {}, "judge_status": "disabled"
        }
        mock_engine.finalize.return_value = mock_result

        # Mock generation
        with patch('chatapp.clinical_rag.pipeline.generation.stream_answer') as mock_gen:
            mock_gen.return_value = iter(["Answer: ", "HbA1c is ", "a marker."])

            events = self._collect_events("What is HbA1c in type 1 diabetes?")

        # Verify event sequence
        event_types = self._get_event_types(events)
        self.assertEqual(event_types[0], "scope")
        self.assertEqual(event_types[1], "query")
        self.assertEqual(event_types[2], "sources")
        self.assertIn("sufficiency", event_types)
        self.assertIn("token", event_types)

        # Verify scope event content
        scope_event = events[0]
        self.assertEqual(scope_event["decision"], "IN_SCOPE")
        self.assertEqual(scope_event["label_id"], 1)
        self.assertIsNone(scope_event["message"])

        # Verify condensation called with original question and empty history
        mock_condense.assert_called_once_with("What is HbA1c in type 1 diabetes?", [])

        # Verify retrieval called exactly once with condensed question
        mock_search.assert_called_once()

        # Verify sufficiency engine used
        mock_engine.evaluate_partial.assert_called_once()

        # Verify tokens generated
        token_events = [e for e in events if e["type"] == "token"]
        self.assertEqual(len(token_events), 3)

    @patch('chatapp.clinical_rag.pipeline.generation.condense_question')
    @patch('chatapp.clinical_rag.pipeline.retrieval.search')
    @patch('chatapp.clinical_rag.pipeline.scope.classify_scope')
    def test_out_of_scope_stops_early(self, mock_classify, mock_search, mock_condense):
        """Test B: OUT_OF_SCOPE - stops after scope event, no downstream calls."""
        mock_classify.return_value = ScopeQueryResponse(
            decision=ScopeDecision.OUT_OF_SCOPE,
            message=OUT_OF_SCOPE_MESSAGE,
            scope_prediction=create_mock_prediction(0, "OUT_OF_SCOPE"),
        )

        events = self._collect_events("What is the capital of France?")

        # Verify event sequence: scope -> abstain -> done (no query, sources, etc.)
        event_types = self._get_event_types(events)
        self.assertEqual(event_types, ["scope", "abstain"])

        # Verify scope event
        scope_event = events[0]
        self.assertEqual(scope_event["decision"], "OUT_OF_SCOPE")
        self.assertEqual(scope_event["message"], OUT_OF_SCOPE_MESSAGE)

        # Verify abstain event has formal message
        abstain_event = events[1]
        self.assertEqual(abstain_event["type"], "abstain")
        self.assertEqual(abstain_event["reason"], OUT_OF_SCOPE_MESSAGE)
        self.assertEqual(abstain_event["details"]["scope_decision"], "OUT_OF_SCOPE")

        # Verify downstream never called
        mock_condense.assert_not_called()
        mock_search.assert_not_called()

    @patch('chatapp.clinical_rag.pipeline.generation.condense_question')
    @patch('chatapp.clinical_rag.pipeline.retrieval.search')
    @patch('chatapp.clinical_rag.pipeline.scope.classify_scope')
    def test_needs_clarification_stops_early(self, mock_classify, mock_search, mock_condense):
        """Test C: NEEDS_CLARIFICATION - stops after scope event, no downstream calls."""
        mock_classify.return_value = ScopeQueryResponse(
            decision=ScopeDecision.NEEDS_CLARIFICATION,
            message=NEEDS_CLARIFICATION_MESSAGE,
            scope_prediction=create_mock_prediction(2, "NEEDS_CLARIFICATION"),
        )

        events = self._collect_events("diabetes?")

        event_types = self._get_event_types(events)
        self.assertEqual(event_types, ["scope", "abstain"])

        scope_event = events[0]
        self.assertEqual(scope_event["decision"], "NEEDS_CLARIFICATION")
        self.assertEqual(scope_event["message"], NEEDS_CLARIFICATION_MESSAGE)

        abstain_event = events[1]
        self.assertEqual(abstain_event["reason"], NEEDS_CLARIFICATION_MESSAGE)
        self.assertEqual(abstain_event["details"]["scope_decision"], "NEEDS_CLARIFICATION")

        mock_condense.assert_not_called()
        mock_search.assert_not_called()

    @patch('chatapp.clinical_rag.pipeline.scope.classify_scope')
    def test_scope_model_failure_fail_closed(self, mock_classify):
        """Test D: Scope model failure - fail closed, no downstream processing."""
        mock_classify.return_value = ScopeQueryResponse(
            decision=None,
            message=SCOPE_DETECTION_ERROR_MESSAGE,
            scope_prediction=None,
            error="Model loading failed",
        )

        events = self._collect_events("What is HbA1c?")

        event_types = self._get_event_types(events)
        self.assertEqual(event_types, ["scope", "abstain"])

        scope_event = events[0]
        self.assertIsNone(scope_event["decision"])

        abstain_event = events[1]
        self.assertEqual(abstain_event["reason"], SCOPE_DETECTION_ERROR_MESSAGE)
        self.assertIn("scope_error", abstain_event["details"])

    @patch('chatapp.clinical_rag.pipeline.generation.condense_question')
    @patch('chatapp.clinical_rag.pipeline.retrieval.search')
    @patch('chatapp.clinical_rag.pipeline.SufficiencyEngine')
    @patch('chatapp.clinical_rag.pipeline.scope.classify_scope')
    def test_streaming_compatibility_in_scope(self, mock_classify, mock_engine_cls, mock_search, mock_condense):
        """Test E: Existing streaming contract preserved for IN_SCOPE."""
        mock_classify.return_value = ScopeQueryResponse(
            decision=ScopeDecision.IN_SCOPE,
            message=None,
            scope_prediction=create_mock_prediction(1, "IN_SCOPE"),
        )
        mock_condense.return_value = "Test question"
        mock_search.return_value = [
            {"content": "Content", "metadata": {"source": "test.txt"}, "similarity": 0.9, "distance": 0.1}
        ]

        mock_engine = MagicMock()
        mock_engine_cls.return_value = mock_engine
        mock_partial = MagicMock()
        mock_partial.model_c_probability = 0.9
        mock_partial.cosine_similarity = 0.85
        mock_partial.model_c_details = {}
        mock_partial.cosine_details = {}
        mock_partial.llm_judge_future = None
        mock_partial.llm_judge_enabled = False
        mock_partial.judge_model = "test"
        mock_partial.gate_disabled = False
        mock_engine.evaluate_partial.return_value = mock_partial

        mock_result = MagicMock()
        mock_result.sufficient = True
        mock_result.combined_score = 0.85
        mock_result.llm_judge_score = 0.8
        mock_result.model_c_probability = 0.9
        mock_result.cosine_similarity = 0.85
        mock_result.threshold = 0.7
        mock_result.check_details = {
            "model_c": {}, "cosine": {}, "llm_judge": {"status": "disabled"},
            "weights": {}, "contributions": {}, "judge_status": "disabled"
        }
        mock_engine.finalize.return_value = mock_result

        with patch('chatapp.clinical_rag.pipeline.generation.stream_answer') as mock_gen:
            mock_gen.return_value = iter(["Token 1", "Token 2"])
            events = self._collect_events("Test question")

        # Verify event types in order
        event_types = self._get_event_types(events)
        expected_sequence = ["scope", "query", "sources", "sufficiency", "token", "token"]
        self.assertEqual(event_types, expected_sequence)

        # Verify all events have required fields
        for event in events:
            self.assertIn("type", event)

        scope_event = events[0]
        self.assertIn("decision", scope_event)
        self.assertIn("label_id", scope_event)
        self.assertIn("probabilities", scope_event)
        self.assertIn("message", scope_event)

        query_event = events[1]
        self.assertIn("text", query_event)

        sources_event = events[2]
        self.assertIn("sources", sources_event)
        self.assertIn("confidence", sources_event)

        suff_event = events[3]
        self.assertIn("details", suff_event)
        self.assertIn("combined_score", suff_event["details"])
        self.assertIn("threshold", suff_event["details"])
        self.assertIn("sufficient", suff_event["details"])

        # Verify no 'done' event (that's added by the view)
        self.assertNotIn("done", event_types)


# ============================================================
# View Integration Tests
# ============================================================

class ViewIntegrationTests(TestCase):
    """Test the Django view integration with scope detection."""

    def setUp(self):
        self.user = User.objects.create_user(username="testuser", password="testpass")
        self.client.force_login(self.user)

    @patch('chatapp.views.pipeline.stream_rag_answer')
    def test_out_of_scope_persists_formal_message(self, mock_stream_rag):
        """Test that OUT_OF_SCOPE formal message is persisted correctly."""
        # Mock the full pipeline to return scope + abstain events
        mock_stream_rag.return_value = iter([
            {"type": "scope", "decision": "OUT_OF_SCOPE", "label_id": 0,
             "probabilities": {"OUT_OF_SCOPE": 0.9}, "message": OUT_OF_SCOPE_MESSAGE},
            {"type": "abstain", "reason": OUT_OF_SCOPE_MESSAGE,
             "details": {"scope_decision": "OUT_OF_SCOPE", "scope_probabilities": {"OUT_OF_SCOPE": 0.9}}},
        ])

        response = self.client.post("/chat/ask/", {"question": "What is the capital of France?"})
        self.assertEqual(response.status_code, 200)

        # Parse streaming response
        content = b"".join(response.streaming_content).decode("utf-8")
        events = [json.loads(line) for line in content.strip().split("\n") if line.strip()]

        # Check abstain event has formal message
        abstain_events = [e for e in events if e["type"] == "abstain"]
        self.assertEqual(len(abstain_events), 1)
        self.assertEqual(abstain_events[0]["reason"], OUT_OF_SCOPE_MESSAGE)

        # Check ChatMessage was created with formal message
        from chatapp.models import ChatMessage
        messages = ChatMessage.objects.filter(owner=self.user, role=ChatMessage.Role.ASSISTANT)
        self.assertEqual(messages.count(), 1)
        self.assertEqual(messages.first().content, OUT_OF_SCOPE_MESSAGE)

    @patch('chatapp.views.pipeline.stream_rag_answer')
    def test_needs_clarification_persists_formal_message(self, mock_stream_rag):
        """Test that NEEDS_CLARIFICATION formal message is persisted correctly."""
        mock_stream_rag.return_value = iter([
            {"type": "scope", "decision": "NEEDS_CLARIFICATION", "label_id": 2,
             "probabilities": {"NEEDS_CLARIFICATION": 0.8}, "message": NEEDS_CLARIFICATION_MESSAGE},
            {"type": "abstain", "reason": NEEDS_CLARIFICATION_MESSAGE,
             "details": {"scope_decision": "NEEDS_CLARIFICATION", "scope_probabilities": {"NEEDS_CLARIFICATION": 0.8}}},
        ])

        response = self.client.post("/chat/ask/", {"question": "diabetes?"})
        self.assertEqual(response.status_code, 200)

        content = b"".join(response.streaming_content).decode("utf-8")
        events = [json.loads(line) for line in content.strip().split("\n") if line.strip()]

        abstain_events = [e for e in events if e["type"] == "abstain"]
        self.assertEqual(len(abstain_events), 1)
        self.assertEqual(abstain_events[0]["reason"], NEEDS_CLARIFICATION_MESSAGE)

        from chatapp.models import ChatMessage
        messages = ChatMessage.objects.filter(owner=self.user, role=ChatMessage.Role.ASSISTANT)
        self.assertEqual(messages.count(), 1)
        self.assertEqual(messages.first().content, NEEDS_CLARIFICATION_MESSAGE)

    @patch('chatapp.views.pipeline.stream_rag_answer')
    def test_in_scope_persists_answer(self, mock_stream_rag):
        """Test that IN_SCOPE persists the generated answer."""
        mock_stream_rag.return_value = iter([
            {"type": "scope", "decision": "IN_SCOPE", "label_id": 1,
             "probabilities": {"IN_SCOPE": 0.8}, "message": None},
            {"type": "query", "text": "Test question"},
            {"type": "sources", "sources": [], "confidence": 0.9},
            {"type": "sufficiency", "details": {
                "combined_score": 0.85, "threshold": 0.7, "sufficient": True,
                "model_c_probability": 0.9, "cosine_similarity": 0.85, "llm_judge_score": 0.8,
                "check_details": {}, "weights": {}, "contributions": {}
            }},
            {"type": "token", "text": "Generated answer"},
        ])

        response = self.client.post("/chat/ask/", {"question": "What is HbA1c?"})
        self.assertEqual(response.status_code, 200)

        # Consume the streaming response to trigger persistence
        content = b"".join(response.streaming_content).decode("utf-8")

        from chatapp.models import ChatMessage
        messages = ChatMessage.objects.filter(owner=self.user, role=ChatMessage.Role.ASSISTANT)
        self.assertEqual(messages.count(), 1)
        self.assertEqual(messages.first().content, "Generated answer")


# ============================================================
# Configuration Tests
# ============================================================

class ConfigTests(TestCase):
    """Test scope detection configuration."""

    def test_scope_config_exists(self):
        """Verify scope config values are defined."""
        from chatapp.clinical_rag import config
        self.assertTrue(hasattr(config, "SCOPE_MODEL_PATH"))
        self.assertTrue(hasattr(config, "SCOPE_MAX_LENGTH"))
        self.assertTrue(hasattr(config, "SCOPE_DETECTION_ENABLED"))

    def test_scope_model_path_correct(self):
        """Verify scope model path points to correct location."""
        from chatapp.clinical_rag import config
        expected = config.PROJECT_ROOT / "classification" / "scope_detection" / "model" / "pubmedbert_best"
        self.assertEqual(config.SCOPE_MODEL_PATH, expected)

    def test_scope_defaults_enabled(self):
        """Verify scope detection is enabled by default."""
        from chatapp.clinical_rag import config
        self.assertTrue(config.SCOPE_DETECTION_ENABLED)

    def test_scope_max_length(self):
        """Verify scope max length is 64."""
        from chatapp.clinical_rag import config
        self.assertEqual(config.SCOPE_MAX_LENGTH, 64)


if __name__ == "__main__":
    import django
    django.setup()
    import unittest
    unittest.main()