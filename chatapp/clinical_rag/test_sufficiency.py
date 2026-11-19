"""
Unit tests for the sufficiency engine and checks.

Run with: python -m pytest chatapp/clinical_rag/test_sufficiency.py -v
Or: python chatapp/clinical_rag/test_sufficiency.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from chatapp.clinical_rag import config
from chatapp.clinical_rag.sufficiency_checks import (
    ModelCCheck,
    CosineCheck,
    LLMJudgeCheck,
    CheckResult,
    ModelCCheckpointError,
    ModelCInferenceError,
    ModelCOutputError,
    CosineSimilarityError,
    LLMJudgeError,
    _build_judge_prompt,
    _parse_judge_response,
    _GeminiJudge,
    _GroqJudge,
    get_model_c,
)
from chatapp.clinical_rag.sufficiency import (
    PartialSufficiencyResult,
    SufficiencyEngine,
    SufficiencyResult,
    calculate_combined_score,
    calculate_fallback_score,
    NoChecksConfiguredError,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singletons between tests."""
    # Reset Model C singleton
    from chatapp.clinical_rag.sufficiency_checks import _ModelCSingleton
    _ModelCSingleton._instance = None
    _ModelCSingleton._initialized = False
    yield
    _ModelCSingleton._instance = None
    _ModelCSingleton._initialized = False


@pytest.fixture(autouse=True)
def disable_llm_judge_by_default():
    """Disable LLM Judge by default for tests that don't explicitly test it."""
    from chatapp.clinical_rag import config
    original = config.LLM_JUDGE_ENABLED
    config.LLM_JUDGE_ENABLED = False
    yield
    config.LLM_JUDGE_ENABLED = original


@pytest.fixture
def sample_hits():
    """Sample retrieval hits for testing."""
    return [
        {
            "id": "doc1_chunk_0",
            "content": "Type 2 diabetes in adolescents is associated with obesity and insulin resistance.",
            "metadata": {"source": "doc1.txt", "chunk_index": 0},
            "distance": 0.15,
            "similarity": 0.85,
            "rank": 1,
        },
        {
            "id": "doc2_chunk_0",
            "content": "Glycemic control can be improved with continuous glucose monitoring.",
            "metadata": {"source": "doc2.txt", "chunk_index": 0},
            "distance": 0.30,
            "similarity": 0.70,
            "rank": 2,
        },
        {
            "id": "doc3_chunk_0",
            "content": "Screening for prediabetes in youth uses HbA1c and fasting glucose.",
            "metadata": {"source": "doc3.txt", "chunk_index": 0},
            "distance": 0.45,
            "similarity": 0.55,
            "rank": 3,
        },
    ]


@pytest.fixture
def mock_model_c():
    """Mock Model C singleton for testing without loading real model."""
    with patch("chatapp.clinical_rag.sufficiency_checks._ModelCSingleton._load_model"):
        singleton = get_model_c()
        # Mock the predict_proba method
        singleton.predict_proba = MagicMock(return_value=0.92)
        yield singleton


# ============================================================
# Tests for calculate_combined_score
# ============================================================

class TestCombinedScore:
    """Tests for the canonical combined scoring formula."""

    def test_formula_with_three_components(self):
        """Verify formula: 0.10 * model_c + 0.45 * cosine + 0.45 * llm_judge"""
        model_c_prob = 0.90
        cosine_sim = 0.80
        llm_judge_score = 1.0
        expected = round(0.10 * model_c_prob + 0.45 * cosine_sim + 0.45 * llm_judge_score, 4)
        actual = calculate_combined_score(model_c_prob, cosine_sim, llm_judge_score)
        assert actual == expected

    def test_formula_with_zero_llm_judge(self):
        """Verify formula with LLM judge score of 0."""
        model_c_prob = 0.90
        cosine_sim = 0.80
        llm_judge_score = 0.0
        expected = round(0.10 * model_c_prob + 0.45 * cosine_sim + 0.45 * llm_judge_score, 4)
        actual = calculate_combined_score(model_c_prob, cosine_sim, llm_judge_score)
        assert actual == expected

    def test_rounding_behavior(self):
        """Verify rounding to 4 decimal places."""
        # 0.10 * 0.5 + 0.45 * 0.5 + 0.45 * 0.5 = 0.5 exactly
        assert calculate_combined_score(0.5, 0.5, 0.5) == 0.5

        # 0.10 * 0.3333 + 0.45 * 0.6666 + 0.45 * 0.7777 = 0.6499 -> 0.6499
        result = calculate_combined_score(0.3333, 0.6666, 0.7777)
        assert result == round(0.10 * 0.3333 + 0.45 * 0.6666 + 0.45 * 0.7777, 4)

    def test_edge_cases(self):
        """Test boundary values."""
        assert calculate_combined_score(0.0, 0.0, 0.0) == 0.0
        assert calculate_combined_score(1.0, 1.0, 1.0) == 1.0
        assert calculate_combined_score(0.0, 1.0, 0.0) == round(0.45, 4)
        assert calculate_combined_score(1.0, 0.0, 0.0) == round(0.10, 4)
        assert calculate_combined_score(0.0, 0.0, 1.0) == round(0.45, 4)

    def test_weights_are_from_config(self):
        """Verify weights come from config, not hardcoded."""
        original_mc = config.MODEL_C_WEIGHT
        original_cos = config.COSINE_WEIGHT
        original_llm = config.LLM_JUDGE_WEIGHT
        try:
            config.MODEL_C_WEIGHT = 0.20
            config.COSINE_WEIGHT = 0.30
            config.LLM_JUDGE_WEIGHT = 0.50
            result = calculate_combined_score(1.0, 0.0, 0.0)
            assert result == 0.20
            result = calculate_combined_score(0.0, 1.0, 0.0)
            assert result == 0.30
            result = calculate_combined_score(0.0, 0.0, 1.0)
            assert result == 0.50
        finally:
            config.MODEL_C_WEIGHT = original_mc
            config.COSINE_WEIGHT = original_cos
            config.LLM_JUDGE_WEIGHT = original_llm


class TestFallbackScore:
    """Tests for fallback score when LLM judge is unavailable."""

    def test_fallback_renormalizes_weights(self):
        """Fallback should use renormalized weights: 0.10/0.55 and 0.45/0.55"""
        model_c_prob = 1.0
        cosine_sim = 0.0
        # model_c weight = 0.10/0.55 = 0.1818..., cosine weight = 0.45/0.55 = 0.8181...
        expected = round((0.10/0.55) * 1.0 + (0.45/0.55) * 0.0, 4)
        actual = calculate_fallback_score(model_c_prob, cosine_sim)
        assert actual == expected

    def test_fallback_with_both_scores(self):
        """Fallback with both model_c and cosine."""
        model_c_prob = 0.80
        cosine_sim = 0.60
        fallback_mc_weight = 0.10 / 0.55
        fallback_cos_weight = 0.45 / 0.55
        expected = round(fallback_mc_weight * model_c_prob + fallback_cos_weight * cosine_sim, 4)
        actual = calculate_fallback_score(model_c_prob, cosine_sim)
        assert actual == expected


# ============================================================
# Tests for CosineCheck (Rank-Weighted)
# ============================================================

class TestCosineCheck:
    """Tests for rank-weighted cosine similarity check."""

    def test_weighted_cosine_three_passages(self, sample_hits):
        """Three passages use weights [1.0, 0.5, 0.25] with denominator 1.75."""
        # sample_hits: P1=0.85, P2=0.70, P3=0.55
        # weighted = (1.0*0.85 + 0.5*0.70 + 0.25*0.55) / 1.75 = 1.3375 / 1.75 = 0.7642857 -> 0.7643
        check = CosineCheck()
        result = check.evaluate("test question", sample_hits)
        assert result.name == "cosine"
        assert result.score == 0.7643
        assert result.details["weights"] == [1.0, 0.5, 0.25]
        assert result.details["weight_sum"] == 1.75
        assert result.details["passages_used"] == 3
        assert result.details["passage_similarities"] == [0.85, 0.70, 0.55]

    def test_weighted_cosine_two_passages(self):
        """Two passages use weights [1.0, 0.5] with denominator 1.50."""
        # P1=0.80, P2=0.60 -> (1.0*0.80 + 0.5*0.60) / 1.50 = 1.10 / 1.50 = 0.7333
        check = CosineCheck()
        hits = [
            {"similarity": 0.80, "content": "pass 1"},
            {"similarity": 0.60, "content": "pass 2"},
        ]
        result = check.evaluate("test question", hits)
        assert result.score == 0.7333
        assert result.details["weights"] == [1.0, 0.5]
        assert result.details["weight_sum"] == 1.5
        assert result.details["passages_used"] == 2
        assert result.details["passage_similarities"] == [0.80, 0.60]

    def test_weighted_cosine_one_passage(self):
        """One passage uses weight [1.0] with denominator 1.00."""
        check = CosineCheck()
        hits = [{"similarity": 0.85, "content": "pass 1"}]
        result = check.evaluate("test question", hits)
        assert result.score == 0.85
        assert result.details["weights"] == [1.0]
        assert result.details["weight_sum"] == 1.0
        assert result.details["passages_used"] == 1
        assert result.details["passage_similarities"] == [0.85]

    def test_clamps_similarity_to_valid_range(self):
        """Similarities outside [0, 1] are clamped."""
        check = CosineCheck()
        hits_high = [{"similarity": 1.5, "content": "test"}]
        hits_low = [{"similarity": -0.5, "content": "test"}]
        result_high = check.evaluate("q", hits_high)
        result_low = check.evaluate("q", hits_low)
        assert result_high.score == 1.0
        assert result_low.score == 0.0

    def test_preserves_retrieval_rank_order(self):
        """Rank 1 has highest weight (1.0) while Rank 3 has lowest weight (0.25)."""
        check = CosineCheck()
        hits_descending = [
            {"similarity": 0.90, "content": "p1"},
            {"similarity": 0.50, "content": "p2"},
            {"similarity": 0.10, "content": "p3"},
        ]
        hits_ascending = [
            {"similarity": 0.10, "content": "p1"},
            {"similarity": 0.50, "content": "p2"},
            {"similarity": 0.90, "content": "p3"},
        ]
        res_desc = check.evaluate("q", hits_descending)
        res_asc = check.evaluate("q", hits_ascending)

        # desc: (1.0*0.9 + 0.5*0.5 + 0.25*0.1) / 1.75 = (0.9 + 0.25 + 0.025) / 1.75 = 1.175 / 1.75 = 0.6714
        # asc:  (1.0*0.1 + 0.5*0.5 + 0.25*0.9) / 1.75 = (0.1 + 0.25 + 0.225) / 1.75 = 0.575 / 1.75 = 0.3286
        assert res_desc.score == 0.6714
        assert res_asc.score == 0.3286
        assert res_desc.score > res_asc.score

    def test_raises_on_empty_hits(self):
        """Empty hits raises CosineSimilarityError."""
        check = CosineCheck()
        with pytest.raises(CosineSimilarityError):
            check.evaluate("question", [])

    def test_computes_similarity_from_distance_fallback(self):
        """If 'similarity' key is missing, calculates 1 - distance."""
        check = CosineCheck()
        hits = [{"distance": 0.25, "content": "test"}]
        result = check.evaluate("q", hits)
        assert result.score == 0.75
        assert result.details["passage_similarities"] == [0.75]

    def test_raises_on_missing_similarity_and_distance(self):
        """Missing both similarity and distance raises CosineSimilarityError."""
        check = CosineCheck()
        hits = [{"content": "test"}]
        with pytest.raises(CosineSimilarityError):
            check.evaluate("question", hits)

    def test_raises_on_non_numeric_similarity(self):
        """Non-numeric similarity raises CosineSimilarityError."""
        check = CosineCheck()
        hits = [{"similarity": "high", "content": "test"}]
        with pytest.raises(CosineSimilarityError):
            check.evaluate("question", hits)

    def test_details_structure(self, sample_hits):
        """Check result details contain expected fields."""
        check = CosineCheck()
        result = check.evaluate("question", sample_hits)
        assert "similarity" in result.details
        assert "raw_similarity" in result.details
        assert "passage_similarities" in result.details
        assert "weights" in result.details
        assert "weight_sum" in result.details
        assert "passages_used" in result.details
        assert "formula" in result.details
        assert result.details["passages_used"] == 3


# ============================================================
# Tests for ModelCCheck (Multi-Passage with Mocked Model)
# ============================================================

class TestModelCCheck:
    """Tests for multi-passage Model C check."""

    def test_uses_all_three_passages(self, mock_model_c, sample_hits):
        """Model C receives all 3 retrieved passages in original order."""
        check = ModelCCheck()
        result = check.evaluate("test question", sample_hits)
        mock_model_c.predict_proba.assert_called_once()
        call_args = mock_model_c.predict_proba.call_args
        assert call_args[0][0] == "test question"
        expected_passages = [h["content"] for h in sample_hits]
        assert call_args[0][1] == expected_passages
        assert result.details["passages_used"] == 3

    def test_two_retrieved_passages(self, mock_model_c, sample_hits):
        """Model C handles two retrieved passages correctly."""
        check = ModelCCheck()
        result = check.evaluate("test question", sample_hits[:2])
        call_args = mock_model_c.predict_proba.call_args
        assert call_args[0][1] == [sample_hits[0]["content"], sample_hits[1]["content"]]
        assert result.details["passages_used"] == 2

    def test_one_retrieved_passage(self, mock_model_c, sample_hits):
        """Model C handles single retrieved passage correctly."""
        check = ModelCCheck()
        result = check.evaluate("test question", sample_hits[:1])
        call_args = mock_model_c.predict_proba.call_args
        assert call_args[0][1] == [sample_hits[0]["content"]]
        assert result.details["passages_used"] == 1

    def test_returns_probability_of_label_1(self, mock_model_c, sample_hits):
        """Score is P(sufficient) = probability of label 1."""
        mock_model_c.predict_proba.return_value = 0.75
        check = ModelCCheck()
        result = check.evaluate("test question", sample_hits)
        assert result.score == 0.75
        assert result.details["probability_sufficient"] == 0.75
        assert result.details["probability_insufficient"] == 0.25

    def test_raises_on_empty_hits(self, mock_model_c):
        """Empty hits raises ModelCInferenceError."""
        check = ModelCCheck()
        with pytest.raises(ModelCInferenceError):
            check.evaluate("question", [])

    def test_raises_on_all_empty_passages(self, mock_model_c):
        """All empty passage contents raises ModelCInferenceError."""
        check = ModelCCheck()
        hits = [{"content": ""}, {"content": "   "}]
        with pytest.raises(ModelCInferenceError, match="All retrieved passages are empty"):
            check.evaluate("question", hits)

    def test_raises_on_missing_checkpoint(self):
        """Missing checkpoint raises ModelCCheckpointError."""
        with patch.object(config, "MODEL_C_CHECKPOINT", Path("/nonexistent/path")):
            check = ModelCCheck()
            with pytest.raises(ModelCCheckpointError):
                check.evaluate("question", [{"content": "test", "similarity": 0.8}])

    def test_invalid_probability_raises(self, mock_model_c, sample_hits):
        """Probability outside [0,1] raises ModelCOutputError."""
        mock_model_c.predict_proba.return_value = 1.5
        check = ModelCCheck()
        with pytest.raises(ModelCOutputError):
            check.evaluate("question", sample_hits)

    def test_details_structure(self, mock_model_c, sample_hits):
        """Check result details contain expected fields."""
        check = ModelCCheck()
        result = check.evaluate("question", sample_hits)
        assert "probability_sufficient" in result.details
        assert "probability_insufficient" in result.details
        assert "passages_used" in result.details
        assert "max_length" in result.details
        assert result.details["passages_used"] == 3
        assert result.details["max_length"] == 512


# ============================================================
# Tests for SufficiencyEngine
# ============================================================

class TestSufficiencyEngine:
    """Tests for the sufficiency engine."""

    def test_sufficient_when_score_above_threshold(self, mock_model_c, sample_hits):
        """Score >= 0.70 returns sufficient=True (with LLM judge disabled -> fallback)."""
        # Set all similarities to 0.85 -> weighted cosine = 0.85
        for h in sample_hits:
            h["similarity"] = 0.85
        mock_model_c.predict_proba.return_value = 0.92
        engine = SufficiencyEngine()
        partial = engine.evaluate_partial("test question", sample_hits)
        result = engine.finalize(partial)
        assert result.sufficient is True
        assert result.combined_score == 0.8627
        assert result.llm_judge_score is None
        assert result.check_details["judge_status"] == "disabled"

    def test_insufficient_when_score_below_threshold(self, mock_model_c, sample_hits):
        """Score < 0.70 returns sufficient=False (with LLM judge disabled -> fallback)."""
        for h in sample_hits:
            h["similarity"] = 0.50
        mock_model_c.predict_proba.return_value = 0.30
        engine = SufficiencyEngine()
        partial = engine.evaluate_partial("test question", sample_hits)
        result = engine.finalize(partial)
        assert result.sufficient is False
        assert result.combined_score == 0.4636
        assert result.llm_judge_score is None

    def test_exact_threshold_is_sufficient(self, mock_model_c, sample_hits):
        """Score exactly at threshold (0.70) returns sufficient=True (fallback)."""
        # If mc=1.0: (0.10/0.55)*1.0 + (0.45/0.55)*cos = 0.70 -> cos = 0.6333333333333333
        mock_model_c.predict_proba.return_value = 1.0
        for h in sample_hits:
            h["similarity"] = 0.6333333333333333
        engine = SufficiencyEngine()
        partial = engine.evaluate_partial("test question", sample_hits)
        result = engine.finalize(partial)
        assert result.sufficient is True
        assert result.combined_score >= 0.70

    def test_combined_score_matches_canonical_formula(self, mock_model_c, sample_hits):
        """Engine uses calculate_fallback_score when LLM judge disabled (single source of truth)."""
        mock_model_c.predict_proba.return_value = 0.88
        for h in sample_hits:
            h["similarity"] = 0.78
        engine = SufficiencyEngine()
        partial = engine.evaluate_partial("test question", sample_hits)
        result = engine.finalize(partial)
        expected = calculate_fallback_score(0.88, 0.78)
        assert result.combined_score == expected

    def test_result_contains_all_required_fields(self, mock_model_c, sample_hits):
        """SufficiencyResult has all required fields."""
        mock_model_c.predict_proba.return_value = 0.90
        engine = SufficiencyEngine()
        partial = engine.evaluate_partial("test question", sample_hits)
        result = engine.finalize(partial)
        assert hasattr(result, "sufficient")
        assert hasattr(result, "combined_score")
        assert hasattr(result, "threshold")
        assert hasattr(result, "model_c_probability")
        assert hasattr(result, "cosine_similarity")
        assert hasattr(result, "llm_judge_score")
        assert hasattr(result, "check_details")

    def test_check_details_contains_component_info(self, mock_model_c, sample_hits):
        """check_details has weights, contributions, and component details."""
        mock_model_c.predict_proba.return_value = 0.90
        engine = SufficiencyEngine()
        partial = engine.evaluate_partial("test question", sample_hits)
        result = engine.finalize(partial)
        details = result.check_details
        assert "model_c" in details
        assert "cosine" in details
        assert "llm_judge" in details
        assert "weights" in details
        assert "contributions" in details
        assert "judge_status" in details
        # With LLM judge disabled, weights are renormalized
        assert abs(details["weights"]["model_c"] - (0.10/0.55)) < 1e-9
        assert abs(details["weights"]["cosine"] - (0.45/0.55)) < 1e-9
        assert details["weights"]["llm_judge"] == 0.0

    def test_disabled_gate_allows_generation(self, mock_model_c, sample_hits):
        """SUFFICIENCY_ENABLED=False returns sufficient with neutral scores."""
        original = config.SUFFICIENCY_ENABLED
        config.SUFFICIENCY_ENABLED = False
        try:
            engine = SufficiencyEngine()
            partial = engine.evaluate_partial("test question", sample_hits)
            result = engine.finalize(partial)
            assert result.sufficient is True
            assert result.combined_score == 1.0
            assert result.model_c_probability == 1.0
            assert result.cosine_similarity == 1.0
            assert result.llm_judge_score == 1.0
            assert result.check_details.get("disabled") is True
        finally:
            config.SUFFICIENCY_ENABLED = original

    def test_model_c_failure_propagates(self, mock_model_c, sample_hits):
        """Model C failure raises error (fail-closed)."""
        mock_model_c.predict_proba.side_effect = ModelCInferenceError("Model failed")
        engine = SufficiencyEngine()
        with pytest.raises(ModelCInferenceError):
            engine.evaluate_partial("test question", sample_hits)

    def test_cosine_failure_propagates(self, mock_model_c, sample_hits):
        """Cosine failure raises error (fail-closed)."""
        sample_hits[0]["similarity"] = None
        sample_hits[0].pop("distance", None)
        engine = SufficiencyEngine()
        with pytest.raises(CosineSimilarityError):
            engine.evaluate_partial("test question", sample_hits)
        with pytest.raises(CosineSimilarityError):
            engine.evaluate_partial("test question", sample_hits)

    def test_no_checks_configured_raises(self):
        """Engine with no checks raises NoChecksConfiguredError."""
        engine = SufficiencyEngine()
        # Replace checks with empty list (avoids Model C torch loading)
        engine._model_c_check = None
        engine._cosine_check = None
        engine._llm_judge_check = None
        with pytest.raises(NoChecksConfiguredError):
            engine.evaluate_partial("question", [{"content": "test", "similarity": 0.8}])

    def test_model_loading_reused_not_repeated(self, mock_model_c, sample_hits):
        """Model C singleton is reused across evaluations."""
        mock_model_c.predict_proba.return_value = 0.90
        engine = SufficiencyEngine()
        # First evaluation
        partial1 = engine.evaluate_partial("question 1", sample_hits)
        result1 = engine.finalize(partial1)
        # Second evaluation
        partial2 = engine.evaluate_partial("question 2", sample_hits)
        result2 = engine.finalize(partial2)
        # predict_proba should be called twice but _load_model only once
        assert mock_model_c.predict_proba.call_count == 2


class TestSufficiencyEngineWithLLMJudge:
    """Tests for the sufficiency engine with LLM judge enabled (mocked)."""

    @pytest.fixture
    def engine_with_mocked_judge(self, mock_model_c, sample_hits):
        """Create engine with mocked LLM judge."""
        # Enable LLM judge for this test class
        from chatapp.clinical_rag import config
        config.LLM_JUDGE_ENABLED = True
        engine = SufficiencyEngine()
        # Replace LLM judge with mock
        mock_judge = MagicMock()
        mock_judge.evaluate = MagicMock(return_value=CheckResult(
            name="llm_judge",
            score=1.0,
            details={"label": 1, "reason": "test", "status": "done", "model": "gemini-3.5-flash"}
        ))
        engine._llm_judge_check = mock_judge
        yield engine
        config.LLM_JUDGE_ENABLED = False

    def test_llm_judge_score_included_when_done(self, engine_with_mocked_judge, mock_model_c, sample_hits):
        """LLM judge score included in combined score when successful."""
        mock_model_c.predict_proba.return_value = 0.90
        for h in sample_hits:
            h["similarity"] = 0.80
        engine = engine_with_mocked_judge
        partial = engine.evaluate_partial("test question", sample_hits)
        result = engine.finalize(partial)
        assert result.llm_judge_score == 1.0
        assert result.check_details["judge_status"] == "done"
        # combined = 0.10*0.90 + 0.45*0.80 + 0.45*1.0 = 0.09 + 0.36 + 0.45 = 0.90
        assert result.combined_score == 0.90

    def test_llm_judge_timeout_falls_back(self, mock_model_c, sample_hits):
        """LLM judge timeout falls back to renormalized Model C + Cosine."""
        from chatapp.clinical_rag import config
        config.LLM_JUDGE_ENABLED = True
        try:
            mock_model_c.predict_proba.return_value = 0.90
            for h in sample_hits:
                h["similarity"] = 0.80

            engine = SufficiencyEngine()
            # Mock LLM judge to timeout
            mock_judge = MagicMock()
            import concurrent.futures
            mock_judge.evaluate = MagicMock(side_effect=concurrent.futures.TimeoutError())
            engine._llm_judge_check = mock_judge

            partial = engine.evaluate_partial("test question", sample_hits)
            result = engine.finalize(partial)

            assert result.llm_judge_score is None
            assert result.check_details["judge_status"] == "timeout"
            # Fallback: model_c=0.90, cosine=0.80
            # renorm weights: model_c=0.10/0.55=0.1818..., cosine=0.45/0.55=0.8181...
            # fallback = 0.1818*0.90 + 0.8181*0.80 = 0.1636 + 0.6545 = 0.8181
            expected_fallback = calculate_fallback_score(0.90, 0.80)
            assert result.combined_score == expected_fallback
            assert result.sufficient is True  # 0.8181 > 0.70
        finally:
            config.LLM_JUDGE_ENABLED = False

    def test_llm_judge_error_falls_back(self, mock_model_c, sample_hits):
        """LLM judge error falls back to renormalized Model C + Cosine."""
        from chatapp.clinical_rag import config
        config.LLM_JUDGE_ENABLED = True
        try:
            mock_model_c.predict_proba.return_value = 0.90
            for h in sample_hits:
                h["similarity"] = 0.80

            engine = SufficiencyEngine()
            mock_judge = MagicMock()
            mock_judge.evaluate = MagicMock(side_effect=LLMJudgeError("API error"))
            engine._llm_judge_check = mock_judge

            partial = engine.evaluate_partial("test question", sample_hits)
            result = engine.finalize(partial)

            assert result.llm_judge_score is None
            assert result.check_details["judge_status"] == "error"
            expected_fallback = calculate_fallback_score(0.90, 0.80)
            assert result.combined_score == expected_fallback
        finally:
            config.LLM_JUDGE_ENABLED = False

    def test_llm_judge_insufficient_score(self, engine_with_mocked_judge, mock_model_c, sample_hits):
        """LLM judge returning 0 (insufficient) lowers combined score."""
        mock_model_c.predict_proba.return_value = 0.90
        for h in sample_hits:
            h["similarity"] = 0.80

        engine = engine_with_mocked_judge
        engine._llm_judge_check.evaluate.return_value = CheckResult(
            name="llm_judge",
            score=0.0,
            details={"label": 0, "reason": "insufficient", "status": "done", "model": "gemini-3.5-flash"}
        )

        partial = engine.evaluate_partial("test question", sample_hits)
        result = engine.finalize(partial)

        assert result.llm_judge_score == 0.0
        # combined = 0.10*0.90 + 0.45*0.80 + 0.45*0.0 = 0.09 + 0.36 + 0 = 0.45
        assert result.combined_score == 0.45
        assert result.sufficient is False


# ============================================================
# Tests for LLM Judge Multi-Passage Evaluation
# ============================================================

class TestLLMJudgeCheckMultiPassage:
    """Comprehensive unit tests for multi-passage LLM Judge evaluation."""

    def test_prompt_building_single_passage(self):
        """Prompt formats a single passage correctly with principles and headers."""
        prompt = _build_judge_prompt("What is DKA?", "Diabetic ketoacidosis is an acute complication.")
        assert "--- Retrieved Passage 1 ---" in prompt
        assert "Diabetic ketoacidosis is an acute complication." in prompt
        assert "SPECIFIC INFORMATION OVER GENERAL TOPIC" in prompt
        assert "COLLECTIVE & COMPLEMENTARY EVIDENCE" in prompt
        assert "IRRELEVANT PASSAGES" in prompt
        assert "CONTRADICTIONS" in prompt

    def test_prompt_building_multi_passages_ordering(self):
        """Prompt preserves exact retrieval order of up to 3 passages."""
        passages = [
            "First hit: diagnostic criteria for diabetes.",
            "Second hit: pediatric insulin dosages.",
            "Third hit: hypoglycemia management.",
        ]
        prompt = _build_judge_prompt("Pediatric diabetes protocol?", passages)
        idx_1 = prompt.find("--- Retrieved Passage 1 ---\nFirst hit: diagnostic criteria for diabetes.")
        idx_2 = prompt.find("--- Retrieved Passage 2 ---\nSecond hit: pediatric insulin dosages.")
        idx_3 = prompt.find("--- Retrieved Passage 3 ---\nThird hit: hypoglycemia management.")

        assert idx_1 != -1
        assert idx_2 != -1
        assert idx_3 != -1
        assert idx_1 < idx_2 < idx_3

    def test_prompt_building_filters_empty_passages(self):
        """Prompt filters out empty strings and whitespace passages while indexing remaining ones."""
        passages = ["Valid first passage.", "", "   ", "Valid second passage."]
        prompt = _build_judge_prompt("Test question?", passages)
        assert "--- Retrieved Passage 1 ---\nValid first passage." in prompt
        assert "--- Retrieved Passage 2 ---\nValid second passage." in prompt
        assert "--- Retrieved Passage 3 ---" not in prompt

    def test_prompt_building_all_empty_raises(self):
        """Passing only empty passages raises LLMJudgeError."""
        with pytest.raises(LLMJudgeError, match="No non-empty passages provided"):
            _build_judge_prompt("Question?", ["", "   "])

    def test_json_parsing_valid(self):
        """Valid JSON parses correctly into score and reason."""
        json_text = '{"sufficiency_score": 0.85, "reason": "Passages 1 and 2 collectively provide the answer."}'
        score, reason = _parse_judge_response(json_text)
        assert score == 0.85
        assert reason == "Passages 1 and 2 collectively provide the answer."

    def test_json_parsing_markdown_code_fences(self):
        """JSON enclosed in markdown code fences is cleanly parsed."""
        wrapped = '```json\n{\n  "sufficiency_score": 0.95,\n  "reason": "Direct evidence present."\n}\n```'
        score, reason = _parse_judge_response(wrapped)
        assert score == 0.95
        assert reason == "Direct evidence present."

    def test_json_parsing_invalid_cases(self):
        """Malformed or out-of-range JSON triggers LLMJudgeError."""
        with pytest.raises(LLMJudgeError, match="empty response"):
            _parse_judge_response("")
        with pytest.raises(LLMJudgeError, match="not valid JSON"):
            _parse_judge_response("Not JSON at all")
        with pytest.raises(LLMJudgeError, match="not a JSON object"):
            _parse_judge_response('["array"]')
        with pytest.raises(LLMJudgeError, match="Missing 'sufficiency_score'"):
            _parse_judge_response('{"reason": "no score"}')
        with pytest.raises(LLMJudgeError, match="Invalid score type"):
            _parse_judge_response('{"sufficiency_score": "high", "reason": "invalid type"}')
        with pytest.raises(LLMJudgeError, match="Score must be in"):
            _parse_judge_response('{"sufficiency_score": 1.5, "reason": "too high"}')
        with pytest.raises(LLMJudgeError, match="Score must be in"):
            _parse_judge_response('{"sufficiency_score": -0.1, "reason": "too low"}')

    def test_one_passage_sufficient_evidence(self):
        """Single retrieved passage containing full evidence yields high sufficiency score."""
        check = LLMJudgeCheck()
        mock_backend = MagicMock()
        mock_backend.model = "mock-model"
        mock_backend.evaluate.return_value = (1.0, "Passage 1 completely answers the question.")
        check._backend = mock_backend

        hits = [
            {"content": "A1C target for pediatric type 1 diabetes is < 7.5% per ADA guidelines.", "similarity": 0.75}
        ]
        result = check.evaluate("What is the pediatric A1C target?", hits)

        assert result.score == 1.0
        assert result.details["passages_used"] == 1
        assert result.details["sufficiency_score"] == 1.0
        mock_backend.evaluate.assert_called_once_with(
            "What is the pediatric A1C target?",
            ["A1C target for pediatric type 1 diabetes is < 7.5% per ADA guidelines."]
        )

    def test_two_complementary_passages(self):
        """Two passages providing complementary parts of an answer are passed in order."""
        check = LLMJudgeCheck()
        mock_backend = MagicMock()
        mock_backend.model = "mock-model"
        mock_backend.evaluate.return_value = (0.95, "Passage 1 gives symptoms; Passage 2 gives lab thresholds.")
        check._backend = mock_backend

        hits = [
            {"content": "DKA symptoms in children include polyuria, polydipsia, and vomiting.", "similarity": 0.80},
            {"content": "DKA diagnosis requires blood glucose > 200 mg/dL and venous pH < 7.3.", "similarity": 0.72},
        ]
        result = check.evaluate("What are the clinical symptoms and diagnostic criteria for DKA in children?", hits)

        assert result.score == 0.95
        assert result.details["passages_used"] == 2
        mock_backend.evaluate.assert_called_once_with(
            "What are the clinical symptoms and diagnostic criteria for DKA in children?",
            [
                "DKA symptoms in children include polyuria, polydipsia, and vomiting.",
                "DKA diagnosis requires blood glucose > 200 mg/dL and venous pH < 7.3.",
            ]
        )

    def test_three_complementary_passages(self):
        """Three passages collectively answering a multifaceted question."""
        check = LLMJudgeCheck()
        mock_backend = MagicMock()
        mock_backend.model = "mock-model"
        mock_backend.evaluate.return_value = (1.0, "Passages 1, 2, and 3 together cover screening, diagnosis, and treatment.")
        check._backend = mock_backend

        hits = [
            {"content": "Celiac screening in T1D starts with tTG-IgA testing at diagnosis.", "similarity": 0.82},
            {"content": "If tTG-IgA is elevated, confirmation requires intestinal biopsy or high-titer IgA panel.", "similarity": 0.74},
            {"content": "Management of confirmed celiac disease in children consists of a strict gluten-free diet.", "similarity": 0.68},
        ]
        result = check.evaluate("How is celiac disease screened, confirmed, and managed in children with T1D?", hits)

        assert result.score == 1.0
        assert result.details["passages_used"] == 3
        mock_backend.evaluate.assert_called_once_with(
            "How is celiac disease screened, confirmed, and managed in children with T1D?",
            [h["content"] for h in hits]
        )

    def test_three_passages_correct_topic_but_lacking_answer(self):
        """Passages discussing the right topic (DKA/diabetes) without answering the exact question."""
        check = LLMJudgeCheck()
        mock_backend = MagicMock()
        mock_backend.model = "mock-model"
        mock_backend.evaluate.return_value = (0.15, "Passages discuss DKA prevalence but do not state the fluid replacement rate.")
        check._backend = mock_backend

        hits = [
            {"content": "Trends in DKA prevalence show 30% of new T1D diagnoses present in ketoacidosis.", "similarity": 0.73},
            {"content": "DKA prevalence is higher in ethnic minority youth and families without insurance.", "similarity": 0.70},
            {"content": "Temporal trends remained stable between 2002 and 2010 in the SEARCH study.", "similarity": 0.67},
        ]
        result = check.evaluate("What is the initial intravenous fluid infusion rate in ml/kg/h for pediatric DKA resuscitation?", hits)

        assert result.score == 0.15
        assert result.details["passages_used"] == 3
        assert result.score < config.COMBINED_THRESHOLD

    def test_relevant_passage_with_irrelevant_passages(self):
        """One highly relevant passage accompanied by two irrelevant distractors retains high sufficiency."""
        check = LLMJudgeCheck()
        mock_backend = MagicMock()
        mock_backend.model = "mock-model"
        mock_backend.evaluate.return_value = (0.90, "Passage 1 answers the question completely; Passages 2 and 3 are irrelevant.")
        check._backend = mock_backend

        hits = [
            {"content": "Canagliflozin reduced HbA1c by 0.8% in adolescents with type 2 diabetes.", "similarity": 0.78},
            {"content": "Parking garage fees at Royal Children's Hospital averaged 15 dollars.", "similarity": 0.45},
            {"content": "Bavarian nurse educators wear blue scrubs instead of green.", "similarity": 0.40},
        ]
        result = check.evaluate("What was the glycemic effect of canagliflozin in adolescents with type 2 diabetes?", hits)

        assert result.score == 0.90
        assert result.details["passages_used"] == 3

    def test_contradictory_evidence_across_passages(self):
        """Conflicting clinical recommendations lower the sufficiency score."""
        check = LLMJudgeCheck()
        mock_backend = MagicMock()
        mock_backend.model = "mock-model"
        mock_backend.evaluate.return_value = (0.30, "Passages present conflicting A1C targets: Passage 1 says <7.0% while Passage 2 says <7.5%.")
        check._backend = mock_backend

        hits = [
            {"content": "Guideline A recommends target A1C under 7.0% for all pediatric patients.", "similarity": 0.79},
            {"content": "Guideline B recommends target A1C under 7.5% due to severe hypoglycemia risks.", "similarity": 0.77},
        ]
        result = check.evaluate("What is the agreed consensus target A1C for children with diabetes?", hits)

        assert result.score == 0.30
        assert "conflicting" in result.details["reason"].lower()

    def test_empty_or_missing_passages_raises(self):
        """Empty hits list or hits with empty content raises LLMJudgeError."""
        check = LLMJudgeCheck()
        with pytest.raises(LLMJudgeError, match="No passages retrieved"):
            check.evaluate("Question?", [])

        with pytest.raises(LLMJudgeError, match="All retrieved passages are empty"):
            check.evaluate("Question?", [{"content": ""}, {"content": "   "}])

    def test_groq_backend_receives_multi_passage_prompt(self):
        """_GroqJudge formats all passages and passes them to the Groq client."""
        judge = _GroqJudge()
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [
            MagicMock(message=MagicMock(content='{"sufficiency_score": 0.88, "reason": "All 3 passages evaluated."}'))
        ]
        mock_client.chat.completions.create.return_value = mock_completion
        judge._client = mock_client

        passages = ["Evidence chunk 1", "Evidence chunk 2", "Evidence chunk 3"]
        score, reason = judge.evaluate("Clinical question?", passages)

        assert score == 0.88
        assert reason == "All 3 passages evaluated."
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]["messages"]
        prompt_content = messages[0]["content"]
        assert "--- Retrieved Passage 1 ---\nEvidence chunk 1" in prompt_content
        assert "--- Retrieved Passage 2 ---\nEvidence chunk 2" in prompt_content
        assert "--- Retrieved Passage 3 ---\nEvidence chunk 3" in prompt_content

    def test_gemini_backend_receives_multi_passage_prompt(self):
        """_GeminiJudge formats all passages and passes them to the Gemini client."""
        judge = _GeminiJudge()
        mock_client = MagicMock()
        mock_response = MagicMock(text='{"sufficiency_score": 0.92, "reason": "Multi-passage evidence sufficient."}')
        mock_client.models.generate_content.return_value = mock_response
        judge._client = mock_client

        passages = ["Gemini passage 1", "Gemini passage 2"]
        score, reason = judge.evaluate("Gemini question?", passages)

        assert score == 0.92
        assert reason == "Multi-passage evidence sufficient."
        call_args = mock_client.models.generate_content.call_args
        prompt_content = call_args[1]["contents"]
        assert "--- Retrieved Passage 1 ---\nGemini passage 1" in prompt_content
        assert "--- Retrieved Passage 2 ---\nGemini passage 2" in prompt_content


# ============================================================
# Integration-style test (requires real Model C checkpoint)
# ============================================================

class TestRealModelCIntegration:
    """Integration test with real Model C (skipped if checkpoint missing or torch fails)."""

    @pytest.fixture(autouse=True)
    def check_checkpoint(self):
        """Skip if Model C checkpoint not available."""
        if not config.MODEL_C_CHECKPOINT.exists():
            pytest.skip("Model C checkpoint not found")

    def test_real_model_c_loads_and_predicts(self, sample_hits):
        """Real Model C loads and returns valid probability."""
        pytest.skip("Requires torch DLL working - run manually in working environment")

    def test_real_engine_produces_decision(self, sample_hits):
        """Real engine produces a valid sufficiency decision."""
        pytest.skip("Requires torch DLL working - run manually in working environment")


# ============================================================
# Run tests when executed directly
# ============================================================

if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))