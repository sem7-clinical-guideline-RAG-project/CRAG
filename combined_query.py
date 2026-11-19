"""
combined_query.py — Combined RAG Evidence Sufficiency Detector

Combines Model C's evidence sufficiency classification score with the
cosine similarity score of passage 1 from retrieval to make a final
sufficiency decision for the Diabetes RAG pipeline.

Usage:
    python combined_query.py
    python combined_query.py --combined-threshold 0.75
"""

import argparse
import os
import sys
from pathlib import Path

import requests
import torch
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from chatapp.clinical_rag import retrieval
from classification.common.data_loader import (
    LABEL_MAP,
    format_dataset_b_input,
    format_multi_passage_input,
)

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"

MODEL_C_CHECKPOINT = Path(__file__).parent / "classification" / "model_c_v2" / "checkpoint" / "best"
MODEL_C_MAX_LENGTH = 512

# Thresholds and weights (easily configurable here)
DEFAULT_COMBINED_THRESHOLD = 0.70
MODEL_C_THRESHOLD = 0.50
COSINE_THRESHOLD = 0.70

MODEL_C_WEIGHT = 0.15
COSINE_WEIGHT = 0.85

LABEL_NAMES = {0: "insufficient", 1: "sufficient"}


# ============================================================
# STEP 1: Load Model C (Evidence Sufficiency Classifier)
# ============================================================

print(f"Loading Model C from {MODEL_C_CHECKPOINT}...")

if not MODEL_C_CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Model C checkpoint not found at {MODEL_C_CHECKPOINT}. "
        f"Train Model C first (python classification/model_c/train.py)."
    )

_model_c_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_C_CHECKPOINT))
_model_c_model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_C_CHECKPOINT))

_model_c_device = "cuda" if torch.cuda.is_available() else "cpu"
_model_c_model.to(_model_c_device)
_model_c_model.eval()

print(f"  Model C loaded on {_model_c_device}")
print(f"  Max length: {MODEL_C_MAX_LENGTH}")
print(f"  Label mapping: {LABEL_MAP}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def search_knowledge_base(question: str, top_k: int = 3):
    """Search knowledge base using existing RAG retrieval (ChromaDB + MedCPT)."""
    hits = retrieval.search(question, top_k=top_k)
    return {
        "ids": [[h["id"] for h in hits]],
        "documents": [[h["content"] for h in hits]],
        "metadatas": [[h["metadata"] for h in hits]],
        "distances": [[h["distance"] for h in hits]],
    }


def display_retrieved_passages(question: str, results: dict):
    """Pretty-print retrieved passages."""
    print("=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    if not ids:
        print("\n  (No passages retrieved)")
        print("=" * 70)
        return

    for rank, (doc_id, doc, meta, dist) in enumerate(
        zip(ids, docs, metas, distances), start=1
    ):
        similarity = 1 - dist
        source = meta.get("source", "Unknown")

        print(f"\n--- Retrieved Passage {rank} ---")
        print(f"  Source:     {source}")
        print(f"  Similarity: {similarity:.4f}")
        print("  Content:")
        try:
            print(f"    {doc}")
        except UnicodeEncodeError:
            print(f"    {doc.encode('ascii', 'replace').decode('ascii')}")

    print("\n" + "=" * 70)


def classify_evidence_sufficiency(
    question: str,
    passages: list,
    passages_mode: int = 3,
) -> dict:
    """Classify evidence sufficiency using Model C."""
    passage_1 = passages[0] if len(passages) > 0 else ""
    passage_2 = passages[1] if len(passages) > 1 else ""
    passage_3 = passages[2] if len(passages) > 2 else ""

    if passages_mode == 1:
        # Legacy single-passage format (passage_1 only via format_dataset_b_input)
        row = {
            "question": question,
            "passage_1": passage_1,
            "passage_2": passage_2,
            "passage_3": passage_3,
        }
        formatted_text = format_dataset_b_input(row, max_length=MODEL_C_MAX_LENGTH)
    else:
        # 3-passage mode matching Model C V2 training data format
        formatted_text = format_multi_passage_input(
            question=question,
            passages=passages,
            max_length=MODEL_C_MAX_LENGTH,
        )

    inputs = _model_c_tokenizer(
        formatted_text,
        padding="max_length",
        truncation=True,
        max_length=MODEL_C_MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {k: v.to(_model_c_device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = _model_c_model(**inputs)

    logits = outputs.logits.cpu()
    probs = torch.softmax(logits, dim=-1)[0].numpy()

    return {
        "scores": {
            "insufficient": round(float(probs[0]), 4),
            "sufficient": round(float(probs[1]), 4),
        },
        "formatted_text": formatted_text,
    }


def generate_groq_response(question: str, passages: list, api_key: str) -> str:
    """Generate answer using Groq API."""
    if not api_key or api_key == "YOUR_GROQ_API_KEY_HERE":
        return "\n[!] Groq API key not set. Skipping LLM generation."

    context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(passages)])

    system_prompt = (
        "You are a helpful medical assistant specializing in diabetes for children and adolescents. "
        "Use ONLY the provided context to answer the user's question. "
        "If the answer cannot be found in the context, say 'I cannot find the answer in the provided documents.'"
    )

    user_prompt = "Context information is below.\n---------------------\n" + context + "\n---------------------\n\nQuestion: " + question + "\nAnswer:"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
    }

    try:
        response = requests.post(GROQ_ENDPOINT, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        return f"Groq API HTTP error: {e}\nDetails: {e.response.text}"
    except requests.exceptions.Timeout:
        return "Groq API request timed out (30s)."
    except Exception as e:
        return f"Groq API error: {e}"


# ============================================================
# MAIN QUERY LOOP
# ============================================================

def run_query_loop(combined_threshold: float, passages_mode: int = 3):
    """Interactive query loop using combined sufficiency scoring."""
    print("\n" + "=" * 70)
    print("DIABETES RAG — COMBINED SUFFICIENCY QUERY")
    print("=" * 70)
    print(f"  Combined threshold:  {combined_threshold:.2f}")
    print(f"  Passages mode:       {passages_mode} passage(s)")
    print(f"  Model C weight:      {MODEL_C_WEIGHT}")
    print(f"  Cosine weight:       {COSINE_WEIGHT}")
    print(f"  Model C:             {MODEL_C_CHECKPOINT}")
    print(f"  Groq model:          {GROQ_MODEL}")
    print("=" * 70)
    print("Type 'quit', 'exit', or 'q' to stop.\n")

    while True:
        try:
            question = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            print("  (Please enter a question)")
            continue

        if question.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        # Step 1: Retrieve passages
        print("\n[1/3] Retrieving passages from knowledge base...")
        try:
            results = search_knowledge_base(question, top_k=3)
            similarity_scores = {
                f"passage_{i+1}": round(1 - distance, 4)
                for i, distance in enumerate(results["distances"][0])
            }
        except Exception as e:
            print(f"  Retrieval failed: {e}")
            print("\n" + "=" * 70)
            print("ABSTENTION: Evidence retrieval failed.")
            print("-" * 70)
            print("No answer generated due to retrieval error.")
            print("=" * 70)
            continue

        display_retrieved_passages(question, results)

        docs = results["documents"][0]
        distances = results["distances"][0]

        if not docs or not docs[0]:
            print("\n" + "=" * 70)
            print("ABSTENTION: No passages retrieved.")
            print("-" * 70)
            print("No answer generated.")
            print("=" * 70)
            continue

        # Step 2: Classify evidence sufficiency with Model C
        print("[2/3] Classifying evidence sufficiency with Model C...")
        try:
            classification = classify_evidence_sufficiency(
                question=question,
                passages=docs,
                passages_mode=passages_mode,
            )
            classification_scores = classification["scores"]
        except Exception as e:
            print(f"  Classification failed: {e}")
            print("\n" + "=" * 70)
            print("ABSTENTION: Evidence classification failed.")
            print("-" * 70)
            print("No answer generated due to classification error.")
            print("=" * 70)
            continue

        # Step 3: Calculate combined score and evaluate gate

        clamped_sims = [
            max(0.0, min(1.0, 1.0 - float(dist)))
            for dist in distances[:3]
        ]

        if passages_mode == 1:
            # Preserve legacy single-passage behavior
            active_cosine = clamped_sims[0] if clamped_sims else 0.0

        else:
            # Weighted cosine: P1 > P2 > P3
            weights = [1.0, 0.5, 0.25][:len(clamped_sims)]

            active_cosine = (
                sum(score * weight for score, weight in zip(clamped_sims, weights))
                / sum(weights)
                if weights else 0.0
            )
        classification_contribution = MODEL_C_WEIGHT * classification_scores["sufficient"]
        cosine_contribution = COSINE_WEIGHT * active_cosine

        combined_score = round(classification_contribution + cosine_contribution, 4)

        model_c_label = int(classification_scores["sufficient"] >= MODEL_C_THRESHOLD)
        model_c_decision = LABEL_NAMES[model_c_label].upper()

        cosine_label = int(active_cosine >= COSINE_THRESHOLD)
        cosine_decision = "SUFFICIENT" if cosine_label == 1 else "INSUFFICIENT"

        print("\n" + "-" * 50)
        print("EVIDENCE SUFFICIENCY EVALUATION & COMBINED SCORE")
        print("-" * 50)
        print("A. Model C Classification:")
        if passages_mode == 1:
            print("  Model C input:            Passage 1 only (legacy format)")
        else:
            print(f"  Model C input:            Passages 1 to {len(docs)} (training format)")
        print(f"  Insufficient probability: {classification_scores['insufficient']:.4f}")
        print(f"  Sufficient probability:   {classification_scores['sufficient']:.4f}")
        print(f"  Model C Decision:         {model_c_decision}")

        print("\nB. Retrieval Cosine Similarity:")
        for idx_hit, sim_val in enumerate(clamped_sims, start=1):
            flag = "" if passages_mode == 3 else ("" if idx_hit == 1 else " (info only)")
            print(f"  Passage {idx_hit} similarity:     {sim_val:.4f}{flag}")
        if passages_mode == 3:
            print(f"  Weighted Cosine Similarity: {active_cosine:.4f} (used in gate, n={len(clamped_sims)})")
        print(f"  Cosine Decision:          {cosine_decision}")

        print("\nC. Combined Decision:")
        print(f"  Classification score (w={MODEL_C_WEIGHT}): {round(classification_contribution, 4):.4f}")
        print(f"  Cosine score (w={COSINE_WEIGHT}):         {round(cosine_contribution, 4):.4f}")
        print(f"  Final Combined Score:              {combined_score:.4f}")
        print(f"  Combined Threshold:                {combined_threshold:.2f}")

        will_proceed = combined_score >= combined_threshold
        combined_decision = "SUFFICIENT" if will_proceed else "INSUFFICIENT"
        print(f"  Final Combined Decision:           {combined_decision}")
        print("-" * 50)

        # Step 4: Generate or abstain
        if will_proceed:
            print(f"\n[3/3] Combined score ({combined_score:.4f}) >= threshold ({combined_threshold:.2f}). PASSED sufficiency check.")
            print("Generating answer with Groq...")
            answer = generate_groq_response(question, docs, GROQ_API_KEY)
            print("\n" + "=" * 70)
            print("LLM ANSWER (RAG):")
            print("-" * 70)
            print(answer)
            print("=" * 70)
        else:
            print("\n" + "=" * 70)
            print("ABSTENTION: Evidence was not sufficient.")
            print("-" * 70)
            print("The combined sufficiency score is below the required threshold.")
            print("The retrieved passages were not considered sufficient to answer the question.")
            print("No LLM answer was generated.")
            print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Combined RAG Evidence Sufficiency Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--combined-threshold",
        type=float,
        default=DEFAULT_COMBINED_THRESHOLD,
        help=f"Combined threshold for sufficiency. Default: {DEFAULT_COMBINED_THRESHOLD}",
    )
    parser.add_argument(
        "--passages",
        type=int,
        choices=[1, 3],
        default=3,
        help="Number of retrieved passages to use in sufficiency gate: 1 (legacy passage 1 only) or 3 (all retrieved passages, default 3).",
    )
    args = parser.parse_args()

    if not 0.0 <= args.combined_threshold <= 1.0:
        print(f"Error: Combined threshold must be between 0.0 and 1.0, got {args.combined_threshold}")
        sys.exit(1)

    run_query_loop(combined_threshold=args.combined_threshold, passages_mode=args.passages)


if __name__ == "__main__":
    main()
