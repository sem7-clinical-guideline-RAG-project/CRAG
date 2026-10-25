"""
Verification Script: MedCPT Embedding Storage

Verifies that MedCPT Article Encoder embeddings
are correctly stored in ChromaDB.

Does NOT implement:
  - MedCPT Query Encoder
  - RAG retrieval pipeline
  - LLM generation
  - Reranking / Cross Encoder
"""

import sys
import io

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8"
)

import os
import chromadb
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch


def verify():

    print("=" * 55)
    print("MedCPT Embedding Storage Verification")
    print("=" * 55)

    issues = []
    passed = []


    # ------------------------------------------
    # CHECK 1: MedCPT Article Encoder loads
    # ------------------------------------------

    print("\n[Check 1] MedCPT Article Encoder...")

    model_name = "ncbi/MedCPT-Article-Encoder"

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )
        model = AutoModel.from_pretrained(
            model_name
        )
        model.eval()

        passed.append(
            "MedCPT Article Encoder loaded"
        )
        print("  PASSED: Model loaded.")

    except Exception as e:
        issues.append(
            f"Cannot load MedCPT Article Encoder: {e}"
        )
        print(f"  FAILED: {e}")
        # Cannot continue without model
        print_final(passed, issues)
        return


    # ------------------------------------------
    # CHECK 2: ChromaDB persistent at ./vector_db
    # ------------------------------------------

    print("\n[Check 2] ChromaDB persistence...")

    db_path = "./vector_db"

    if os.path.isdir(db_path):
        passed.append(
            f"Persistent DB exists at {db_path}"
        )
        print(f"  PASSED: {db_path} exists.")
    else:
        issues.append(
            f"DB directory not found: {db_path}"
        )
        print(f"  FAILED: {db_path} not found.")
        print_final(passed, issues)
        return

    sqlite_path = os.path.join(
        db_path, "chroma.sqlite3"
    )

    if os.path.isfile(sqlite_path):
        size_kb = (
            os.path.getsize(sqlite_path) / 1024
        )
        passed.append(
            f"chroma.sqlite3 exists ({size_kb:.0f} KB)"
        )
        print(
            f"  PASSED: chroma.sqlite3 "
            f"({size_kb:.0f} KB)."
        )
    else:
        issues.append("chroma.sqlite3 not found.")
        print("  FAILED: chroma.sqlite3 missing.")


    # ------------------------------------------
    # CHECK 3: Connect and access collection
    # ------------------------------------------

    print("\n[Check 3] Collection: research_abstracts...")

    try:
        client = chromadb.PersistentClient(
            path=db_path
        )

        collection = client.get_collection(
            name="research_abstracts"
        )

        passed.append(
            "Collection 'research_abstracts' accessible"
        )
        print("  PASSED: Collection found.")

    except Exception as e:
        issues.append(
            f"Collection not found: {e}"
        )
        print(f"  FAILED: {e}")
        print_final(passed, issues)
        return


    # ------------------------------------------
    # CHECK 4: Document and embedding counts
    # ------------------------------------------

    print("\n[Check 4] Document and embedding counts...")

    total_docs = collection.count()

    if total_docs > 0:
        passed.append(
            f"{total_docs} documents stored"
        )
        print(
            f"  PASSED: {total_docs} documents stored."
        )
    else:
        issues.append("No documents in collection.")
        print("  FAILED: Collection is empty.")
        print_final(passed, issues)
        return

    # Fetch all data
    all_data = collection.get(
        include=[
            "documents",
            "metadatas",
            "embeddings",
        ]
    )

    ids = all_data["ids"]
    documents = all_data["documents"]
    metadatas = all_data["metadatas"]
    embeddings = all_data["embeddings"]

    num_embeddings = sum(
        1 for e in embeddings if e is not None
    )

    if num_embeddings == total_docs:
        passed.append(
            f"{num_embeddings} embeddings stored "
            f"(matches document count)"
        )
        print(
            f"  PASSED: {num_embeddings} embeddings "
            f"match {total_docs} documents."
        )
    else:
        issues.append(
            f"Embedding count ({num_embeddings}) != "
            f"document count ({total_docs})"
        )
        print(
            f"  FAILED: {num_embeddings} embeddings "
            f"vs {total_docs} documents."
        )


    # ------------------------------------------
    # CHECK 5: All embeddings have dimension 768
    # ------------------------------------------

    print("\n[Check 5] Embedding dimension = 768...")

    wrong_dims = []

    for i, emb in enumerate(embeddings):
        if emb is None:
            wrong_dims.append(
                (ids[i], "None")
            )
        elif len(emb) != 768:
            wrong_dims.append(
                (ids[i], len(emb))
            )

    if len(wrong_dims) == 0:
        passed.append(
            "All embeddings have dimension 768"
        )
        print(
            f"  PASSED: All {total_docs} embeddings "
            f"are 768-dimensional."
        )
    else:
        for doc_id, dim in wrong_dims:
            issues.append(
                f"Wrong dim for {doc_id}: {dim}"
            )
            print(
                f"  FAILED: {doc_id} has dim {dim}."
            )


    # ------------------------------------------
    # CHECK 6: Unique IDs
    # ------------------------------------------

    print("\n[Check 6] Unique IDs...")

    unique_ids = set(ids)

    if len(unique_ids) == len(ids):
        passed.append(
            f"All {len(ids)} IDs are unique"
        )
        print(
            f"  PASSED: {len(ids)} unique IDs."
        )
    else:
        duplicates = len(ids) - len(unique_ids)
        issues.append(
            f"{duplicates} duplicate IDs found"
        )
        print(
            f"  FAILED: {duplicates} duplicate IDs."
        )


    # ------------------------------------------
    # CHECK 7: Every document has required
    # metadata (source, chunk_index)
    # ------------------------------------------

    print("\n[Check 7] Metadata fields...")

    missing_meta = []

    for i, meta in enumerate(metadatas):

        missing_fields = []

        if "source" not in meta:
            missing_fields.append("source")

        if "chunk_index" not in meta:
            missing_fields.append("chunk_index")

        if missing_fields:
            missing_meta.append(
                (ids[i], missing_fields)
            )

    if len(missing_meta) == 0:
        passed.append(
            "All documents have 'source' and "
            "'chunk_index' metadata"
        )
        print(
            "  PASSED: All documents have "
            "'source' and 'chunk_index'."
        )
    else:
        for doc_id, fields in missing_meta:
            issues.append(
                f"{doc_id} missing: {fields}"
            )
            print(
                f"  FAILED: {doc_id} missing "
                f"{fields}."
            )


    # ------------------------------------------
    # CHECK 8: Every document has chunk text
    # ------------------------------------------

    print("\n[Check 8] Document text present...")

    empty_docs = []

    for i, doc in enumerate(documents):
        if not doc or len(doc.strip()) == 0:
            empty_docs.append(ids[i])

    if len(empty_docs) == 0:
        passed.append(
            "All documents have non-empty text"
        )
        print("  PASSED: All documents have text.")
    else:
        for doc_id in empty_docs:
            issues.append(
                f"Empty document: {doc_id}"
            )
            print(
                f"  FAILED: Empty doc: {doc_id}."
            )


    # ------------------------------------------
    # CHECK 9: Sample document inspection
    # ------------------------------------------

    print("\n[Check 9] Sample document...")

    sample_idx = 0
    sample_id = ids[sample_idx]
    sample_doc = documents[sample_idx]
    sample_meta = metadatas[sample_idx]
    sample_emb = embeddings[sample_idx]
    sample_emb_arr = np.array(sample_emb)

    print(f"  ID:        {sample_id}")
    print(f"  Source:    {sample_meta.get('source')}")
    print(
        f"  Chunk:     {sample_meta.get('chunk_index')}"
    )
    print(f"  Text len:  {len(sample_doc)} chars")
    print(
        f"  Text:      {sample_doc[:120]}..."
    )
    print(f"  Emb dim:   {len(sample_emb)}")
    print(
        f"  Emb first 5: "
        f"{sample_emb[:5]}"
    )
    print(
        f"  Emb norm:  {np.linalg.norm(sample_emb_arr):.4f}"
    )

    passed.append("Sample document inspected")


    # ------------------------------------------
    # CHECK 10: Similarity search test
    # (Using MedCPT Article Encoder ONLY
    #  — NOT the Query Encoder)
    # ------------------------------------------

    print("\n[Check 10] Similarity search test...")
    print(
        "  (Using MedCPT Article Encoder "
        "as temporary query encoder)"
    )

    try:
        # Pick the first document's text
        # as the query to search for itself
        query_text = documents[0]

        # Generate embedding using Article Encoder
        encoded = tokenizer(
            [["", query_text]],
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        )

        with torch.no_grad():
            output = model(**encoded)

        query_emb = (
            output.last_hidden_state[:, 0, :]
            .squeeze()
            .numpy()
            .tolist()
        )

        # Run similarity search
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=3,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        top_id = results["ids"][0][0]
        top_dist = results["distances"][0][0]
        top_meta = results["metadatas"][0][0]

        print(f"  Query: first stored document")
        print(f"  Top result ID: {top_id}")
        print(f"  Top distance:  {top_dist:.6f}")
        print(
            f"  Top source:    "
            f"{top_meta.get('source')}"
        )

        # The top result should be the same
        # document (distance ≈ 0)
        if top_id == ids[0] and top_dist < 0.01:
            passed.append(
                "Similarity search returned "
                "correct top match (self)"
            )
            print(
                "  PASSED: Top match is the query "
                "document itself."
            )
        else:
            issues.append(
                f"Unexpected top result: {top_id} "
                f"(distance {top_dist})"
            )
            print(
                "  WARNING: Top match is not "
                "the query document."
            )

        # Show top 3 results
        print("\n  Top 3 results:")

        for rank in range(
            len(results["ids"][0])
        ):
            r_id = results["ids"][0][rank]
            r_dist = results["distances"][0][rank]
            r_source = (
                results["metadatas"][0][rank]
                .get("source", "N/A")
            )

            print(
                f"    {rank + 1}. {r_source} "
                f"(distance: {r_dist:.6f})"
            )

    except Exception as e:
        issues.append(
            f"Similarity search failed: {e}"
        )
        print(f"  FAILED: {e}")


    # ------------------------------------------
    # Final Summary
    # ------------------------------------------

    print_final(passed, issues)

    # Print the requested summary block
    print("\n" + "-" * 55)
    print("VERIFICATION SUMMARY")
    print("-" * 55)

    sim_ok = any(
        "Similarity search" in p for p in passed
    )

    print(
        f"  Collection: research_abstracts"
    )
    print(
        f"  Documents stored: {total_docs}"
    )
    print(
        f"  Embeddings stored: {num_embeddings}"
    )
    print(
        f"  Embedding dimension: 768"
    )
    print(
        f"  Sample ID: {ids[0]}"
    )
    print(
        f"  Sample metadata: {metadatas[0]}"
    )
    print(
        f"  Similarity search successful: "
        f"{'Yes' if sim_ok else 'No'}"
    )

    print("-" * 55)


def print_final(passed, issues):
    """Print pass/fail summary."""

    print("\n" + "=" * 55)

    if len(issues) == 0:
        print(
            f"ALL {len(passed)} CHECKS PASSED"
        )
    else:
        print(
            f"{len(passed)} PASSED, "
            f"{len(issues)} ISSUES"
        )

    print("=" * 55)

    if passed:
        print("\nPassed:")
        for p in passed:
            print(f"  [OK] {p}")

    if issues:
        print("\nIssues:")
        for issue in issues:
            print(f"  [!!] {issue}")


if __name__ == "__main__":
    verify()
