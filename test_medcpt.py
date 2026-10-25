"""
Test script to verify MedCPT Article Encoder
integration with ChromaDB.

Verifies:
  1. MedCPT Article Encoder loads successfully
  2. A sample biomedical chunk produces an embedding
  3. The embedding dimension is 768
  4. The embedding can be inserted into ChromaDB
  5. The stored document can be retrieved
"""

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import chromadb


def test_medcpt_integration():

    print("=" * 50)
    print("MedCPT Article Encoder Integration Test")
    print("=" * 50)

    all_passed = True


    # ------------------------------------------
    # Test 1: Load MedCPT Article Encoder
    # ------------------------------------------

    print("\n[Test 1] Loading MedCPT Article Encoder...")

    try:
        model_name = "ncbi/MedCPT-Article-Encoder"

        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        model = AutoModel.from_pretrained(
            model_name
        )

        model.eval()

        print("  PASSED: Model loaded successfully.")

    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
        return  # Cannot continue without model


    # ------------------------------------------
    # Test 2: Generate embedding from sample
    # biomedical text
    # ------------------------------------------

    print("\n[Test 2] Generating embedding for a "
          "sample biomedical chunk...")

    sample_text = (
        "This study evaluates the effect of "
        "continuous glucose monitoring on glycemic "
        "control in adolescents with type 1 diabetes. "
        "Results showed significant improvement in "
        "HbA1c levels over 6 months of CGM use "
        "compared to the control group using "
        "traditional blood glucose monitoring."
    )

    try:
        # Tokenize with title + abstract pair
        # (empty title since we only have abstract)
        encoded = tokenizer(
            [["", sample_text]],
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=512,
        )

        with torch.no_grad():
            output = model(**encoded)

        # CLS token representation
        embedding = (
            output.last_hidden_state[:, 0, :]
        )

        embedding_list = (
            embedding.squeeze().numpy().tolist()
        )

        print(
            f"  Embedding type: {type(embedding_list)}"
        )

        print(
            f"  Embedding length: {len(embedding_list)}"
        )

        print(
            f"  First 5 values: "
            f"{embedding_list[:5]}"
        )

        print("  PASSED: Embedding generated.")

    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False
        return


    # ------------------------------------------
    # Test 3: Verify embedding dimension is 768
    # ------------------------------------------

    print("\n[Test 3] Verifying embedding dimension...")

    if len(embedding_list) == 768:
        print("  PASSED: Dimension is 768.")
    else:
        print(
            f"  FAILED: Expected 768, "
            f"got {len(embedding_list)}."
        )
        all_passed = False


    # ------------------------------------------
    # Test 4: Insert embedding into ChromaDB
    # ------------------------------------------

    print("\n[Test 4] Inserting embedding into "
          "ChromaDB...")

    try:
        # Use a temporary in-memory client for testing
        # so we don't pollute the production vector_db
        test_client = chromadb.Client()

        test_collection = (
            test_client.get_or_create_collection(
                name="test_medcpt",
                metadata={"hnsw:space": "cosine"}
            )
        )

        test_collection.upsert(
            ids=["test_chunk_0"],
            embeddings=[embedding_list],
            documents=[sample_text],
            metadatas=[{
                "source": "test_sample.txt",
                "chunk_index": 0,
            }],
        )

        count = test_collection.count()
        print(f"  Documents in collection: {count}")

        if count == 1:
            print("  PASSED: Embedding stored.")
        else:
            print("  FAILED: Unexpected count.")
            all_passed = False

    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False


    # ------------------------------------------
    # Test 5: Retrieve the stored document
    # ------------------------------------------

    print("\n[Test 5] Retrieving stored document...")

    try:
        result = test_collection.get(
            ids=["test_chunk_0"],
            include=[
                "documents",
                "metadatas",
                "embeddings",
            ],
        )

        retrieved_doc = result["documents"][0]
        retrieved_meta = result["metadatas"][0]
        retrieved_emb = result["embeddings"][0]

        print(
            f"  Retrieved document (first 80 chars): "
            f"{retrieved_doc[:80]}..."
        )

        print(
            f"  Retrieved metadata: {retrieved_meta}"
        )

        print(
            f"  Retrieved embedding dimension: "
            f"{len(retrieved_emb)}"
        )

        # Verify retrieved data matches
        doc_match = (retrieved_doc == sample_text)
        meta_match = (
            retrieved_meta["source"]
            == "test_sample.txt"
        )
        emb_match = (len(retrieved_emb) == 768)

        if doc_match and meta_match and emb_match:
            print("  PASSED: Retrieved data matches.")
        else:
            print("  FAILED: Data mismatch.")
            all_passed = False

    except Exception as e:
        print(f"  FAILED: {e}")
        all_passed = False


    # ------------------------------------------
    # Final Result
    # ------------------------------------------

    print("\n" + "=" * 50)

    if all_passed:
        print( 
            "ALL TESTS PASSED - MedCPT integration "
            "is working correctly!"
        )
    else:
        print(
            "SOME TESTS FAILED - Review the output "
            "above for details."
        )

    print("=" * 50)


if __name__ == "__main__":
    test_medcpt_integration()
