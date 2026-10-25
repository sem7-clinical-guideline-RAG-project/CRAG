"""
Inspect ChromaDB vector database contents.

Displays all stored documents, metadata,
and embedding details from the
'research_abstracts' collection.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8"
)

import chromadb
import numpy as np


def inspect_database():

    print("=" * 55)
    print("ChromaDB Data Inspector")
    print("=" * 55)


    # ------------------------------------------
    # Connect to persistent ChromaDB
    # ------------------------------------------

    client = chromadb.PersistentClient(
        path="./vector_db"
    )


    # ------------------------------------------
    # List all collections
    # ------------------------------------------

    collections = client.list_collections()

    print(f"\nCollections found: {len(collections)}")

    for col in collections:
        print(f"  - {col.name}")


    # ------------------------------------------
    # Access the research_abstracts collection
    # ------------------------------------------

    try:
        collection = client.get_collection(
            name="research_abstracts"
        )
    except Exception as e:
        print(f"\nERROR: Could not access collection: {e}")
        return

    total_docs = collection.count()

    print(f"\nTotal documents in collection: {total_docs}")


    if total_docs == 0:
        print("No documents found. Run ingest.py first.")
        return


    # ------------------------------------------
    # Fetch all data (documents, metadata,
    # embeddings)
    # ------------------------------------------

    results = collection.get(
        include=[
            "documents",
            "metadatas",
            "embeddings",
        ]
    )

    ids = results["ids"]
    documents = results["documents"]
    metadatas = results["metadatas"]
    embeddings = results["embeddings"]


    # ------------------------------------------
    # Display each document's details
    # ------------------------------------------

    print("\n" + "-" * 55)
    print("STORED DOCUMENTS")
    print("-" * 55)

    for i in range(len(ids)):

        print(f"\n{'='*55}")
        print(f"Document {i + 1} of {len(ids)}")
        print(f"{'='*55}")


        # --- ID ---
        print(f"\n  ID: {ids[i]}")


        # --- Metadata ---
        meta = metadatas[i]
        print(f"\n  Metadata:")

        for key, value in meta.items():
            print(f"    {key}: {value}")


        # --- Document text ---
        doc = documents[i]
        print(f"\n  Document length: {len(doc)} chars")
        print(f"\n  Document text (first 200 chars):")
        print(f"    {doc[:200]}...")


        # --- Full document text ---
        print(f"\n  Full document text:")

        # Word-wrap at 50 chars for readability
        words = doc.split()
        line = "    "

        for word in words:

            if len(line) + len(word) + 1 > 54:
                print(line)
                line = "    " + word
            else:
                line += " " + word if line.strip() else "    " + word

        if line.strip():
            print(line)


        # --- Embedding details ---
        emb = embeddings[i]
        emb_array = np.array(emb)

        print(f"\n  Embedding:")
        print(f"    Dimension: {len(emb)}")
        print(f"    First 10 values:")
        print(f"      {emb[:10]}")
        print(f"    Last 5 values:")
        print(f"      {emb[-5:]}")
        print(f"    Min value:  {emb_array.min():.6f}")
        print(f"    Max value:  {emb_array.max():.6f}")
        print(f"    Mean value: {emb_array.mean():.6f}")
        print(f"    Std dev:    {emb_array.std():.6f}")
        print(f"    Norm (L2):  {np.linalg.norm(emb_array):.6f}")


    # ------------------------------------------
    # Summary table
    # ------------------------------------------

    print(f"\n\n{'='*55}")
    print("SUMMARY TABLE")
    print(f"{'='*55}")

    print(
        f"\n  {'ID':<30} {'Source':<20} "
        f"{'Chunk':<6} {'Chars':<6} "
        f"{'Emb Dim':<8}"
    )

    print(f"  {'-'*30} {'-'*20} {'-'*6} {'-'*6} {'-'*8}")

    for i in range(len(ids)):

        doc_id = ids[i]
        source = metadatas[i].get("source", "N/A")
        chunk_idx = metadatas[i].get(
            "chunk_index", "N/A"
        )
        doc_len = len(documents[i])
        emb_dim = len(embeddings[i])

        print(
            f"  {doc_id:<30} {source:<20} "
            f"{chunk_idx:<6} {doc_len:<6} "
            f"{emb_dim:<8}"
        )


    print(f"\n{'='*55}")
    print("INSPECTION COMPLETE")
    print(f"{'='*55}")


if __name__ == "__main__":
    inspect_database()
