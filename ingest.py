import chromadb
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


# ==========================================
# STEP 1: Load MedCPT Article Encoder
# ==========================================

ARTICLE_MODEL_NAME = "ncbi/MedCPT-Article-Encoder"

tokenizer = AutoTokenizer.from_pretrained(
    ARTICLE_MODEL_NAME
)

model = AutoModel.from_pretrained(
    ARTICLE_MODEL_NAME
)

# Set model to evaluation mode
# (disables dropout, etc.)
model.eval()

print("MedCPT Article Encoder loaded successfully!")


# ==========================================
# STEP 2: Define Embedding Function
# ==========================================

def generate_embedding(text, title=""):
    """
    Generate a 768-dimensional MedCPT embedding
    for a document chunk.

    MedCPT Article Encoder expects title + abstract
    pairs. If no title is available, pass an empty
    string as the title.

    Args:
        text:  The document chunk (abstract text).
        title: The article title (empty string if
               not available).

    Returns:
        A list of 768 floats (the embedding).
    """

    # Tokenize the title + text pair
    # MedCPT expects a list of [title, abstract] pairs
    encoded = tokenizer(
        [[title, text]],
        truncation=True,
        padding=True,
        return_tensors="pt",
        max_length=512,
    )

    # Generate embedding without computing gradients
    # (inference only, not training)
    with torch.no_grad():
        output = model(**encoded)

    # Extract the CLS token representation
    # Shape: (1, 768) -> (768,)
    cls_embedding = output.last_hidden_state[:, 0, :]

    # Convert to a Python list for ChromaDB storage
    embedding_list = cls_embedding.squeeze().numpy().tolist()

    return embedding_list


# ==========================================
# STEP 3: Connect to ChromaDB
# ==========================================

client = chromadb.PersistentClient(
    path="./vector_db"
)

print("ChromaDB connected successfully!")


# ==========================================
# STEP 4: Create / Access Collection
# ==========================================

# Delete existing collection if it exists,
# because old embeddings (384-dim from
# all-MiniLM-L6-v2) are incompatible with
# new MedCPT embeddings (768-dim).
try:
    client.delete_collection(
        name="research_abstracts"
    )
    print("Deleted old collection (dimension mismatch).")
except Exception:
    pass

collection = client.get_or_create_collection(
    name="research_abstracts",
    metadata={"hnsw:space": "cosine"}
)

print("Collection created successfully!")


# ==========================================
# STEP 5: Find TXT Files
# ==========================================

KNOWLEDGE_BASE = Path("knowledge_base")

# Find all .txt files
txt_files = list(
    KNOWLEDGE_BASE.glob("*.txt")
)

print(
    f"Found {len(txt_files)} TXT files."
)


# ==========================================
# STEP 6: Clean Text
# ==========================================

def clean_text(text):

    # Remove whitespace from beginning
    # and end of the text
    text = text.strip()

    # Remove unnecessary multiple spaces,
    # tabs and line breaks
    text = " ".join(text.split())

    return text


# ==========================================
# STEP 7: Adaptive Chunking
# ==========================================

# Reduced from 3000 to 2000 characters to stay
# safely within MedCPT's 512-token input limit.
#
# Biomedical text averages ~4-5 characters per
# token. At 2000 characters, chunks will be
# approximately 400-500 tokens, fitting within
# the 512-token limit. The tokenizer's
# truncation=True acts as a safety net for any
# edge cases.
MAX_CHUNK_SIZE = 2000


def create_chunks(
    text,
    max_size=MAX_CHUNK_SIZE
):

    # --------------------------------------
    # Case 1: Short text
    # --------------------------------------

    # If the entire abstract is small enough,
    # keep it as one chunk.
    if len(text) <= max_size:

        return [text]


    # --------------------------------------
    # Case 2: Long text
    # --------------------------------------

    # Split the text into sentences.
    sentences = text.split(". ")


    chunks = []

    current_chunk = ""


    # Go through each sentence
    for sentence in sentences:

        sentence = sentence.strip()


        # Ignore empty sentences
        if not sentence:
            continue


        # Check whether the sentence can
        # fit into the current chunk
        if (
            len(current_chunk)
            + len(sentence)
            + 1
            <= max_size
        ):

            current_chunk += (
                sentence + ". "
            )


        else:

            # Store the current chunk
            if current_chunk:

                chunks.append(
                    current_chunk.strip()
                )


            # Start a new chunk
            current_chunk = (
                sentence + ". "
            )


    # --------------------------------------
    # Store the final chunk
    # --------------------------------------

    if current_chunk:

        chunks.append(
            current_chunk.strip()
        )


    return chunks


# ==========================================
# STEP 8: Process All TXT Files
# ==========================================

total_chunks_stored = 0
files_processed = 0
files_failed = 0

for file_path in txt_files:

    try:

        print(
            f"\nProcessing: {file_path.name}"
        )


        # --------------------------------------
        # Read the TXT file
        # --------------------------------------

        text = file_path.read_text(
            encoding="utf-8"
        )


        # --------------------------------------
        # Clean the text
        # --------------------------------------

        cleaned_text = clean_text(text)


        # --------------------------------------
        # Create chunks
        # --------------------------------------

        chunks = create_chunks(
            cleaned_text
        )

        print(
            f"  Number of chunks: {len(chunks)}"
        )


        # --------------------------------------
        # Generate embeddings and store
        # in ChromaDB
        # --------------------------------------

        # Use the filename stem (without .txt)
        # for deterministic IDs
        file_stem = file_path.stem

        chunk_ids = []
        chunk_embeddings = []
        chunk_documents = []
        chunk_metadatas = []

        for i, chunk in enumerate(chunks):

            # Generate MedCPT embedding
            # Using empty title since our TXT
            # files contain abstracts only.
            # To use titles later, pass:
            #   generate_embedding(chunk, title="...")
            embedding = generate_embedding(
                text=chunk,
                title=""
            )

            print(
                f"  Embedding generated for chunk {i}: "
                f"dimension = {len(embedding)}"
            )

            # Build deterministic ID
            chunk_id = f"{file_stem}_chunk_{i}"

            # Build metadata
            metadata = {
                "source": file_path.name,
                "chunk_index": i,
            }

            chunk_ids.append(chunk_id)
            chunk_embeddings.append(embedding)
            chunk_documents.append(chunk)
            chunk_metadatas.append(metadata)


        # Upsert all chunks for this file
        # at once (safe to rerun without
        # creating duplicates)
        collection.upsert(
            ids=chunk_ids,
            embeddings=chunk_embeddings,
            documents=chunk_documents,
            metadatas=chunk_metadatas,
        )

        print(
            f"  Stored {len(chunks)} chunk(s) "
            f"for {file_path.name}"
        )

        total_chunks_stored += len(chunks)
        files_processed += 1


    except Exception as e:

        print(
            f"  ERROR processing "
            f"{file_path.name}: {e}"
        )

        files_failed += 1


# ==========================================
# STEP 9: Summary
# ==========================================

print("\n==========================================")
print("INGESTION COMPLETE")
print("==========================================")
print(f"Files processed successfully: {files_processed}")
print(f"Files failed: {files_failed}")
print(f"Total chunks stored: {total_chunks_stored}")
print(
    f"Total documents in collection: "
    f"{collection.count()}"
)
print("==========================================")