"""
Chunk Token-Length Validation & Visualization

Validates that adaptive chunking keeps all chunks
within MedCPT's 512-token limit.

Workflow:
  1. Pull stored chunk texts from ChromaDB
  2. Tokenize with MedCPT (truncation=False)
  3. Compute token-length statistics
  4. Flag edge cases exceeding 512 tokens
  5. Plot token-length histogram
  6. Generate a markdown report
"""

import sys
import io

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding="utf-8"
)

import os
import chromadb
import numpy as np
from transformers import AutoTokenizer
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime


# ==========================================
# CONFIG
# ==========================================

CHROMA_DB_PATH = "./vector_db"
COLLECTION_NAME = "research_abstracts"
MEDCPT_MODEL = "ncbi/MedCPT-Article-Encoder"
MAX_TOKENS = 512
OUTPUT_DIR = "."
CHART_FILENAME = "token_length_distribution.png"
REPORT_FILENAME = "chunk_validation_report.md"


# ==========================================
# STEP 1: Load chunks from ChromaDB
# ==========================================

print("=" * 55)
print("Chunk Token-Length Validation")
print("=" * 55)

print("\n[Step 1] Loading chunks from ChromaDB...")

client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)

total_docs = collection.count()
print(f"  Documents in collection: {total_docs}")

results = collection.get(
    include=["documents", "metadatas"]
)

ids = results["ids"]
documents = results["documents"]
metadatas = results["metadatas"]

print(f"  Retrieved {len(ids)} chunks.")


# ==========================================
# STEP 2: Load MedCPT tokenizer
# ==========================================

print("\n[Step 2] Loading MedCPT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MEDCPT_MODEL
)

print("  Tokenizer loaded.")
print(
    f"  Model max length: "
    f"{tokenizer.model_max_length}"
)


# ==========================================
# STEP 3: Tokenize every chunk (no truncation)
# ==========================================

print("\n[Step 3] Tokenizing all chunks "
      "(truncation=False)...")

token_lengths = []
chunk_details = []

for i in range(len(ids)):

    chunk_id = ids[i]
    chunk_text = documents[i]
    meta = metadatas[i]

    # MedCPT expects [title, abstract] pairs.
    # We use empty title to match ingest.py behavior.
    encoded = tokenizer(
        [["", chunk_text]],
        truncation=False,
        padding=False,
        return_tensors="pt",
    )

    num_tokens = encoded["input_ids"].shape[1]

    token_lengths.append(num_tokens)

    chunk_details.append({
        "id": chunk_id,
        "token_length": num_tokens,
        "char_length": len(chunk_text),
        "source": meta.get("source", "N/A"),
        "chunk_index": meta.get("chunk_index", "N/A"),
        "text_snippet": chunk_text[:120],
    })

token_lengths = np.array(token_lengths)

print(f"  Tokenized {len(token_lengths)} chunks.")


# ==========================================
# STEP 4: Compute statistics
# ==========================================

print("\n[Step 4] Computing statistics...")

total_chunks = len(token_lengths)
min_tokens = int(np.min(token_lengths))
max_tokens = int(np.max(token_lengths))
mean_tokens = float(np.mean(token_lengths))
median_tokens = float(np.median(token_lengths))
std_tokens = float(np.std(token_lengths))

within_limit = int(
    np.sum(token_lengths <= MAX_TOKENS)
)
exceeding_limit = int(
    np.sum(token_lengths > MAX_TOKENS)
)

pct_within = (within_limit / total_chunks) * 100
pct_exceeding = (exceeding_limit / total_chunks) * 100

print(f"  Total chunks:    {total_chunks}")
print(f"  Min tokens:      {min_tokens}")
print(f"  Max tokens:      {max_tokens}")
print(f"  Mean tokens:     {mean_tokens:.1f}")
print(f"  Median tokens:   {median_tokens:.1f}")
print(f"  Std deviation:   {std_tokens:.1f}")
print(f"  Within 512:      {within_limit} "
      f"({pct_within:.1f}%)")
print(f"  Exceeding 512:   {exceeding_limit} "
      f"({pct_exceeding:.1f}%)")


# ==========================================
# STEP 5: Flag edge cases
# ==========================================

print("\n[Step 5] Flagging edge cases "
      f"(>{MAX_TOKENS} tokens)...")

edge_cases = [
    d for d in chunk_details
    if d["token_length"] > MAX_TOKENS
]

if edge_cases:
    print(
        f"  WARNING: {len(edge_cases)} chunk(s) "
        f"exceed {MAX_TOKENS} tokens:"
    )
    for ec in edge_cases:
        print(
            f"\n    ID:     {ec['id']}"
        )
        print(
            f"    Tokens: {ec['token_length']}"
        )
        print(
            f"    Chars:  {ec['char_length']}"
        )
        print(
            f"    Source: {ec['source']}"
        )
        print(
            f"    Text:   {ec['text_snippet']}..."
        )
else:
    print(
        f"  All chunks are within the "
        f"{MAX_TOKENS}-token limit."
    )


# ==========================================
# STEP 6: Visualization — Histogram
# ==========================================

print("\n[Step 6] Generating histogram...")

fig, ax = plt.subplots(figsize=(12, 6))

# Color palette
BAR_COLOR = "#4A90D9"
EDGE_COLOR = "#2C5F8A"
LINE_COLOR = "#E74C3C"
BG_COLOR = "#FAFBFC"
GRID_COLOR = "#E0E0E0"

fig.patch.set_facecolor(BG_COLOR)
ax.set_facecolor(BG_COLOR)

# Histogram
bin_count = min(50, max(10, total_chunks // 2))

n, bins, patches = ax.hist(
    token_lengths,
    bins=bin_count,
    color=BAR_COLOR,
    edgecolor=EDGE_COLOR,
    linewidth=0.8,
    alpha=0.85,
    zorder=3,
)

# Color bars exceeding 512 in red
for patch, left_edge in zip(patches, bins[:-1]):
    if left_edge >= MAX_TOKENS:
        patch.set_facecolor("#E74C3C")
        patch.set_edgecolor("#B03A2E")
        patch.set_alpha(0.9)

# 512-token cutoff line
ax.axvline(
    x=MAX_TOKENS,
    color=LINE_COLOR,
    linestyle="--",
    linewidth=2.5,
    zorder=4,
    label=f"MedCPT max_length = {MAX_TOKENS}",
)

# Annotate the cutoff line
ax.annotate(
    f"MedCPT max_length = {MAX_TOKENS}",
    xy=(MAX_TOKENS, ax.get_ylim()[1] * 0.92),
    xytext=(MAX_TOKENS + 15, ax.get_ylim()[1] * 0.92),
    fontsize=10,
    fontweight="bold",
    color=LINE_COLOR,
    ha="left",
    va="top",
    arrowprops=dict(
        arrowstyle="->",
        color=LINE_COLOR,
        lw=1.5,
    ),
)

# Labels and title
ax.set_xlabel(
    "Token Length",
    fontsize=13,
    fontweight="bold",
    labelpad=10,
)

ax.set_ylabel(
    "Number of Chunks",
    fontsize=13,
    fontweight="bold",
    labelpad=10,
)

ax.set_title(
    "Token Length Distribution Across "
    "All Document Chunks\n"
    f"(MedCPT Article Encoder — "
    f"{total_chunks} chunks from "
    f"ChromaDB)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)

# Stats text box
stats_text = (
    f"Total: {total_chunks}  |  "
    f"Min: {min_tokens}  |  "
    f"Max: {max_tokens}  |  "
    f"Mean: {mean_tokens:.1f}  |  "
    f"Median: {median_tokens:.1f}  |  "
    f"Std: {std_tokens:.1f}\n"
    f"Within limit: {within_limit} "
    f"({pct_within:.1f}%)  |  "
    f"Exceeding: {exceeding_limit} "
    f"({pct_exceeding:.1f}%)"
)

ax.text(
    0.5, -0.15,
    stats_text,
    transform=ax.transAxes,
    fontsize=9.5,
    ha="center",
    va="top",
    color="#555555",
    fontstyle="italic",
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="#F0F0F0",
        edgecolor="#CCCCCC",
        alpha=0.9,
    ),
)

# Grid
ax.grid(
    axis="y",
    alpha=0.4,
    color=GRID_COLOR,
    linestyle="-",
    zorder=1,
)

ax.legend(
    loc="upper right",
    fontsize=11,
    framealpha=0.9,
)

ax.tick_params(labelsize=11)

plt.tight_layout()
plt.subplots_adjust(bottom=0.22)

# Save chart
chart_path = os.path.join(
    OUTPUT_DIR, CHART_FILENAME
)

fig.savefig(
    chart_path,
    dpi=150,
    bbox_inches="tight",
    facecolor=fig.get_facecolor(),
)

print(f"  Chart saved: {chart_path}")

plt.close(fig)


# ==========================================
# STEP 7: Generate Markdown Report
# ==========================================

print("\n[Step 7] Generating markdown report...")

timestamp = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)

# Determine verdict
if exceeding_limit == 0:
    verdict = (
        "✅ **PASS** — Adaptive chunking "
        "successfully kept 100% of chunks within "
        f"MedCPT's {MAX_TOKENS}-token limit. "
        "No adjustments needed."
    )
elif pct_exceeding <= 5:
    verdict = (
        f"⚠️ **MARGINAL** — {pct_exceeding:.1f}% "
        f"of chunks ({exceeding_limit} of "
        f"{total_chunks}) exceed the "
        f"{MAX_TOKENS}-token limit and were "
        f"silently truncated during embedding. "
        f"Consider reducing MAX_CHUNK_SIZE."
    )
else:
    verdict = (
        f"❌ **FAIL** — {pct_exceeding:.1f}% "
        f"of chunks ({exceeding_limit} of "
        f"{total_chunks}) exceed the "
        f"{MAX_TOKENS}-token limit. "
        f"MAX_CHUNK_SIZE must be reduced."
    )

# Edge cases section
if edge_cases:
    edge_case_rows = ""
    for ec in edge_cases:
        snippet = (
            ec["text_snippet"]
            .replace("|", "\\|")
            .replace("\n", " ")
        )
        edge_case_rows += (
            f"| `{ec['id']}` "
            f"| {ec['source']} "
            f"| {ec['token_length']} "
            f"| {ec['char_length']} "
            f"| {snippet}... |\n"
        )

    edge_cases_section = f"""## Edge Cases (Chunks Exceeding {MAX_TOKENS} Tokens)

| Chunk ID | Source File | Tokens | Chars | Text Snippet |
|----------|------------|--------|-------|--------------|
{edge_case_rows}

> These chunks were silently truncated by the tokenizer during embedding generation
> (`truncation=True, max_length=512` in `ingest.py`), meaning the tail end of
> their content was **not** encoded into the embedding vector.
"""
else:
    edge_cases_section = f"""## Edge Cases

No chunks exceed {MAX_TOKENS} tokens. All chunk content was fully encoded into the embedding vectors without truncation.
"""


report = f"""# Chunk Token-Length Validation Report

**Generated:** {timestamp}
**Project:** Diabetes-in-Adolescents RAG System

---

## Methodology

This validation tests whether the adaptive chunking strategy (implemented in `ingest.py`)
produces chunks that fit within MedCPT Article Encoder's **{MAX_TOKENS}-token** input limit.

### What was tested
- **{total_chunks} chunks** retrieved from the `{COLLECTION_NAME}` collection in ChromaDB (`{CHROMA_DB_PATH}`)
- Each chunk was tokenized using the **MedCPT Article Encoder tokenizer** (`{MEDCPT_MODEL}`)
- Tokenization used `truncation=False` to capture **true** token counts (not clipped values)
- The tokenizer was called with `[["", chunk_text]]` pairs (empty title + abstract) to match the encoding used during ingestion

### Chunking configuration
- `MAX_CHUNK_SIZE = 2000` characters (set in `ingest.py`)
- Adaptive sentence-boundary splitting for texts exceeding this limit
- Rationale: biomedical text averages ~4–5 characters per token, so 2000 chars ≈ 400–500 tokens

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total chunks | {total_chunks} |
| Min token length | {min_tokens} |
| Max token length | {max_tokens} |
| Mean token length | {mean_tokens:.1f} |
| Median token length | {median_tokens:.1f} |
| Standard deviation | {std_tokens:.1f} |
| Within {MAX_TOKENS}-token limit | {within_limit} ({pct_within:.1f}%) |
| Exceeding {MAX_TOKENS}-token limit | {exceeding_limit} ({pct_exceeding:.1f}%) |

---

{edge_cases_section}

---

## Token Length Distribution

![Token Length Distribution](token_length_distribution.png)

---

## Verdict

{verdict}
"""

report_path = os.path.join(
    OUTPUT_DIR, REPORT_FILENAME
)

with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"  Report saved: {report_path}")


# ==========================================
# DONE
# ==========================================

print("\n" + "=" * 55)
print("VALIDATION COMPLETE")
print("=" * 55)
print(f"  Chart:  {os.path.abspath(chart_path)}")
print(f"  Report: {os.path.abspath(report_path)}")
print("=" * 55)
