from transformers import AutoTokenizer

# MedCPT tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "ncbi/MedCPT-Article-Encoder"
)

import json

with open(
    "data/processed/processed_output.json",
    "r",
    encoding="utf-8"
) as f:

    document = json.load(f)


MAX_TOKENS = 450    
OVERLAP = 50


def count_tokens(text):
    return len(tokenizer(text)["input_ids"])


def token_split(text, max_tokens=MAX_TOKENS, overlap=OVERLAP):

    tokens = tokenizer.tokenize(text)

    chunks = []

    start = 0

    while start < len(tokens):

        end = min(start + max_tokens, len(tokens))

        token_chunk = tokens[start:end]

        chunk_text = tokenizer.convert_tokens_to_string(
            token_chunk
        )

        chunks.append(chunk_text)

        if end == len(tokens):
            break

        start = end - overlap

    return chunks


def semantic_split(text):
    """
    Placeholder.

    Later this function can use
    SentenceTransformers
    cosine similarity
    or LangChain SemanticChunker.

    For now
    return whole paragraph.
    """

    return [text]


def hierarchical_chunk(document):

    final_chunks = []

    if "sections" in document and document["sections"]:
        sections = document["sections"]
    else:
        sections = [{
            "section": "document",
            "paragraphs": [{
                "id": "p1",
                "text": document.get("text", "")
            }]
        }]

    chapter = document.get("chapter", document.get("metadata", {}).get("title", "document"))

    for section in sections:

        paragraphs = section.get("paragraphs", [])

        if not paragraphs and "text" in section:
            paragraphs = [{"id": "p1", "text": section["text"]}]

        for paragraph in paragraphs:

            text = paragraph.get("text", "")

            token_length = count_tokens(text)

            # -------------------------
            # Paragraph is already good
            # -------------------------

            if token_length <= MAX_TOKENS:

                final_chunks.append({

                    "id": paragraph.get("id", "p1"),

                    "text": text,

                    "metadata":{

                        "chapter": chapter,

                        "section": section.get("section", "document"),

                        "level":"paragraph"
                    }
                })

                continue

            # -------------------------
            # Semantic Chunking
            # -------------------------

            semantic_chunks = semantic_split(text)

            for i, sem_chunk in enumerate(semantic_chunks):

                sem_tokens = count_tokens(sem_chunk)

                if sem_tokens <= MAX_TOKENS:

                    final_chunks.append({

                        "id":f'{paragraph.get("id", "p1")}_S{i+1}',

                        "text":sem_chunk,

                        "metadata":{

                            "chapter": chapter,

                            "section": section.get("section", "document"),

                            "level":"semantic"
                        }
                    })

                else:

                    token_chunks = token_split(sem_chunk)

                    for j, chunk in enumerate(token_chunks):

                        final_chunks.append({

                            "id":f'{paragraph.get("id", "p1")}_S{i+1}_T{j+1}',

                            "text":chunk,

                            "metadata":{

                                "chapter": chapter,

                                "section": section.get("section", "document"),

                                "level":"token"
                            }
                        })

    return final_chunks

with open("data/chunks/chunks.json", "w", encoding="utf-8") as f:
    json.dump(hierarchical_chunk(document), f, ensure_ascii=False, indent=4)