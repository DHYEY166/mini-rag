from __future__ import annotations

import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "embeddinggemma")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gemma3:4b")
EMBED_DIM = 768
REFUSAL = "The provided policy does not answer this question."
DEFAULT_POLICY = Path("data/policy.md")
PROMPT_PATH = Path("prompts/grounded.v1.md")
SECTION_HEADING = re.compile(r"^## (\d+)\. (.+)$", re.MULTILINE)


def split_policy(markdown: str) -> list[dict]:
    """One chunk per numbered heading. Body text is unchanged."""
    version_match = re.search(r"Version (\d+\.\d+)", markdown)
    version = version_match.group(1) if version_match else "2.0"
    headings = list(SECTION_HEADING.finditer(markdown))
    chunks = []
    for i, match in enumerate(headings):
        start = match.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(markdown)
        section = match.group(1)
        title = match.group(2).strip()
        chunks.append(
            {
                "chunk_id": f"expense-policy:v{version}:section-{section}",
                "document": "Employee Expense Policy",
                "version": version,
                "section": section,
                "section_title": title,
                "text": markdown[start:end].strip(),
            }
        )
    return chunks


def cosine_distance(a: list[float], b: list[float]) -> float:
    """1 - cosine similarity. Smaller is closer. Used by offline tests."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return 1.0 - (dot / (norm_a * norm_b))


def rank_by_distance(query: list[float], chunks: list[dict], k: int = 3) -> list[dict]:
    ranked = []
    for chunk in chunks:
        item = dict(chunk)
        item["distance"] = cosine_distance(query, chunk["vector"])
        ranked.append(item)
    ranked.sort(key=lambda c: c["distance"])
    return ranked[:k]


def section_label(chunk: dict) -> str:
    return f"{chunk['section']}. {chunk['section_title']}"


def build_response(answer: str, retrieved: list[dict]) -> dict:
    """Code owns the citation. Refusal still returns JSON."""
    answer = answer.strip()
    citation = None
    if answer != REFUSAL and retrieved:
        top = retrieved[0]
        citation = {
            "document": top["document"],
            "version": top["version"],
            "section": section_label(top),
        }
    return {
        "answer": answer,
        "citation": citation,
        "retrieved_chunks": [
            {"section": section_label(c), "distance": float(c["distance"])}
            for c in retrieved
        ],
    }


def _ollama_post(path: str, payload: dict) -> dict:
    request = urllib.request.Request(
        f"{OLLAMA_HOST}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise SystemExit(f"Ollama is not reachable at {OLLAMA_HOST}: {exc}") from exc


def embed_text(text: str, role: str) -> list[float]:
    """Same function for chunks and questions. role is 'document' or 'query'."""
    if role == "document":
        prefixed = text
    elif role == "query":
        prefixed = f"task: search result | query: {text}"
    else:
        raise ValueError("role must be 'document' or 'query'")

    data = _ollama_post("/api/embed", {"model": EMBED_MODEL, "input": prefixed})
    vector = data["embeddings"][0]
    if len(vector) != EMBED_DIM:
        raise SystemExit(f"Expected {EMBED_DIM} dimensions, got {len(vector)}")
    return vector


def embed_chunk(chunk: dict) -> list[float]:
    prefixed = f"title: {chunk['section_title']} | text: {chunk['text']}"
    return embed_text(prefixed, role="document")


def connect():
    conn = psycopg.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def ingest(policy_path: str) -> None:
    markdown = Path(policy_path).read_text(encoding="utf-8")
    chunks = split_policy(markdown)
    if len(chunks) != 6:
        raise SystemExit(f"Expected 6 sections, found {len(chunks)}")

    sql = """
        INSERT INTO mini_rag.chunks
          (chunk_id, document, version, section, section_title, text, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (chunk_id) DO UPDATE SET
          document = EXCLUDED.document,
          version = EXCLUDED.version,
          section = EXCLUDED.section,
          section_title = EXCLUDED.section_title,
          text = EXCLUDED.text,
          embedding = EXCLUDED.embedding
    """
    with connect() as conn:
        with conn.cursor() as cur:
            for chunk in chunks:
                cur.execute(
                    sql,
                    (
                        chunk["chunk_id"],
                        chunk["document"],
                        chunk["version"],
                        chunk["section"],
                        chunk["section_title"],
                        chunk["text"],
                        embed_chunk(chunk),
                    ),
                )
        conn.commit()
    print(f"Ingested {len(chunks)} chunks.")


def retrieve(question: str) -> list[dict]:
    query_vector = embed_text(question, role="query")
    sql = """
        SELECT document, version, section, section_title, text,
               embedding <=> %s AS distance
        FROM mini_rag.chunks
        ORDER BY embedding <=> %s ASC
        LIMIT 3
    """
    with connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (query_vector, query_vector))
            rows = cur.fetchall()
    return [
        {
            "document": row[0],
            "version": row[1],
            "section": row[2],
            "section_title": row[3],
            "text": row[4],
            "distance": float(row[5]),
        }
        for row in rows
    ]


def generate_answer(question: str, retrieved: list[dict]) -> str:
    system = PROMPT_PATH.read_text(encoding="utf-8").strip()
    excerpts = "\n\n".join(
        f"[{section_label(c)}]\n{c['text']}" for c in retrieved
    )
    user = (
        f"<<<QUESTION>>>\n{question}\n<<<END>>>\n\n"
        f"<<<EXCERPTS>>>\n{excerpts}\n<<<END>>>"
    )
    data = _ollama_post(
        "/api/chat",
        {
            "model": CHAT_MODEL,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
    )
    return data["message"]["content"].strip()


def ask(question: str) -> dict:
    retrieved = retrieve(question)
    answer = generate_answer(question, retrieved)
    return build_response(answer, retrieved)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"ingest", "ask"}:
        print("Usage:")
        print("  python main.py ingest")
        print("  python main.py ingest --policy data/policy.md")
        print('  python main.py ask "Your question"')
        raise SystemExit(2)

    if sys.argv[1] == "ingest":
        path = str(DEFAULT_POLICY)
        if "--policy" in sys.argv:
            path = sys.argv[sys.argv.index("--policy") + 1]
        ingest(path)
        return

    question = " ".join(sys.argv[2:]).strip()
    if not question:
        raise SystemExit("Provide a question after ask.")
    print(json.dumps(ask(question), indent=2))


if __name__ == "__main__":
    main()