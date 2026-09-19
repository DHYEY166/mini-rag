import json
from pathlib import Path
from pgvector import Vector

import pytest

import main

CASES = json.loads(Path("tests/required_questions.json").read_text(encoding="utf-8"))
REFUSAL = "The provided policy does not answer this question."


def _live_ready() -> bool:
    try:
        main.embed_text("ping", role="query")
        with main.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _live_ready(), reason="Ollama or Postgres is down")


@pytest.mark.parametrize("case", CASES, ids=[c["question"][:40] for c in CASES])
def test_expected_section_in_top_three(case):
    if case["section"] is None:
        return
    retrieved = main.retrieve(case["question"])
    assert len(retrieved) <= 3
    distances = [c["distance"] for c in retrieved]
    assert distances == sorted(distances)
    labels = [main.section_label(c) for c in retrieved]
    assert case["section"] in labels


def test_food_ranks_meals_above_deadline():
    food = Vector(main.embed_text("How much can I spend on food each day?", role="query"))
    with main.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT section_title, embedding <=> %s AS distance
                FROM mini_rag.chunks
                WHERE section_title IN ('Meals', 'Submission Deadline')
                """,
                (food,),
            )
            rows = {title: float(dist) for title, dist in cur.fetchall()}
    assert rows["Meals"] < rows["Submission Deadline"]


@pytest.mark.parametrize("case", CASES, ids=[c["question"][:40] for c in CASES])
def test_answers_match_policy(case):
    response = main.ask(case["question"])
    if case["section"] is None:
        assert response["answer"] == REFUSAL
        assert response["citation"] is None
        return
    text = response["answer"].lower()
    for fact in case["must_include"]:
        assert fact.lower() in text
    assert response["citation"] is not None
    assert response["citation"]["section"] == case["section"]