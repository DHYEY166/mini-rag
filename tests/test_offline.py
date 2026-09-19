from pathlib import Path

import main

POLICY = Path("data/policy.md").read_text(encoding="utf-8")
REFUSAL = "The provided policy does not answer this question."


def test_split_policy_makes_six_citable_chunks():
    chunks = main.split_policy(POLICY)
    assert len(chunks) == 6
    assert [c["section"] for c in chunks] == ["1", "2", "3", "4", "5", "6"]
    assert chunks[0]["chunk_id"] == "expense-policy:v2.0:section-1"
    assert chunks[0]["section_title"] == "Meals"
    assert chunks[0]["document"] == "Employee Expense Policy"
    assert chunks[0]["version"] == "2.0"
    assert "Employees may claim" in chunks[0]["text"]
    assert not chunks[0]["text"].startswith("##")


def test_rank_sorts_smallest_distance_first():
    query = [1.0, 0.0, 0.0]
    chunks = [
        {"section": "far", "vector": [0.0, 1.0, 0.0]},
        {"section": "near", "vector": [0.9, 0.1, 0.0]},
        {"section": "mid", "vector": [0.5, 0.5, 0.0]},
    ]
    ranked = main.rank_by_distance(query, chunks, k=3)
    assert [c["section"] for c in ranked] == ["near", "mid", "far"]
    assert ranked[0]["distance"] <= ranked[1]["distance"] <= ranked[2]["distance"]
    assert len(ranked) <= 3


def test_citation_from_row_when_answered():
    retrieved = [
        {
            "document": "Employee Expense Policy",
            "version": "2.0",
            "section": "1",
            "section_title": "Meals",
            "distance": 0.08,
        }
    ]
    response = main.build_response("Employees may claim up to $65 per day.", retrieved)
    assert response["citation"] == {
        "document": "Employee Expense Policy",
        "version": "2.0",
        "section": "1. Meals",
    }
    assert response["retrieved_chunks"][0]["distance"] == 0.08
    assert isinstance(response["retrieved_chunks"][0]["distance"], float)


def test_refusal_has_no_citation():
    retrieved = [
        {
            "document": "Employee Expense Policy",
            "version": "2.0",
            "section": "1",
            "section_title": "Meals",
            "distance": 0.4,
        }
    ]
    response = main.build_response(REFUSAL, retrieved)
    assert response["answer"] == REFUSAL
    assert response["citation"] is None