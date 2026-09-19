# Mini RAG — Expense Policy Assistant

Answers expense questions from data/policy.md only. Refuses when the policy does not say.

## How to run it
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker start fde-postgres
docker exec -i fde-postgres psql -U fde -d fde < sql/001_init.sql
python main.py ingest
python main.py ask "How much can I spend on food each day?"

## How to test it
python -m pytest tests/test_offline.py
python -m pytest tests/test_required_questions.py

## Configuration
DATABASE_URL, OLLAMA_HOST, EMBED_MODEL=embeddinggemma, CHAT_MODEL=gemma3:4b

## Known limitations
Needs local Ollama and the fde-postgres container. Distance is a rank, not confidence.