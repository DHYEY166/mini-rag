CREATE EXTENSION IF NOT EXISTS vector;

CREATE SCHEMA IF NOT EXISTS mini_rag;

CREATE TABLE IF NOT EXISTS mini_rag.chunks (
  chunk_id      TEXT PRIMARY KEY,
  document      TEXT NOT NULL,
  version       TEXT NOT NULL,
  section       TEXT NOT NULL,
  section_title TEXT NOT NULL,
  text          TEXT NOT NULL,
  embedding     vector(768) NOT NULL
);