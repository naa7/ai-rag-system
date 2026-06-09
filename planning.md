# Project 1 Planning: The Unofficial Guide

## Domain

Student reviews of Computer Science professors. The official course catalog tells you a class's
title, credits, and prerequisites — but never the things that actually decide your semester: Are
the exams curved? Does attendance matter? Is the workload survivable alongside your other classes?
Who gives genuinely useful feedback? That knowledge lives in scattered RateMyProfessor reviews and
word-of-mouth. It is hard to find through official channels because it is unstructured, anonymous,
spread across hundreds of individual opinions, and never aggregated into something you can ask a
direct question of. This system makes that informal, student-generated knowledge searchable and
answerable, with citations.

> **Note on the corpus:** the documents in `data/raw/` are **synthetic, RMP-style sample reviews**
> written for this project (labeled as synthetic in every file header), not scraped from a real
> site. They are formatted exactly like real reviews, so real `.txt` files can be dropped in with
> no code changes.

---

## Documents

12 documents, one per professor — each a `.txt` file with a `Key: value` header (professor, course,
department), a `---` separator, and five short reviews as blank-line-separated paragraphs (the
shape of a real RateMyProfessor professor page). Sources span different *kinds* of questions —
workload, grading, exam style, feedback quality, beginner-friendliness — for query variety.

| #  | Source | Description | URL or location |
|----|--------|-------------|-----------------|
| 1  | Prof. Smith reviews | CS 201 Data Structures — curved exams, slide-based, attendance | `data/raw/rmp_smith.txt` |
| 2  | Prof. Nguyen reviews | CS 101 Intro Programming — beginner-friendly, patient | `data/raw/rmp_nguyen.txt` |
| 3  | Prof. Alvarez reviews | CS 340 Software Eng — very heavy workload, projects | `data/raw/rmp_alvarez.txt` |
| 4  | Prof. Johnson reviews | CS 220 Computer Org — detailed feedback, responsive | `data/raw/rmp_johnson.txt` |
| 5  | Prof. Okafor reviews | CS 310 Algorithms — harsh grader, tough but fair | `data/raw/rmp_okafor.txt` |
| 6  | Prof. Patel reviews | CS 150 Discrete Math — boring lectures, easy tests | `data/raw/rmp_patel.txt` |
| 7  | Prof. Lee reviews | CS 410 Operating Systems — hard, office hours gold | `data/raw/rmp_lee.txt` |
| 8  | Prof. Garcia reviews | CS 360 Web Dev — group-project heavy, free-rider risk | `data/raw/rmp_garcia.txt` |
| 9  | Prof. Murphy reviews | CS 330 Databases — disorganized, good material | `data/raw/rmp_murphy.txt` |
| 10 | Prof. Chen reviews | CS 470 Machine Learning — research-focused, distant | `data/raw/rmp_chen.txt` |
| 11 | Prof. Roberts reviews | CS 380 Networks — practical, industry-relevant | `data/raw/rmp_roberts.txt` |
| 12 | Prof. Williams reviews | CS 250 Theory of Comp — dry, proof-heavy | `data/raw/rmp_williams.txt` |

---

## Chunking Strategy

**Chunk size:** One review per chunk (review/paragraph-based). Reviews longer than a **600-character**
threshold are sentence-split; in practice almost none hit it, so chunks average ~182 chars.

**Overlap:** ~80 characters, word-aligned, applied **only** when a long review is sentence-split
(it carries the tail of the previous segment forward so a fact spanning the split stays retrievable).

**Reasoning:** Each review is a short, self-contained opinion, so the individual review is the
natural semantic unit — splitting on blank-line boundaries gives one complete, retrievable thought
per chunk. A 200-char fixed split would fragment a review ("Smith's exams are heavily"), which is
unretrievable alone; packing many reviews into one chunk would dilute the embedding so a query about
exams also matches office-hours and workload text. I first implemented a greedy packer (~600-char
target combining reviews); it produced only **26 chunks** from 12 docs and sliced words mid-token at
overlap boundaries, so I switched to one-review-per-chunk → **60 clean chunks**.

---

## Retrieval Approach

**Embedding model:** `all-MiniLM-L6-v2` via `sentence-transformers` — local, no API key, no rate
limits, 384-dim, fast on CPU, well-suited to short English opinion text. Vectors are stored in a
persistent ChromaDB collection using cosine similarity.

**Top-k:** 5. With ~130–250-char reviews, 5 chunks gives several perspectives for the LLM to
synthesize without burying the signal. k=1–2 risks missing the relevant review if the single best
match is slightly off; k=15 floods the prompt with loosely-related chunks that pull the answer
off-target and waste tokens.

**Production tradeoff reflection:** If cost were no object I'd weigh **accuracy on domain text** (a
larger hosted model — OpenAI `text-embedding-3-large`, Voyage, Cohere — handles slang better);
**context length** (irrelevant here with short chunks, but it'd matter for long guides/syllabi);
**multilingual** support (needed if reviews arrive in multiple languages); and **local vs. API /
privacy & latency** (reviews can be sensitive — the local model keeps data on-device with zero
per-query cost and no rate limits). For this use case the local model's privacy, cost, and speed
outweigh the marginal accuracy gain of a hosted model.

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What do students say about Professor Smith's exams? | Exams based on lecture slides (not the textbook); midterms curved but the final is not; attendance / showing up matters. |
| 2 | Which professor is best for someone brand new to programming? | Nguyen (CS 101) — extremely patient, assumes zero background, beginner-friendly, slow pace good for first-timers. |
| 3 | How heavy is the workload in Professor Alvarez's software engineering course? | Very heavy — four major projects, ~15–20 hrs/week each near deadlines, time-consuming but rewarding; deadlines relentless. |
| 4 | Which professor gives the most useful feedback on assignments? | Johnson — detailed paragraph-long comments on code and design docs, very responsive over email. |
| 5 | Do any CS professors offer extra credit, and how? | **(Designed hard case)** No document describes extra credit. Closest fact is Patel sometimes drops the lowest quiz — *not* extra credit. Correct behavior = decline / say not enough info. |

---

## Anticipated Challenges

1. **Coverage gaps + the retriever's "always return k" behavior.** Semantic search returns its k
   nearest chunks even when none are actually relevant (e.g. the extra-credit question, which no
   document covers). A weakly-grounded LLM would paraphrase a near-miss chunk into a confident wrong
   answer, so the system prompt must *enforce*, not suggest, context-only answering.

2. **One-review-per-chunk isolates each fact.** Clean chunks help most queries, but a single
   sub-fact phrased unlike the query (e.g. Smith's "curved … no mercy" review vs. the word "exams")
   can be out-competed for a top-k slot by loosely-related chunks from other professors — producing
   a partially-complete answer. Near-duplicate phrasing across professors ("tough but fair," "office
   hours help") compounds this; source citations are what let a reader catch a wrong-professor match.

---

## Architecture

```
                         THE UNOFFICIAL GUIDE — RAG PIPELINE

  ┌────────────────┐   ┌────────────────┐   ┌─────────────────────┐
  │ 1. INGESTION   │   │ 2. CHUNKING    │   │ 3. EMBED + STORE    │
  │                │   │                │   │                     │
  │ data/raw/*.txt │──▶│ one review =   │──▶│ all-MiniLM-L6-v2    │
  │ load + clean   │   │ one chunk      │   │ (sentence-          │
  │ (src/ingest.py)│   │ (split >600ch, │   │  transformers)      │
  │                │   │  ~80 overlap)  │   │   → ChromaDB        │
  └────────────────┘   │ (src/ingest.py)│   │ (src/store.py)      │
                       └────────────────┘   └──────────┬──────────┘
                                                        │
                       user query                       ▼
  ┌────────────────┐   ┌────────────────┐   ┌─────────────────────┐
  │ 5. GENERATION  │   │ 4. RETRIEVAL   │   │  vector index       │
  │                │◀──│                │◀──│  (persisted on disk)│
  │ Groq           │   │ top-k = 5      │   └─────────────────────┘
  │ llama-3.3-70b  │   │ semantic       │
  │ grounded +     │   │ similarity     │
  │ cited sources  │   │ (src/store.py) │
  │ (src/generate) │   └────────────────┘
  └───────┬────────┘
          │
          ▼
  ┌────────────────┐
  │ Gradio UI      │   answer + sources + retrieved-chunk preview
  │ (app.py)       │
  └────────────────┘
```

---

## AI Tool Plan

I (the student) am using **Claude** as the AI coding tool throughout.

**Milestone 3 — Ingestion and chunking:** Give Claude the *Documents* and *Chunking Strategy*
sections above plus the architecture diagram; ask it to implement `load_documents()`, `clean_text()`,
and a review-based `chunk_document()` matching the spec. Expect `src/ingest.py` with a CLI that
prints chunk count + 5 samples. **Verify:** inspect the 5 samples — each must be self-contained, and
the count must clear ~50; if not, the chunking unit is wrong.

**Milestone 4 — Embedding and retrieval:** Give Claude the *Retrieval Approach* section; ask for
`build_index()` (MiniLM → ChromaDB with metadata) and `retrieve(query, k)`. Expect `src/store.py`
+ a one-shot `build_index.py`. **Verify:** run 3 eval questions through retrieval and confirm
top distances < 0.5 and on-topic chunks before adding any LLM.

**Milestone 5 — Generation and interface:** Give Claude the grounding requirement (answer from
context only; exact "not enough information" fallback) and the Gradio skeleton; ask for a strict
system prompt, an `ask()` that **appends sources programmatically**, and the UI wiring. Expect
`src/generate.py`, `src/pipeline.py`, `app.py`. **Verify:** read the prompt to confirm it *enforces*
grounding, and test the Q5 no-coverage case to confirm the system declines instead of hallucinating.

I will **not** ask Claude to write this planning.md or the README's reflection sections — those are
my own.
