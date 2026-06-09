# The Unofficial Guide — Project 1

A Retrieval-Augmented Generation (RAG) system that makes student-generated knowledge about CS
professors searchable and answerable. Ask a plain-language question and get a grounded, **cited**
answer drawn only from a corpus of student reviews.

> ### About the documents
>
> Reviews are scraped from RateMyProfessors via `scrape_rmp.py` and stored in `documents/` — 15
> Georgia Tech CS professors, 25 real reviews each (382 chunks total). To refresh or expand the
> corpus, re-run `python scrape_rmp.py`, then `python build_index.py`.

### Quickstart

```bash
python -m venv .venv && source .venv/bin/activate    # Python 3.10–3.12
pip install -r requirements.txt
cp .env.example .env                                 # then set GROQ_API_KEY=... (free: console.groq.com)
python build_index.py                                # build the vector index (run once)
python app.py                                        # open http://localhost:7860
```

Other entry points: `python src/ingest.py` (inspect chunks) · `python src/store.py "question"`
(retrieval only, no key needed) · `python evaluate.py` (run the 5-question evaluation).

---

## Domain

Student reviews of Computer Science professors. The official course catalog gives a class's title,
credits, and prerequisites — but not the things that actually decide your semester: whether exams
are curved, whether attendance matters, whether the workload is survivable, who gives genuinely
useful feedback. That knowledge lives in scattered RateMyProfessor reviews and word-of-mouth. It is
hard to find through official channels because it is unstructured, anonymous, spread across hundreds
of individual opinions, and never aggregated into something you can ask a direct question of.

---

## Document Sources

15 documents scraped from RateMyProfessors — Georgia Tech CS professors with the most ratings (82–282
each). 25 real reviews per professor, 382 chunks total. Scraped via the public RMP GraphQL API using
`scrape_rmp.py`.

| #  | Professor               | Course  | Ratings | File                                  |
|----|-------------------------|---------|---------|---------------------------------------|
| 1  | Melinda McDaniel        | CS4400  | 282     | `documents/rmp_melinda_mcdaniel.txt`  |
| 2  | David Joyner            | CS1301  | 186     | `documents/rmp_david_joyner.txt`      |
| 3  | Olufisayo Omojokun      | CS1331  | 179     | `documents/rmp_olufisayo_omojokun.txt`|
| 4  | Abrahim Ladha           | CS2050  | 178     | `documents/rmp_abrahim_ladha.txt`     |
| 5  | Mary Hudachek-Buswell   | CS1332  | 174     | `documents/rmp_mary_hudachek_buswell.txt`|
| 6  | Ronnie Howard           | CS2050  | 168     | `documents/rmp_ronnie_howard.txt`     |
| 7  | Aibek Musaev            | CS1331  | 167     | `documents/rmp_aibek_musaev.txt`      |
| 8  | Cedric Stallworth       | CS1371  | 153     | `documents/rmp_cedric_stallworth.txt` |
| 9  | Frederic Faulkner       | CS1332  | 140     | `documents/rmp_frederic_faulkner.txt` |
| 10 | Iretta Kearse           | CS1301  | 104     | `documents/rmp_iretta_kearse.txt`     |
| 11 | Mark Moss               | CS2110  | 100     | `documents/rmp_mark_moss.txt`         |
| 12 | Thad Starner            | CS3600  | 97      | `documents/rmp_thad_starner.txt`      |
| 13 | Daniel Forsyth          | CS2200  | 90      | `documents/rmp_daniel_forsyth.txt`    |
| 14 | Rodrigo Borela          | CS1301  | 84      | `documents/rmp_rodrigo_borela.txt`    |
| 15 | Kristine Nagel          | CS3803  | 82      | `documents/rmp_kristine_nagel.txt`    |

**Adding more data:** run `python scrape_rmp.py` to re-scrape or adjust `MIN_RATINGS`,
`MAX_PROFESSORS`, or `SCHOOL_ID` at the top of the script to target a different school. Each `.txt`
file follows the format: a `Key: Value` metadata header, then `---`, then reviews as blank-line-separated paragraphs.

---

## Chunking Strategy

**Chunk size:** One review per chunk (review/paragraph-based). Reviews longer than a **600-character**
threshold are sentence-split; almost none reach it, so chunks average **~182 characters**.

**Overlap:** ~80 characters, word-aligned — applied **only** when a long review is sentence-split, to
carry a boundary-spanning fact forward. Independent reviews need no overlap between them.

**Why these choices fit your documents:** Each review is a short, self-contained opinion, so the
individual review is the natural semantic unit — one complete, retrievable thought per chunk (the
"good chunk" from the rubric, e.g. a full opinion about Joyner's exams). A 200-char fixed split would
fragment a review; packing many reviews together would dilute the embedding so an "exams" query also
matches office-hours and workload text. Preprocessing before chunking: `clean_text()` strips HTML
tags, decodes HTML entities (`&amp;`, `&#39;`, …), and collapses whitespace — no-ops on the clean
synthetic corpus but exactly what real scraped reviews need.

**Final chunk count:** **60 chunks** across 12 documents.

---

## Embedding Model

**Model used:** `all-MiniLM-L6-v2` via `sentence-transformers` — local, no API key, no rate limits,
384-dim, fast on CPU, well-suited to short English opinion text. Vectors live in a persistent
ChromaDB collection (cosine similarity); retrieval returns **top-k = 5**.

**Production tradeoff reflection:** With cost no object I'd weigh: **accuracy on domain text** — a
larger hosted model (OpenAI `text-embedding-3-large`, Voyage, Cohere) handles slang/domain phrasing
better, worth it if retrieval is the bottleneck; **context length** — irrelevant here with short
chunks, but it would matter for long-form guides/syllabi; **multilingual** support — needed if
reviews arrive in multiple languages; **local vs. API / privacy & latency** — reviews can be
sensitive and anonymous, so the local model keeps data on-device with zero per-query cost and no
rate limits. For this use case the local model's privacy, cost, and speed outweigh the marginal
accuracy gain of a hosted model.

---

## Grounded Generation

**System prompt grounding instruction:** The prompt in `src/generate.py` _enforces_ grounding rather
than suggesting it. Verbatim rules given to the model: use **ONLY** information in the CONTEXT block;
do **NOT** use any outside or prior knowledge about these professors/courses; if the context is
insufficient, reply with **exactly** the sentence _"I don't have enough information on that."_ and
nothing else; do not guess, extrapolate, or fill gaps with plausible-sounding claims; if reviews
only partially address the question, answer only the supported part and say what isn't covered.
Retrieved chunks are passed in a labeled `[Source N: file | professor | course]` CONTEXT block at
temperature 0.1.

**How source attribution is surfaced in the response:** Source attribution is appended
**programmatically** in `src/pipeline.py` (`ask()`), not left to the LLM — it de-duplicates the
retrieved chunks' source metadata in retrieval order and returns them in a `sources` list. The
Gradio UI shows this list in a "Retrieved from" panel, and the raw retrieved chunks (with distance
scores) in an expandable panel, so every answer is traceable to its documents.

---

## Evaluation Report

Run with `python evaluate.py` (Groq `llama-3.3-70b-versatile`, top-k = 5). Full transcript is saved
in `eval_results.txt`. Q5 is a **deliberately hard case** with no supporting document.

| #   | Question                                              | Expected answer                                                                   | System response (summarized)                                                                         | Retrieval quality                                           | Response accuracy                 |
| --- | ----------------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- | --------------------------------- |
| 1   | What do students say about Professor Joyner's exams?  | All-or-nothing coding tests, no partial credit; very hard for an intro course     | All-or-nothing tests, no partial credit; difficult without prior CS background — **omits async note**| Partially relevant (async chunk fell outside top-5)         | **Partially accurate**            |
| 2   | Best professor for someone brand new to programming?  | Borela — patient, zero-background, beginner-friendly                              | "Rodrigo Borela… no prior coding experience needed… explains Python clearly… very approachable"      | Relevant (#1 correct)                                       | **Accurate**                      |
| 3   | How heavy is Starner's AI course workload?            | Very heavy — difficult projects, 20+ hrs/wk, self-teaching required              | "Very heavy… 20+ hours per week… projects difficult and time consuming… start very early"            | Relevant                                                    | **Accurate**                      |
| 4   | Which professor gives the most useful feedback?       | Moss — engaging lectures, accessible at office hours, helpful during labs         | "Mark Moss… very nice… accessible at office hours… labs and projects well-supported"                 | Relevant (top-3 all Moss)                                   | **Accurate**                      |
| 5   | Do any CS professors offer extra credit, and how?     | Borela (+2 pts end of term) and Starner (auto-graded projects + extra credit)     | Names both; explains Borela's 2-point bonus and Starner's auto-graded extra credit                   | Relevant (chunks present, distances < 0.4)                  | **Accurate**                      |

**Retrieval quality:** Relevant / Partially relevant / Off-target
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Case Analysis

**Question that failed:** Q1 — "What do students say about Professor Joyner's exams?" (_partially
accurate_).

**What the system returned:** A confident, fluent answer covering the all-or-nothing coding tests
and the lack of partial credit — but it **omitted the async/inaccessibility detail** (professor
completely unavailable, course is async, TAs handle everything), which is a defining characteristic
students consistently flag.

**Root cause (tied to a specific pipeline stage):** A **retrieval ranking miss driven by the
chunking strategy** — not a generation bug. Because each review is its own chunk, the async/access
complaint lives in isolated chunks whose wording ("unavailable," "async," "TA's handle everything")
shares little vocabulary with the query word "exams," so those chunks' embedding distances landed
just outside the top-5 — ranked below distractors from other professors whose reviews mention
"exams" more directly. The generator can only ground in what retrieval hands it, so the
inaccessibility context never reached the model.

**What you would change to fix it:** Add a lightweight **keyword / BM25 hybrid search** so terms
like "unavailable" or "async" match lexically even when semantically distant; and/or **embed the
professor's name into each chunk's text** so same-professor chunks cluster and out-rank
cross-professor distractors. Raising top-k would also surface it but admits more noise and costs
tokens.

---

## Spec Reflection

**One way the spec helped you during implementation:** Writing the Chunking Strategy section
_before_ coding forced an explicit commitment to "one review = one self-contained thought." When the
first implementation (a greedy packer matching the spec's early ~600-char wording) produced only 26
chunks and word-sliced overlaps, having that written intent made the problem obvious the instant I
inspected the sample chunks — I knew what "good" was supposed to look like, so I caught the mismatch
in minutes instead of discovering it later as bad retrieval.

**One way your implementation diverged from the spec, and why:** The spec originally described
greedy paragraph-_packing_ toward a ~600-char target. In practice that combined multiple reviews per
chunk, yielded too few chunks (26, below the ~50 floor), and sliced words mid-token at overlap
boundaries. I switched to **one-review-per-chunk** (600 becoming a split _threshold_ for unusually
long reviews, not a packing target) and updated `planning.md` to match — producing 60 clean,
self-contained chunks.

---

## AI Usage

**Instance 1**

- _What I gave the AI:_ My planning.md Documents and Chunking Strategy sections plus the architecture
  diagram, asking it to implement the ingestion + chunking module.
- _What it produced:_ A `chunk_document()` that **greedily packed multiple reviews** up to a ~600-char
  target with raw character-slice overlap — matching the spec's literal early wording.
- _What I changed or overrode:_ After running the inspection CLI (26 chunks; overlaps starting
  mid-word like "s rusty,"), I overrode the approach and directed it to chunk **one review per
  paragraph**, with overlap only for over-long reviews and snapped to word boundaries. Result: 60
  clean chunks, and I updated planning.md to reflect the change.

**Instance 2**

- _What I gave the AI:_ The grounding requirement — answer from retrieved context only, with
  guaranteed source attribution — and asked it to write the generation system prompt and `ask()`.
- _What it produced:_ A first draft that _suggested_ using the documents and asked the model to cite
  its own sources.
- _What I changed or overrode:_ I tightened it so grounding is **enforced**: the "not enough
  information" fallback is an exact required string, I added an explicit rule against filling gaps
  with plausible claims, and I moved **source attribution out of the LLM into code** (`pipeline.py`
  appends sources from chunk metadata) so citations can never be omitted or invented. This is what
  made the Q5 no-coverage case decline correctly instead of hallucinating.

---

## Project Structure

```
.
├── documents/           # professor review .txt files (one per professor)
├── src/
│   ├── ingest.py        # load + clean + chunk (one review per chunk)
│   ├── store.py         # embed (MiniLM) + ChromaDB + retrieve
│   ├── generate.py      # grounded generation via Groq
│   └── pipeline.py      # ask(): retrieve -> generate -> attribute
├── build_index.py       # one-shot indexer
├── app.py               # Gradio UI
├── evaluate.py          # 5-question evaluation harness
├── eval_results.txt     # saved evaluation transcript
├── planning.md          # spec + architecture diagram
├── requirements.txt
└── .env.example
```
