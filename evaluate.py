"""
Evaluation harness for The Unofficial Guide.

Runs 5 test questions against the live system and prints, for each: the question,
the expected (ground-truth) answer, the system's actual response, the retrieved
chunks, and a slot for an accuracy judgment.

The questions follow the design intent of planning.md's eval set (a mix of question
kinds plus a deliberate "hard case") but are adapted to this corpus of real GT
professor reviews. Q4 and Q5 are both hard cases that test honest abstention: Q4
asks about a topic the corpus does not cover at all (assignment feedback), and Q5
asks about one the corpus only resembles (makeup exams vs. abundant exam talk) — the
system should decline rather than paraphrase a near-miss chunk into a wrong answer.

Run:
    python evaluate.py
Prerequisite: build the index once with `python build_index.py`.

Note: accuracy judgments are filled in by a human after reading the output (see
README "Evaluation Report"). The script structures the comparison; it does not
auto-grade.
"""
from src.pipeline import ask

TESTS = [
    {
        "question": "What do students say about Professor Joyner's exams?",
        "expected": "Exams are all-or-nothing coding tests with no partial credit; "
                    "very difficult for an intro course; professor is largely inaccessible "
                    "and the course is async — TAs handle most interactions.",
    },
    {
        "question": "Which professor is best for someone brand new to programming?",
        "expected": "Borela (CS1301) — highly rated, assumes no prior coding experience, "
                    "explains Python clearly, approachable, plenty of practice exams.",
    },
    {
        "question": "How heavy is the workload in Professor Starner's AI course?",
        "expected": "Very heavy — 20+ hours per week, difficult projects that must be started "
                    "early, largely self-taught; plenty of extra credit available via "
                    "auto-graded projects.",
    },
    {
        "question": "Which professor gives the most useful feedback on assignments?",
        "expected": "(Hard case) No document describes a professor giving detailed feedback on "
                    "assignments — the word 'feedback' only appears in the sense of professors "
                    "receiving student feedback. Correct behavior = decline / say not enough info. "
                    "Distractors: several professors are described as accessible at office hours "
                    "(Moss, Borela), which a weakly-grounded system may paraphrase into a false answer.",
    },
    {
        "question": "Do any CS professors offer makeup exams for students who miss one?",
        "expected": "(Hard case) No document describes makeup exams. The corpus is dense with exam "
                    "talk (difficulty, curving, all-or-nothing grading, final exemptions), so a "
                    "near-miss chunk can tempt a confident wrong answer. Correct behavior = decline / "
                    "say not enough info.",
    },
]


def main():
    for i, t in enumerate(TESTS, 1):
        result = ask(t["question"])
        print("=" * 80)
        print(f"Q{i}: {t['question']}")
        print(f"\nEXPECTED:\n  {t['expected']}")
        print(f"\nSYSTEM ANSWER:\n  {result['answer']}")
        print("\nRETRIEVED CHUNKS:")
        for c in result["chunks"]:
            print(f"  - [{c['label']} | dist={c['distance']}] "
                  f"{c['text'][:110]}...")
        print(f"\nSOURCES CITED:\n  {', '.join(result['sources'])}")
        print("\nACCURACY JUDGMENT: ____ (accurate / partially accurate / inaccurate)")
        print()


if __name__ == "__main__":
    main()
