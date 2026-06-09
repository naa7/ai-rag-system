"""
Domain configuration.

This is the ONE place that knows what the corpus is about. Everything else in the
pipeline (chunking, embedding, retrieval, generation, UI) is domain-agnostic and
reads its domain behaviour from the DomainConfig selected here.

To point the system at a different corpus (product reviews, legal docs, support
tickets, ...), write a new DomainConfig and set ACTIVE_CONFIG to it. No other file
needs to change.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class DomainConfig:
    """Everything that varies by corpus / use-case."""

    # Used to namespace the vector collection so different domains don't collide.
    name: str

    # Which header keys (lower-cased) to lift into structured metadata. The header
    # is parsed generically as "Key: value"; these are the keys worth keeping and
    # surfacing in source labels / chunk displays. Empty list = keep all header keys.
    metadata_fields: list[str] = field(default_factory=list)

    # System prompt for grounded generation. {persona} is filled from `persona`.
    # Keep the grounding rules domain-neutral; only the persona/subject changes.
    persona: str = "a helpful assistant"
    subject: str = "the provided documents"

    # UI strings.
    ui_title: str = "RAG Assistant"
    ui_description: str = (
        "Ask a plain-language question. Answers are grounded in the source "
        "documents and cite where they came from."
    )
    ui_placeholder: str = "Ask a question about the documents…"
    examples: list[str] = field(default_factory=list)

    def label_for(self, metadata: dict) -> str:
        """
        Human-readable label for a chunk's source, used in citations and the UI.
        Joins the configured metadata fields that are present, e.g.
        "rmp_joyner.txt (David Joyner — CS1301)".
        """
        source = metadata.get("source", "?")
        parts = [
            str(metadata[f])
            for f in self.metadata_fields
            if f != "source" and metadata.get(f) not in (None, "", "Unknown")
        ]
        return f"{source} ({' — '.join(parts)})" if parts else source

    def system_prompt(self, no_info: str) -> str:
        return (
            f"You are {self.persona}, answering questions about {self.subject} "
            "using ONLY the information provided in the CONTEXT block.\n\n"
            "Rules:\n"
            "1. Use ONLY information stated in the CONTEXT. Do NOT use any outside or "
            "prior knowledge.\n"
            "2. If the CONTEXT does not contain enough information to answer, reply with "
            f'EXACTLY this sentence and nothing else: "{no_info}"\n'
            "3. Do not guess, extrapolate, or fill gaps with plausible-sounding claims. "
            "If the context only partially addresses the question, answer only the part "
            "that is supported and say what is not covered.\n"
            "4. Attribute claims to the source they came from when relevant.\n"
            "5. Be concise (2-5 sentences)."
        )


# --------------------------------------------------------------------------- #
# Built-in domains
# --------------------------------------------------------------------------- #
PROFESSOR_RATINGS = DomainConfig(
    name="unofficial_guide",
    metadata_fields=["source", "professor", "course"],
    persona="The Unofficial Guide, a helpful upperclassman",
    subject="university professors, based on student reviews",
    ui_title="🎓 The Unofficial Guide",
    ui_description=(
        "Ask a plain-language question about CS professors. Answers are grounded "
        "in student reviews and cite their sources."
    ),
    ui_placeholder="e.g. What are Professor Joyner's exams like?",
    examples=[
        "What do students say about Professor Joyner's exams?",
        "Which professor is best for someone brand new to programming?",
        "How heavy is the workload in Professor Starner's AI course?",
        "What do students think of Professor McDaniel's database class?",
        "Do any CS professors offer extra credit, and how?",
    ],
)


# A generic config for an arbitrary corpus with no special header schema.
GENERIC = DomainConfig(
    name="generic_corpus",
    metadata_fields=["source"],
)

# Registry so a domain can be selected by name via the DOMAIN env var.
CONFIGS = {
    "professor_ratings": PROFESSOR_RATINGS,
    "generic": GENERIC,
}

ACTIVE_CONFIG = CONFIGS.get(os.getenv("DOMAIN", "professor_ratings"), PROFESSOR_RATINGS)
