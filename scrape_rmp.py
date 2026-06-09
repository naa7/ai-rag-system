"""
Scrape RateMyProfessors reviews via their public GraphQL API.

Targets Georgia Tech CS professors with the most ratings, saves each as a .txt
file in documents/ using the format expected by src/ingest.py:

    Professor: <name>
    Course: <course>
    School: <school>
    Department: <department>
    ---
    <review paragraphs separated by blank lines>

Run: python scrape_rmp.py
"""
import json
import os
import re
import time
import requests

GQL_URL = "https://www.ratemyprofessors.com/graphql"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Authorization": "Basic dGVzdDp0ZXN0",
    "Content-Type": "application/json",
}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "documents")
SCHOOL_ID = "U2Nob29sLTM2MQ=="   # Georgia Institute of Technology
TARGET_DEPT = "Computer Science"
MIN_RATINGS = 10
MAX_PROFESSORS = 15
REVIEWS_PER_PROF = 25


def gql(query: str, variables: dict) -> dict:
    resp = requests.post(GQL_URL, json={"query": query, "variables": variables},
                         headers=HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_cs_professors() -> list[dict]:
    query = """
    query SearchProfessors($schoolId: ID!) {
      newSearch {
        teachers(query: {text: "", schoolID: $schoolId}, first: 1000) {
          edges {
            node {
              id legacyId firstName lastName department
              avgRating avgDifficulty numRatings wouldTakeAgainPercent
            }
          }
        }
      }
    }
    """
    data = gql(query, {"schoolId": SCHOOL_ID})
    edges = data["data"]["newSearch"]["teachers"]["edges"]
    cs_profs = [
        e["node"] for e in edges
        if e["node"]["department"] == TARGET_DEPT
        and e["node"]["numRatings"] >= MIN_RATINGS
    ]
    cs_profs.sort(key=lambda p: p["numRatings"], reverse=True)
    return cs_profs[:MAX_PROFESSORS]


def fetch_ratings(prof_id: str) -> list[dict]:
    query = """
    query ProfRatings($id: ID!) {
      node(id: $id) {
        ... on Teacher {
          ratings(first: %d) {
            edges {
              node {
                comment class qualityRating difficultyRatingRounded
                wouldTakeAgain grade date
              }
            }
          }
        }
      }
    }
    """ % REVIEWS_PER_PROF
    data = gql(query, {"id": prof_id})
    edges = data["data"]["node"]["ratings"]["edges"]
    return [e["node"] for e in edges if e["node"]["comment"].strip()]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save_professor(prof: dict, ratings: list[dict]) -> str:
    name = f"{prof['firstName']} {prof['lastName']}"
    # Use the most common course code across ratings as the "course" field
    courses = [r["class"] for r in ratings if r.get("class")]
    course = max(set(courses), key=courses.count) if courses else "N/A"

    lines = [
        f"Professor: {name}",
        f"Course: {course}",
        f"School: Georgia Institute of Technology",
        f"Department: {prof['department']}",
        f"Avg Rating: {prof['avgRating']} / 5",
        f"Avg Difficulty: {prof['avgDifficulty']} / 5",
        f"Num Ratings: {prof['numRatings']}",
        "---",
    ]

    for r in ratings:
        comment = r["comment"].strip()
        if not comment:
            continue
        # Attach brief metadata as a prefix so each chunk carries context
        meta_parts = []
        if r.get("class"):
            meta_parts.append(r["class"])
        if r.get("qualityRating"):
            meta_parts.append(f"Quality {r['qualityRating']}/5")
        if r.get("difficultyRatingRounded"):
            meta_parts.append(f"Difficulty {r['difficultyRatingRounded']}/5")
        if r.get("grade"):
            meta_parts.append(f"Grade: {r['grade']}")
        prefix = f"[{', '.join(meta_parts)}] " if meta_parts else ""
        lines.append(prefix + comment)
        lines.append("")  # blank line between reviews

    fname = os.path.join(OUT_DIR, f"rmp_{slug(name)}.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return fname


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Fetching Georgia Tech CS professors with >= {MIN_RATINGS} ratings...")
    profs = fetch_cs_professors()
    print(f"Found {len(profs)} professors to scrape\n")

    for i, prof in enumerate(profs, 1):
        name = f"{prof['firstName']} {prof['lastName']}"
        print(f"[{i}/{len(profs)}] {name} ({prof['numRatings']} ratings)...", end=" ", flush=True)
        try:
            ratings = fetch_ratings(prof["id"])
            path = save_professor(prof, ratings)
            print(f"saved {len(ratings)} reviews → {os.path.basename(path)}")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(0.4)  # be polite

    print(f"\nDone. Files written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
