"""
Text credibility pipeline.

Two stages, kept deliberately separate (see project notes on why
squashing these into one score is a bad idea):

  1. is_claim()      -> is this even a checkable factual claim, or just
                         opinion / small talk? (rule-based starter --
                         swap in a fine-tuned BERT classifier later,
                         see /train/train_text_classifier.py)

  2. search_factcheck_db() -> has this specific claim already been
                         fact-checked by a real publisher? Uses
                         Google's free Fact Check Tools API.

evaluate_text() combines both into a result that keeps
"no match found" clearly separate from "matched a known false claim".
"""

import os
import re
import requests

FACT_CHECK_API_KEY = os.environ.get("GOOGLE_FACT_CHECK_API_KEY", "")
FACT_CHECK_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Words that suggest opinion/sentiment rather than a checkable claim.
OPINION_MARKERS = {
    "i think", "i feel", "i believe", "in my opinion", "i love", "i hate",
    "best", "worst", "amazing", "terrible", "beautiful", "ugly",
}

# A checkable claim usually has: a subject doing/being something, and
# often a number, date, or named entity. This is a coarse starter --
# not a real claim-detection model. Good enough to filter out obvious
# non-claims cheaply before spending an API call.
CLAIM_HINTS = re.compile(
    r"\b(\d{4}|\d+%|percent|million|billion|died|caused|announced|"
    r"banned|approved|study shows|according to|confirmed|reported)\b",
    re.IGNORECASE,
)


def is_claim(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in OPINION_MARKERS):
        return False
    if CLAIM_HINTS.search(text):
        return True
    # Fallback heuristic: reasonably long declarative sentences with a
    # named-entity-looking capitalized word are worth checking.
    has_capitalized_word = bool(re.search(r"(?<!^)[A-Z][a-z]{2,}", text))
    return len(text.split()) > 6 and has_capitalized_word


def search_factcheck_db(claim_text: str) -> list[dict]:
    """Query Google's Fact Check Tools API for existing fact-checks
    that match this claim. Returns a list of {publisher, rating, url,
    title} dicts, or [] if nothing matches or no API key is set."""
    if not FACT_CHECK_API_KEY:
        return []

    params = {
        "query": claim_text[:200],  # API has a query length limit
        "key": FACT_CHECK_API_KEY,
        "languageCode": "en",
    }
    try:
        resp = requests.get(FACT_CHECK_ENDPOINT, params=params, timeout=5)
        resp.raise_for_status()
    except requests.RequestException:
        # Fail open: if the API is unreachable, report "no match found"
        # rather than crashing the whole check.
        return []

    data = resp.json()
    matches = []
    for claim in data.get("claims", [])[:5]:
        for review in claim.get("claimReview", []):
            matches.append({
                "publisher": review.get("publisher", {}).get("name", "unknown"),
                "rating": review.get("textualRating", "unrated"),
                "url": review.get("url", ""),
                "title": claim.get("text", claim_text),
            })
    return matches


def evaluate_text(text: str) -> dict:
    claim_detected = is_claim(text)

    if not claim_detected:
        return {
            "type": "text",
            "is_claim": False,
            "verdict": "not_a_claim",
            "message": "This looks like opinion or commentary, not a checkable factual claim.",
            "matches": [],
        }

    matches = search_factcheck_db(text)

    if not matches:
        return {
            "type": "text",
            "is_claim": True,
            "verdict": "unverified",
            "message": "This looks like a factual claim, but no existing fact-check was found. "
                       "That does not mean it's true -- it just hasn't been checked yet.",
            "matches": [],
        }

    return {
        "type": "text",
        "is_claim": True,
        "verdict": "matched",
        "message": f"Found {len(matches)} existing fact-check(s) for a similar claim.",
        "matches": matches,
    }
