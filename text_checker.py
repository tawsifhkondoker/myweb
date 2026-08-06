"""
Text credibility pipeline with trained DistilBERT + rule-based fallback.
"""
import os
import re
import requests

FACT_CHECK_API_KEY = os.environ.get("GOOGLE_FACT_CHECK_API_KEY", "")
FACT_CHECK_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# ------------------------------------------------------------------
# 1. Load the trained DistilBERT model from Hugging Face
# ------------------------------------------------------------------
try:
    from transformers import pipeline

    # 👇 CHANGE THIS TO YOUR ACTUAL HUGGING FACE MODEL ID
    # If you uploaded to HF, use: "tawsifhkondoker/claim-classifier"
    # If you haven't uploaded yet, keep using the local folder for now.
    MODEL_ID = "tawsif23984/claim-classifier"  # local folder fallback

    claim_pipeline = pipeline("text-classification", model=MODEL_ID)
    USE_ML = True
    print("✅ Loaded trained claim classifier (DistilBERT).")
except Exception as e:
    USE_ML = False
    print(f"⚠️ Text ML model not found, using rule-based heuristic: {e}")

# ------------------------------------------------------------------
# 2. Rule-based fallback (kept for when model isn't available)
# ------------------------------------------------------------------
OPINION_MARKERS = {
    "i think", "i feel", "i believe", "in my opinion", "i love", "i hate",
    "best", "worst", "amazing", "terrible", "beautiful", "ugly",
}
CLAIM_HINTS = re.compile(
    r"\b(\d{4}|\d+%|percent|million|billion|died|caused|announced|"
    r"banned|approved|study shows|according to|confirmed|reported)\b",
    re.IGNORECASE,
)

def is_claim_heuristic(text: str) -> bool:
    lowered = text.lower()
    if any(marker in lowered for marker in OPINION_MARKERS):
        return False
    if CLAIM_HINTS.search(text):
        return True
    has_capitalized_word = bool(re.search(r"(?<!^)[A-Z][a-z]{2,}", text))
    return len(text.split()) > 6 and has_capitalized_word

def is_claim_ml(text: str) -> bool:
    result = claim_pipeline(text)[0]
    # LABEL_1 = claim, LABEL_0 = not claim
    return result["label"] == "LABEL_1" and result["score"] > 0.5

def is_claim(text: str) -> bool:
    if USE_ML:
        return is_claim_ml(text)
    else:
        return is_claim_heuristic(text)

# ------------------------------------------------------------------
# 3. Fact-check API search (unchanged)
# ------------------------------------------------------------------
def search_factcheck_db(claim_text: str) -> list[dict]:
    if not FACT_CHECK_API_KEY:
        return []
    params = {
        "query": claim_text[:200],
        "key": FACT_CHECK_API_KEY,
        "languageCode": "en",
    }
    try:
        resp = requests.get(FACT_CHECK_ENDPOINT, params=params, timeout=5)
        resp.raise_for_status()
    except requests.RequestException:
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
            "message": "This looks like a factual claim, but no existing fact-check was found. That does not mean it's true -- it just hasn't been checked yet.",
            "matches": [],
        }
    return {
        "type": "text",
        "is_claim": True,
        "verdict": "matched",
        "message": f"Found {len(matches)} existing fact-check(s) for a similar claim.",
        "matches": matches,
    }
