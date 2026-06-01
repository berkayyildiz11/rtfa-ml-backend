import re
from dataclasses import dataclass, replace
from typing import Iterable


RELEVANCE_THRESHOLD = 0.55


@dataclass(frozen=True)
class AliasRule:
    alias: str
    weight: float
    strength: str
    ambiguous: bool = False
    ticker_alias: bool = False


COMPANY_ALIASES = {
    "AAPL": ["apple", "aapl", "iphone", "macbook", "tim cook"],
    "MSFT": ["microsoft", "msft", "windows", "azure", "satya nadella"],
    "GOOGL": ["google", "googl", "alphabet", "sundar pichai", "youtube"],
    "AMZN": ["amazon", "amzn", "aws", "andy jassy", "jeff bezos"],
    "META": ["meta", "facebook", "zuckerberg", "instagram", "whatsapp"],
    "TSLA": ["tesla", "tsla", "elon musk", "model 3", "cybertruck"],
    "NVDA": ["nvidia", "nvda", "jensen huang", "gpu", "geforce"],
    "NFLX": ["netflix", "nflx", "ted sarandos", "streaming"],
    "ADBE": ["adobe", "adbe", "photoshop", "shantanu narayen"],
    "INTC": ["intel", "intc", "pat gelsinger", "pentium", "xeon"],
    "CSCO": ["cisco", "csco", "chuck robbins", "networking"],
    "PEP": ["pepsico", "pepsi", "pep", "ramon laguarta"],
    "AVGO": ["broadcom", "avgo", "hock tan", "vmware"],
    "TXN": ["texas instruments", "txn", "haviv ilan"],
    "QCOM": ["qualcomm", "qcom", "snapdragon", "cristiano amon"],
    "COST": ["costco", "cost", "craig jelinek", "wholesale"],
    "TMUS": ["t-mobile", "tmus", "mike sievert", "sprint"],
    "AMGN": ["amgen", "amgn", "robert bradway"],
    "SBUX": ["starbucks", "sbux", "laxman narasimhan", "coffee"],
    "ISRG": ["intuitive surgical", "isrg", "da vinci", "gary guthart"],
}


STRONG_ALIASES = {
    "AAPL": ["apple", "aapl"],
    "MSFT": ["microsoft", "msft"],
    "GOOGL": ["google", "googl", "alphabet"],
    "AMZN": ["amazon", "amzn"],
    "META": ["facebook", "zuckerberg"],
    "TSLA": ["tesla", "tsla", "elon musk"],
    "NVDA": ["nvidia", "nvda", "jensen huang"],
    "NFLX": ["netflix", "nflx"],
    "ADBE": ["adobe", "adbe"],
    "INTC": ["intel", "intc"],
    "CSCO": ["cisco", "csco"],
    "PEP": ["pepsico", "pepsi"],
    "AVGO": ["broadcom", "avgo"],
    "TXN": ["texas instruments", "txn"],
    "QCOM": ["qualcomm", "qcom"],
    "COST": ["costco"],
    "TMUS": ["t-mobile", "tmus"],
    "AMGN": ["amgen", "amgn"],
    "SBUX": ["starbucks", "sbux"],
    "ISRG": ["intuitive surgical", "isrg"],
}


AMBIGUOUS_ALIASES = {"aws", "cost", "meta", "pep"}


def _build_alias_rules(ticker: str) -> list[AliasRule]:
    aliases = COMPANY_ALIASES.get(ticker, [])
    strong_aliases = set(STRONG_ALIASES.get(ticker, []))
    rules = []

    for alias in aliases:
        is_ambiguous = alias in AMBIGUOUS_ALIASES
        if is_ambiguous:
            weight = 0.35
            strength = "ambiguous"
        elif alias == ticker.lower():
            weight = 0.95
            strength = "ticker"
        elif alias in strong_aliases:
            weight = 0.85
            strength = "company"
        else:
            weight = 0.45
            strength = "weak"

        rules.append(AliasRule(alias, weight, strength, is_ambiguous, alias == ticker.lower()))

    return rules


def _alias_pattern(alias: str) -> re.Pattern:
    escaped_alias = re.escape(alias)
    return re.compile(rf"(?<![A-Za-z0-9]){escaped_alias}(?![A-Za-z0-9])", re.IGNORECASE)


def _matching_rules(text: str, rules: Iterable[AliasRule]) -> list[AliasRule]:
    matches = []

    for rule in rules:
        strongest_match = None
        for match in _alias_pattern(rule.alias).finditer(text):
            matched_text = match.group(0)
            current_match = rule

            if rule.ambiguous and rule.ticker_alias and matched_text == rule.alias.upper():
                current_match = replace(rule, weight=0.95, strength="ticker")
            elif rule.ambiguous and rule.alias == "meta" and matched_text == "Meta":
                current_match = replace(rule, weight=0.85, strength="company")

            if strongest_match is None or current_match.weight > strongest_match.weight:
                strongest_match = current_match

        if strongest_match is not None:
            matches.append(strongest_match)

    return matches


def _score_matches(headline_matches: list[AliasRule], summary_matches: list[AliasRule]) -> float:
    score = 0.0

    for rule in headline_matches:
        score += rule.weight

    summary_only_matches = [
        rule for rule in summary_matches if rule.alias not in {match.alias for match in headline_matches}
    ]
    for rule in summary_only_matches:
        score += rule.weight * 0.65

    if len({rule.alias for rule in headline_matches + summary_matches}) > 1:
        score += 0.15

    return round(min(score, 1.0), 4)


def _relevance_reason(headline_matches: list[AliasRule], summary_matches: list[AliasRule]) -> str:
    all_matches = headline_matches + summary_matches
    if not all_matches:
        return "no_alias_match"

    headline_strengths = {rule.strength for rule in headline_matches}
    unique_aliases = {rule.alias for rule in all_matches}

    if "ticker" in headline_strengths:
        return "headline_ticker_match"
    if "company" in headline_strengths:
        return "headline_company_match"
    if len(unique_aliases) > 1:
        return "summary_multiple_alias_match"
    if headline_matches:
        return "headline_weak_match"
    return "summary_alias_match"


def analyze_article_relevance(ticker: str, headline: str = "", summary: str = "") -> dict:
    """
    Returns relevance metadata for a news article and ticker.
    """
    ticker = ticker.upper()
    rules = _build_alias_rules(ticker)

    if not rules:
        return {
            "is_relevant": True,
            "relevance_score": 1.0,
            "relevance_reason": "unknown_ticker_allowed",
            "matched_aliases": [],
        }

    headline_text = headline or ""
    summary_text = summary or ""
    headline_matches = _matching_rules(headline_text, rules)
    summary_matches = _matching_rules(summary_text, rules)
    score = _score_matches(headline_matches, summary_matches)
    matched_aliases = sorted({rule.alias for rule in headline_matches + summary_matches})

    return {
        "is_relevant": score >= RELEVANCE_THRESHOLD,
        "relevance_score": score,
        "relevance_reason": _relevance_reason(headline_matches, summary_matches),
        "matched_aliases": matched_aliases,
    }


def is_article_relevant(ticker: str, headline: str, summary: str) -> bool:
    """
    Compatibility wrapper for callers that only need a boolean result.
    """
    return analyze_article_relevance(ticker, headline, summary)["is_relevant"]
