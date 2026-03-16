# triage_engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import re
import os
from typing import Optional
from llm_client_hf_serverless import (
    call_hf_serverless_json as call_hf_chat_json,
    LLMError,
)

# from llm_client_hf import call_hf_chat_json, LLMError
from llm_prompts import QUESTIONS_SYSTEM, MEMO_SYSTEM


def _is_json_parse_error(err: Exception) -> bool:
    # Be broad: HF can throw different wrappers; we just retry once on "looks like parsing"
    msg = str(err).lower()
    return (
        ("json" in msg)
        or ("expecting value" in msg)
        or ("did not return a json" in msg)
    )


def _retry_suffix() -> str:
    return (
        "\n\nSTRICT OUTPUT RULE: Return ONLY a single JSON object. "
        "No markdown. No prose. No backticks. Start with '{' and end with '}'."
    )


def use_llm() -> bool:
    return os.getenv("USE_LLM", "0") == "1"


def locate_quote_span(text: str, quote: str) -> Optional[dict]:
    if not quote or quote == "unknown":
        return None
    idx = text.lower().find(quote.lower())
    if idx == -1:
        return None
    return {
        "quote": text[idx : idx + len(quote)],
        "start": idx,
        "end": idx + len(quote),
    }


def enforce_question_grounding(text: str, questions: list) -> list:
    out = []
    for q in questions:
        q = dict(q)
        quote = (q.get("evidence_quote") or "unknown").strip()

        # NEW: allow "missing" to mean "this info is absent; question is about missing data"
        if quote.lower() == "missing":
            q["evidence"] = []
            q["evidence_quote"] = "missing"
            out.append(q)
            continue

        span = locate_quote_span(text, quote)
        q["evidence"] = [span] if span else []
        if not span:
            q["evidence_quote"] = "unknown"
        out.append(q)
    return out


RISK_CATEGORIES = [
    "account_takeover",
    "fraud",
    "complaint",
    "privacy",
    "market_abuse",
    "suitability",
    "other",
    "unknown",
]


def find_spans(
    text: str, patterns: List[str], max_spans: int = 3
) -> List[Dict[str, Any]]:
    """
    Return evidence spans (quote + start/end offsets) for the first matches of patterns.
    """
    evidence = []
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            start, end = m.start(), m.end()
            quote = text[start:end]
            evidence.append({"quote": quote, "start": start, "end": end})
        if len(evidence) >= max_spans:
            break
    return evidence


def extract_amount(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    m = re.search(r"\$[\s]*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?)", text)
    if not m:
        return "unknown", []
    start, end = m.start(), m.end()
    return text[start:end], [{"quote": text[start:end], "start": start, "end": end}]


def classify_case(case: Dict[str, Any]) -> Dict[str, Any]:
    text = case["text"]

    # Simple keyword signals (deterministic + demo-friendly)
    signals = {
        "account_takeover": [
            r"login attempt",
            r"didn't try to log in",
            r"code I didn't request",
            r"lock my account",
            r"new phone number",
            r"new device",
            r"locked out",
        ],
        "fraud": [
            r"withdrawal",
            r"didn't request",
            r"pending",
            r"cancel",
            r"unauthorized",
        ],
        "complaint": [
            r"complaint",
            r"compensation",
            r"unacceptable",
            r"misled",
            r"blame",
        ],
        "privacy": [
            r"someone.*looked at my account",
            r"data",
            r"accessed",
            r"knew my exact holdings",
            r"how is that possible",
        ],
        "market_abuse": [
            r"spoof",
            r"manipulat",
            r"40 orders",
            r"restricted",
            r"day trader",
        ],
        "suitability": [r"low risk", r"retiring", r"90% equities", r"onboarding quiz"],
    }

    matched = []
    for cat, pats in signals.items():
        ev = find_spans(text, pats, max_spans=1)
        if ev:
            matched.append((cat, ev))

    if not matched:
        category = "unknown"
        evidence = []
        confidence = 0.25
    else:
        # pick best by a rough priority order
        priority_order = [
            "account_takeover",
            "fraud",
            "privacy",
            "market_abuse",
            "suitability",
            "complaint",
            "other",
        ]
        matched.sort(
            key=lambda x: priority_order.index(x[0]) if x[0] in priority_order else 999
        )
        category, evidence = matched[0]
        confidence = 0.72 if category in ("account_takeover", "fraud") else 0.62

    routing = {
        "account_takeover": ("P0", "ATO"),
        "fraud": ("P0", "Fraud"),
        "privacy": ("P1", "Privacy"),
        "market_abuse": ("P1", "MarketIntegrity"),
        "suitability": ("P2", "Suitability"),
        "complaint": ("P2", "Complaints"),
        "other": ("P3", "GeneralReview"),
        "unknown": ("P3", "GeneralReview"),
    }
    prio, queue = routing[category]

    return {
        "risk_category": category,
        "priority": prio,
        "routing_queue": queue,
        "confidence": confidence,
        "rationale_evidence": evidence,
    }


def extract_facts(case: Dict[str, Any], risk_category: str) -> List[Dict[str, Any]]:
    text = case["text"]
    facts: List[Dict[str, Any]] = []

    amount, amount_ev = extract_amount(text)
    facts.append({"field": "mentioned_amount", "value": amount, "evidence": amount_ev})

    # Common factual fields
    def add(field: str, value: str, ev: List[Dict[str, Any]]):
        facts.append({"field": field, "value": value, "evidence": ev})

    # Device / login related
    ev_login = find_spans(
        text,
        [
            r"login attempt",
            r"locked out",
            r"code I didn't request",
            r"changed phones",
            r"new phone number",
        ],
        2,
    )
    add("login_or_device_issue", "yes" if ev_login else "unknown", ev_login)

    # Unauthorized action hints
    ev_unauth = find_spans(
        text,
        [r"didn't request", r"unauthorized", r"I didn't .*request", r"didn't try"],
        2,
    )
    add("unauthorized_claim", "yes" if ev_unauth else "unknown", ev_unauth)

    # Destination bank hints
    ev_dest = find_spans(text, [r"withdraw.*to", r"Bank of", r"RBC"], 2)
    add("destination_mentioned", "yes" if ev_dest else "unknown", ev_dest)

    # Complaint / compensation
    ev_comp = find_spans(
        text, [r"complaint", r"compensation", r"misled", r"unacceptable"], 2
    )
    add("complaint_signal", "yes" if ev_comp else "unknown", ev_comp)

    # Privacy allegation
    ev_priv = find_spans(
        text,
        [r"looked at my account", r"data was accessed", r"knew my exact holdings"],
        2,
    )
    add("privacy_signal", "yes" if ev_priv else "unknown", ev_priv)

    return facts


def missing_questions(case: Dict[str, Any], risk_category: str) -> List[Dict[str, Any]]:
    text = case["text"]

    def fallback(source: str = "fallback", err: str | None = None):
        qs = [
            {
                "question": "Can you confirm what happened step-by-step and what outcome you want?",
                "why": "Clarifies intent and the safest next action.",
                "cost": "cheap",
                "evidence_quote": "unknown",
                "evidence": [],
                "_source": source,
            }
        ]
        if err:
            qs[0]["_error"] = err
        return qs

    if not use_llm():
        return fallback(source="fallback")

    # Tight prompt to reduce token usage + improve JSON adherence
    base_user = (
        "Return ONLY JSON with key missing_info_questions.\n"
        "Each item MUST have: question, why, cost, evidence_quote.\n"
        "cost must be one of: cheap|medium|expensive.\n"
        "evidence_quote must be either:\n"
        "- a SHORT exact quote from the case text that motivates the question, OR\n"
        '- the string "missing" if the question is about information not present in the case text.\n'
        "Ask 3–5 questions. Prefer cheap questions first.\n"
        "No extra keys.\n\n"
        f"risk_category: {risk_category}\n"
        f"metadata: product={case.get('product')}, flags={case.get('known_flags')}, account_age_days={case.get('account_age_days')}\n"
        f"case_text:\n{text}"
    )

    def call_once(user_msg: str):
        return call_hf_chat_json(
            system=QUESTIONS_SYSTEM,
            user=user_msg,
            temperature=0.15,
            max_tokens=350,
        )

    try:
        resp = call_once(base_user)
    except (LLMError, Exception) as e:
        return fallback(source="fallback_error", err=str(e))

    # If we got a response but parsing/shape is weird, retry once with stricter suffix.
    try:
        questions = resp.get("missing_info_questions", [])
        if not isinstance(questions, list):
            raise ValueError("missing_info_questions is not a list")
    except Exception as e:
        if _is_json_parse_error(e):
            try:
                resp = call_once(base_user + _retry_suffix())
                questions = resp.get("missing_info_questions", [])
                if not isinstance(questions, list):
                    raise ValueError("missing_info_questions is not a list")
            except Exception as e2:
                return fallback(source="fallback_error", err=str(e2))
        else:
            return fallback(source="fallback_error", err=str(e))

    # Grounding pass + limit
    questions = enforce_question_grounding(text, questions)[:5]

    cleaned: List[Dict[str, Any]] = []
    for q in questions:
        cost = (q.get("cost") or "cheap").strip().lower()
        if cost not in ("cheap", "medium", "expensive"):
            cost = "cheap"

        cleaned.append(
            {
                "question": (q.get("question") or "unknown").strip(),
                "why": (q.get("why") or "unknown").strip(),
                "cost": cost,
                "evidence_quote": (q.get("evidence_quote") or "unknown").strip(),
                "evidence": q.get("evidence", []),
                "_source": "hf_llm",
            }
        )

    return (
        cleaned
        if cleaned
        else fallback(source="fallback_error", err="Empty questions list from model")
    )


def draft_memo(
    case: Dict[str, Any],
    triage: Dict[str, Any],
    facts: List[Dict[str, Any]],
    questions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    text = case["text"]

    def fallback(source: str = "fallback", err: str | None = None):
        ev = triage.get("rationale_evidence", [])
        m = {
            "summary": f"Case {case['case_id']} triaged as {triage['risk_category'].replace('_', ' ')}.",
            "recommendation": "Gather missing information and route to the appropriate queue; do not take irreversible actions without human approval.",
            "risks": [
                "Incomplete information; unsupported claims treated as unknown.",
                "Human must approve high-impact actions (restriction/escalation).",
            ],
            "evidence": ev,
            "_source": source,
        }
        if err:
            m["_error"] = err
        return m

    if not use_llm():
        return fallback(source="fallback")

    base_user = (
        "Return ONLY JSON with key memo.\n"
        "memo MUST have: summary, recommendation, risks, evidence_quotes.\n"
        "summary: max 2 sentences.\n"
        "recommendation: concise, actionable (1–3 short bullets as a single string is OK).\n"
        "risks: list of 2–4 short strings.\n"
        "evidence_quotes: list of 1–3 SHORT exact quotes from the case text (or empty list if none).\n"
        "No extra keys.\n\n"
        f"triage: category={triage['risk_category']}, priority={triage['priority']}, queue={triage['routing_queue']}\n"
        f"case_text:\n{text}"
    )

    def call_once(user_msg: str):
        return call_hf_chat_json(
            system=MEMO_SYSTEM,
            user=user_msg,
            temperature=0.15,
            max_tokens=450,
        )

    try:
        resp = call_once(base_user)
    except (LLMError, Exception) as e:
        return fallback(source="fallback_error", err=str(e))

    try:
        memo = resp.get("memo", {})
        if not isinstance(memo, dict):
            raise ValueError("memo is not an object")
    except Exception as e:
        if _is_json_parse_error(e):
            try:
                resp = call_once(base_user + _retry_suffix())
                memo = resp.get("memo", {})
                if not isinstance(memo, dict):
                    raise ValueError("memo is not an object")
            except Exception as e2:
                return fallback(source="fallback_error", err=str(e2))
        else:
            return fallback(source="fallback_error", err=str(e))

    # Convert evidence_quotes → spans
    ev_spans = []
    for q in (memo.get("evidence_quotes") or [])[:3]:
        if not isinstance(q, str):
            continue
        span = locate_quote_span(text, q.strip())
        if span:
            ev_spans.append(span)

    risks = memo.get("risks", ["unknown"])
    if not isinstance(risks, list):
        risks = ["unknown"]
    risks = [str(r).strip() for r in risks][:4]

    return {
        "summary": str(memo.get("summary", "unknown")).strip(),
        "recommendation": str(memo.get("recommendation", "unknown")).strip(),
        "risks": risks if risks else ["unknown"],
        "evidence": ev_spans if ev_spans else triage.get("rationale_evidence", []),
        "_source": "hf_llm",
    }


def human_gate_for(category: str) -> Dict[str, Any]:
    return {
        "decision": "Approve any account restriction / money-movement stop / regulatory escalation",
        "why_human": "Client-impact + legal/regulatory consequences require human judgment and context beyond the ticket.",
    }


def run_full_triage(case: Dict[str, Any]) -> Dict[str, Any]:
    tri = classify_case(case)
    facts = extract_facts(case, tri["risk_category"])
    qs = missing_questions(case, tri["risk_category"])
    memo = draft_memo(case, tri, facts, qs)

    return {
        "case_id": case["case_id"],
        "risk_category": tri["risk_category"],
        "priority": tri["priority"],
        "routing_queue": tri["routing_queue"],
        "confidence": tri["confidence"],
        "facts": facts,
        "missing_info_questions": qs,
        "memo": memo,
        "human_only_decision": human_gate_for(tri["risk_category"]),
        # for display/debug
        "rationale_evidence": tri.get("rationale_evidence", []),
    }
