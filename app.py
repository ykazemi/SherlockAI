# app.py
from dotenv import load_dotenv

load_dotenv()
import json
import os
from pathlib import Path
import streamlit as st

from triage_engine import run_full_triage
from audit import append_audit

CASES_PATH = Path("cases.jsonl")


def load_cases():
    cases = []
    for line in CASES_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def evidence_block(ev_list):
    if not ev_list:
        st.caption("No evidence spans found.")
        return
    for ev in ev_list:
        st.code(f"“{ev['quote']}”\n[start={ev['start']}, end={ev['end']}]")


st.set_page_config(page_title="SherlockAI", layout="wide")
st.title("SherlockAI: AI-powered case triage")

cases = load_cases()
case_ids = [c["case_id"] for c in cases]

with st.sidebar:
    import os

with st.sidebar:
    st.header("Case Selector")
    selected_id = st.selectbox("Choose a case", case_ids, index=0)
    run_btn = st.button("Run triage", type="primary")

    st.divider()
    st.subheader("LLM config (debug)")
    st.write("USE_LLM =", os.getenv("USE_LLM"))
    st.write("HF_MODEL =", os.getenv("HF_MODEL"))
    st.write("HF_BASE_URL =", os.getenv("HF_BASE_URL"))
    st.write("HF_TOKEN set? =", "yes" if os.getenv("HF_TOKEN") else "no")

case = next(c for c in cases if c["case_id"] == selected_id)

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Input case")
    st.json(case)
    st.divider()
    st.subheader("Raw text")
    st.text_area("Case text", value=case["text"], height=220)

if "triage_out" not in st.session_state:
    st.session_state.triage_out = None

if run_btn:
    st.session_state.triage_out = run_full_triage(case)

out = st.session_state.triage_out

with right:
    st.subheader("AI output")
    if not out:
        st.info("Click **Run triage** to generate the AI draft outputs.")
    else:
        tabs = st.tabs(
            ["Triage", "Facts", "Questions", "Memo", "Human Decision", "Audit JSON"]
        )

        # Show HF errors (if any)
        q_err = None
        if out.get("missing_info_questions"):
            q_err = out["missing_info_questions"][0].get("_error")
        m_err = out.get("memo", {}).get("_error")

        if q_err or m_err:
            st.error(f"LLM error (questions): {q_err}\n\nLLM error (memo): {m_err}")

        src_q = (
            out["missing_info_questions"][0].get("_source")
            if out.get("missing_info_questions")
            else "unknown"
        )
        src_m = out["memo"].get("_source", "unknown")
        st.caption(f"Questions source: **{src_q}** | Memo source: **{src_m}**")

        with tabs[0]:
            st.metric("Risk Category", out["risk_category"])
            st.metric("Priority", out["priority"])
            st.metric("Routing Queue", out["routing_queue"])
            st.metric("Confidence", f"{out['confidence']:.2f}")
            st.markdown("**Rationale evidence**")
            evidence_block(out.get("rationale_evidence", []))

        with tabs[1]:
            st.markdown("**Structured facts (each should have evidence spans)**")
            st.json(out["facts"])

        with tabs[2]:
            st.markdown("**Missing-info questions**")
            # for q in out["missing_info_questions"]:
            #     st.write(
            #         f"- **{q['question']}**  \n  _Why:_ {q['why']}  \n  _Cost:_ `{q['cost']}`"
            #     )
            for q in out["missing_info_questions"]:
                st.write(
                    f"- **{q['question']}**  \n  _Why:_ {q['why']}  \n  _Cost:_ `{q['cost']}`"
                )
                if q.get("evidence"):
                    for ev in q["evidence"]:
                        st.code(
                            f"“{ev['quote']}”\n[start={ev['start']}, end={ev['end']}]"
                        )
                else:
                    st.caption("No grounding quote found (unknown).")

        with tabs[3]:
            st.markdown("**Internal memo**")
            st.write(out["memo"]["summary"])
            st.write("**Recommendation:**", out["memo"]["recommendation"])
            st.write("**Risks:**")
            for r in out["memo"]["risks"]:
                st.write(f"- {r}")
            st.markdown("**Evidence**")
            evidence_block(out["memo"].get("evidence", []))

        with tabs[4]:
            st.markdown("### Human-only gate")
            st.write("**Decision:**", out["human_only_decision"]["decision"])
            st.write("**Why human:**", out["human_only_decision"]["why_human"])

            st.divider()
            st.markdown("### Reviewer action")
            final_queue = st.selectbox(
                "Final routing queue",
                [
                    "ATO",
                    "Fraud",
                    "Complaints",
                    "Privacy",
                    "MarketIntegrity",
                    "Suitability",
                    "GeneralReview",
                ],
                index=[
                    "ATO",
                    "Fraud",
                    "Complaints",
                    "Privacy",
                    "MarketIntegrity",
                    "Suitability",
                    "GeneralReview",
                ].index(out["routing_queue"]),
            )
            final_priority = st.selectbox(
                "Final priority",
                ["P0", "P1", "P2", "P3"],
                index=["P0", "P1", "P2", "P3"].index(out["priority"]),
            )
            approve = st.radio(
                "Approve AI recommendation?", ["Approve", "Override"], horizontal=True
            )
            reason = st.text_area(
                "Reviewer note / justification",
                height=100,
                placeholder="Why approve or override?",
            )

            if st.button("Save decision to audit log"):
                append_audit(
                    {
                        "case_id": out["case_id"],
                        "ai": {
                            "priority": out["priority"],
                            "routing_queue": out["routing_queue"],
                            "risk_category": out["risk_category"],
                        },
                        "human": {
                            "priority": final_priority,
                            "routing_queue": final_queue,
                            "action": approve,
                            "note": reason,
                        },
                    }
                )
                st.success("Saved to audit_log.jsonl")

        with tabs[5]:
            st.json(out)
