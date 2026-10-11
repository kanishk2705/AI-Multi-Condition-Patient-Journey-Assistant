"""Streamlit interface for the educational patient-journey prototype."""
from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from patient_journey_assistant.answer_generation import answer_question
from patient_journey_assistant.manifest import ALLOWED_CONDITIONS, ALLOWED_STAGES
from patient_journey_assistant.retrieval import create_persistent_retriever

load_dotenv()

CONDITION_LABELS = {
    "asthma": "Asthma",
    "type_2_diabetes": "Type 2 diabetes",
}
STAGE_LABELS = {
    "understanding_condition": "Understanding the condition",
    "appointment_preparation": "Preparing for an appointment or test",
    "ongoing_care_information": "General ongoing-care information",
}

st.set_page_config(
    page_title="Patient Journey Assistant",
    page_icon="🩺",
    layout="centered",
)
st.title("AI Multi-Condition Patient Journey Assistant")
st.caption("Educational health information grounded in a small collection of MedlinePlus sources.")

st.warning(
    "Educational use only. This prototype cannot diagnose, triage emergencies, interpret individual "
    "test results, prescribe, or recommend personal medication/treatment changes. Do not enter names, "
    "medical-record details, or other identifying or personal health information."
)
st.markdown(
    "Choose a condition and information stage, then ask a general question. The app uses the selected "
    "source passages to prepare a concise explanation and displays source links for every presented claim."
)

with st.form("question_form", clear_on_submit=False):
    condition = st.selectbox(
        "Condition",
        options=sorted(ALLOWED_CONDITIONS),
        format_func=lambda value: CONDITION_LABELS[value],
    )
    stage = st.selectbox(
        "Information stage",
        options=sorted(ALLOWED_STAGES),
        format_func=lambda value: STAGE_LABELS[value],
    )
    question = st.text_area(
        "General educational question",
        placeholder="For example: What happens during lung function testing?",
        height=110,
        help="Do not enter personal medical details or ask for diagnosis, treatment, or medication advice.",
    )
    submitted = st.form_submit_button("Find an evidence-grounded answer", type="primary")

if submitted:
    if not question.strip():
        st.info("Enter a general educational question to continue.")
    else:
        with st.spinner("Checking scope and searching approved sources…"):
            response = answer_question(
                question,
                condition=condition,
                stage=stage,
                retriever=lambda: create_persistent_retriever(
                    index_dir=os.getenv("PJA_INDEX_DIR", "data/local/chroma"),
                    collection_name=os.getenv("PJA_COLLECTION_NAME", "patient_journey_chunks"),
                ),
            )

        if response.status == "answered":
            st.subheader("Answer")
            st.caption(response.message)
            source_by_id = {item.chunk_id: item for item in response.sources}
            for claim in response.claims:
                st.markdown(claim.text)
                citation_links = []
                for chunk_id in claim.source_chunk_ids:
                    source = source_by_id.get(chunk_id)
                    if source:
                        citation_links.append(
                            f"[{source.title} — source passage](<{source.source_url}>)"
                        )
                if citation_links:
                    st.caption("Evidence: " + " · ".join(citation_links))
            if response.model_name:
                st.caption(f"Generation model: {response.model_name}")
        elif response.status in {"redirected", "professional_redirect"}:
            st.warning(response.message)
        elif response.status == "configuration_missing":
            st.error(response.message)
        elif response.status == "service_unavailable":
            st.error(response.message)
        else:
            st.info(response.message)

st.divider()
st.caption(
    "Privacy and scope: submitted questions are used transiently for this request and are not written "
    "to an application database or custom logs by this prototype. For an in-scope request, the question "
    "and retrieved passages are sent to the configured language-model provider; follow that provider's "
    "terms and do not submit personal or identifying health information."
)
