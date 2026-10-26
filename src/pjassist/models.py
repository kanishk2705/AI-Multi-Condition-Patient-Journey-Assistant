"""Model identifiers and text formatting shared by CI scripts, the evaluation and the backend."""
EMB_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge retrieval instruction
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"   # labels: 0 contradiction, 1 entailment, 2 neutral
NLI_ENTAIL_IDX = 1
LLM_MODEL = "openai/gpt-oss-120b"
LLM_SETTINGS = {"temperature": 0.0, "max_tokens": 700, "reasoning_effort": "low", "seed": 7}


def passage_text(p: dict) -> str:
    """What the dense and cross-encoder models see: title, section heading and text."""
    head = p["title"] + (" - " + p["section"] if p.get("section") else "")
    return head + ". " + p["text"]
