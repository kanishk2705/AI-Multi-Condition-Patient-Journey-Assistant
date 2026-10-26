from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"
SEED = 20261024

CONDITIONS = ("asthma", "diabetes")          # "both" = passage/question concerns the two together
STAGES = ("understand", "manage", "warning")  # PROTOCOL.md section 2 defines them
