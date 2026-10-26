set -e
pip install -q -r requirements-ci.txt onnxruntime tokenizers
python scripts/check_onnx.py
