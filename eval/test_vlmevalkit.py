"""Smoke test: instantiate NanoVLM adapter and run generate_inner on a synthetic sample."""
import sys
import os

# Ensure VLMEvalKit is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
vlmevalkit_path = os.path.join(repo_root, 'VLMEvalKit')
if vlmevalkit_path not in sys.path:
    sys.path.insert(0, vlmevalkit_path)

from vlmeval.config import supported_VLM

MODEL_NAME = os.environ.get('TEST_MODEL', 'nanoVLM-460M-8k')
print(f"=== Step 1: Instantiate {MODEL_NAME} ===")
model_cls = supported_VLM[MODEL_NAME]
model = model_cls()
print("Model loaded successfully on CUDA")

print("\n=== Step 2: Test with a dummy text-only message ===")
message_text = [{"type": "text", "value": "What is 2 + 2?"}]
result_text = model.generate_inner(message_text, dataset=None)
print(f"Text-only result: '{result_text}'")
assert len(result_text) > 0, "FAIL: empty text-only response"
print("PASS: non-empty text response")

print("\n=== Step 3: Test with image + text ===")
asset_path = os.path.join(repo_root, 'assets', 'image.png')
if os.path.exists(asset_path):
    message_img = [
        {"type": "image", "value": asset_path},
        {"type": "text", "value": "What is in this image?"},
    ]
    result_img = model.generate_inner(message_img, dataset=None)
    print(f"Image+text result: '{result_img}'")
    assert len(result_img) > 0, "FAIL: empty image response"
    print("PASS: non-empty image response")
else:
    print(f"SKIP: test asset not found at {asset_path}")

print("\n=== Step 4: Test MCQ prompt formatting (MMStar-style) ===")
if os.path.exists(asset_path):
    message_mcq = [
        {"type": "image", "value": asset_path},
        {"type": "text", "value": "What animal is in the image?\nOptions:\nA. Dog\nB. Cat\nC. Bird\nD. Fish\nPlease select the correct answer from the options above."},
    ]
    result_mcq = model.generate_inner(message_mcq, dataset="MMStar")
    print(f"MCQ result: '{result_mcq}'")
    assert len(result_mcq) > 0, "FAIL: empty MCQ response"
    print("PASS: non-empty MCQ response")

print("\n=== ALL TESTS PASSED ===")
