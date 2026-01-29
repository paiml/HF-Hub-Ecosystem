# QA Falsification Protocol: Capstone - Multi-Modal Content Analyzer

**Document ID:** QA-CAP-001
**Date:** 2026-01-29
**Audience:** External QA Engineering Team
**Philosophy:** Popperian Falsification & Toyota Production System (Genchi Genbutsu)

---

## 1. The Core Hypothesis (H0)
> **Hypothesis:** "The provided Capstone Project documentation and code snippets constitute a reproducible, robust, and achievable engineering task for a Course 1 graduate using standard consumer hardware (CPU-only, 8GB RAM), completing within 6-8 hours."

**QA Mandate:** Your job is **NOT** to prove this hypothesis true. Your job is to attempt to **falsify** it. We adhere to the principle that a system is only valid if it survives rigorous attempts to break it. If you find a single blocking error, ambiguity, or resource violation, the Hypothesis is **REJECTED**.

---

## 2. Falsification Vectors (The "Five Whys")

We will attack the project from five distinct vectors. Failure in any vector constitutes a "Stop the Line" (Jidoka) event.

### Vector 1: The "Fresh Install" Falsification
**Hypothesis:** The environment setup instructions are insufficient.
*   **Attack:** Execute the project in a completely clean, isolated container (Docker/Podman) or a fresh `uv` environment.
*   **Success Condition:** The code fails to import `transformers`, `torch`, or `datasets`.
*   **Test Command:**
    ```bash
    # Try to break the setup
    git clone https://github.com/paiml/HF-Hub-Ecosystem.git
    cd HF-Hub-Ecosystem
    make setup
    source .venv/bin/activate
    python3 -c "import transformers; import torch; import datasets; print('Imports successful')"
    ```

### Vector 2: The "Deprecation" Falsification
**Hypothesis:** The hardcoded model IDs in the instruction are already deprecated, gated, or deleted from the Hub.
*   **Attack:** Attempt to load every specific model ID listed in the Phase 3 & 5 snippets.
*   **Target Models:**
    *   `distilbert-base-uncased-finetuned-sst-2-english`
    *   `google/vit-base-patch16-224`
    *   `openai/whisper-tiny`
    *   `Salesforce/blip-image-captioning-base`
*   **Genchi Genbutsu (Go and See):** Do not assume they work. Run a script to pull them.
*   **Pass Criteria:** All models download without an authentication token (public access).

### Vector 3: The "Resource Hog" Falsification
**Hypothesis:** The project requires a GPU or massive RAM, violating the "Standard Hardware" constraint.
*   **Attack:** Run the `MultiModalAnalyzer` initialization and the `analyze_content` function on a constrained VM (e.g., 2 vCPUs, 4GB RAM) or enforce a memory limit.
*   **Success Condition (for QA):** The process receives a `SIGKILL` (OOM) or takes > 30 seconds for a single inference, rendering the "interactive" nature of the demo false.
*   **Key Test:** `whisper-tiny` and `blip` running concurrently in memory.

### Vector 4: The "Copy-Paste" Falsification
**Hypothesis:** The code snippets in the README/Notebook contain syntax errors, undefined variables, or indentation issues when copied directly.
*   **Attack:** Literally copy-paste the code blocks from the "Phase 3" and "Phase 5" sections into a `test_capstone.py` file and run it.
*   **Success Condition (for QA):** Python throws a `NameError` (e.g., `device` is defined in one block but used in another without context) or `IndentationError`.

### Vector 5: The "Streaming" Falsification
**Hypothesis:** The Common Voice dataset (Phase 2) is too large to handle without precise streaming implementation, and the provided snippet is flaky.
*   **Attack:** Run the dataset loading snippet.
*   **Success Condition (for QA):** The script hangs indefinitely or downloads the full dataset despite `streaming=True`.

---

## 3. The QA Execution Script (Genchi Genbutsu)

To validate the hypothesis, execute the following "falsification script". If this script exits with code 0, the project survives. If it fails, the project is rejected.

**File:** `tests/qa_falsify_capstone.py` (You must create and run this)

```python
import sys
import time
import torch
from transformers import pipeline
from datasets import load_dataset
from huggingface_hub import HfApi

def log(msg):
    print(f"[QA-FALSIFY] {msg}")

def fail(msg):
    print(f"\n[FATAL] Falsification Successful: {msg}")
    sys.exit(1)

log("Initiating Stress Test of Capstone Project...")

# 1. Dependency Check
try:
    import transformers, datasets, torch
    log("Dependencies loaded.")
except ImportError as e:
    fail(f"Missing dependency: {e}")

# 2. Model Availability Check (The 'Gatekeeper' Test)
models = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "google/vit-base-patch16-224",
    "openai/whisper-tiny",
    "Salesforce/blip-image-captioning-base"
]
api = HfApi()
for m in models:
    try:
        info = api.model_info(m)
        if info.private or info.gated:
             fail(f"Model {m} is gated/private. Student cannot access without auth.")
        log(f"Model {m} is public and available.")
    except Exception as e:
        fail(f"Model {m} not found on Hub: {e}")

# 3. Pipeline Instantiation (The 'Memory' Test)
pipelines = {}
try:
    log("Attempting to load all pipelines into memory (CPU)...")
    device = "cpu"
    
    pipelines['text'] = pipeline("text-classification", model=models[0], device=device)
    log("Text loaded.")
    
    pipelines['image'] = pipeline("image-classification", model=models[1], device=device)
    log("Image loaded.")
    
    pipelines['asr'] = pipeline("automatic-speech-recognition", model=models[2], device=device)
    log("ASR loaded.")
    
    pipelines['caption'] = pipeline("image-to-text", model=models[3], device=device)
    log("Caption loaded.")

except Exception as e:
    fail(f"Pipeline instantiation crashed (likely OOM or device error): {e}")

# 4. Streaming Data Test (The 'Network' Test)
try:
    log("Testing Common Voice streaming...")
    cv = load_dataset("mozilla-foundation/common_voice_11_0", "en", split="test", streaming=True, trust_remote_code=True)
    sample = next(iter(cv))
    if 'audio' not in sample:
        fail("Common Voice sample missing 'audio' key.")
    log("Streaming successful.")
except Exception as e:
    fail(f"Streaming dataset failed: {e}")

log(">>> CAPSTONE SURVIVED FALSIFICATION <<<")
log("The Hypothesis (H0) stands. Project is technically sound.")
sys.exit(0)
```

---

## 4. Reporting Standards

Report findings using the **Toyota 3-Type** format:

1.  **Problem Description:** Exactly what failed? (e.g., "ASR Pipeline initialization caused System Kill (OOM) on 4GB Container")
2.  **Root Cause:** Why? (e.g., "Whisper-tiny loads full model weights + overhead, exceeding container limits alongside ViT")
3.  **Countermeasure:** How to fix instructions? (e.g., "Instruction update: Advise students to clear memory between phases, or use `lazy_load` techniques.")

**Signed:**
*The QA Falsification Team*
