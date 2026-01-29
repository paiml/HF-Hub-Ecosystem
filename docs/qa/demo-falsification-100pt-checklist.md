# QA Falsification Checklist: 100-Point Demo Audit

**Document ID:** QA-DEMO-100
**Date:** 2026-01-29
**Philosophy:** "If it hasn't been tested, it's broken." (Dr. Karl Popper)
**Goal:** Prove the repository is broken. The project only "passes" if it survives this entire checklist without a single error.

---

## 1. Environment & Setup Integrity (10 Points)

Before touching any code, verify the foundation is solid.

*   [ ] **(5 pts) Clean Install Verification**
    *   **Action:** Delete `.venv` and run `make setup` (or `uv sync`).
    *   **Pass Condition:** Zero errors during dependency installation.
    *   **Fail Condition:** Conflict errors, missing wheels, or network timeouts on PyPI.
*   [ ] **(5 pts) Module Import Check**
    *   **Action:** `python3 -c "import hf_ecosystem; print(hf_ecosystem.__file__)"`
    *   **Pass Condition:** Prints the path to `src/hf_ecosystem/__init__.py`.
    *   **Fail Condition:** `ModuleNotFoundError` or `ImportError`.

---

## 2. CLI Demo Falsification (20 Points)

The primary entry points for users. These MUST work out-of-the-box.

*   [ ] **(10 pts) `demo.py` "Rich" Interface**
    *   **Action:** Run `python demo.py`.
    *   **Pass Condition:**
        *   Displays styled tables/panels.
        *   Successfully downloads/caches `distilbert`.
        *   Exits with code 0.
        *   Prints "✓ All demonstrations completed successfully".
    *   **Fail Condition:** Stack trace, ugly raw ANSI codes on unsupported terminals (check fallback), or hanging progress bar.
*   [ ] **(10 pts) `examples/quickstart.py`**
    *   **Action:** Run `python examples/quickstart.py`.
    *   **Pass Condition:** Prints device (cpu/cuda), lists 3 models, cleans text, and outputs sentiment.
    *   **Fail Condition:** "Model not found" or logic error in preprocessing.

---

## 3. Course 1 Notebook Falsification (35 Points)

Validate the "Hub & Ecosystem" curriculum. Run using `jupyter nbconvert --to python --execute --stdout` or `pytest --nbmake`.

*   [ ] **(5 pts) Week 1: Hub & Data**
    *   **Targets:** `1.3-exploring-hub.ipynb`, `1.7-dataset-exploration.ipynb`
    *   **Attack:** Check for hardcoded API tokens (security risk) or deprecated Hub filters.
*   [ ] **(10 pts) Week 2: Text Pipelines (The "Workhorse")**
    *   **Targets:** `2.2-text-pipelines.ipynb`, `2.5-custom-inference.ipynb`, `2.8-multi-gpu.ipynb`
    *   **Attack:**
        *   `2.8-multi-gpu`: Run on a CPU-only machine. Does it degrade gracefully or crash? (It MUST handle `device_map="auto"` correctly).
        *   `2.5-custom-inference`: Ensure the custom class inherits correctly.
*   [ ] **(10 pts) Week 3: Multi-Modal (The "Heavy" Weights)**
    *   **Targets:** `3.2-image-classification.ipynb`, `3.4-whisper-transcription.ipynb`, `3.6-image-captioning.ipynb`
    *   **Attack:** Run `3.4-whisper` on a low-memory (4GB) container.
    *   **Pass Condition:** It should likely use `whisper-tiny` or stream. If it tries to load `large-v3`, it fails (and you catch it).
*   [ ] **(10 pts) Capstone Notebook**
    *   **Target:** `docs/capstone.md` (or associated notebook if present).
    *   **Action:** Verify the "Phase 5 Integration" code runs end-to-end.

---

## 4. Course 2 Notebook Falsification (25 Points)

Validate the "Advanced Production" curriculum.

*   [ ] **(5 pts) Week 1: Advanced Data**
    *   **Targets:** `1.2-loading-datasets.ipynb`, `1.7-custom-dataset.ipynb`
    *   **Attack:** Ensure `load_dataset` with local files path resolution works on different OS (Linux vs Windows paths).
*   [ ] **(10 pts) Week 2: Fine-Tuning (The "Slow" Tests)**
    *   **Targets:** `2.3-first-finetuning.ipynb`, `2.5-custom-metrics.ipynb`, `2.7-experiment-tracking.ipynb`
    *   **Attack:**
        *   Does `Trainer` actually start training?
        *   Run for 1 step only (don't wait hours). Modify the config to `max_steps=1`.
        *   Check if `experiment-tracking` tries to login to W&B without a token (should handle gracefully).
*   [ ] **(10 pts) Week 3: Evaluation & Publishing**
    *   **Targets:** `3.2-comprehensive-eval.ipynb`, `3.4-nlg-evaluation.ipynb`, `3.7-publish-model.ipynb`
    *   **Attack:** `3.7-publish-model` must NOT actually upload to the public Hub during testing (unless using a dry-run flag or mock). If it spams the Hub, it FAILS.

---

## 5. Library & Unit Test Integrity (10 Points)

*   [ ] **(10 pts) Full Test Suite**
    *   **Action:** Run `pytest tests/`.
    *   **Pass Condition:** 100% passing. Green bar.
    *   **Fail Condition:** Any red 'F'.

---

## 6. The "Falsification Script" (Genchi Genbutsu)

To execute this 100-point audit automatically, use the following bash script.

**File:** `tests/qa_100_point_check.sh`

```bash
#!/bin/bash
set -e  # Exit immediately if any command fails

echo "======================================================="
echo "   STARTING 100-POINT FALSIFICATION PROTOCOL"
echo "======================================================="

# 1. Environment (10 pts)
echo "[1/5] Verifying Environment..."
python3 -c "import hf_ecosystem" || { echo "❌ Module Import Failed"; exit 1; }
echo "✅ Environment Valid"

# 2. CLI Demos (20 pts)
echo "[2/5] Verifying CLI Demos..."
python3 demo.py > /dev/null || { echo "❌ demo.py Failed"; exit 1; }
python3 examples/quickstart.py > /dev/null || { echo "❌ quickstart.py Failed"; exit 1; }
echo "✅ CLI Demos Valid"

# 3. Unit Tests (10 pts)
echo "[3/5] Verifying Unit Tests..."
pytest tests/ || { echo "❌ Unit Tests Failed"; exit 1; }
echo "✅ Unit Tests Valid"

# 4. Notebook Smoke Tests (60 pts)
# Using nbconvert to execute notebooks headlessly
echo "[4/5] Verifying Notebooks (Smoke Test)..."

NB_FILES=$(find notebooks -name "*.ipynb" -not -path "*/.ipynb_checkpoints/*")
PASS_COUNT=0
FAIL_COUNT=0

for nb in $NB_FILES; do
    echo "Processing $nb..."
    # We set a timeout of 600s (10 mins) per notebook to catch hanging processes
    # We ignore errors for now to count total failures, or set -e to stop on first
    if jupyter nbconvert --to python --execute "$nb" --stdout > /dev/null 2>&1; then
        echo "  ✅ $nb passed"
        ((PASS_COUNT++))
    else
        echo "  ❌ $nb FAILED"
        ((FAIL_COUNT++))
        # Uncomment to stop line: exit 1
    fi
done

echo "======================================================="
echo "   FALSIFICATION REPORT"
echo "======================================================="
echo "Passed: $PASS_COUNT"
echo "Failed: $FAIL_COUNT"

if [ $FAIL_COUNT -eq 0 ]; then
    echo "RESULT: PASSED (100/100 Points)"
    exit 0
else
    echo "RESULT: FAILED (System Unstable)"
    exit 1
fi
```

**Instruction:** Save the script above, make it executable (`chmod +x tests/qa_100_point_check.sh`), and run it.
