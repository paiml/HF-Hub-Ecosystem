# Course Outline: Hugging Face Hub and Ecosystem Fundamentals

**Courses 1-2 of 5** | Hugging Face ML Specialization

---

## Course 1: Hub and Ecosystem Fundamentals

### Module 1.1: Introduction to the Hugging Face Ecosystem

**Duration:** 2 hours

**Description:**
This module introduces the Hugging Face ecosystem, including the Hub, Transformers library, and the broader open-source ML community. Students learn to navigate the Hub, understand model cards, and discover models and datasets for their projects.

**Learning Objectives:**
- Understand the architecture of the Hugging Face Hub
- Navigate and search for models using filters and tags
- Read and interpret model cards for informed model selection
- Understand licensing implications for model usage
- Configure authentication and API access

**Topics:**
1. Hugging Face ecosystem overview
2. Hub architecture and organization
3. Model discovery and search
4. Model cards and documentation standards
5. Licensing (Apache-2.0, MIT, CC-BY, etc.)
6. API authentication with `huggingface-cli`

---

### Module 1.2: Working with Datasets

**Duration:** 2 hours

**Description:**
Students learn to discover, load, and explore datasets from the Hugging Face Hub. The module covers streaming for large datasets, data splits, and preprocessing fundamentals.

**Learning Objectives:**
- Search and filter datasets on the Hub
- Load datasets using the `datasets` library
- Understand train/validation/test splits
- Use streaming for memory-efficient data loading
- Explore dataset statistics and distributions

**Topics:**
1. Dataset Hub navigation
2. `load_dataset()` API
3. Dataset splits and configurations
4. Streaming large datasets
5. Dataset inspection and statistics
6. Common dataset formats (CSV, JSON, Parquet)

---

### Module 1.3: Transformers Fundamentals

**Duration:** 3 hours

**Description:**
This module covers the core Transformers library APIs. Students learn to use pipelines for quick inference, understand AutoClasses for flexible model loading, and perform custom inference with tokenizers and models.

**Learning Objectives:**
- Use `pipeline()` for common NLP tasks
- Understand task types (classification, generation, QA, etc.)
- Load models with `AutoModel` and `AutoTokenizer`
- Perform batched inference efficiently
- Handle tokenization edge cases

**Topics:**
1. Pipeline API for quick inference
2. Task types and model selection
3. AutoClasses (`AutoModel`, `AutoTokenizer`, `AutoConfig`)
4. Tokenization fundamentals
5. Batched inference patterns
6. Handling long sequences and truncation

---

### Module 1.4: Device Management and Optimization

**Duration:** 2 hours

**Description:**
Students learn to manage compute resources effectively, including GPU detection, device placement, and memory optimization strategies for inference.

**Learning Objectives:**
- Detect available hardware (CPU, CUDA, MPS)
- Configure device placement for models
- Use `device_map="auto"` for multi-GPU inference
- Estimate memory requirements
- Optimize inference for resource constraints

**Topics:**
1. Hardware detection with PyTorch
2. Device placement strategies
3. `accelerate` library integration
4. Memory estimation and profiling
5. CPU fallback strategies
6. Mixed precision inference

---

### Module 1.5: Multi-Modal Models

**Duration:** 3 hours

**Description:**
This module introduces multi-modal models that work with images, audio, and text. Students learn to use Vision Transformers, Whisper for speech recognition, and image captioning models.

**Learning Objectives:**
- Perform image classification with ViT and CLIP
- Transcribe audio with Whisper
- Generate image captions with BLIP
- Understand multi-modal preprocessing
- Chain models for complex workflows

**Topics:**
1. Vision Transformers (ViT) for image classification
2. CLIP for zero-shot classification
3. Whisper for speech-to-text
4. Audio preprocessing with `librosa`
5. BLIP for image captioning
6. Visual question answering (VQA)

---

## Course 2: Fine-Tuning Transformers

### Module 2.1: Data Preparation for Fine-Tuning

**Duration:** 3 hours

**Description:**
Students learn to prepare datasets for fine-tuning, including loading from various sources, preprocessing pipelines, and creating custom datasets from scratch.

**Learning Objectives:**
- Load datasets from Hub, local files, and remote URLs
- Build preprocessing pipelines with `map()`
- Tokenize text data for transformer models
- Create custom datasets with `from_dict()` and `from_generator()`
- Handle imbalanced datasets

**Topics:**
1. Dataset loading patterns
2. `map()` for parallel preprocessing
3. Tokenization for fine-tuning
4. Dynamic padding vs. fixed padding
5. Custom dataset creation
6. Data augmentation strategies

---

### Module 2.2: The Trainer API

**Duration:** 4 hours

**Description:**
This module covers the Trainer API for fine-tuning transformer models. Students learn to configure training arguments, implement training loops, and monitor training progress.

**Learning Objectives:**
- Configure `TrainingArguments` for different scenarios
- Initialize and run `Trainer`
- Implement custom `compute_metrics` functions
- Use callbacks for training hooks
- Save and resume training checkpoints

**Topics:**
1. `TrainingArguments` configuration
2. `Trainer` initialization and execution
3. Learning rate schedules
4. Gradient accumulation
5. Mixed precision training
6. Checkpointing strategies

---

### Module 2.3: Custom Metrics and Loss Functions

**Duration:** 2 hours

**Description:**
Students learn to implement custom evaluation metrics and loss functions for specialized tasks. The module covers the `evaluate` library and integration with Trainer.

**Learning Objectives:**
- Use the `evaluate` library for standard metrics
- Implement `compute_metrics` for Trainer
- Combine multiple metrics (accuracy, F1, precision, recall)
- Create weighted loss functions
- Handle multi-label classification metrics

**Topics:**
1. `evaluate` library fundamentals
2. Classification metrics (accuracy, F1, precision, recall)
3. Custom metric functions
4. Weighted and focal loss
5. Multi-label metrics
6. Metric aggregation strategies

---

### Module 2.4: Experiment Tracking

**Duration:** 2 hours

**Description:**
This module covers experiment tracking with TensorBoard and Weights & Biases. Students learn to log metrics, visualize training, and compare experiments.

**Learning Objectives:**
- Configure TensorBoard logging in Trainer
- Visualize training curves
- Track hyperparameters and configurations
- Compare multiple training runs
- Log custom metrics and artifacts

**Topics:**
1. TensorBoard integration
2. Logging configuration in `TrainingArguments`
3. Custom logging callbacks
4. Weights & Biases integration
5. Experiment comparison
6. Hyperparameter tracking

---

### Module 2.5: Model Evaluation

**Duration:** 3 hours

**Description:**
Students learn comprehensive model evaluation techniques, including confusion matrices, error analysis, and task-specific metrics like ROUGE and BLEU for generation tasks.

**Learning Objectives:**
- Generate and interpret confusion matrices
- Perform error analysis on misclassified examples
- Compute ROUGE scores for summarization
- Compute BLEU scores for translation
- Evaluate perplexity for language models

**Topics:**
1. Confusion matrix visualization
2. Classification report analysis
3. ROUGE for summarization evaluation
4. BLEU for translation evaluation
5. Perplexity and cross-entropy
6. Human evaluation strategies

---

### Module 2.6: Publishing Models

**Duration:** 2 hours

**Description:**
This module covers publishing fine-tuned models to the Hugging Face Hub, including model cards, repository configuration, and best practices for sharing.

**Learning Objectives:**
- Use `push_to_hub()` to publish models
- Write effective model cards
- Configure repository visibility and settings
- Version models with Git LFS
- Share tokenizers and configurations

**Topics:**
1. `push_to_hub()` API
2. Model card structure and metadata
3. Repository creation with `create_repo()`
4. Git LFS for large files
5. Model versioning strategies
6. Community guidelines and best practices

---

## Learning Path Summary

| Week | Course | Focus | Hours |
|------|--------|-------|-------|
| 1 | Course 1 | Hub Navigation, Datasets | 4 |
| 2 | Course 1 | Transformers, Pipelines | 5 |
| 3 | Course 1 | Multi-Modal Models | 3 |
| 4 | Course 2 | Data Preparation | 3 |
| 5 | Course 2 | Trainer API, Metrics | 6 |
| 6 | Course 2 | Evaluation, Publishing | 5 |

**Total Duration:** ~26 hours

---

## Prerequisites

- Python programming (intermediate level)
- Basic understanding of machine learning concepts
- Familiarity with NumPy and pandas
- Command line proficiency

## Recommended Background

- Introduction to deep learning
- Basic NLP concepts (tokenization, embeddings)
- PyTorch fundamentals (helpful but not required)

---

## Repository Demo Structure

```
HF-Hub-Ecosystem/
├── .github/
│   └── workflows/
│       └── ci.yml                          # GitHub Actions CI pipeline
├── docs/
│   ├── diagrams/
│   │   └── hero-image.svg                  # Course hero image
│   ├── readings/                           # Supplementary readings
│   ├── roadmaps/
│   │   └── roadmap.yaml                    # PMAT work tickets
│   ├── specifications/
│   │   └── demo-repo.md                    # Full specification document
│   └── outline.md                          # This file
├── notebooks/
│   ├── course1/
│   │   ├── week1/
│   │   │   ├── 1.3-exploring-hub.ipynb     # Hub navigation and search
│   │   │   └── 1.7-dataset-exploration.ipynb # Dataset loading and streaming
│   │   ├── week2/
│   │   │   ├── 2.2-text-pipelines.ipynb    # Pipeline API for NLP tasks
│   │   │   ├── 2.5-custom-inference.ipynb  # AutoModel and tokenization
│   │   │   └── 2.8-multi-gpu.ipynb         # Device management
│   │   └── week3/
│   │       ├── 3.2-image-classification.ipynb # ViT and CLIP
│   │       ├── 3.4-whisper-transcription.ipynb # Speech-to-text
│   │       └── 3.6-image-captioning.ipynb  # BLIP captioning
│   └── course2/
│       ├── week1/
│       │   ├── 1.2-loading-datasets.ipynb  # Dataset sources and formats
│       │   └── 1.7-custom-dataset.ipynb    # Creating custom datasets
│       ├── week2/
│       │   ├── 2.3-first-finetuning.ipynb  # Trainer API basics
│       │   ├── 2.5-custom-metrics.ipynb    # Evaluation metrics
│       │   └── 2.7-experiment-tracking.ipynb # TensorBoard logging
│       └── week3/
│           ├── 3.2-comprehensive-eval.ipynb # Confusion matrix, metrics
│           ├── 3.4-nlg-evaluation.ipynb    # ROUGE, BLEU scores
│           └── 3.7-publish-model.ipynb     # push_to_hub() workflow
├── src/
│   └── hf_ecosystem/
│       ├── __init__.py                     # Package exports
│       ├── hub/
│       │   ├── __init__.py
│       │   ├── search.py                   # search_models(), search_datasets()
│       │   └── cards.py                    # parse_model_card()
│       ├── inference/
│       │   ├── __init__.py
│       │   ├── pipelines.py                # create_pipeline()
│       │   └── device.py                   # get_device(), get_device_map()
│       ├── data/
│       │   ├── __init__.py
│       │   ├── preprocessing.py            # preprocess_text(), tokenize_batch()
│       │   └── streaming.py                # stream_dataset()
│       └── training/
│           ├── __init__.py
│           ├── trainer.py                  # create_trainer()
│           └── metrics.py                  # compute_metrics()
├── tests/
│   ├── __init__.py
│   ├── conftest.py                         # Shared pytest fixtures
│   ├── unit/
│   │   ├── test_device.py                  # Device detection tests
│   │   ├── test_preprocessing.py           # Text preprocessing tests
│   │   └── test_pipelines.py               # Pipeline creation tests
│   ├── integration/                        # Cross-component tests
│   └── notebooks/                          # Notebook execution tests
├── .gitignore
├── .pmat-metrics.toml                      # Quality thresholds
├── .pre-commit-config.yaml                 # Pre-commit hooks
├── CLAUDE.md                               # AI assistant instructions
├── LICENSE                                 # Apache-2.0
├── Makefile                                # Build targets
├── README.md                               # Project overview
├── pyproject.toml                          # uv/Python configuration
└── uv.lock                                 # Locked dependencies
```

### Notebook Naming Convention

```
{lesson_number}-{kebab-case-title}.ipynb
```

Each notebook follows a consistent structure:
1. **Metadata cell** - Title, objective, duration, learning outcomes
2. **Setup cell** - Path configuration, version check
3. **Import cell** - All required imports
4. **Content cells** - Markdown headers with code examples
5. **Verification cell** - Assertions to validate completion
6. **Cleanup cell** - Memory cleanup (GPU notebooks)

### Quality Gates

| Check | Command | Threshold |
|-------|---------|-----------|
| Linting | `make lint` | Zero violations |
| Type checking | `pyright` | Strict mode, zero errors |
| Unit tests | `make test` | 95% coverage |
| Notebook execution | `pytest --nbval-lax` | All pass |

### Key Make Targets

```bash
make setup      # Install dependencies with uv
make lint       # Run ruff and pyright
make format     # Auto-format code
make test       # Run pytest with coverage
make check      # Full CI equivalent
make lab        # Launch Jupyter Lab
```

---

*Last Updated: 2026-01-29*
