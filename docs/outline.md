# Course Outline: Hugging Face Hub and Ecosystem Fundamentals

**Course 1 of 5** | Hugging Face ML Specialization

---

## Course Overview

This course introduces the Hugging Face ecosystem, teaching you to navigate the Hub, evaluate models and datasets, and build multi-modal inference pipelines. You'll progress from browsing and discovery to hands-on implementation through a capstone project.

| | |
|---|---|
| **Duration** | ~2.5 hours |
| **Modules** | 3 |
| **Videos** | 8 |
| **Role Plays** | 2 |
| **Capstone** | 1 |

---

## Module 1: Navigate the Hub

**Duration:** ~40 minutes

**Description:**
This module introduces the Hugging Face ecosystem and teaches you to find, evaluate, and select models for your projects. You'll learn to navigate the Hub's search and filtering system, understand model file structures, evaluate licensing for your use case, and configure authentication for programmatic access.

**Learning Objectives:**
- Understand the architecture of the Hugging Face Hub
- Navigate and search for models using filters and tags
- Read and interpret model cards for informed model selection
- Understand model file structures (safetensors, config, tokenizer files)
- Evaluate licensing implications for commercial and research use
- Configure authentication and API access

### Lessons

| Lesson | Type | Title | Duration |
|--------|------|-------|----------|
| 1.1.1 | Video | What Is Hugging Face | 5 min |
| 1.1.2 | Video | Searching and Filtering Models | 5 min |
| 1.1.3 | Video | Your Account and Organizations | 5 min |
| 1.2.1 | Video | Model Files | 5 min |
| 1.2.2 | Video | Licensing and Usage Rights | 5 min |
| 1.2.3 | Role Play | Model Selection Sprint | 15 min |

### Lesson Details

**1.1.1 What Is Hugging Face**
- Hub as "GitHub for ML" — models, datasets, spaces
- Homepage navigation: Models, Datasets, Spaces tabs
- Scale: 2.5M+ models, 800K+ datasets
- Ecosystem overview: Transformers, Datasets, Hub libraries
- Major organizations publishing on Hub (Meta, Google, Microsoft)

**1.1.2 Searching and Filtering Models**
- Basic text search functionality
- Task filters (text classification, generation, translation, etc.)
- Library filters (PyTorch, TensorFlow, JAX, ONNX)
- Language filters for NLP models
- License filters (Apache 2.0, MIT, etc.)
- Sorting options: trending, most likes, recently updated

**1.1.3 Your Account and Organizations**
- Individual users vs. organizations
- Verified badges and why they matter
- Organization pages: members, published models
- Community discussions on model pages
- Creating an account
- Access tokens: read vs. write permissions
- CLI login with `huggingface-cli`

**1.2.1 Model Files**
- Model repositories contain multiple files
- Weight files: SafeTensors vs. PyTorch .bin
- SafeTensors advantages: security, speed, memory mapping
- config.json: hyperparameters and architecture
- Tokenizer files: vocab, special tokens
- Version history and commits

**1.2.2 Licensing and Usage Rights**
- Why licensing matters for production
- License spectrum: permissive → copyleft → restrictive
- Apache 2.0: commercial use, patent grant
- MIT: permissive, attribution required
- GPL: copyleft, share modifications
- Custom licenses: Llama Community License, RAIL
- Checking licenses in model metadata

**1.2.3 Role Play: Model Selection Sprint**
- Scenario: ML Engineer selecting models for healthcare document triage
- Apply Hub search and filtering skills
- Evaluate model cards and licenses
- Document selection rationale
- Practice systematic model evaluation

---

## Module 2: Work with Datasets

**Duration:** ~30 minutes

**Description:**
This module covers discovering and loading datasets from the Hugging Face Hub. You'll learn to explore the datasets hub, load data programmatically with Python, and use streaming for datasets too large to download.

**Learning Objectives:**
- Search and filter datasets on the Hub
- Preview datasets using the dataset viewer
- Load datasets using the `datasets` library
- Access splits, index examples, and filter data
- Use streaming for memory-efficient loading of large datasets

### Lessons

| Lesson | Type | Title | Duration |
|--------|------|-------|----------|
| 1.3.1 | Video | Exploring Datasets Hub | 5 min |
| 1.3.2 | Video | Loading Datasets With Python | 5 min |
| 1.3.3 | Video | Streaming Large Datasets | 5 min |
| 1.3.4 | Role Play | Format Converter | 15 min |

### Lesson Details

**1.3.1 Exploring Datasets Hub**
- Dataset hub layout and navigation
- Filters: task, size, language, license, modality
- Dataset cards: structure, splits, documentation
- Dataset viewer: preview without downloading
- Viewing statistics and distributions
- Code snippets in dataset cards

**1.3.2 Loading Datasets With Python**
- Installing the `datasets` library
- `load_dataset()` function basics
- Dataset structure: splits (train, test, validation)
- Accessing splits with dictionary syntax
- Indexing individual examples
- Iterating through datasets
- Filtering with lambda functions
- Dataset configurations

**1.3.3 Streaming Large Datasets**
- Problem: massive datasets (100s of GB)
- Solution: `streaming=True` parameter
- Iterable datasets vs. regular datasets
- Iterating with `take()` for samples
- Filtering streams lazily
- When to stream: disk limits, sampling, exploration
- Trade-offs: latency vs. storage

**1.3.4 Role Play: Format Converter**
- Scenario: Data Engineer unifying clinical notes from multiple formats
- Load CSV, JSON Lines, and Parquet files
- Align schemas across sources
- Concatenate into unified dataset
- Create stratified train/val/test splits
- Generate data quality report

---

## Module 3: Capstone Project

**Duration:** ~75 minutes

**Description:**
Code-only capstone combining Hub navigation and datasets into a unified multi-modal analyzer. Search the Hub programmatically, load sample datasets, build inference pipelines, and create a single `analyze_content()` function for text sentiment, image classification, and captioning.

**Notebook:** `notebooks/course1/week3/3.8-capstone-multimodal-analyzer.ipynb`

**Learning Objectives:**
- Apply Hub navigation to discover models across modalities
- Load and explore datasets for text, image, and audio
- Use the `pipeline()` API for inference
- Implement device detection (CUDA, MPS, CPU)
- Build a unified multi-modal analysis workflow
- Compare model performance and trade-offs

### Lessons

| Lesson | Type | Title | Duration |
|--------|------|-------|----------|
| 1.4.1 | Reading | Key Concepts Review | 10 min |
| 1.4.2 | Project | Multi-Modal Content Analyzer | 55 min |
| 1.4.3 | Assignment | Final Graded Quiz | 10 min |

### Capstone Project Phases

**Phase 1: Hub Exploration and Model Discovery** (60 min)
- Search Hub for text classification models
- Search Hub for image classification models
- Search Hub for speech recognition models
- Search Hub for image captioning models
- Analyze model cards for top candidates
- Document model selection with rationale

**Phase 2: Dataset Exploration** (45 min)
- Load text dataset (IMDb)
- Load image dataset (CIFAR-10)
- Stream audio dataset (Common Voice)
- Compute dataset statistics
- Understand data structures across modalities

**Phase 3: Building Inference Pipelines** (60 min)
- Device detection: CUDA, MPS, CPU
- Text classification pipeline
- Image classification pipeline
- Speech recognition pipeline
- Image captioning pipeline
- Unified `MultiModalAnalyzer` class

**Phase 4: Performance Comparison** (45 min)
- Benchmark multiple models per task
- Measure inference latency
- Document speed vs. accuracy trade-offs

**Phase 5: Integration Demo** (30 min)
- End-to-end analysis workflow
- Process mixed content types
- Generate formatted analysis report

### Skills Introduced in Capstone

The capstone teaches these new concepts through guided implementation:

| Skill | Description |
|-------|-------------|
| `pipeline()` API | Load and run models with single function |
| Device detection | `torch.cuda.is_available()`, MPS, CPU fallback |
| Tokenization basics | Padding, truncation, return_tensors |
| Multi-modal processing | Image loading, audio handling |
| Batched inference | Processing multiple inputs efficiently |

---

## Prerequisites

- Python programming (intermediate level)
- Basic understanding of machine learning concepts
- Familiarity with command line
- Hugging Face account (free)

---

## Technical Requirements

| Requirement | Specification |
|-------------|---------------|
| Python | 3.11+ |
| RAM | 8 GB minimum, 16 GB recommended |
| GPU | Optional (CPU works for all exercises) |
| Disk | 10 GB free space |

---

## Learning Path Summary

| Module | Focus | Duration |
|--------|-------|----------|
| 1 | Navigate the Hub | 40 min |
| 2 | Work with Datasets | 30 min |
| 3 | Capstone Project | 75 min |
| **Total** | | **~2.5 hours** |

---

## Assessment

| Component | Weight |
|-----------|--------|
| Model Selection Sprint (Role Play) | 15% |
| Format Converter (Role Play) | 15% |
| Capstone Project | 50% |
| Final Quiz | 20% |

**Passing Score:** 70%

---

## Resources

- [Hugging Face Hub](https://huggingface.co)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Datasets Documentation](https://huggingface.co/docs/datasets)
- [Hub Python Library](https://huggingface.co/docs/huggingface_hub)

---

*Last Updated: January 2025*
