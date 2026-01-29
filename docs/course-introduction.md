# Course Introduction: Hugging Face Hub and Ecosystem Fundamentals

**Course 1 of 5** | Hugging Face ML Specialization

---

## Welcome

Welcome to the Hugging Face Hub and Ecosystem Fundamentals course. This hands-on course teaches you to leverage the world's largest open-source machine learning platform for model discovery, inference, and multi-modal applications.

By the end of this course, you'll be able to:

- Navigate the Hugging Face Hub to find models and datasets
- Build inference pipelines for text, image, and audio tasks
- Understand model cards, licensing, and best practices
- Create multi-modal applications using pre-trained models

---

## Course Resources

### Repository

| Resource | Link |
|----------|------|
| **Course Repository** | [github.com/paiml/HF-Hub-Ecosystem](https://github.com/paiml/HF-Hub-Ecosystem) |
| **Course Outline** | [docs/outline.md](outline.md) |
| **Capstone Project** | [docs/capstone.md](capstone.md) |
| **Specification** | [docs/specifications/demo-repo.md](specifications/demo-repo.md) |

### Hugging Face Documentation

| Resource | Description |
|----------|-------------|
| [Hugging Face Hub](https://huggingface.co/docs/hub) | Model and dataset hosting platform |
| [Transformers](https://huggingface.co/docs/transformers) | State-of-the-art ML library |
| [Datasets](https://huggingface.co/docs/datasets) | Dataset loading and processing |
| [Hub Python Library](https://huggingface.co/docs/huggingface_hub) | Programmatic Hub access |
| [Tokenizers](https://huggingface.co/docs/tokenizers) | Fast tokenization library |
| [Accelerate](https://huggingface.co/docs/accelerate) | Multi-GPU and mixed precision |

### External Resources

| Resource | Description |
|----------|-------------|
| [PyTorch Documentation](https://pytorch.org/docs/) | Deep learning framework |
| [Python 3.11 Docs](https://docs.python.org/3.11/) | Python language reference |
| [uv Documentation](https://docs.astral.sh/uv/) | Fast Python package manager |

---

## Course Structure

### Duration and Format

| | |
|---|---|
| **Total Duration** | 12 hours (3 weeks) |
| **Format** | Self-paced with Jupyter notebooks |
| **Notebooks** | 8 hands-on labs |
| **Capstone** | Multi-Modal Content Analyzer |

### Weekly Breakdown

```
Week 1: Hub Navigation (4 hours)
├── Module 1.1: Introduction to the HF Ecosystem
├── Module 1.2: Working with Datasets
└── Labs: exploring-hub, dataset-exploration

Week 2: Transformers Fundamentals (5 hours)
├── Module 1.3: Transformers Fundamentals
├── Module 1.4: Device Management and Optimization
└── Labs: text-pipelines, custom-inference, multi-gpu

Week 3: Multi-Modal Models (3 hours)
├── Module 1.5: Multi-Modal Models
└── Labs: image-classification, whisper-transcription, image-captioning

Capstone: Multi-Modal Content Analyzer (6-8 hours)
```

---

## Prerequisites

### Required Knowledge

| Skill | Level | Notes |
|-------|-------|-------|
| Python programming | Intermediate | Classes, functions, list comprehensions |
| Command line | Basic | Navigation, running commands |
| Git | Basic | Clone, commit (for setup) |

### Recommended Background

| Topic | Helpful For |
|-------|-------------|
| Machine learning basics | Understanding model concepts |
| NumPy/Pandas | Dataset exploration |
| PyTorch fundamentals | Custom inference (Module 1.3) |
| Basic NLP concepts | Tokenization understanding |

### Not Required

- Deep learning expertise
- GPU/CUDA experience
- Prior Hugging Face experience

---

## Environment Setup

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB |
| **Storage** | 10 GB free | 20 GB free |
| **Python** | 3.11+ | 3.11 or 3.12 |
| **GPU** | Not required | NVIDIA with 8GB+ VRAM |

### Installation

#### 1. Install uv (Package Manager)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```

#### 2. Clone the Repository

```bash
git clone https://github.com/paiml/HF-Hub-Ecosystem.git
cd HF-Hub-Ecosystem
```

#### 3. Setup Environment

```bash
# Install all dependencies
make setup

# Verify installation
make check
```

#### 4. Configure Hugging Face (Optional)

```bash
# Login for private models and increased rate limits
huggingface-cli login

# Set cache directory (optional)
export HF_HOME=/path/to/cache
```

### Verify Setup

Run the demo to verify everything works:

```bash
make demo
```

Expected output:
```
╔══════════════════════════════════════════════════════════╗
║   🤗 HF Hub Ecosystem Demo                               ║
╚══════════════════════════════════════════════════════════╝

▶ Device Detection
  ✓ Detected device: cpu

▶ Sentiment Analysis Pipeline
  ✓ Model loaded successfully
  ✓ Inference complete

════════════════════════════════════════════════════════════
║ Demo Complete!
════════════════════════════════════════════════════════════
```

---

## Notebooks Overview

### Course 1 Labs

| Week | Notebook | Duration | Key Concepts |
|------|----------|----------|--------------|
| 1 | `1.3-exploring-hub.ipynb` | 20 min | `list_models()`, filters, model cards |
| 1 | `1.7-dataset-exploration.ipynb` | 25 min | `load_dataset()`, streaming, splits |
| 2 | `2.2-text-pipelines.ipynb` | 30 min | `pipeline()`, task types, batching |
| 2 | `2.5-custom-inference.ipynb` | 25 min | `AutoModel`, `AutoTokenizer`, forward pass |
| 2 | `2.8-multi-gpu.ipynb` | 20 min | `device_map="auto"`, memory estimation |
| 3 | `3.2-image-classification.ipynb` | 25 min | ViT, CLIP, preprocessing |
| 3 | `3.4-whisper-transcription.ipynb` | 25 min | Whisper, chunking, language detection |
| 3 | `3.6-image-captioning.ipynb` | 30 min | BLIP, VQA |

### Running Notebooks

```bash
# Launch Jupyter Lab
make notebook

# Navigate to notebooks/course1/week1/ to begin
```

### Notebook Structure

Every notebook follows this structure:

1. **Metadata** - Title, objective, duration, learning outcomes
2. **Setup** - Path configuration, version check
3. **Imports** - Required libraries
4. **Content** - Instructional content with code examples
5. **Verification** - Assertions to validate completion

---

## Key Concepts

### The Hugging Face Hub

The Hub is a platform hosting:

- **200,000+** pre-trained models
- **50,000+** datasets
- **100,000+** demo Spaces

```python
from huggingface_hub import HfApi

api = HfApi()

# Search for models
models = api.list_models(task="text-classification", limit=10)

# Get model details
info = api.model_info("bert-base-uncased")
print(f"Downloads: {info.downloads:,}")
```

### Transformers Pipelines

Pipelines provide high-level inference APIs:

```python
from transformers import pipeline

# Text classification
classifier = pipeline("text-classification")
result = classifier("I love this course!")
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Image classification
classifier = pipeline("image-classification")
result = classifier("cat.jpg")
# [{'label': 'tabby cat', 'score': 0.95}]

# Speech recognition
asr = pipeline("automatic-speech-recognition")
result = asr("audio.mp3")
# {'text': 'Hello, how are you?'}
```

### AutoClasses

AutoClasses automatically load the right model architecture:

```python
from transformers import AutoModel, AutoTokenizer

# Load any model/tokenizer by name
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# Tokenize text
inputs = tokenizer("Hello, world!", return_tensors="pt")

# Run inference
outputs = model(**inputs)
```

### Datasets Library

Load and process datasets efficiently:

```python
from datasets import load_dataset

# Load from Hub
dataset = load_dataset("imdb")

# Stream large datasets
dataset = load_dataset("common_voice", streaming=True)

# Process with map
def tokenize(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized = dataset.map(tokenize, batched=True)
```

---

## Models Used in This Course

| Task | Model | Parameters | Hub ID |
|------|-------|------------|--------|
| Text Classification | DistilBERT | 66M | `distilbert-base-uncased-finetuned-sst-2-english` |
| Image Classification | ViT | 86M | `google/vit-base-patch16-224` |
| Speech Recognition | Whisper | 39M | `openai/whisper-tiny` |
| Image Captioning | BLIP | 247M | `Salesforce/blip-image-captioning-base` |
| Zero-Shot | BART | 407M | `facebook/bart-large-mnli` |

---

## Datasets Used in This Course

| Dataset | Modality | Samples | Hub ID |
|---------|----------|---------|--------|
| IMDb | Text | 50K | `imdb` |
| CIFAR-10 | Image | 60K | `cifar10` |
| Common Voice | Audio | 1M+ | `mozilla-foundation/common_voice_11_0` |
| ImageNet | Image | 1.2M | `imagenet-1k` |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Run `make setup` to install dependencies |
| Model download slow | Set `HF_HOME` to SSD location |
| Out of memory | Use smaller models or `device_map="auto"` |
| CUDA not detected | Check PyTorch CUDA installation |
| Rate limited | Run `huggingface-cli login` |

### Getting Help

1. Check the [Hugging Face Forums](https://discuss.huggingface.co/)
2. Search [GitHub Issues](https://github.com/huggingface/transformers/issues)
3. Review [Transformers Documentation](https://huggingface.co/docs/transformers)

---

## Learning Path

### This Course (Course 1)

```
Hub Navigation → Pipelines → Multi-Modal → Capstone
```

### Full Specialization

| Course | Title | Prerequisites |
|--------|-------|---------------|
| **1** | Hub and Ecosystem Fundamentals | Python basics |
| 2 | Fine-Tuning Transformers (Bonus) | Course 1 |
| 3 | Advanced NLP | Courses 1-2 |
| 4 | Advanced Fine-Tuning (LoRA, PEFT) | Courses 1-3 |
| 5 | Production ML | Courses 1-4 |

---

## Assessment

### Capstone Project

The capstone project demonstrates mastery by building a **Multi-Modal Content Analyzer**:

- Search and select models from the Hub
- Load and explore multi-modal datasets
- Build inference pipelines for 4 tasks
- Compare model performance
- Create a unified analysis workflow

**Duration:** 6-8 hours
**Passing Score:** 70/100 points

See [docs/capstone.md](capstone.md) for full details.

---

## Quick Reference

### Make Targets

```bash
make setup      # Install dependencies
make demo       # Run demo
make notebook   # Launch Jupyter Lab
make lint       # Run linting
make test       # Run tests
make check      # Full CI check
```

### Key Imports

```python
# Hub API
from huggingface_hub import HfApi, ModelFilter

# Transformers
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer, AutoConfig

# Datasets
from datasets import load_dataset

# Device detection
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
```

### Environment Variables

```bash
HF_HOME=/path/to/cache      # Model cache location
HF_TOKEN=hf_xxxxx           # Authentication token
CUDA_VISIBLE_DEVICES=0      # GPU selection
```

---

## Next Steps

1. **Complete setup** - Run `make setup` and `make demo`
2. **Start Week 1** - Open `notebooks/course1/week1/1.3-exploring-hub.ipynb`
3. **Progress through modules** - Follow the weekly structure
4. **Complete capstone** - Build your Multi-Modal Content Analyzer

Welcome to the course. Let's get started!

---

*Course 1 of the Hugging Face ML Specialization*
*Repository: [github.com/paiml/HF-Hub-Ecosystem](https://github.com/paiml/HF-Hub-Ecosystem)*
*Last Updated: January 2026*
