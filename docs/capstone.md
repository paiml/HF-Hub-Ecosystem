# Capstone Project: Multi-Modal Content Analyzer

**Course 1 Final Project** | Hugging Face Hub and Ecosystem Fundamentals

📦 **Course Repository:** [github.com/paiml/HF-Hub-Ecosystem](https://github.com/paiml/HF-Hub-Ecosystem)

---

## Overview

This capstone project demonstrates mastery of the Hugging Face Hub and ecosystem by building a multi-modal content analysis system. You'll discover and compare models across modalities, explore datasets, build inference pipelines, and create a unified analysis workflow.

| | |
|---|---|
| **Duration** | 6–8 hours |
| **Prerequisites** | Completion of Course 1 modules |
| **Requirements** | Hugging Face account, Python 3.11+ |

### What You'll Build

A content analysis tool for a media company that:

1. **Classifies text** sentiment from article headlines
2. **Classifies images** into content categories
3. **Transcribes audio** clips from podcasts
4. **Generates captions** for images

---

## Phase 1: Hub Exploration and Model Discovery

**⏱️ Duration: 1.5 hours**

### Learning Objectives

- Search the Hub programmatically for models across four tasks
- Compare model options using the Hub API
- Analyze model cards to make informed selections

### 1.1 Searching for Models

Use the Hub API to discover models for each task:

```python
from huggingface_hub import HfApi, ModelFilter

api = HfApi()

# Text classification models
text_models = api.list_models(
    filter=ModelFilter(task="text-classification", library="transformers"),
    sort="downloads",
    direction=-1,
    limit=15,
)

# Image classification models
image_models = api.list_models(
    filter=ModelFilter(task="image-classification", library="transformers"),
    sort="downloads",
    direction=-1,
    limit=15,
)

# Speech recognition models
asr_models = api.list_models(
    filter=ModelFilter(task="automatic-speech-recognition", library="transformers"),
    sort="downloads",
    direction=-1,
    limit=15,
)

# Image captioning models
caption_models = api.list_models(
    filter=ModelFilter(task="image-to-text", library="transformers"),
    sort="downloads",
    direction=-1,
    limit=15,
)
```

### 1.2 Analyzing Model Cards

Extract key information to compare candidates:

```python
def analyze_model_card(model_id):
    """Extract key information from model card."""
    info = api.model_info(model_id)
    return {
        "model_id": model_id,
        "downloads": info.downloads,
        "likes": info.likes,
        "license": info.card_data.license if info.card_data else "Unknown",
        "pipeline_tag": info.pipeline_tag,
        "tags": info.tags[:5] if info.tags else [],
    }

# Compare candidates
text_analysis = analyze_model_card("distilbert-base-uncased-finetuned-sst-2-english")
print(f"Text Model: {text_analysis}")
```

### 1.3 Deliverable: Selection Table

Document your model choices:

| Task | Model Selected | Parameters | License | Rationale |
|------|----------------|------------|---------|-----------|
| Text Classification | | | | |
| Image Classification | | | | |
| Speech Recognition | | | | |
| Image Captioning | | | | |

---

## Phase 2: Dataset Exploration

**⏱️ Duration: 1.5 hours**

### Learning Objectives

- Load datasets across text, image, and audio modalities
- Use streaming for memory-efficient loading of large datasets
- Compute and interpret dataset statistics

### 2.1 Loading Multi-Modal Datasets

```python
from datasets import load_dataset

# Text: IMDb sentiment dataset
text_dataset = load_dataset("imdb", split="test[:1000]")
print(f"Text samples: {len(text_dataset)}")
print(f"Features: {text_dataset.features}")

# Image: CIFAR-10 classification dataset
image_dataset = load_dataset("cifar10", split="test[:100]")
print(f"Image samples: {len(image_dataset)}")
print(f"Labels: {image_dataset.features['label'].names}")

# Audio: Common Voice with streaming (large dataset)
audio_dataset = load_dataset(
    "mozilla-foundation/common_voice_11_0",
    "en",
    split="test",
    streaming=True,
    trust_remote_code=True,
)
```

### 2.2 Computing Dataset Statistics

```python
import pandas as pd

# Text analysis
text_df = text_dataset.to_pandas()
text_df['length'] = text_df['text'].str.len()
print(f"Label distribution:\n{text_df['label'].value_counts()}")
print(f"Text length: mean={text_df['length'].mean():.0f}, max={text_df['length'].max()}")

# Image analysis
image_df = image_dataset.to_pandas()
print(f"Label distribution:\n{image_df['label'].value_counts()}")
```

---

## Phase 3: Building Inference Pipelines

**⏱️ Duration: 2 hours**

### Learning Objectives

- Create pipelines for text, image, audio, and captioning tasks
- Implement proper device detection and placement
- Build a unified multi-modal analyzer class

### 3.1 Device Detection

```python
import torch

def get_device():
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

device = get_device()
print(f"Using device: {device}")
```

### 3.2 Individual Pipelines

```python
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# Use device from 3.1
device = get_device()

# Text classification
text_classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
)

# Image classification
image_classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
    device=device,
)

# Speech recognition
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device=device,
)

# Image captioning (use "image-text-to-text" in transformers 5.0+)
# For pure captioning, use manual inference:
from transformers import BlipProcessor, BlipForConditionalGeneration
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
```

### 3.3 Unified Analyzer Class

> **Memory Note:** Loading all four models requires ~2-3GB RAM. On memory-constrained systems, consider loading models on-demand or processing one modality at a time.

```python
class MultiModalAnalyzer:
    """Unified multi-modal content analyzer."""

    def __init__(self, device="cpu"):
        self.device = device
        self.text_classifier = pipeline(
            "text-classification",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device,
        )
        self.image_classifier = pipeline(
            "image-classification",
            model="google/vit-base-patch16-224",
            device=device,
        )
        # ASR using manual inference (transformers 5.0 pipeline has issues)
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

        # Captioning using manual inference (transformers 5.0 renamed task)
        from transformers import BlipProcessor, BlipForConditionalGeneration
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    def analyze_text(self, text):
        """Classify text sentiment."""
        return self.text_classifier(text)

    def analyze_image(self, image):
        """Classify image and generate caption."""
        classification = self.image_classifier(image)
        # Manual captioning inference
        inputs = self.blip_processor(image, return_tensors="pt")
        output = self.blip_model.generate(**inputs, max_new_tokens=50)
        caption = self.blip_processor.decode(output[0], skip_special_tokens=True)
        return {
            "classification": classification[:3],
            "caption": caption,
        }

    def transcribe_audio(self, audio):
        """Transcribe audio to text."""
        inputs = self.whisper_processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt")
        generated_ids = self.whisper_model.generate(inputs["input_features"], language="en", task="transcribe")
        return {"text": self.whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]}
```

---

## Phase 4: Performance Comparison

**⏱️ Duration: 1.5 hours**

### Learning Objectives

- Benchmark multiple models for each task
- Measure and compare inference latency
- Document speed vs. accuracy trade-offs

### 4.1 Benchmarking Models

```python
import time
from transformers import pipeline

device = get_device()  # From Section 3.1

def benchmark_model(task, model_id, test_input, n_iterations=10):
    """Benchmark a model's inference latency."""
    pipe = pipeline(task, model=model_id, device=device)

    # Warm-up
    _ = pipe(test_input)

    # Measure
    start = time.time()
    for _ in range(n_iterations):
        result = pipe(test_input)
    elapsed = time.time() - start

    return {
        "model": model_id,
        "latency_ms": (elapsed / n_iterations) * 1000,
        "sample_result": result,
    }

# Compare text models
text_models = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
]

for model_id in text_models:
    result = benchmark_model(
        "text-classification",
        model_id,
        "This product is amazing!",
    )
    print(f"{model_id}: {result['latency_ms']:.1f}ms")
```

### 4.2 Deliverable: Comparison Table

| Task | Model | Latency | Quality Notes |
|------|-------|---------|---------------|
| Text | distilbert-sst-2 | — ms | Fast, English only |
| Text | roberta-sentiment | — ms | More accurate |
| Image | vit-base | — ms | Good accuracy |
| Image | resnet-50 | — ms | Faster |
| ASR | whisper-tiny | — s | Fastest |
| ASR | whisper-base | — s | Better accuracy |

---

## Phase 5: Integration Demo

**⏱️ Duration: 1 hour**

### Learning Objectives

- Create an end-to-end analysis workflow
- Process mixed content types
- Generate a formatted analysis report

### 5.1 Full Analysis Workflow

```python
import requests
from PIL import Image
from io import BytesIO

def analyze_content(analyzer, content):
    """Run full multi-modal analysis on content."""
    report = {"text_analysis": [], "image_analysis": [], "summary": {}}

    # Analyze headlines
    for headline in content["headlines"]:
        result = analyzer.analyze_text(headline)
        report["text_analysis"].append({
            "text": headline,
            "sentiment": result[0]["label"],
            "confidence": result[0]["score"],
        })

    # Analyze images
    for url in content["image_urls"]:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        result = analyzer.analyze_image(image)
        report["image_analysis"].append({
            "url": url,
            "classification": result["classification"][0]["label"],
            "caption": result["caption"],
        })

    # Generate summary
    sentiments = [r["sentiment"] for r in report["text_analysis"]]
    report["summary"] = {
        "total_headlines": len(content["headlines"]),
        "positive_count": sentiments.count("POSITIVE"),
        "negative_count": sentiments.count("NEGATIVE"),
        "images_processed": len(content["image_urls"]),
    }

    return report
```

### 5.2 Demo Content

```python
demo_content = {
    "headlines": [
        "Scientists discover breakthrough in renewable energy",
        "Stock market crashes amid economic uncertainty",
        "Local team wins championship after dramatic final",
        "New study raises concerns about social media usage",
    ],
    "image_urls": [
        "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    ],
}

analyzer = MultiModalAnalyzer(device=device)
report = analyze_content(analyzer, demo_content)
```

### 5.3 Report Generation

```python
def print_report(report):
    """Print formatted analysis report."""
    print("\n" + "=" * 60)
    print("MULTI-MODAL CONTENT ANALYSIS REPORT")
    print("=" * 60)

    print("\n📝 TEXT ANALYSIS")
    print("-" * 40)
    for item in report["text_analysis"]:
        emoji = "✅" if item["sentiment"] == "POSITIVE" else "❌"
        print(f"{emoji} {item['text'][:50]}...")
        print(f"   Sentiment: {item['sentiment']} ({item['confidence']:.1%})")

    print("\n🖼️ IMAGE ANALYSIS")
    print("-" * 40)
    for item in report["image_analysis"]:
        print(f"Classification: {item['classification']}")
        print(f"Caption: {item['caption']}")

    print("\n📊 SUMMARY")
    print("-" * 40)
    print(f"Headlines analyzed: {report['summary']['total_headlines']}")
    print(f"  Positive: {report['summary']['positive_count']}")
    print(f"  Negative: {report['summary']['negative_count']}")
    print(f"Images processed: {report['summary']['images_processed']}")

print_report(report)
```

---

## Submission Requirements

### Code Deliverables

| Deliverable | Description |
|-------------|-------------|
| `capstone.ipynb` | Jupyter notebook with all cells executed and documented |
| `analyzer.py` | Python module with `MultiModalAnalyzer` class, type hints, and docstrings |

### Written Deliverables

| Section | Length | Content |
|---------|--------|---------|
| Model Selection Report | 200–300 words | Models chosen, selection criteria, trade-offs |
| Performance Analysis | 150–200 words | Latency comparison, use-case recommendations |
| Reflection | 100–150 words | Key learnings, challenges, extension ideas |

---

## Evaluation Criteria

| Criterion | Points | Requirements |
|-----------|--------|--------------|
| Hub Exploration | 15 | Systematic search, documented comparison |
| Dataset Exploration | 15 | Three modalities, statistics, streaming |
| Pipeline Implementation | 25 | All four tasks working correctly |
| Performance Comparison | 20 | Multiple models compared, latency measured |
| Integration Demo | 15 | End-to-end workflow, clean output |
| Code Quality | 10 | Clean, documented, type hints |
| **Total** | **100** | **Passing: 70 points** |

---

## Bonus Challenges

### Zero-Shot Classification (+10 points)

```python
zero_shot = pipeline("zero-shot-classification", device=device)
candidate_labels = ["politics", "sports", "technology", "entertainment"]
result = zero_shot("Apple announces new iPhone with AI features", candidate_labels)
```

### Visual Question Answering (+10 points)

```python
vqa = pipeline("visual-question-answering", device=device)
result = vqa(image, "What animal is in the image?")
```

### Batch Processing Optimization (+5 points)

```python
results = text_classifier(texts, batch_size=8)
```

### Memory Profiling (+5 points)

```python
from transformers import AutoModel

def get_model_memory(model_id):
    """Estimate model memory usage in MB."""
    model = AutoModel.from_pretrained(model_id)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_size / (1024 ** 2)
```

---

## Reference Materials

### Recommended Models

| Task | Model | Size | Notes |
|------|-------|------|-------|
| Text Classification | `distilbert-base-uncased-finetuned-sst-2-english` | 66M | Fast, accurate |
| Image Classification | `google/vit-base-patch16-224` | 86M | Good balance |
| Speech Recognition | `openai/whisper-tiny` | 39M | Fastest |
| Image Captioning | `Salesforce/blip-image-captioning-base` | 247M | Quality captions |

### Sample Datasets

| Dataset | Modality | Size | Hub ID |
|---------|----------|------|--------|
| IMDb | Text | 50K | `imdb` |
| CIFAR-10 | Image | 60K | `cifar10` |
| Common Voice | Audio | 1M+ | `mozilla-foundation/common_voice_11_0` |

### Documentation

- [Transformers Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [Hub Python Library](https://huggingface.co/docs/huggingface_hub)
- [Datasets Library](https://huggingface.co/docs/datasets)

---

## FAQ

**Do I need a GPU?**
No. All tasks run on CPU, though a GPU significantly speeds up image and audio inference.

**What about memory usage?**
Loading all four models simultaneously requires ~2-3GB RAM. If you have limited memory (4GB), load models one at a time and delete them when done:
```python
del analyzer.captioner  # Free memory
import gc; gc.collect()  # Force garbage collection
```

**Can I use different models?**
Yes! Exploring the Hub and documenting your choices is part of the exercise.

**What if a model download fails?**
Check your connection and try a smaller model. Some models require authentication—run `huggingface-cli login` if needed.

**Can I use my own content?**
Absolutely. Using your own images or audio demonstrates real-world application.

---

## Getting Started

Clone the course repository to access starter code, utilities, and additional resources:

```bash
git clone https://github.com/paiml/HF-Hub-Ecosystem.git
cd HF-Hub-Ecosystem
make setup
```

See the [repository README](https://github.com/paiml/HF-Hub-Ecosystem) for full setup instructions.

---

*Course 1 of the Hugging Face ML Specialization*
*Last Updated: January 2026*
