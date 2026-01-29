# Capstone Project: Multi-Modal Content Analyzer

**Course 1 Final Project** | Hugging Face Hub and Ecosystem Fundamentals

---

## Project Overview

### Objective

Build a multi-modal content analysis system that demonstrates mastery of the Hugging Face Hub and ecosystem. You will discover and compare models across modalities, explore relevant datasets, build inference pipelines for text, images, and audio, and create a unified analysis workflow.

### Duration

**Estimated Time:** 6-8 hours

### Prerequisites

- Completion of Course 1 (Hub and Ecosystem Fundamentals)
- Hugging Face account (read access)

---

## Project Scenario

You are a machine learning engineer building a content analysis tool for a media company. The tool needs to:

1. **Classify text** sentiment from article headlines
2. **Classify images** into content categories
3. **Transcribe audio** clips from podcasts
4. **Generate captions** for images

Your task is to build this system using pre-trained models from the Hugging Face Hub.

---

## Phase 1: Hub Exploration and Model Discovery

**Duration:** 1.5 hours

### Objectives

- Search the Hub for models across four tasks
- Compare model options using Hub API
- Document selection criteria and final choices

### Tasks

#### 1.1 Text Classification Models

Search for sentiment analysis models:

```python
from huggingface_hub import HfApi, ModelFilter

api = HfApi()

# Search for text classification models
text_models = api.list_models(
    filter=ModelFilter(
        task="text-classification",
        library="transformers",
    ),
    sort="downloads",
    direction=-1,
    limit=15,
)

print("Top Text Classification Models:")
for model in text_models:
    print(f"  {model.id}: {model.downloads:,} downloads")
```

#### 1.2 Image Classification Models

Search for image classification models:

```python
# Search for image classification models
image_models = api.list_models(
    filter=ModelFilter(
        task="image-classification",
        library="transformers",
    ),
    sort="downloads",
    direction=-1,
    limit=15,
)

print("\nTop Image Classification Models:")
for model in image_models:
    print(f"  {model.id}: {model.downloads:,} downloads")
```

#### 1.3 Speech Recognition Models

Search for automatic speech recognition models:

```python
# Search for ASR models
asr_models = api.list_models(
    filter=ModelFilter(
        task="automatic-speech-recognition",
        library="transformers",
    ),
    sort="downloads",
    direction=-1,
    limit=15,
)

print("\nTop ASR Models:")
for model in asr_models:
    print(f"  {model.id}: {model.downloads:,} downloads")
```

#### 1.4 Image Captioning Models

Search for image-to-text models:

```python
# Search for image captioning models
caption_models = api.list_models(
    filter=ModelFilter(
        task="image-to-text",
        library="transformers",
    ),
    sort="downloads",
    direction=-1,
    limit=15,
)

print("\nTop Image Captioning Models:")
for model in caption_models:
    print(f"  {model.id}: {model.downloads:,} downloads")
```

#### 1.5 Model Card Analysis

For each task, analyze the top 3 model cards:

```python
from huggingface_hub import model_info

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

# Analyze top candidates
text_analysis = analyze_model_card("distilbert-base-uncased-finetuned-sst-2-english")
print(f"Text Model: {text_analysis}")
```

#### 1.6 Selection Documentation

Create a selection table:

| Task | Model Selected | Parameters | License | Rationale |
|------|----------------|------------|---------|-----------|
| Text Classification | | | | |
| Image Classification | | | | |
| Speech Recognition | | | | |
| Image Captioning | | | | |

### Deliverables

- [ ] List of 10+ candidate models per task
- [ ] Model card analysis for top 3 per task
- [ ] Selection table with rationale
- [ ] Written justification (150-200 words)

---

## Phase 2: Dataset Exploration

**Duration:** 1.5 hours

### Objectives

- Load and explore datasets for each modality
- Understand data formats and structures
- Use streaming for large datasets

### Tasks

#### 2.1 Text Dataset

Load and explore a text classification dataset:

```python
from datasets import load_dataset

# Load IMDb for sentiment
text_dataset = load_dataset("imdb", split="test[:1000]")

print(f"Text Dataset: {len(text_dataset)} samples")
print(f"Features: {text_dataset.features}")
print(f"\nSample:")
print(f"  Text: {text_dataset[0]['text'][:200]}...")
print(f"  Label: {text_dataset[0]['label']}")
```

#### 2.2 Image Dataset

Load and explore an image dataset:

```python
# Load CIFAR-10 for images
image_dataset = load_dataset("cifar10", split="test[:100]")

print(f"\nImage Dataset: {len(image_dataset)} samples")
print(f"Features: {image_dataset.features}")
print(f"Labels: {image_dataset.features['label'].names}")

# Display sample image info
sample_image = image_dataset[0]["img"]
print(f"Image size: {sample_image.size}")
```

#### 2.3 Audio Dataset

Load and explore an audio dataset with streaming:

```python
# Load Common Voice with streaming (large dataset)
audio_dataset = load_dataset(
    "mozilla-foundation/common_voice_11_0",
    "en",
    split="test",
    streaming=True,
    trust_remote_code=True,
)

# Get first few samples
print("\nAudio Dataset (streaming):")
for i, sample in enumerate(audio_dataset):
    if i >= 3:
        break
    print(f"  Sample {i}: {sample['sentence'][:50]}...")
    print(f"    Audio: {sample['audio']['sampling_rate']} Hz")
```

#### 2.4 Dataset Statistics

Analyze dataset distributions:

```python
import pandas as pd

# Text dataset analysis
text_df = text_dataset.to_pandas()
print("\nText Dataset Statistics:")
print(f"  Label distribution:\n{text_df['label'].value_counts()}")
text_df['length'] = text_df['text'].str.len()
print(f"  Text length: mean={text_df['length'].mean():.0f}, max={text_df['length'].max()}")

# Image dataset analysis
image_df = image_dataset.to_pandas()
print(f"\nImage Dataset Statistics:")
print(f"  Label distribution:\n{image_df['label'].value_counts()}")
```

### Deliverables

- [ ] Three datasets loaded (text, image, audio)
- [ ] Feature exploration for each dataset
- [ ] Statistics summary table
- [ ] Streaming demonstration for large dataset

---

## Phase 3: Building Inference Pipelines

**Duration:** 2 hours

### Objectives

- Create pipelines for all four tasks
- Handle device placement
- Process batched inputs

### Tasks

#### 3.1 Device Detection

Set up device management:

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

#### 3.2 Text Classification Pipeline

```python
from transformers import pipeline

# Create text classification pipeline
text_classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
)

# Test single prediction
result = text_classifier("This movie was absolutely fantastic!")
print(f"Text Classification: {result}")

# Test batch prediction
texts = [
    "I love this product, it's amazing!",
    "Terrible experience, would not recommend.",
    "It's okay, nothing special.",
]
batch_results = text_classifier(texts)
for text, result in zip(texts, batch_results):
    print(f"  '{text[:40]}...' -> {result['label']} ({result['score']:.3f})")
```

#### 3.3 Image Classification Pipeline

```python
from PIL import Image
import requests
from io import BytesIO

# Create image classification pipeline
image_classifier = pipeline(
    "image-classification",
    model="google/vit-base-patch16-224",
    device=device,
)

# Load sample image
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
response = requests.get(url)
image = Image.open(BytesIO(response.content))

# Classify
results = image_classifier(image)
print(f"\nImage Classification (top 5):")
for r in results[:5]:
    print(f"  {r['label']}: {r['score']:.3f}")
```

#### 3.4 Speech Recognition Pipeline

```python
# Create ASR pipeline
asr = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device=device,
)

# Test with sample audio
audio_sample = next(iter(audio_dataset))
transcription = asr(audio_sample["audio"]["array"])
print(f"\nSpeech Recognition:")
print(f"  Expected: {audio_sample['sentence']}")
print(f"  Transcribed: {transcription['text']}")
```

#### 3.5 Image Captioning Pipeline

```python
# Create image captioning pipeline
captioner = pipeline(
    "image-to-text",
    model="Salesforce/blip-image-captioning-base",
    device=device,
)

# Generate caption
caption = captioner(image)
print(f"\nImage Caption: {caption[0]['generated_text']}")
```

#### 3.6 Unified Pipeline Class

Create a unified interface:

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
        self.asr = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-tiny",
            device=device,
        )
        self.captioner = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
            device=device,
        )

    def analyze_text(self, text):
        """Classify text sentiment."""
        return self.text_classifier(text)

    def analyze_image(self, image):
        """Classify image and generate caption."""
        classification = self.image_classifier(image)
        caption = self.captioner(image)
        return {
            "classification": classification[:3],
            "caption": caption[0]["generated_text"],
        }

    def transcribe_audio(self, audio):
        """Transcribe audio to text."""
        return self.asr(audio)

# Initialize analyzer
analyzer = MultiModalAnalyzer(device=device)
print("MultiModalAnalyzer initialized!")
```

### Deliverables

- [ ] Four working pipelines (text, image, ASR, caption)
- [ ] Device detection and placement
- [ ] Batched inference demonstration
- [ ] Unified `MultiModalAnalyzer` class

---

## Phase 4: Performance Comparison

**Duration:** 1.5 hours

### Objectives

- Compare multiple models per task
- Measure inference latency
- Document trade-offs

### Tasks

#### 4.1 Text Model Comparison

Compare sentiment models:

```python
import time

text_models = [
    "distilbert-base-uncased-finetuned-sst-2-english",
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "nlptown/bert-base-multilingual-uncased-sentiment",
]

test_texts = [
    "This is the best thing I've ever seen!",
    "Absolutely terrible, waste of money.",
    "It's fine, nothing special.",
] * 10  # 30 samples

print("Text Model Comparison:")
print("-" * 60)

for model_id in text_models:
    classifier = pipeline("text-classification", model=model_id, device=device)

    # Warm up
    _ = classifier(test_texts[0])

    # Measure latency
    start = time.time()
    results = classifier(test_texts)
    elapsed = time.time() - start

    print(f"{model_id}:")
    print(f"  Latency: {elapsed:.3f}s ({elapsed/len(test_texts)*1000:.1f}ms/sample)")
    print(f"  Sample: {results[0]}")
    print()
```

#### 4.2 Image Model Comparison

Compare vision models:

```python
image_models = [
    "google/vit-base-patch16-224",
    "microsoft/resnet-50",
    "facebook/deit-base-distilled-patch16-224",
]

print("\nImage Model Comparison:")
print("-" * 60)

for model_id in image_models:
    classifier = pipeline("image-classification", model=model_id, device=device)

    # Warm up
    _ = classifier(image)

    # Measure latency (10 iterations)
    start = time.time()
    for _ in range(10):
        results = classifier(image)
    elapsed = time.time() - start

    print(f"{model_id}:")
    print(f"  Latency: {elapsed/10*1000:.1f}ms/image")
    print(f"  Top prediction: {results[0]['label']} ({results[0]['score']:.3f})")
    print()
```

#### 4.3 ASR Model Comparison

Compare speech recognition models:

```python
asr_models = [
    "openai/whisper-tiny",
    "openai/whisper-base",
]

print("\nASR Model Comparison:")
print("-" * 60)

for model_id in asr_models:
    asr_pipeline = pipeline("automatic-speech-recognition", model=model_id, device=device)

    # Get audio sample
    audio_sample = next(iter(audio_dataset))

    # Measure latency
    start = time.time()
    result = asr_pipeline(audio_sample["audio"]["array"])
    elapsed = time.time() - start

    print(f"{model_id}:")
    print(f"  Latency: {elapsed:.3f}s")
    print(f"  Transcription: {result['text'][:100]}...")
    print()
```

#### 4.4 Results Summary

Create a comparison table:

| Task | Model | Latency | Quality Notes |
|------|-------|---------|---------------|
| Text | distilbert-sst-2 | Xms | Fast, English only |
| Text | roberta-sentiment | Xms | More accurate |
| Image | vit-base | Xms | Good accuracy |
| Image | resnet-50 | Xms | Faster |
| ASR | whisper-tiny | Xs | Fastest |
| ASR | whisper-base | Xs | Better accuracy |

### Deliverables

- [ ] Comparison of 2-3 models per task
- [ ] Latency measurements
- [ ] Trade-off analysis (speed vs. accuracy)
- [ ] Recommendation summary

---

## Phase 5: Integration Demo

**Duration:** 1 hour

### Objectives

- Create end-to-end demo workflow
- Process mixed content
- Generate analysis report

### Tasks

#### 5.1 Demo Content Preparation

```python
# Prepare demo content
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
```

#### 5.2 Full Analysis Workflow

```python
def analyze_content(analyzer, content):
    """Run full multi-modal analysis on content."""
    report = {
        "text_analysis": [],
        "image_analysis": [],
        "summary": {},
    }

    # Analyze headlines
    print("Analyzing headlines...")
    for headline in content["headlines"]:
        result = analyzer.analyze_text(headline)
        report["text_analysis"].append({
            "text": headline,
            "sentiment": result[0]["label"],
            "confidence": result[0]["score"],
        })

    # Analyze images
    print("Analyzing images...")
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

# Run analysis
report = analyze_content(analyzer, demo_content)
```

#### 5.3 Report Generation

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

### Deliverables

- [ ] End-to-end demo workflow
- [ ] Mixed content processing
- [ ] Formatted analysis report
- [ ] Working `MultiModalAnalyzer` demonstration

---

## Submission Requirements

### Code Deliverables

1. **Jupyter Notebook** (`capstone.ipynb`)
   - All code cells executed with outputs
   - Markdown documentation between sections
   - Verification cell at end confirming completion

2. **Python Module** (`analyzer.py`)
   - `MultiModalAnalyzer` class
   - Helper functions
   - Type hints and docstrings

### Written Deliverables

1. **Model Selection Report** (200-300 words)
   - Models chosen for each task
   - Selection criteria and rationale
   - Trade-offs considered

2. **Performance Analysis** (150-200 words)
   - Latency comparison summary
   - Recommendations for different use cases
   - Bottlenecks identified

3. **Reflection** (100-150 words)
   - What you learned about the HF ecosystem
   - Challenges encountered
   - Ideas for extending the project

### Evaluation Criteria

| Criterion | Points | Requirements |
|-----------|--------|--------------|
| **Hub Exploration** | 15 | Systematic search, documented comparison |
| **Dataset Exploration** | 15 | Three modalities, statistics, streaming |
| **Pipeline Implementation** | 25 | All four tasks working correctly |
| **Performance Comparison** | 20 | Multiple models compared, latency measured |
| **Integration Demo** | 15 | End-to-end workflow, clean output |
| **Code Quality** | 10 | Clean, documented, type hints |

**Total:** 100 points

**Passing Score:** 70 points

---

## Bonus Challenges

### Bonus 1: Zero-Shot Classification (+10 points)

Add zero-shot classification capability:

```python
from transformers import pipeline

zero_shot = pipeline("zero-shot-classification", device=device)

candidate_labels = ["politics", "sports", "technology", "entertainment"]
result = zero_shot("Apple announces new iPhone with AI features", candidate_labels)
print(result)
```

### Bonus 2: Visual Question Answering (+10 points)

Add VQA to image analysis:

```python
vqa = pipeline("visual-question-answering", device=device)
result = vqa(image, "What animal is in the image?")
print(result)
```

### Bonus 3: Batch Processing Optimization (+5 points)

Optimize for batch processing:

```python
# Process all texts in a single batch
results = text_classifier(texts, batch_size=8)
```

### Bonus 4: Memory Profiling (+5 points)

Profile memory usage:

```python
import torch

def get_model_memory(model_id):
    """Estimate model memory usage."""
    model = AutoModel.from_pretrained(model_id)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_size / (1024 ** 2)  # MB

print(f"Model size: {get_model_memory('distilbert-base-uncased'):.1f} MB")
```

---

## Resources

### Documentation

- [Transformers Pipelines](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [Hub Python Library](https://huggingface.co/docs/huggingface_hub)
- [Datasets Library](https://huggingface.co/docs/datasets)

### Recommended Models

| Task | Model | Size | Notes |
|------|-------|------|-------|
| Text Classification | `distilbert-base-uncased-finetuned-sst-2-english` | 66M | Fast, accurate |
| Image Classification | `google/vit-base-patch16-224` | 86M | Good balance |
| ASR | `openai/whisper-tiny` | 39M | Fastest |
| Image Captioning | `Salesforce/blip-image-captioning-base` | 247M | Quality captions |

### Sample Datasets

| Dataset | Modality | Size | Hub ID |
|---------|----------|------|--------|
| IMDb | Text | 50K | `imdb` |
| CIFAR-10 | Image | 60K | `cifar10` |
| Common Voice | Audio | 1M+ | `mozilla-foundation/common_voice_11_0` |

---

## Frequently Asked Questions

**Q: Do I need a GPU?**

A: No, all tasks can run on CPU. A GPU will significantly speed up inference, especially for image and audio models.

**Q: Can I use different models than suggested?**

A: Yes! Part of the exercise is exploring the Hub and finding suitable models. Document your choices.

**Q: What if a model download fails?**

A: Check your internet connection and try a smaller model. Some models require authentication - run `huggingface-cli login` if needed.

**Q: Can I use my own images/audio?**

A: Absolutely! Using your own content demonstrates real-world application.

---

*Last Updated: 2026-01-29*
