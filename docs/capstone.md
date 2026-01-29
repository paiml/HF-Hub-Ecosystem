# Capstone Project: End-to-End Sentiment Analysis System

**Courses 1-2 Final Project** | Hugging Face ML Specialization

---

## Project Overview

### Objective

Build a complete sentiment analysis system from scratch, demonstrating mastery of the Hugging Face ecosystem. You will discover and evaluate base models, prepare a custom dataset, fine-tune a transformer, evaluate performance comprehensively, and publish the final model to the Hub.

### Duration

**Estimated Time:** 8-10 hours

### Prerequisites

- Completion of Course 1 (Hub and Ecosystem Fundamentals)
- Completion of Course 2 (Fine-Tuning Transformers)
- Hugging Face account with write access token

---

## Project Scenario

You are a machine learning engineer at a product company. The product team needs a sentiment classifier for customer reviews that:

1. Classifies reviews as **positive**, **negative**, or **neutral**
2. Works on short-form text (1-3 sentences)
3. Achieves >85% accuracy on held-out test data
4. Is published internally for team access

Your task is to build this system using the Hugging Face ecosystem.

---

## Phase 1: Model Discovery and Selection

**Duration:** 1 hour

### Objectives

- Search the Hub for suitable base models
- Compare model architectures and sizes
- Select an appropriate model for fine-tuning

### Tasks

#### 1.1 Search for Candidate Models

Use the Hub API to find text classification models:

```python
from huggingface_hub import HfApi

api = HfApi()

# Search for sentiment models
models = api.list_models(
    task="text-classification",
    sort="downloads",
    direction=-1,
    limit=20
)

for model in models:
    print(f"{model.id}: {model.downloads:,} downloads")
```

#### 1.2 Evaluate Model Cards

For your top 3 candidates, analyze:

- Training data and domain
- Model size (parameters)
- Reported performance metrics
- License compatibility
- Last update date

#### 1.3 Select Base Model

Document your selection with justification:

| Criterion | Model A | Model B | Model C |
|-----------|---------|---------|---------|
| Size (params) | | | |
| Training domain | | | |
| Reported F1 | | | |
| License | | | |
| **Selection** | | | |

### Deliverables

- [ ] List of 10+ candidate models from Hub search
- [ ] Comparison table for top 3 models
- [ ] Written justification for final selection (100-200 words)

---

## Phase 2: Dataset Preparation

**Duration:** 2 hours

### Objectives

- Load and explore a sentiment dataset
- Preprocess text for transformer input
- Create train/validation/test splits
- Handle class imbalance

### Tasks

#### 2.1 Load Dataset

Use a multi-class sentiment dataset:

```python
from datasets import load_dataset

# Option 1: Amazon reviews (3-class)
dataset = load_dataset("amazon_polarity", split="train[:50000]")

# Option 2: Create 3-class from SST-2 + neutral samples
# Option 3: Use your own customer review data
```

#### 2.2 Explore Data Distribution

```python
import pandas as pd

# Analyze class distribution
df = dataset.to_pandas()
print(df['label'].value_counts())

# Analyze text lengths
df['length'] = df['text'].str.len()
print(df['length'].describe())
```

#### 2.3 Preprocess Pipeline

Implement a preprocessing function:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("your-base-model")

def preprocess(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

tokenized = dataset.map(preprocess, batched=True)
```

#### 2.4 Create Splits

```python
# 80% train, 10% validation, 10% test
splits = tokenized.train_test_split(test_size=0.2, seed=42)
test_valid = splits["test"].train_test_split(test_size=0.5, seed=42)

dataset_dict = {
    "train": splits["train"],
    "validation": test_valid["train"],
    "test": test_valid["test"],
}
```

### Deliverables

- [ ] Dataset loaded with at least 10,000 samples
- [ ] Class distribution analysis with visualization
- [ ] Tokenized dataset with train/val/test splits
- [ ] Preprocessing function documented

---

## Phase 3: Model Fine-Tuning

**Duration:** 3 hours

### Objectives

- Configure training arguments
- Implement custom metrics
- Train with early stopping
- Monitor training progress

### Tasks

#### 3.1 Configure Training

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=100,
    report_to=["tensorboard"],
)
```

#### 3.2 Implement Metrics

```python
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")
f1 = evaluate.load("f1")
precision = evaluate.load("precision")
recall = evaluate.load("recall")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy.compute(
            predictions=predictions,
            references=labels
        )["accuracy"],
        "f1": f1.compute(
            predictions=predictions,
            references=labels,
            average="weighted"
        )["f1"],
        "precision": precision.compute(
            predictions=predictions,
            references=labels,
            average="weighted"
        )["precision"],
        "recall": recall.compute(
            predictions=predictions,
            references=labels,
            average="weighted"
        )["recall"],
    }
```

#### 3.3 Initialize Trainer

```python
from transformers import AutoModelForSequenceClassification, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "your-base-model",
    num_labels=3,
    id2label={0: "negative", 1: "neutral", 2: "positive"},
    label2id={"negative": 0, "neutral": 1, "positive": 2},
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_dict["train"],
    eval_dataset=dataset_dict["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

#### 3.4 Train and Monitor

```python
# Train
trainer.train()

# View TensorBoard
# tensorboard --logdir ./logs
```

### Deliverables

- [ ] Training arguments configured with justification
- [ ] Custom compute_metrics function implemented
- [ ] Model trained for at least 3 epochs
- [ ] TensorBoard logs showing training curves
- [ ] Best checkpoint saved

---

## Phase 4: Comprehensive Evaluation

**Duration:** 2 hours

### Objectives

- Evaluate on held-out test set
- Generate confusion matrix
- Perform error analysis
- Document model limitations

### Tasks

#### 4.1 Test Set Evaluation

```python
# Evaluate on test set
results = trainer.evaluate(dataset_dict["test"])
print(f"Test Accuracy: {results['eval_accuracy']:.3f}")
print(f"Test F1: {results['eval_f1']:.3f}")
```

#### 4.2 Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Get predictions
predictions = trainer.predict(dataset_dict["test"])
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

# Plot confusion matrix
cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(cm, display_labels=["negative", "neutral", "positive"])
disp.plot(cmap="Blues")
plt.title("Sentiment Classification Confusion Matrix")
plt.savefig("confusion_matrix.png")
```

#### 4.3 Error Analysis

Analyze misclassified examples:

```python
# Find misclassified samples
errors = []
for i, (pred, label) in enumerate(zip(preds, labels)):
    if pred != label:
        errors.append({
            "text": dataset_dict["test"][i]["text"],
            "predicted": pred,
            "actual": label,
        })

# Analyze patterns
print(f"Total errors: {len(errors)}")
print(f"Error rate: {len(errors) / len(labels):.2%}")

# Show examples
for error in errors[:10]:
    print(f"Text: {error['text'][:100]}...")
    print(f"Predicted: {error['predicted']}, Actual: {error['actual']}\n")
```

#### 4.4 Document Limitations

Based on error analysis, document:

1. **Common failure modes** (e.g., sarcasm, mixed sentiment)
2. **Domain limitations** (e.g., specific to product reviews)
3. **Length constraints** (e.g., performance on very short/long text)
4. **Bias considerations** (e.g., demographic or cultural biases)

### Deliverables

- [ ] Test set metrics (accuracy ≥85%, F1 ≥0.83)
- [ ] Confusion matrix visualization
- [ ] Error analysis with 10+ example failures
- [ ] Written limitations section (200-300 words)

---

## Phase 5: Model Publication

**Duration:** 1.5 hours

### Objectives

- Write comprehensive model card
- Push model to Hub
- Configure repository settings
- Test published model

### Tasks

#### 5.1 Create Model Card

```python
MODEL_CARD = """
---
license: apache-2.0
language: en
tags:
- text-classification
- sentiment-analysis
- transformers
datasets:
- amazon_polarity
metrics:
- accuracy
- f1
pipeline_tag: text-classification
---

# Sentiment Classifier (3-Class)

## Model Description

Fine-tuned [base-model] for 3-class sentiment classification (positive, neutral, negative).

## Intended Use

- Customer review sentiment analysis
- Product feedback classification
- Social media sentiment monitoring

## Training Data

- **Dataset**: Amazon Polarity (subset)
- **Size**: 40,000 training samples
- **Classes**: Positive, Neutral, Negative

## Evaluation Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.XX |
| F1 (weighted) | 0.XX |
| Precision | 0.XX |
| Recall | 0.XX |

## Limitations

- Trained on English product reviews only
- May not generalize to other domains (news, social media)
- Limited handling of sarcasm and irony
- Maximum input length: 128 tokens

## How to Use

```python
from transformers import pipeline

classifier = pipeline("text-classification", model="your-username/sentiment-classifier")
result = classifier("This product is amazing!")
print(result)
```

## Training Procedure

- **Base model**: [base-model-name]
- **Learning rate**: 2e-5
- **Batch size**: 16
- **Epochs**: 3
- **Optimizer**: AdamW with weight decay 0.01

## Citation

If you use this model, please cite:

```bibtex
@misc{sentiment-classifier-2026,
  author = {Your Name},
  title = {3-Class Sentiment Classifier},
  year = {2026},
  publisher = {Hugging Face Hub},
}
```
"""
```

#### 5.2 Push to Hub

```python
# Save model card
with open("./results/README.md", "w") as f:
    f.write(MODEL_CARD)

# Push model and tokenizer
model.push_to_hub("your-username/sentiment-classifier-3class")
tokenizer.push_to_hub("your-username/sentiment-classifier-3class")
```

#### 5.3 Test Published Model

```python
from transformers import pipeline

# Load from Hub
classifier = pipeline(
    "text-classification",
    model="your-username/sentiment-classifier-3class"
)

# Test predictions
test_texts = [
    "This product exceeded my expectations!",
    "It's okay, nothing special.",
    "Terrible quality, complete waste of money.",
]

for text in test_texts:
    result = classifier(text)
    print(f"{text}\n  -> {result}\n")
```

### Deliverables

- [ ] Model card with all required sections
- [ ] Model pushed to Hub (public or private)
- [ ] Tokenizer pushed alongside model
- [ ] Published model tested and working

---

## Submission Requirements

### Code Deliverables

1. **Jupyter Notebook** (`capstone.ipynb`)
   - All code cells executed with outputs
   - Markdown documentation between sections
   - Verification cell at end confirming completion

2. **Model Artifacts**
   - Trained model pushed to Hub
   - TensorBoard logs in `./logs/`
   - Confusion matrix saved as PNG

### Written Deliverables

1. **Model Selection Justification** (100-200 words)
2. **Dataset Analysis Summary** (150-250 words)
3. **Training Configuration Rationale** (100-200 words)
4. **Error Analysis and Limitations** (200-300 words)
5. **Model Card** (complete, following template)

### Evaluation Criteria

| Criterion | Points | Requirements |
|-----------|--------|--------------|
| **Model Discovery** | 10 | Systematic search, documented comparison |
| **Data Preparation** | 15 | Clean splits, proper preprocessing |
| **Training Setup** | 20 | Appropriate hyperparameters, metrics |
| **Model Performance** | 20 | Accuracy ≥85%, F1 ≥0.83 |
| **Evaluation Quality** | 15 | Confusion matrix, error analysis |
| **Model Card** | 10 | Complete, accurate, well-written |
| **Code Quality** | 10 | Clean, documented, reproducible |

**Total:** 100 points

**Passing Score:** 75 points

---

## Bonus Challenges

### Bonus 1: Multi-GPU Training (+5 points)

Configure training to use multiple GPUs with accelerate:

```python
from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)
```

### Bonus 2: Hyperparameter Search (+5 points)

Use Optuna for hyperparameter optimization:

```python
def hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "num_train_epochs": trial.suggest_int("epochs", 2, 5),
    }

trainer.hyperparameter_search(
    hp_space=hp_space,
    n_trials=10,
    direction="maximize",
)
```

### Bonus 3: Inference Optimization (+5 points)

Quantize the model for faster inference:

```python
from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained(
    "your-username/sentiment-classifier-3class",
    torch_dtype=torch.float16,
)
```

### Bonus 4: Custom Dataset (+10 points)

Instead of using an existing dataset, create your own:

1. Collect 1,000+ reviews from a specific domain
2. Label manually or with weak supervision
3. Document collection methodology
4. Analyze inter-annotator agreement (if applicable)

---

## Resources

### Documentation

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Datasets Documentation](https://huggingface.co/docs/datasets)
- [Hub Documentation](https://huggingface.co/docs/hub)

### Recommended Models for Base

| Model | Parameters | Best For |
|-------|------------|----------|
| `distilbert-base-uncased` | 66M | Fast training, limited compute |
| `bert-base-uncased` | 110M | Balanced performance |
| `roberta-base` | 125M | Higher accuracy |
| `deberta-v3-small` | 44M | State-of-the-art efficiency |

### Recommended Datasets

| Dataset | Classes | Size | Domain |
|---------|---------|------|--------|
| `amazon_polarity` | 2 | 3.6M | Product reviews |
| `yelp_review_full` | 5 | 650K | Restaurant reviews |
| `imdb` | 2 | 50K | Movie reviews |
| `tweet_eval` | 3 | 60K | Twitter |

---

## Frequently Asked Questions

**Q: Can I use a different task instead of sentiment analysis?**

A: Yes, you may choose text classification tasks like topic classification, intent detection, or spam detection. The rubric remains the same.

**Q: What if I don't have GPU access?**

A: Use Google Colab (free GPU), or train on CPU with a smaller model (DistilBERT) and reduced dataset size (5,000 samples minimum).

**Q: Can I fine-tune a larger model like RoBERTa-large?**

A: Yes, but ensure you have adequate compute resources. Larger models require more VRAM and training time.

**Q: What if my accuracy is below 85%?**

A: Document your attempts to improve (hyperparameter tuning, data augmentation, different base models). Partial credit is awarded for systematic effort.

---

*Last Updated: 2026-01-29*
