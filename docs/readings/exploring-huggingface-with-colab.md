# Exploring Hugging Face with Colab Notebooks

Google Colab provides free GPU access, making it ideal for experimenting with Hugging Face models without local setup.

---

## Getting Started

### 1. Open a New Notebook

Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

### 2. Enable GPU Runtime

```
Runtime → Change runtime type → T4 GPU → Save
```

Verify GPU access:

```python
!nvidia-smi
```

### 3. Install Libraries

```python
!pip install -q transformers datasets huggingface_hub accelerate
```

The `-q` flag suppresses verbose output.

---

## Authentication

For private models or pushing to Hub, authenticate with your token:

```python
from huggingface_hub import login
login()  # Opens interactive prompt
```

Or use a secret (recommended):

```python
from google.colab import userdata
from huggingface_hub import login

login(token=userdata.get('HF_TOKEN'))
```

To set up secrets: Click the key icon in Colab's left sidebar → Add `HF_TOKEN`.

---

## Common Patterns

### Load a Model Pipeline

```python
from transformers import pipeline

classifier = pipeline("text-classification", device=0)  # device=0 uses GPU
result = classifier("I love this course!")
print(result)
```

### Load a Dataset

```python
from datasets import load_dataset

dataset = load_dataset("imdb", split="train[:100]")
print(dataset[0])
```

### Check Device Placement

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

---

## Memory Management

Colab's free tier has limited GPU memory (~15GB on T4). Use these techniques:

### Clear Cache Between Models

```python
import torch
import gc

del model  # Delete the model variable
gc.collect()
torch.cuda.empty_cache()
```

### Use Smaller Model Variants

| Full Model | Smaller Alternative |
|------------|---------------------|
| `bert-base-uncased` | `distilbert-base-uncased` |
| `gpt2-medium` | `gpt2` |
| `facebook/bart-large` | `facebook/bart-base` |

### Load in Lower Precision

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)
```

---

## Colab-Specific Tips

### Prevent Timeout

Colab disconnects after ~90 minutes of inactivity. For long training runs:

```python
# Add to a cell and run
import IPython
IPython.display.Javascript('''
function ClickConnect(){
    console.log("Keeping alive...");
    document.querySelector("colab-connect-button").click()
}
setInterval(ClickConnect, 60000)
''')
```

### Mount Google Drive

Save models and checkpoints to persist across sessions:

```python
from google.colab import drive
drive.mount('/content/drive')

# Save model
model.save_pretrained('/content/drive/MyDrive/my_model')
```

### Download Files

```python
from google.colab import files
files.download('output.csv')
```

---

## Example: Quick Model Exploration

```python
# 1. Setup
!pip install -q transformers datasets

# 2. Search for models
from huggingface_hub import list_models

models = list(list_models(pipeline_tag="text-classification", sort="downloads", limit=5))
for m in models:
    print(f"{m.id}: {m.downloads:,} downloads")

# 3. Try the top model
from transformers import pipeline

classifier = pipeline("text-classification", model=models[0].id, device=0)
print(classifier("This is amazing!"))
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `CUDA out of memory` | Restart runtime, use smaller model, reduce batch size |
| `Model not found` | Check model ID spelling, ensure you're logged in for gated models |
| `Slow downloads` | Models are cached; subsequent runs are faster |
| `Session disconnected` | Remount Drive, rerun setup cells |

---

## Resources

- [Colab + Hugging Face Guide](https://huggingface.co/docs/transformers/notebooks)
- [Free GPU Comparison](https://colab.research.google.com/notebooks/gpu.ipynb)
- [Colab Pro Features](https://colab.research.google.com/signup)

---

*Estimated reading time: 10 minutes*
