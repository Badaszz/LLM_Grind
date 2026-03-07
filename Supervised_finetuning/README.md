# Personal Qwen2.5 Model – "Badasz's Number 1 AI Fan"

This repository contains a **fine-tuned Qwen2.5 model** trained with **QLoRA** to act as my personal AI assistant — essentially my #1 fan!  

## About

The model was fine-tuned using a dataset of my:

- Resume and professional profile
- GitHub and Hashnode project READMEs
- Blog posts and technical write-ups

It can answer questions about:

- My robotics and machine learning projects  
- Drone mapping and computer vision work  
- My research interests and blog topics  

## Project Structure

- **`data/`** - Contains `dataset.jsonl`, the training dataset with instruction-response pairs
- **`personal_qwen/`** - Fine-tuned model weights and configuration files
- **`personal_qwen.py`** - CLI interface to interact with the model
- **`QWEN2_5_finetune.ipynb`** - Jupyter notebook for the fine-tuning process

## Quick Start

### Using the CLI

Run the interactive CLI to chat with the model:

```bash
python personal_qwen.py
```

### Using in Python

```python
from unsloth import FastLanguageModel
from transformers import pipeline

model, tokenizer = FastLanguageModel.from_pretrained(
    "personal_qwen",
    load_in_4bit=True,
    max_seq_length=1024
)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

prompt = """### Instruction:
Who is Yusuf Solomon Olumide?

### Response:
"""
print(pipe(prompt, max_new_tokens=120)[0]["generated_text"])
```

## Training Data

The training data in the `data/` folder consists of JSONL format files with instruction-response pairs derived from:

- Personal resume and work experience
- Project README files  
- Technical blog posts and write-ups

## Model Details

- **Model Name:** Qwen2.5 (fine-tuned with QLoRA)
- **Location:** `personal_qwen/` folder
- **Max Sequence Length:** 1024 tokens
- **Training Method:** QLoRA (Quantized LoRA for efficient fine-tuning)
