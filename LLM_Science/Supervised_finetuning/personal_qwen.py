from unsloth import FastLanguageModel
from transformers import pipeline

### GPU is needed for this

# fine-tuned model
model, tokenizer = FastLanguageModel.from_pretrained(
    "personal_qwen",
    load_in_4bit=True,       
    max_seq_length=1024
)

# Create text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

#### CLI
while True:
    q = input("Question: ")
    if q.lower() in ["exit", "quit"]:
        break
    prompt = f"### Instruction:\n{q}\n\n### Response:\n"
    print(pipe(prompt, max_new_tokens=120)[0]["generated_text"])