# LLM_Grind

A personal workspace documenting experiments and notes while following the LLM course (https://github.com/mlabonne/llm-course). The repository collects notebooks, scripts, datasets, local model files and experiment outputs used to learn and experiment with language models.

## Overview
- Purpose: experimental playground for learning LLM concepts, decoding strategies, small model analyses, and sequence models.
- Structure: organized into three main areas for engineering, fundamentals, and science-focused experiments.

## Project structure

- `LLM_Engineering/` — prompt engineering experiments and notebooks for designing, testing, and refining prompts.
- `LLM_Fundamentals/` — core NLP tutorials and classification examples using standard NLP tools like spaCy.
- `LLM_Science/` — research-style investigations, decoding strategy analysis, model training workflows, local model artifacts, and fine-tuning experiments.

Each folder includes its own `README.md` for a focused introduction to the notebook and experiment sets it contains.

## Important files and folders

- `LLM_Engineering/prompting_techniques.ipynb` — notebook exploring prompt engineering methods, examples, and templates.

- `LLM_Fundamentals/NLP_classification.ipynb` — classification experiments and examples.
- `LLM_Fundamentals/NLP_For_classification.ipynb` — classification-focused examples and notes.
- `LLM_Fundamentals/NLP_with_spacy.ipynb` — spaCy-based NLP pipeline examples.

- `LLM_Science/decoder_strategies.ipynb` — visual and programmatic decoding strategy exploration for greedy, beam, top-k, and nucleus sampling.
- `LLM_Science/full_seq2seq_pipeline.py` / `LLM_Science/full_seq2seq_pipeline.ipynb` — end-to-end seq2seq pipeline examples.
- `LLM_Science/CBOW_pytorch.ipynb`, `LLM_Science/RNN_pytorch.ipynb` — classic representation learning and recurrent network tutorials.
- `LLM_Science/gpt2_analysis.ipynb`, `LLM_Science/microgpt-badasz.ipynb`, `LLM_Science/nanogpt-badasz.ipynb` — transformer experiments and small model analysis.
- `LLM_Science/models/gpt2/` — locally saved GPT-2 model artifacts for offline use.
- `LLM_Science/Preference_alignment/` — preference alignment and reward modeling experiments.
- `LLM_Science/Supervised_finetuning/` — fine-tuning workflows and personal assistant model artifacts.
- `LLM_Science/quantization/` — model quantization experiments.
- `LLM_Science/decoding_strategies/` — utilities for sampling and search strategies.

- `eng_text.txt`, `example.txt`, `Random English Sentences.txt`, `metadata.tsv` — small text resources used across experiments.
- `sample_en-fr.txt` and `LLM_Science/data/eng-fra.txt` — parallel English–French data used in translation/seq2seq experiments.
- `yoruba_english_parallel_sample.csv` — English–Yoruba parallel sample data.
- `data/names/` — name lists per language used for toy NLP tasks.
- `us-patent-phrase-to-phrase-matching/` — Kaggle-style dataset and CSVs for a separate NLP task.

## Notes

- Use the `README.md` in each major folder for a quick entry point into that area of the project.
- Keep notebooks and scripts organized by folder so the workspace stays easy to explore.

