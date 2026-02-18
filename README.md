# LLM_Grind

A personal workspace documenting experiments and notes while following the LLM course (https://github.com/mlabonne/llm-course). The repository collects notebooks, scripts, datasets, local model files and experiment outputs used to learn and experiment with language models.

## Overview
- Purpose: experimental playground for learning LLM concepts, decoding strategies, small model analyses, and sequence models.


## Important files and folders

- `decoder_strategies.ipynb` — Visual and programmatic exploration of decoding strategies (greedy, beam search, top-k, nucleus). Contains code to build a small search tree, score tokens, and visualize the search graph. Also includes plotting utilities (uses NetworkX + Graphviz/pydot).

- `full_seq2seq_pipeline.py` / `full_seq2seq_pipeline.ipynb` — End-to-end seq2seq pipeline examples (data preparation, model training / inference flows).

- `CBOW_pytorch.ipynb`, `RNN_pytorch.ipynb` — Classic word-embedding and recurrent network tutorials implemented in PyTorch used to build intuition about representation learning and sequence modeling.

- `gpt2_analysis.ipynb` — Experiments and analysis using GPT-2 via Hugging Face Transformers.

- `nanogpt-badasz.ipynb` — Notebook for training a small gpt model on shakesphere, it generates shakesphere-like text.

- `seq2seq.ipynb` — Notebook exploring sequence-to-sequence setups (encoder-decoder training and testing).

- `eng_text.txt`, `example.txt`, `Random English Sentences.txt`, `metadata.tsv` — Small textual resources used for toy training, tokenization checks and examples.

- `sample_en-fr.txt` and `data/eng-fra.txt` — Parallel data (English–French) used in translation/seq2seq experiments.

- `yoruba_english_parallel_sample.csv` — Parallel sample data for English–Yoruba experiments.

- `data/names/` — Name lists per language (common NLP toy dataset for character-level models or name classification tasks).

- `models/gpt2/` — Locally saved GPT-2 model artifacts (config, tokenizer files and model weights). Useful for offline inference with Transformers.

- `us-patent-phrase-to-phrase-matching/` — Kaggle-style dataset and CSVs for a separate NLP task (included as part of datasets tried).

- `RNN_pytorch.ipynb` — Implementing RNN with pytorch.

