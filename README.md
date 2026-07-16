# Victorian Story Generator — Character-Level Small Language Model

A decoder-only Transformer built **from scratch in PyTorch** (no HuggingFace, no pre-built model classes), trained on public-domain Victorian novels to generate coherent prose continuations. The project covers the full machine learning pipeline: automated data collection, corpus cleaning, model design, training loop implementation, and an interactive Streamlit frontend.

---

## Motivation

Most LLM tutorials hand you a `from_pretrained()` call and call it a day. The goal here was the opposite: to understand every component of a language model by implementing it from first principles — attention, positional encoding, the training loop, and inference — before relying on any abstraction.

Character-level tokenization was chosen deliberately. It removes the tokenizer as a black box, forces the model to learn orthographic patterns, and keeps the vocabulary small enough (~100 characters) that the full system remains interpretable and trainable on a CPU in minutes.

---

## What Was Built

| Component | Description |
|---|---|
| **Data pipeline** | Downloads 7 Victorian novels from Project Gutenberg via HTTP, strips boilerplate with regex, normalizes whitespace, and merges into a single corpus |
| **Transformer model** | Decoder-only architecture with multi-head causal self-attention, Layer Norm, GELU MLP, residual connections, and weight tying |
| **Training loop** | AdamW optimizer, gradient clipping, train/val split, periodic loss evaluation, and checkpoint serialization |
| **Inference** | Autoregressive generation with temperature scaling and top-k sampling |
| **Web app** | Streamlit UI with interactive generation controls |

---

## Model Architecture

The model (`CharGPT`) follows the GPT-2 design at a smaller scale:

```
Input characters → Token Embedding + Positional Embedding
                 → Dropout
                 → N × [LayerNorm → CausalSelfAttention → Residual
                         LayerNorm → MLP (GELU) → Residual]
                 → LayerNorm
                 → Linear projection → character logits
```

| Hyperparameter | Value |
|---|---|
| Layers | 4 |
| Attention heads | 4 |
| Embedding dimension | 256 |
| MLP hidden dimension | 1024 (4× embd) |
| Context window | 256 characters |
| Vocabulary | ~100 characters (derived from corpus) |
| Approximate parameters | ~2.5 M |

**Key design decisions:**

- **Weight tying:** The token embedding matrix is shared with the final linear (LM head), following Press & Wolf (2017) and GPT-2. This reduces parameters and regularizes the embedding space.
- **Manual attention:** Scaled dot-product attention is computed manually rather than via `F.scaled_dot_product_attention` for stability across CPU/Windows PyTorch builds and full transparency.
- **Mixed precision:** `torch.autocast` with `bfloat16` is applied only when a CUDA device is detected, avoiding silent precision issues on CPU.
- **Causal mask as buffer:** The lower-triangular mask is registered as a non-persistent buffer so it moves correctly with `.to(device)` without being saved to the checkpoint.

---

## Project Structure

```
SLM/
├── data/
│   └── victorian.txt               # Assembled Victorian corpus (~2 M chars)
├── checkpoints/
│   └── victorian_char_slm.pt       # Saved model checkpoint
├── prepare_gutenberg_victorian.py  # Downloads and cleans the corpus
├── train_char_slm.py               # Model definition + training loop
├── generate_char_slm.py            # CLI text generation
├── app_streamlit.py                # Streamlit web UI
└── requirements.txt
```

---

## Setup

```bash
git clone <repo-url>
cd SLM

python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.1+. Runs on CPU; CUDA/MPS detected automatically.

---

## Running the Pipeline

### 1. Build the corpus

```bash
python prepare_gutenberg_victorian.py --out data/victorian.txt
```

Downloads Pride and Prejudice, A Tale of Two Cities, Great Expectations, Wuthering Heights, Middlemarch, The Adventures of Sherlock Holmes, and A Christmas Carol. Strips Project Gutenberg headers/footers and normalizes whitespace.

### 2. Train

```bash
python train_char_slm.py --data data/victorian.txt --max_steps 800
```

Saves a checkpoint to `checkpoints/victorian_char_slm.pt` every `--eval_interval` steps. The checkpoint stores the model weights, vocabulary mappings, and the full training configuration — making it self-contained and reproducible.

**Training flags:**

| Flag | Default | Description |
|---|---|---|
| `--max_steps` | 800 | Total gradient update steps |
| `--batch_size` | 16 | Sequences per batch |
| `--block_size` | 256 | Context window length |
| `--n_layer` | 4 | Transformer layers |
| `--n_head` | 4 | Attention heads |
| `--n_embd` | 256 | Embedding dimension |
| `--dropout` | 0.1 | Dropout rate |
| `--learning_rate` | 3e-4 | AdamW learning rate |
| `--device` | auto | `cpu`, `cuda`, or `mps` |

### 3. Generate (CLI)

```bash
python generate_char_slm.py \
  --ckpt checkpoints/victorian_char_slm.pt \
  --prompt "It was in the autumn of the year" \
  --max_new 400 \
  --temperature 0.9 \
  --top_k 80
```

### 4. Run the web app

```bash
streamlit run app_streamlit.py
```

Interactive controls for temperature, top-k, and generation length. Model is cached across requests with `@st.cache_resource`.

---

## Limitations & Future Directions

This is a small proof-of-concept. Honest limitations worth noting:

- **Scale:** ~2.5 M parameters trained for 800 steps on ~2 M characters. Coherence holds for a few sentences but degrades over longer spans.
- **Character-level ceiling:** Without subword tokenization, the model must learn word boundaries implicitly, which is harder and less data-efficient than BPE.
- **No fine-tuning stage:** The model is trained purely with next-character prediction (pretraining). Instruction following or stylistic steering would require supervised fine-tuning.

Natural extensions:
- Replace character-level vocabulary with a BPE tokenizer (e.g. `tiktoken`) and scale up
- Add learning rate warm-up and cosine decay
- Experiment with Flash Attention for longer context windows
- Evaluate generation quality with perplexity on a held-out test set

---

## References

- Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
- Radford et al. (2019). *Language Models are Unsupervised Multitask Learners.* (GPT-2)
- Press & Wolf (2017). *Using the Output Embedding to Improve Language Models.*
- Karpathy, A. (2023). *[Let's build GPT: from scratch, in code, spelled out.](https://youtu.be/kCc8FmEb1nY)* (pedagogical reference for the architecture)

---

## License

Training data sourced from [Project Gutenberg](https://www.gutenberg.org/) (public domain). Code is for academic use.
