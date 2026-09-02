# ABADES-SLM-15M

<p align="center">
  <img src="tinystories-illustration.png" alt="A small GPT-style model reading storybooks" width="70%" />
</p>

A small GPT-style language model trained from scratch on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. Designed as a lightweight, educational SLM (Small Language Model), trained on an NVIDIA H100 GPU.

---

## Model Details

| Property | Value |
|---|---|
| **Architecture** | GPT (decoder-only transformer) |
| **Parameters** | ~15M |
| **Vocab size** | 50,257 (GPT-2 tokenizer) |
| **Context length** | 128 tokens |
| **Layers** | 6 |
| **Attention heads** | 6 |
| **Embedding dim** | 384 |
| **Training dataset** | TinyStories |
| **Training iterations** | 45,000 |
| **License** | OpenRAIL |

---

## Architecture

<p align="center">
  <img src="gpt-architecture.svg" alt="ABADES-SLM-15M decoder-only transformer architecture diagram" width="85%" />
</p>

Token ids are embedded and combined with learned positional embeddings, then passed through 6 stacked
transformer blocks (pre-norm LayerNorm → masked multi-head self-attention → residual → LayerNorm →
feed-forward MLP → residual). A final LayerNorm and linear head project back to the 50,257-token
vocabulary for next-token prediction — the same core recipe as GPT-2, scaled down to ~15M parameters.

---

## Usage

```python
import torch
import tiktoken
from model import GPT, GPTConfig  # your model file

# Load tokenizer
enc = tiktoken.get_encoding("gpt2")

# Load model
config = GPTConfig(
    vocab_size=50257,
    block_size=128,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.0,
    bias=True
)

model = GPT(config)
checkpoint = torch.load("ABADES-SLM-15M.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Generate text
sentence = "Once upon a time there was a little girl"
context = torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(0)

with torch.no_grad():
    output = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=40)

print(enc.decode(output.squeeze().tolist()))
```

---

## Training Details

- **Tokenizer:** GPT-2 BPE (`tiktoken`)
- **Optimizer:** AdamW (`lr=1e-4`, `betas=(0.9, 0.95)`, `weight_decay=0.1`)
- **LR Schedule:** Linear warmup (1000 steps) → Cosine decay
- **Mixed precision:** bfloat16 / float16
- **Gradient accumulation:** 32 steps
- **Gradient clipping:** 0.5
- **Hardware:** NVIDIA H100

<p align="center">
  <img src="h100-gpu-utilization.png" alt="H100 GPU utilization during training" width="55%" />
</p>

<p align="center"><sub>GPU utilization during a training run — the duty-cycle pattern reflects gradient accumulation steps interleaved with data loading.</sub></p>

---

## Example Outputs

**Prompt:** `"Once upon a time there was a pumpkin."`
> Once upon a time there was a pumpkin. It was big and orange and lived in a garden...

**Prompt:** `"A little girl went to the woods"`
> A little girl went to the woods with her dog. They were looking for something fun to do...

---

## Limitations

- Trained only on simple children's stories (TinyStories)
- Context window limited to 128 tokens
- Not suitable for complex reasoning or factual tasks
- May generate repetitive or incoherent text on out-of-domain prompts

---

## Also in this repo — diffusion experiments

This repo also holds a separate, unrelated experiment, `DDPM_COMPLETE.ipynb` (a Denoising Diffusion
Probabilistic Model for image generation) — a different architecture family than the GPT text model above.
Its reference diagrams are kept here too:

<p align="center">
  <img src="latent-diffusion-architecture.png" alt="Latent diffusion model architecture reference" width="46%" />
  <img src="diffusion-process-diagram.jpg" alt="Diffusion process diagram" width="46%" />
</p>

These describe an image-diffusion pipeline (encoder/decoder + denoising U-Net + conditioning), **not**
the ABADES-SLM-15M transformer — they're included as reference material for the diffusion notebook, not
as a description of this model's architecture.

---

## Author

**ABADES / Hicham Abadour** — built as a learning project to understand transformer training from scratch.

Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.
