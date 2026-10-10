# Context-Grounded Question Answering: LoRA Fine-Tuning of Falcon3-1B

Fine-tuning a small open-source language model to answer questions using only the passage it's given.

This project takes **Falcon3-1B**, a 1 billion parameter base model, and fine-tunes it on 2,000 question and answer examples from the SQuAD dataset using **LoRA**, a parameter-efficient method that trains a small set of extra weights instead of the full model. The fine-tuned model is then evaluated on held-out data and wrapped in a LangChain prompt that tells it to stay within the given context.

---

## Why this matters

In healthcare and other high-stakes settings, a model that answers from what it "remembers" can be confidently wrong. A model that answers only from the text in front of it is much easier to trust and check. This project explores how fine-tuning and prompt design can push a model toward grounded answers.

---

## How it works

```
SQuAD (2,000 examples)
      │
      ▼
1. Format       Context / Question / Answer instruction template
      │
      ▼
2. Tokenize     Mask prompt and padding tokens so loss is on answers only
      │
      ▼
3. Fine-tune    LoRA adapters on Falcon3-1B with Hugging Face Trainer
      │
      ▼
4. Evaluate     Validation loss and perplexity on a held-out split
      │
      ▼
5. Serve        Hugging Face pipeline + LangChain prompt template
```

### 1. Load the model
- Base model: `tiiuae/Falcon3-1B-Base`, loaded with `AutoModelForCausalLM`
- The end-of-sequence token is reused as the padding token, since the base model doesn't define one
- Runs on GPU when available

### 2. Add LoRA adapters
Instead of updating all 1 billion parameters, LoRA trains small low-rank matrices on top of selected layers. This cuts memory and training time while keeping the base model frozen.

| Setting | Value |
|---|---|
| Rank (r) | 8 |
| Alpha | 32 |
| Target layers | `k_proj`, `o_proj` |
| Dropout | 0.1 |
| Task type | Causal language modeling |

### 3. Prepare the data
- Loaded the first 2,000 training examples from SQuAD
- Reformatted each one into an instruction-style template:
  ```
  ### Context:
  {passage}

  ### Question:
  {question}

  ### Answer:
  {answer}<eos>
  ```
- Tokenized to a max length of 384 tokens
- **Masked the prompt and padding tokens** (label = -100), so the model is only trained on producing the answer, not on repeating the passage
- Split 90/10 into training and validation sets, then removed any examples where the answer was cut off by truncation, leaving **1,705 training** and **188 validation** examples

### 4. Train
| Setting | Value |
|---|---|
| Epochs | 2 |
| Learning rate | 3e-4 |
| Batch size | 2 (with 4 gradient accumulation steps, effective batch size 8) |
| Warmup steps | 20 |
| Evaluation | Every 50 steps |

Training ran for 428 steps in about 5 minutes on a Colab GPU.

### 5. Serve with LangChain
The fine-tuned model is wrapped in a Hugging Face text generation pipeline (greedy decoding, up to 128 new tokens) and connected to a LangChain prompt template that tells the model to:
- Answer only from the given context
- Say it doesn't know if the answer isn't there
- Keep the answer to one sentence

---

## Results

Validation loss dropped steadily during training, with no sign of overfitting:

| Step | Training loss | Validation loss |
|---|---|---|
| 50 | 0.201 | 0.206 |
| 200 | 0.159 | 0.157 |
| 400 | 0.122 | 0.148 |

**Final evaluation:**
- Validation loss: **0.148**
- Perplexity: **1.16**

**Example on new, unseen text:**

> **Context:** A passage about immune system disorders, immunodeficiency, and autoimmune diseases
> **Question:** What field involves the study of the immune system?
> **Model answer:** Immunology

---

## Tech stack

| Purpose | Tool |
|---|---|
| Model | Falcon3-1B-Base (Hugging Face) |
| Fine-tuning | PEFT (LoRA), Hugging Face Transformers `Trainer` |
| Data | Hugging Face `datasets` (SQuAD) |
| Framework | PyTorch |
| Prompting and serving | LangChain, Hugging Face pipeline |
| Environment | Python, Google Colab (GPU) |

---

## How to run it

1. Open the notebook in Google Colab and set the runtime to GPU.
2. Install the dependencies:
   ```bash
   pip install transformers peft datasets langchain langchain_community
   ```
3. Run all cells. No API key is needed, since everything runs on open-source models.

---

## Limitations

- **Perplexity is measured on answer tokens only.** Because the prompt is masked, and SQuAD answers are short spans copied from the passage, the low perplexity reflects how well the model predicts those short answers. It isn't a full measure of answer quality.
- **Small training set.** Only 2,000 of SQuAD's ~87,000 examples were used, to keep training fast.
- **No side-by-side comparison yet** with the base model before fine-tuning.

## Next steps

- Measure **Exact Match and F1**, the standard SQuAD metrics, on the official validation set
- Compare the base model and the fine-tuned model on the same questions
- Test how often the model correctly says "I don't know" when the answer isn't in the context
- Try a medical question answering dataset such as PubMedQA
- Save and share the trained LoRA adapter on the Hugging Face Hub

## What I learned

- LoRA makes fine-tuning practical on a free GPU by training a tiny fraction of the model's weights.
- Masking the prompt tokens is a small detail that makes a big difference: the model learns to answer, not to copy the question.
- A low loss number on its own isn't enough. Knowing what the metric actually measures is what tells you whether the model is ready to trust.
