# Project 1 Blueprint v3.0: Evaluation of Indic Translation Models
## Fully Fixed + Model Documentation + Claude Code Ready

---

## WHAT CHANGED FROM V2 (ALL ISSUES FIXED)

- IndicTrans2 now uses `IndicProcessor` correctly — raw tokenizer call was wrong
- FLoRes-200 dataset keys verified — blueprint now checks actual key names at runtime
- MADLAD no longer uses `device_map="auto"` — uses explicit `.to(DEVICE)` for P100 stability
- mT5 translations now saved to CSV in Part A — Part B was going to crash looking for `pred_mT5`
- Radar chart fixed — removed raw `source_token_count` / `target_token_count` (noise, not signal)
- `LICENSE` file added to repo structure (was missing, submission guide requires it)
- Every model now has full documentation, paper links, architecture explanation, and HuggingFace link
- Every notebook cell has a short comment explaining what model it uses and why

---

## MODEL DOCUMENTATION (READ THIS BEFORE CODING)

This section is the most important part of the blueprint.
Each model has a different architecture, different tokenizer type, different API, and different quirks.
Claude Code must understand each model before writing a single line of translation code.

---

### MODEL 1 — Helsinki-NLP/opus-mt-en-ta (MarianMT)

**HuggingFace:** https://huggingface.co/Helsinki-NLP/opus-mt-en-ta
**Paper:** https://aclanthology.org/W18-6311/ (Tiedemann & Thottingal, 2020)
**Architecture docs:** https://huggingface.co/docs/transformers/model_doc/marian

#### What It Is
MarianMT is a family of encoder-decoder transformer models trained by the University of Helsinki
on the OPUS corpus — a massive collection of publicly available multilingual text.
The `opus-mt-en-ta` variant is trained specifically and only on English → Tamil.
It is the smallest and fastest model in this comparison.

#### Architecture
- Standard 6-layer encoder + 6-layer decoder transformer
- SentencePiece tokenizer (unigram language model)
- Vocabulary size: ~65,000 tokens
- Model size: ~300MB
- Trained on: OPUS parallel corpora (OpenSubtitles, CCAligned, WikiMatrix, etc.)

#### Key Quirk
No language codes needed. The model was trained on one language pair only, so it already
knows it translates English → Tamil. Do NOT pass `src_lang` or `forced_bos_token_id`.

#### API Pattern
```python
# Simple — no language codes
inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4)
```

#### Expected Behavior
Fastest model. Reasonable BLEU on simple everyday sentences. Degrades on complex/domain text
because OPUS data quality is mixed and the model is small.

---

### MODEL 2 — google/mt5-base (mT5)

**HuggingFace:** https://huggingface.co/google/mt5-base
**Paper:** https://arxiv.org/abs/2010.11934 (Xue et al., 2021)
**Architecture docs:** https://huggingface.co/docs/transformers/model_doc/mt5

#### What It Is
mT5 (Multilingual T5) is a text-to-text transformer pretrained on mC4 — a multilingual version
of the Common Crawl dataset covering 101 languages. It is a GENERAL PURPOSE pretrained backbone,
NOT fine-tuned for translation.

#### Architecture
- T5 architecture: encoder-decoder transformer (text-in, text-out for any task)
- SentencePiece tokenizer (unigram), vocabulary size: 250,000 tokens
- mC4 training data is heavily English/European biased despite "multilingual" label
- Model size (base): ~580MB
- Tamil coverage in mC4: low — Tamil is present but underrepresented vs European languages

#### CRITICAL: Why mT5 is NOT a Translation Model
T5 and mT5 require fine-tuning on a translation task to produce translations.
The base checkpoint has no task conditioning for translation — if you call `.generate()`
on raw mT5-base with an English sentence, it will produce semi-random Tamil-like tokens,
not a real translation. The outputs are NOT semantically meaningful.

#### Why We Include It Anyway
mT5's 250k-token SentencePiece vocabulary is the most interesting tokenizer to analyze
for Tamil subword fragmentation. Because it was trained on general web crawl data
(not Indic-focused), it will fragment Tamil words into more subwords than purpose-built
models. This contrast is the entire point of Part B and Part C.

#### In This Project
- Part A: mT5 is run through the translation loop for completeness, outputs are SAVED
- Part A BLEU scoring: mT5 is EXCLUDED (outputs are not real translations)
- Part B and C: mT5 IS included for tokenization comparison
- Always label mT5 outputs clearly as "tokenization reference only"

#### API Pattern
```python
# No language codes, no task prefix — just generate (output quality not meaningful)
inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
outputs = model.generate(**inputs, max_new_tokens=128)
```

---

### MODEL 3 — facebook/nllb-200-distilled-600M (NLLB-200)

**HuggingFace:** https://huggingface.co/facebook/nllb-200-distilled-600M
**Paper:** https://arxiv.org/abs/2207.04672 (NLLB Team, Meta AI, 2022)
**Architecture docs:** https://huggingface.co/docs/transformers/model_doc/nllb
**Language codes list:** https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200

#### What It Is
NLLB-200 (No Language Left Behind) is Meta AI's massively multilingual translation model
trained to translate between 200 languages. "Distilled-600M" means it is a knowledge-distilled
smaller version of the full 54B parameter model, making it usable on a single GPU.

#### Architecture
- Encoder-decoder transformer with language-adaptive components
- SentencePiece tokenizer, vocabulary: ~256,000 tokens
- Trained on FLORES-200 + CCAligned + WikiMatrix + custom Indic data
- Model size (600M distilled): ~2.4GB
- Tamil support: explicit, direct (Tamil was one of the focus languages)

#### Key Quirk
Language codes are FLORES-200 format: `eng_Latn` for English, `tam_Taml` for Tamil.
The target language is set via `forced_bos_token_id` using `tokenizer.convert_tokens_to_ids()`.
The `src_lang` keyword is passed to the tokenizer, NOT to the model.

#### API Pattern
```python
# src_lang goes to tokenizer, target lang goes to model via forced_bos_token_id
tgt_lang_id = tokenizer.convert_tokens_to_ids("tam_Taml")
inputs = tokenizer(batch, src_lang="eng_Latn", return_tensors="pt", padding=True)
outputs = model.generate(**inputs, forced_bos_token_id=tgt_lang_id, num_beams=4)
```

#### Expected Behavior
Strong BLEU scores, especially for Tamil. Meta specifically invested in Indic language
quality for NLLB. Should outperform Helsinki on complex sentences.

---

### MODEL 4 — ai4bharat/indictrans2-en-indic-1B (IndicTrans2)

**HuggingFace:** https://huggingface.co/ai4bharat/indictrans2-en-indic-1B
**Paper:** https://arxiv.org/abs/2305.16307 (Gala et al., AI4Bharat, 2023)
**IndicTransToolkit GitHub:** https://github.com/VarunGumma/IndicTransToolkit
**IndicNLP Library:** https://github.com/anoopkunchukuttan/indic_nlp_library

#### What It Is
IndicTrans2 is the state-of-the-art English → Indic translation model built by AI4Bharat,
an initiative from IIT Madras. It is trained specifically on 22 Indian languages including
Tamil. The 1B parameter version is the largest model in this comparison.

#### Architecture
- Encoder-decoder transformer (1 billion parameters)
- Custom Indic-aware tokenizer built on SentencePiece — trained on Indic script data
- Vocabulary: ~32,000 tokens (smaller vocab, but Indic-optimized)
- Trained on IndicCorp v2 + Samanantar + custom crawled Indic parallel data
- Model size: ~4GB in fp16

#### CRITICAL QUIRK — IndicProcessor (V2 Blueprint Was Wrong Here)
IndicTrans2 REQUIRES a preprocessing step using `IndicProcessor` from `IndicTransToolkit`.
Without this, punctuation normalization, script normalization, and tokenization alignment
are all wrong and translation quality degrades significantly.

**Install:**
```python
!pip install indic-trans git+https://github.com/VarunGumma/IndicTransToolkit.git -q
```

**Usage:**
```python
from IndicTransToolkit import IndicProcessor
ip = IndicProcessor(inference_mode=True)

# Preprocess batch BEFORE tokenizing
batch_preprocessed = ip.preprocess_batch(
    batch,
    src_lang="eng_Latn",
    tgt_lang="tam_Taml"
)
# Then tokenize the preprocessed batch
inputs = tokenizer(batch_preprocessed, return_tensors="pt", padding=True, truncation=True)
# After decoding, postprocess
decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
translations = ip.postprocess_batch(decoded, lang="tam_Taml")
```

#### Language Codes
Same FLORES-200 format as NLLB: `eng_Latn`, `tam_Taml`

#### Expected Behavior
Best BLEU scores in this comparison. IndicTrans2 was designed specifically for this task
and trained on the most Indic-relevant data. Particularly strong on colloquial Tamil
and complex agglutinated forms.

---

### MODEL 5 — google/madlad400-3b-mt (MADLAD-400)

**HuggingFace:** https://huggingface.co/google/madlad400-3b-mt
**Paper:** https://arxiv.org/abs/2309.04662 (Kudugunta et al., Google, 2023)
**Architecture docs:** https://huggingface.co/docs/transformers/model_doc/t5

#### What It Is
MADLAD-400 (Massively Multilingual Document-Level Translation, 400 languages) is Google's
multilingual translation model trained on 400 languages using data from Common Crawl.
The 3B variant is the medium-sized version (there are also 7B and 10B variants).

#### Architecture
- T5-based encoder-decoder transformer (3 billion parameters)
- SentencePiece tokenizer, vocabulary: ~256,000 tokens
- Trained on cleaned CommonCrawl data across 400 languages
- Model size (3B, fp16): ~6GB — this is the heaviest model in this project
- Tamil support: present but not as deeply optimized as IndicTrans2

#### CRITICAL QUIRK — Task Prefix
MADLAD-400 is a T5-style model that uses task prefixes to know what to do.
For translation, the prefix is `<2{language_code}>`.
For Tamil specifically: `<2ta>`
Without this prefix, the model does not know the target language and produces garbage.

```python
# ALWAYS prepend <2ta> to every input sentence
prefixed = [f"<2ta> {sentence}" for sentence in batch]
inputs = tokenizer(prefixed, return_tensors="pt", padding=True, truncation=True)
```

#### FIXED: Device Loading on Kaggle P100
V2 used `device_map="auto"` which can split layers across CPU/GPU unpredictably on P100.
Use explicit `.to(DEVICE)` instead:
```python
mod = AutoModelForSeq2SeqLM.from_pretrained(
    "google/madlad400-3b-mt",
    torch_dtype=torch.float16,
).to(DEVICE)   # explicit, not device_map="auto"
```
If this causes OOM, free all other models first and verify VRAM with `torch.cuda.memory_allocated()`.

#### Expected Behavior
Good BLEU scores. MADLAD was trained on more data than NLLB but is less Indic-focused
than IndicTrans2. Interesting to compare against NLLB since both are large multilingual models
but trained on different data and with different architectures.

---

## MODEL COMPARISON SUMMARY TABLE

| Model | HuggingFace ID | Params | Tokenizer | Tamil Focus | Use in BLEU | Use in Token Analysis |
|---|---|---|---|---|---|---|
| Helsinki | Helsinki-NLP/opus-mt-en-ta | ~74M | SentencePiece 65k | OPUS data only | ✅ Yes | ✅ Yes |
| mT5 | google/mt5-base | 580M | SentencePiece 250k | Low (web crawl) | ❌ No | ✅ Yes |
| NLLB-200 | facebook/nllb-200-distilled-600M | 600M | SentencePiece 256k | Strong | ✅ Yes | ✅ Yes |
| IndicTrans2 | ai4bharat/indictrans2-en-indic-1B | 1B | Indic SentencePiece 32k | Best | ✅ Yes | ✅ Yes |
| MADLAD | google/madlad400-3b-mt | 3B | SentencePiece 256k | Moderate | ✅ Yes | ✅ Yes |

---

## 1. REPOSITORY STRUCTURE

```
indic-translation-asr-project/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE                          ← REQUIRED by submission guide
│
├── data/
│   └── raw/
│       └── translation_dataset.csv  ← fallback only, FLoRes-200 used as primary
│
├── task1_translation_evaluation/
│   │
│   ├── README.md
│   │
│   ├── part_a_batch_translation/
│   │   ├── part_a_translation_evaluation.ipynb
│   │   ├── translation_outputs.csv
│   │   ├── sacrebleu_results.csv
│   │   └── observations.md
│   │
│   ├── part_b_token_analysis/
│   │   ├── part_b_token_eda.ipynb
│   │   ├── token_counts.csv
│   │   ├── engineered_features.csv
│   │   ├── plots/
│   │   └── observations.md
│   │
│   └── part_c_indic_token_behavior/
│       ├── part_c_indic_token_analysis.ipynb
│       ├── tokenization_comparison.csv
│       ├── tamil_token_patterns.csv
│       ├── plots/
│       └── observations.md
```

---

## 2. RUNTIME SETUP

**Platform:** Kaggle Notebook
**Accelerator:** GPU P100 (16GB VRAM)
**Internet:** On (required for model downloads)

Every notebook starts with this runtime check cell:

```python
# ── Runtime Check ─────────────────────────────────────────────
# Verify GPU is available before loading any models
# Kaggle P100 has 16GB VRAM — enough for all models sequentially in fp16
import torch, gc, os, warnings
warnings.filterwarnings("ignore")

print(f"GPU available : {torch.cuda.is_available()}")
print(f"GPU name      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
print(f"VRAM total    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device  : {DEVICE}")
```

Memory management — call this after EVERY model to avoid OOM:

```python
def clear_memory(model=None, tokenizer=None):
    """
    Delete model and tokenizer objects and free VRAM.
    Must be called after every model in the translation loop.
    MADLAD-3B alone uses ~6GB fp16 — not clearing between models will OOM.
    """
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        used = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM in use after clear: {used:.2f} GB")
```

---

## 3. GLOBAL VISUAL THEME

Paste into every notebook after the runtime check.
All charts across all 3 notebooks must use the same palette for a coherent submission.

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from IPython.display import display, HTML

# ── Palette — one color per model, consistent across all notebooks ─────
PALETTE = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]

MODEL_COLORS = {
    "IndicTrans2" : "#2E86AB",   # blue   — best Indic model
    "NLLB-200"    : "#A23B72",   # purple — Meta multilingual
    "mT5"         : "#F18F01",   # orange — tokenization reference only
    "Helsinki"    : "#C73E1D",   # red    — smallest, fastest
    "MADLAD"      : "#3B1F2B",   # dark   — Google 3B
}

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi"          : 150,
    "figure.facecolor"    : "white",
    "axes.spines.top"     : False,
    "axes.spines.right"   : False,
    "font.family"         : "DejaVu Sans",
})
print("✓ Global theme applied")
```

---

## 4. PART A — Batch Translation & Evaluation

**File:** `part_a_batch_translation/part_a_translation_evaluation.ipynb`

---

### CELL 1 — Install Dependencies

```python
# Install all required packages
# IndicTransToolkit is required for IndicTrans2 preprocessing — NOT optional
# See: https://github.com/VarunGumma/IndicTransToolkit
!pip install transformers sentencepiece sacrebleu datasets accelerate -q
!pip install git+https://github.com/VarunGumma/IndicTransToolkit.git -q
print("✓ Dependencies installed")
```

---

### CELL 2 — All Imports

```python
# ── ALL imports in one cell ────────────────────────────────────
import torch
import gc
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import sacrebleu

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    logging as hf_logging,
)
from datasets import load_dataset
from tqdm.notebook import tqdm
from IPython.display import display, HTML

# IndicProcessor — required for IndicTrans2 only
# Handles script normalization, punctuation, and tokenization alignment
# Docs: https://github.com/VarunGumma/IndicTransToolkit
from IndicTransToolkit import IndicProcessor

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

print("✓ All imports successful")
print(f"  torch  : {torch.__version__}")
print(f"  device : {DEVICE}")
```

---

### CELL 3 — Dataset Loading

```python
# ── Dataset: FLoRes-200 Tamil Evaluation Split ─────────────────
# FLoRes-200 is the standard benchmark for low-resource language translation
# Used by NLLB, IndicTrans2, and MADLAD papers for evaluation
# Source: https://huggingface.co/datasets/facebook/flores
# 1011 sentences in devtest split, professionally translated into 200 languages
# We use eng_Latn (English source) and tam_Taml (Tamil reference)

try:
    dataset = load_dataset("facebook/flores", "eng_Latn-tam_Taml", split="devtest")

    # Verify actual key names — FLoRes key names depend on HF dataset version
    sample = dataset[0]
    print("Available keys:", list(sample.keys()))

    # Use correct keys based on what's actually in the dataset
    eng_key = "sentence_eng_Latn" if "sentence_eng_Latn" in sample else "sentence"
    tam_key = "sentence_tam_Taml" if "sentence_tam_Taml" in sample else "translation"

    df = pd.DataFrame({
        "english"         : [x[eng_key] for x in dataset],
        "reference_tamil" : [x[tam_key] for x in dataset],
    })
    print(f"✓ FLoRes-200 loaded: {len(df)} sentence pairs")

except Exception as e:
    print(f"FLoRes unavailable ({e}), using CSV fallback")
    df = pd.read_csv("../../data/raw/translation_dataset.csv")
    print(f"✓ CSV fallback loaded: {len(df)} rows")

# 100 sentences — meaningful for corpus BLEU, fast enough on Kaggle free tier
df = df.head(100).reset_index(drop=True)
print(f"Using {len(df)} sentences for evaluation")
display(df.head(3))
```

---

### CELL 4 — Model-Specific Translation Functions

```python
# ── Translation Functions — one per model ──────────────────────
# IMPORTANT: each model has a completely different API.
# Do NOT use a single generic translate() function for all models.


def translate_indictrans2(texts, tokenizer, model, ip, batch_size=8):
    """
    IndicTrans2 — ai4bharat/indictrans2-en-indic-1B
    Paper: https://arxiv.org/abs/2305.16307
    Docs:  https://huggingface.co/ai4bharat/indictrans2-en-indic-1B

    REQUIRES IndicProcessor for pre/post processing.
    Without ip.preprocess_batch(), output quality is significantly worse.
    Language codes: eng_Latn (English), tam_Taml (Tamil script)
    """
    translations = []
    for i in tqdm(range(0, len(texts), batch_size), desc="IndicTrans2"):
        batch = texts[i:i+batch_size]

        # Step 1: Preprocess with IndicProcessor (handles script normalization)
        batch_preprocessed = ip.preprocess_batch(
            batch,
            src_lang="eng_Latn",
            tgt_lang="tam_Taml",
        )

        # Step 2: Tokenize preprocessed text
        inputs = tokenizer(
            batch_preprocessed,
            src_lang="eng_Latn",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(DEVICE)

        # Step 3: Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids("tam_Taml"),
                max_new_tokens=256,
                num_beams=4,
            )

        # Step 4: Decode and postprocess with IndicProcessor
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        postprocessed = ip.postprocess_batch(decoded, lang="tam_Taml")
        translations.extend(postprocessed)

    return translations


def translate_nllb(texts, tokenizer, model, batch_size=8):
    """
    NLLB-200 — facebook/nllb-200-distilled-600M
    Paper: https://arxiv.org/abs/2207.04672
    Docs:  https://huggingface.co/facebook/nllb-200-distilled-600M
    Language codes list: https://github.com/facebookresearch/flores/blob/main/flores200/README.md

    src_lang is passed to the tokenizer.
    Target language is set via forced_bos_token_id on the model.
    Language codes use FLORES-200 format: eng_Latn, tam_Taml
    """
    translations = []
    tgt_lang_id = tokenizer.convert_tokens_to_ids("tam_Taml")

    for i in tqdm(range(0, len(texts), batch_size), desc="NLLB-200"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            src_lang="eng_Latn",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=tgt_lang_id,
                max_new_tokens=256,
                num_beams=4,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)
    return translations


def translate_helsinki(texts, tokenizer, model, batch_size=8):
    """
    Helsinki MarianMT — Helsinki-NLP/opus-mt-en-ta
    Paper: https://aclanthology.org/W18-6311/
    Docs:  https://huggingface.co/Helsinki-NLP/opus-mt-en-ta

    Simplest API — no language codes needed at all.
    This model is trained on English→Tamil ONLY so it already knows
    what to translate. SentencePiece tokenizer with ~65k vocab.
    """
    translations = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Helsinki"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)
    return translations


def translate_madlad(texts, tokenizer, model, batch_size=8):
    """
    MADLAD-400 — google/madlad400-3b-mt
    Paper: https://arxiv.org/abs/2309.04662
    Docs:  https://huggingface.co/google/madlad400-3b-mt

    T5-style model — uses task prefix to specify target language.
    Tamil prefix is <2ta> — this MUST be prepended to every sentence.
    Without <2ta>, the model has no idea what language to produce.
    This is different from NLLB and IndicTrans2 which use forced_bos_token_id.
    """
    translations = []
    prefixed = [f"<2ta> {t}" for t in texts]   # CRITICAL — never remove this

    for i in tqdm(range(0, len(prefixed), batch_size), desc="MADLAD"):
        batch = prefixed[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)
    return translations


def translate_mt5(texts, tokenizer, model, batch_size=8):
    """
    mT5 — google/mt5-base
    Paper: https://arxiv.org/abs/2010.11934
    Docs:  https://huggingface.co/google/mt5-base

    NOT a translation model. This is a general pretrained backbone.
    Outputs are NOT real translations — they are meaningless for BLEU.
    We run it only to SAVE its tokenization behavior for Part B and C.
    BLEU scoring for mT5 is intentionally skipped in this notebook.
    mT5 has the largest vocabulary (250k) of all models here — interesting
    to compare how it tokenizes Tamil vs Indic-optimized models.
    """
    translations = []
    for i in tqdm(range(0, len(texts), batch_size), desc="mT5 (tokenization ref)"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128)
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)
    return translations
```

---

### CELL 5 — Translation Loop (One Model at a Time)

```python
# ── Translation Loop — sequential, one model at a time ────────
# Loading all models simultaneously would require ~15GB+ VRAM
# P100 has 16GB — not enough headroom
# Strategy: load → translate → save output → delete → free VRAM → next model

sentences = df["english"].tolist()
all_translations = {}

# ── 1. Helsinki (~300MB, fastest) ─────────────────────────────
# MarianMT en→ta: no language codes, simple API
# Model page: https://huggingface.co/Helsinki-NLP/opus-mt-en-ta
print("\n[1/5] Helsinki-NLP/opus-mt-en-ta")
tok = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ta")
mod = AutoModelForSeq2SeqLM.from_pretrained(
    "Helsinki-NLP/opus-mt-en-ta",
    torch_dtype=torch.float16,
).to(DEVICE)
all_translations["Helsinki"] = translate_helsinki(sentences, tok, mod)
clear_memory(mod, tok)

# ── 2. mT5 (~580MB, tokenization reference only) ──────────────
# General pretrained backbone — NOT a translation model
# Outputs saved for tokenization analysis in Part B/C only
# Model page: https://huggingface.co/google/mt5-base
print("\n[2/5] google/mt5-base (tokenization reference only — NOT scored for BLEU)")
tok = AutoTokenizer.from_pretrained("google/mt5-base")
mod = AutoModelForSeq2SeqLM.from_pretrained(
    "google/mt5-base",
    torch_dtype=torch.float16,
).to(DEVICE)
all_translations["mT5"] = translate_mt5(sentences, tok, mod)
clear_memory(mod, tok)

# ── 3. NLLB-200 (~2.4GB) ──────────────────────────────────────
# Meta AI No Language Left Behind — strong Tamil support
# Model page: https://huggingface.co/facebook/nllb-200-distilled-600M
print("\n[3/5] facebook/nllb-200-distilled-600M")
tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
mod = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/nllb-200-distilled-600M",
    torch_dtype=torch.float16,
).to(DEVICE)
all_translations["NLLB-200"] = translate_nllb(sentences, tok, mod)
clear_memory(mod, tok)

# ── 4. IndicTrans2 (~4GB fp16) ────────────────────────────────
# AI4Bharat — best Indic translation model, requires IndicProcessor
# Model page: https://huggingface.co/ai4bharat/indictrans2-en-indic-1B
# IndicTransToolkit: https://github.com/VarunGumma/IndicTransToolkit
print("\n[4/5] ai4bharat/indictrans2-en-indic-1B")
ip = IndicProcessor(inference_mode=True)   # initialize once, reuse
tok = AutoTokenizer.from_pretrained(
    "ai4bharat/indictrans2-en-indic-1B",
    trust_remote_code=True,
)
mod = AutoModelForSeq2SeqLM.from_pretrained(
    "ai4bharat/indictrans2-en-indic-1B",
    trust_remote_code=True,
    torch_dtype=torch.float16,
).to(DEVICE)
all_translations["IndicTrans2"] = translate_indictrans2(sentences, tok, mod, ip)
clear_memory(mod, tok)

# ── 5. MADLAD-400 (~6GB fp16, heaviest model) ─────────────────
# Google T5-based 3B model — needs <2ta> task prefix for Tamil
# Using explicit .to(DEVICE) instead of device_map="auto" for P100 stability
# Model page: https://huggingface.co/google/madlad400-3b-mt
print("\n[5/5] google/madlad400-3b-mt")
tok = AutoTokenizer.from_pretrained("google/madlad400-3b-mt")
mod = AutoModelForSeq2SeqLM.from_pretrained(
    "google/madlad400-3b-mt",
    torch_dtype=torch.float16,
).to(DEVICE)   # FIXED: explicit .to(DEVICE), not device_map="auto"
all_translations["MADLAD"] = translate_madlad(sentences, tok, mod)
clear_memory(mod, tok)

print("\n✓ All 5 models done")
```

---

### CELL 6 — Save ALL Translations to CSV (Including mT5)

```python
# ── Save all translations including mT5 ───────────────────────
# IMPORTANT: mT5 must be saved here even though it's excluded from BLEU
# Part B notebook loads pred_mT5 from this CSV for tokenization analysis
# If mT5 is not saved here, Part B will crash with KeyError: 'pred_mT5'

bleu_models = ["Helsinki", "NLLB-200", "IndicTrans2", "MADLAD"]
all_models  = ["Helsinki", "mT5", "NLLB-200", "IndicTrans2", "MADLAD"]
# mT5 intentionally excluded from bleu_models — not a translation model

df_results = df.copy()

# Save all 5 model outputs (including mT5)
for model_name in all_models:
    df_results[f"pred_{model_name}"] = all_translations[model_name]

# Compute sentence-level BLEU for translation models only
for model_name in bleu_models:
    df_results[f"bleu_{model_name}"] = df_results.apply(
        lambda row, m=model_name: sacrebleu.sentence_bleu(
            str(row[f"pred_{m}"]),
            [str(row["reference_tamil"])]
        ).score,
        axis=1
    )

df_results.to_csv("sacrebleu_results.csv", index=False)
print("✓ Saved sacrebleu_results.csv — includes all 5 model outputs")
print(f"  Columns: {list(df_results.columns)}")
```

---

### CELL 7 — Corpus BLEU Scores

```python
# ── Corpus BLEU — 4 models only (mT5 excluded) ────────────────
# sacreBLEU corpus_bleu measures overall translation quality
# across all sentences together (more reliable than avg of sentence BLEUs)
# Docs: https://github.com/mjpost/sacrebleu

bleu_results = {}
for model_name in bleu_models:
    score = sacrebleu.corpus_bleu(
        all_translations[model_name],
        [df["reference_tamil"].tolist()]
    )
    bleu_results[model_name] = round(score.score, 2)
    print(f"  {model_name:15s}: BLEU = {score.score:.2f}")

print("\nNote: mT5 excluded — it is not fine-tuned for translation.")
print("See Part B for mT5 tokenization analysis.")
```

---

### CELL 8 — VIZ A1: BLEU Score Bar + KDE

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Corpus BLEU bar chart
ax1 = axes[0]
models = list(bleu_results.keys())
scores = list(bleu_results.values())
colors = [MODEL_COLORS[m] for m in models]

bars = ax1.bar(models, scores, color=colors, edgecolor="white",
               linewidth=1.5, width=0.55)
for bar, score in zip(bars, scores):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.4,
             f"{score:.1f}", ha="center", fontsize=12, fontweight="bold")
ax1.set_title("Corpus BLEU Score by Model\n(mT5 excluded — not a translation model)",
              fontsize=13, fontweight="bold")
ax1.set_ylabel("sacreBLEU Score")
ax1.set_ylim(0, max(scores) * 1.25)

# Right: Sentence-level BLEU distribution (KDE per model)
ax2 = axes[1]
for model_name in bleu_models:
    col = f"bleu_{model_name}"
    df_results[col].plot.kde(
        ax=ax2, label=model_name,
        color=MODEL_COLORS[model_name], linewidth=2,
    )
    ax2.fill_between(
        ax2.lines[-1].get_xdata(),
        ax2.lines[-1].get_ydata(),
        alpha=0.08, color=MODEL_COLORS[model_name],
    )
ax2.set_title("Sentence BLEU Distribution", fontsize=13, fontweight="bold")
ax2.set_xlabel("Sentence BLEU Score")
ax2.legend()

plt.suptitle("Part A: Translation Quality Analysis", fontsize=15,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plots/parta_bleu_analysis.png", bbox_inches="tight", dpi=150)
plt.show()
```

---

### CELL 9 — VIZ A2: Color-Coded Results Table

```python
# Color cells by BLEU score: green=good, yellow=fair, red=poor
def color_bleu(val):
    if val >= 40:   return "background-color: #d4edda; color: #155724"
    elif val >= 20: return "background-color: #fff3cd; color: #856404"
    else:           return "background-color: #f8d7da; color: #721c24"

display_cols = (
    ["english", "reference_tamil"] +
    [f"pred_{m}" for m in bleu_models] +
    [f"bleu_{m}" for m in bleu_models]
)

df_results[display_cols].head(15).style \
    .applymap(color_bleu, subset=[f"bleu_{m}" for m in bleu_models]) \
    .set_caption("🟢 Good (≥40)  |  🟡 Fair (20–40)  |  🔴 Poor (<20)") \
    .format({f"bleu_{m}": "{:.1f}" for m in bleu_models}) \
    .set_table_styles([{
        "selector": "th",
        "props": [("background-color","#2E86AB"),("color","white")]
    }])
```

---

### CELL 10 — VIZ A3: Error Analysis (Markdown Cell)

```markdown
### Error Analysis: Where Models Succeed and Fail

Pick 3 high-BLEU and 3 low-BLEU sentences from IndicTrans2 output and fill this table:

| Quality | English | IndicTrans2 Tamil | BLEU | Reason |
|---------|---------|-------------------|------|--------|
| ✅ | "The cat sat on the mat." | [your output] | [score] | Short, concrete, common vocabulary |
| ✅ | "She went to the market." | [your output] | [score] | Simple past tense, high-frequency words |
| ❌ | "The policy was retrospectively amended." | [your output] | [score] | Abstract legal vocabulary |
| ❌ | "The compound interest accrues quarterly." | [your output] | [score] | Financial domain terms |

**Finding:** All models degrade on abstract/domain-specific vocabulary.
IndicTrans2 maintains advantage on everyday Tamil due to Indic-specific training data.
NLLB-200 performs competitively due to explicit Tamil focus in training.
Helsinki degrades fastest on complex sentences due to small model size and OPUS data quality.
```

---

## 5. PART B — Token-Based Comparative Analysis

**File:** `part_b_token_analysis/part_b_token_eda.ipynb`

---

### CELL 1 — Note on mT5 (Markdown)

```markdown
## Note on mT5 in This Analysis

`google/mt5-base` is a general-purpose pretrained encoder-decoder (paper: https://arxiv.org/abs/2010.11934).
It is NOT fine-tuned for translation — its Tamil outputs are not semantically meaningful.

It is included in Part B and C specifically for TOKENIZATION COMPARISON:
- mT5 has the largest vocabulary of all 5 models: 250,000 SentencePiece tokens
- Its training data (mC4) is heavily English/European biased
- This makes it the best example of how an English-centric tokenizer
  fragments Tamil agglutinated words into many small subwords
- Comparing mT5 fragmentation vs IndicTrans2 fragmentation is the core insight of Part C

BLEU scoring for mT5 is intentionally omitted in Part A.
```

---

### CELL 2 — Load Tokenizers Only

```python
# ── Load tokenizers only — no full models needed ───────────────
# For token metrics we only need the tokenizer, not the full model weights
# This is much cheaper: tokenizers are MBs, not GBs

from transformers import AutoTokenizer

# Each tokenizer ID with its HuggingFace page for reference
TOKENIZER_IDS = {
    "IndicTrans2" : (
        "ai4bharat/indictrans2-en-indic-1B",
        {"trust_remote_code": True},
        # Indic-optimized SentencePiece, vocab ~32k
        # Docs: https://huggingface.co/ai4bharat/indictrans2-en-indic-1B
    ),
    "NLLB-200" : (
        "facebook/nllb-200-distilled-600M",
        {},
        # SentencePiece 256k vocab, FLORES-200 language codes
        # Docs: https://huggingface.co/facebook/nllb-200-distilled-600M
    ),
    "mT5" : (
        "google/mt5-base",
        {},
        # SentencePiece 250k vocab, mC4 training data
        # Docs: https://huggingface.co/google/mt5-base
    ),
    "Helsinki" : (
        "Helsinki-NLP/opus-mt-en-ta",
        {},
        # SentencePiece 65k vocab, OPUS corpus
        # Docs: https://huggingface.co/Helsinki-NLP/opus-mt-en-ta
    ),
    "MADLAD" : (
        "google/madlad400-3b-mt",
        {},
        # SentencePiece 256k vocab, CommonCrawl 400 languages
        # Docs: https://huggingface.co/google/madlad400-3b-mt
    ),
}

tokenizers = {}
for name, (model_id, kwargs, *_) in TOKENIZER_IDS.items():
    print(f"Loading tokenizer: {name}")
    tokenizers[name] = AutoTokenizer.from_pretrained(model_id, **kwargs)
    print(f"  ✓ Vocab size: {tokenizers[name].vocab_size:,}")
```

---

### CELL 3 — Load Translations from Part A

```python
# ── Load Part A saved CSV — do NOT re-run all 5 models ────────
# Part A saved sacrebleu_results.csv with all 5 model outputs
# including mT5 (pred_mT5 column)
# The data flow is: A → B → C via CSV files. No kernel sharing.

df_parta = pd.read_csv("../part_a_batch_translation/sacrebleu_results.csv")
print(f"✓ Loaded Part A results: {len(df_parta)} rows")
print(f"  Columns: {list(df_parta.columns)}")

sentences_en = df_parta["english"].tolist()

model_outputs = {
    "IndicTrans2" : df_parta["pred_IndicTrans2"].tolist(),
    "NLLB-200"    : df_parta["pred_NLLB-200"].tolist(),
    "Helsinki"    : df_parta["pred_Helsinki"].tolist(),
    "MADLAD"      : df_parta["pred_MADLAD"].tolist(),
    "mT5"         : df_parta["pred_mT5"].tolist(),
}
print("✓ All 5 model outputs loaded (including mT5 for tokenization analysis)")
```

---

### CELL 4 — Compute Token Metrics

```python
def clean_tokens(token_list):
    """Strip SentencePiece (▁) and WordPiece (##) artifacts."""
    cleaned = []
    for tok in token_list:
        tok = tok.replace("▁", "").replace("##", "").strip()
        if tok:
            cleaned.append(tok)
    return cleaned


def compute_token_metrics(text_en, text_ta, tokenizer):
    """
    Compute 6 token metrics for one sentence pair.
    text_ta = model output (not reference).

    Metrics:
    - source_token_count: how many tokens English is split into
    - target_token_count: how many tokens Tamil output is split into
    - expansion_ratio: target/source — higher = more tokens for same content
    - avg_word_length: avg characters per Tamil token — lower = more fragmented
    - subword_fragmentation: inverse of avg_word_length — higher = worse
    - unknown_token_rate: % of Tamil tokens that map to UNK
    """
    src_tokens = tokenizer.encode(text_en, add_special_tokens=False)
    tgt_tokens = tokenizer.encode(text_ta, add_special_tokens=False)

    src_count = max(len(src_tokens), 1)
    tgt_count = max(len(tgt_tokens), 1)

    unk_id   = tokenizer.unk_token_id
    unk_rate = (tgt_tokens.count(unk_id) / tgt_count * 100) if unk_id else 0.0

    tamil_chars       = len(str(text_ta).replace(" ", ""))
    avg_chars_per_tok = tamil_chars / tgt_count
    subword_frag      = round(1 / avg_chars_per_tok, 4) if avg_chars_per_tok > 0 else 0

    return {
        "source_token_count"   : src_count,
        "target_token_count"   : tgt_count,
        "expansion_ratio"      : round(tgt_count / src_count, 3),
        "avg_word_length"      : round(avg_chars_per_tok, 3),
        "subword_fragmentation": subword_frag,
        "unknown_token_rate"   : round(unk_rate, 3),
    }


records = []
for model_name, translations in model_outputs.items():
    tok = tokenizers[model_name]
    for idx, (en, ta) in enumerate(zip(sentences_en, translations)):
        metrics = compute_token_metrics(en, str(ta), tok)
        records.append({
            "model"       : model_name,
            "sentence_id" : idx,
            "english"     : en,
            "tamil"       : ta,
            **metrics
        })

token_df = pd.DataFrame(records)
token_df.to_csv("token_counts.csv", index=False)
print(f"✓ Token metrics computed: {len(token_df)} rows")
print(token_df.groupby("model")["expansion_ratio"].mean().round(3))
```

---

### CELL 5 — Feature Engineering

```python
token_df["log_expansion"]       = np.log1p(token_df["expansion_ratio"])
token_df["efficiency_score"]    = token_df["avg_word_length"] / \
                                   token_df["expansion_ratio"]
token_df["fragmentation_class"] = pd.cut(
    token_df["subword_fragmentation"],
    bins=[0, 0.2, 0.5, 1.0, float("inf")],
    labels=["Low", "Medium", "High", "Very High"],
)
token_df.to_csv("engineered_features.csv", index=False)
print("✓ Engineered features saved")
```

---

### CELL 6 — VIZ B1: Radar Chart (FIXED)

```python
# ── Radar Chart — FIXED version ───────────────────────────────
# FIXES from v2:
# 1. Removed source_token_count and target_token_count — these are
#    raw counts that vary by sentence length, not model quality signals.
#    They add noise to the radar without meaning.
# 2. HIGHER_IS_BETTER correctly inverts "lower is better" metrics
#    so that "outer = better" is always true on the chart.

METRICS = [
    "expansion_ratio",
    "avg_word_length",
    "subword_fragmentation",
    "unknown_token_rate",
]
LABELS = [
    "Expansion\nRatio",
    "Avg Word\nLength",
    "Subword\nFragmentation",
    "Unknown\nToken Rate",
]
# True = higher raw value = better (normalize as-is)
# False = lower raw value = better (invert so outer = better)
HIGHER_IS_BETTER = [False, True, False, False]

summary = token_df.groupby("model")[METRICS].mean()

norm = summary.copy()
for col, higher in zip(METRICS, HIGHER_IS_BETTER):
    mn, mx = summary[col].min(), summary[col].max()
    if mx == mn:
        norm[col] = 0.5
    elif higher:
        norm[col] = (summary[col] - mn) / (mx - mn)
    else:
        norm[col] = 1 - (summary[col] - mn) / (mx - mn)

N      = len(METRICS)
angles = [n / N * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

for model_name, color in MODEL_COLORS.items():
    if model_name not in norm.index:
        continue
    values = norm.loc[model_name].values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, "o-", linewidth=2, label=model_name, color=color)
    ax.fill(angles, values, alpha=0.08, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(LABELS, size=10, fontweight="bold")
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["25%","50%","75%","100%"], size=8, color="grey")
ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.5)
ax.set_title("Model Tokenizer Comparison\n(Outer = Better on that metric)",
             size=15, fontweight="bold", pad=25)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=11)

plt.tight_layout()
plt.savefig("plots/partb_radar_chart.png", bbox_inches="tight", dpi=150)
plt.show()
```

---

### CELL 7 — VIZ B2: Bubble Chart

```python
fig, ax = plt.subplots(figsize=(12, 7))

for model_name, color in MODEL_COLORS.items():
    sub = token_df[token_df["model"] == model_name]
    ax.scatter(
        sub["source_token_count"],
        sub["target_token_count"],
        s=sub["expansion_ratio"] * 120,
        c=color, alpha=0.65,
        edgecolors="white", linewidths=0.8,
        label=model_name,
    )

xvals = np.linspace(
    token_df["source_token_count"].min(),
    token_df["source_token_count"].max(), 100
)
ax.plot(xvals, xvals, "k--", linewidth=1, alpha=0.4, label="Ratio = 1.0")

ax.set_xlabel("Source Token Count (English)", fontsize=13)
ax.set_ylabel("Target Token Count (Tamil)", fontsize=13)
ax.set_title("Token Expansion Bubble Chart\n(Bubble size ∝ expansion ratio)",
             fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("plots/partb_bubble_chart.png", bbox_inches="tight", dpi=150)
plt.show()
```

---

### CELL 8 — VIZ B3: Violin Plot

```python
fig, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(
    data=token_df,
    x="model", y="expansion_ratio",
    palette=list(MODEL_COLORS.values()),
    inner="box", linewidth=1.5, ax=ax,
)
ax.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5,
           label="Expansion ratio = 1.0 (no expansion)")
ax.set_title("Token Expansion Ratio Distribution by Model",
             fontsize=15, fontweight="bold")
ax.set_xlabel("Model")
ax.set_ylabel("Expansion Ratio (Target Tokens / Source Tokens)")
ax.legend()
plt.tight_layout()
plt.savefig("plots/partb_violin_expansion.png", bbox_inches="tight", dpi=150)
plt.show()
```

---

### CELL 9 — VIZ B4: Metrics Heatmap

```python
summary_display = token_df.groupby("model")[METRICS].mean().round(3)

fig, ax = plt.subplots(figsize=(11, 5))
sns.heatmap(
    summary_display,
    annot=True, fmt=".2f",
    cmap="RdYlGn_r",
    linewidths=0.5, linecolor="white",
    ax=ax,
    cbar_kws={"label": "Metric Value"},
    annot_kws={"size": 11, "weight": "bold"},
)
ax.set_title("Model × Metric Heatmap (avg across 100 sentences)",
             fontsize=14, fontweight="bold", pad=15)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
plt.tight_layout()
plt.savefig("plots/partb_heatmap.png", bbox_inches="tight", dpi=150)
plt.show()
```

---

## 6. PART C — Indic Token Behavior Analysis

**File:** `part_c_indic_token_behavior/part_c_indic_token_analysis.ipynb`

---

### CELL 1 — Load from Part B CSV + Tokenizers

```python
# ── Load Part B output — do NOT recompute ─────────────────────
# Part B saved token_counts.csv
# Part C loads that file and loads tokenizers only (no full model weights)
# Data flow: A → B → C via saved CSVs. No kernel sharing between notebooks.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from transformers import AutoTokenizer
from IPython.display import display, HTML

token_df = pd.read_csv("../part_b_token_analysis/token_counts.csv")
print(f"✓ token_df loaded from Part B: {len(token_df)} rows")

# Load tokenizers (lightweight — no GPU needed)
TOKENIZER_IDS = {
    "IndicTrans2" : ("ai4bharat/indictrans2-en-indic-1B", {"trust_remote_code": True}),
    "NLLB-200"    : ("facebook/nllb-200-distilled-600M", {}),
    "mT5"         : ("google/mt5-base", {}),
    "Helsinki"    : ("Helsinki-NLP/opus-mt-en-ta", {}),
    "MADLAD"      : ("google/madlad400-3b-mt", {}),
}

tokenizers = {}
for name, (model_id, kwargs) in TOKENIZER_IDS.items():
    tokenizers[name] = AutoTokenizer.from_pretrained(model_id, **kwargs)
    print(f"  ✓ {name} tokenizer loaded — vocab size: {tokenizers[name].vocab_size:,}")
```

---

### CELL 2 — Compute vocab_stats

```python
# ── vocab_stats: how each tokenizer handles Tamil words ────────
# Classifies each Tamil word as:
#   known      = single token (tokenizer "knows" this word)
#   fragmented = multiple tokens (word split into subwords)
#   unknown    = maps to UNK token (tokenizer has never seen this)
#
# This is the key metric for understanding tokenizer quality for Tamil.
# Tamil is agglutinative — words have many suffixes attached.
# Poor tokenizers (like mT5 trained on English-heavy data) will
# fragment even common Tamil words into many subwords.

def compute_vocab_stats(tokenizer, sample_tamil_words):
    known = fragmented = unknown = 0
    unk_id = tokenizer.unk_token_id

    for word in sample_tamil_words:
        toks = tokenizer.encode(word, add_special_tokens=False)
        if not toks:
            continue
        if unk_id and toks[0] == unk_id:
            unknown += 1
        elif len(toks) == 1:
            known += 1
        else:
            fragmented += 1

    return {"known": known, "fragmented": fragmented, "unknown": unknown}


# Tamil test words: mix of simple, agglutinated, and domain-specific
# Agglutination examples: வந்திருக்கிறான் = "he has come" (one word, many morphemes)
sample_tamil_words = [
    # Simple common words
    "மரம்", "நன்றி", "உணவு", "தண்ணீர்", "மக்கள்",
    # Medium complexity — verb forms
    "படிக்கிறாள்", "சென்றார்கள்", "பேசுகிறோம்",
    # High complexity — agglutinated
    "வந்திருக்கிறான்", "செய்துகொண்டிருக்கிறார்கள்",
    # Domain specific
    "அரசாங்கம்", "கணிப்பொறி", "இணையம்", "பொருளாதாரம்",
    # Places and proper nouns
    "தமிழ்நாடு", "சென்னை", "கோயம்புத்தூர்",
    # Rare / technical
    "மக்கள்தொகை", "விவசாயி", "செய்தி",
]

vocab_stats = {}
for name, tok in tokenizers.items():
    vocab_stats[name] = compute_vocab_stats(tok, sample_tamil_words)
    total = max(sum(vocab_stats[name].values()), 1)
    pct   = {k: round(v/total*100, 1) for k, v in vocab_stats[name].items()}
    print(f"  {name:15s}: known={pct['known']}%  frag={pct['fragmented']}%  unk={pct['unknown']}%")
```

---

### CELL 3 — Token Span Visualizer

```python
# ── Token span visualizer — shows how each model splits a Tamil word ──
# Strips special tokens before rendering (▁ SentencePiece, ## WordPiece)

def clean_token_display(tok_str):
    tok_str = tok_str.replace("▁", "")
    tok_str = tok_str.replace("##", "")
    tok_str = tok_str.replace("<unk>", "?")
    tok_str = tok_str.strip()
    return tok_str if tok_str else "?"


def render_token_spans(word, tokens_per_model):
    color_pool = ["#2E86AB","#A23B72","#F18F01","#C73E1D","#3B1F2B",
                  "#44BBA4","#E94F37","#6A0572"]

    html = f"""
    <div style='font-family:monospace; margin:20px 0; padding:16px;
                border:1px solid #eee; border-radius:8px;'>
      <div style='font-size:20px; font-weight:bold; margin-bottom:14px;'>
        Word: <span style='color:#2E86AB'>{word}</span>
      </div>
    """
    for model_name, raw_tokens in tokens_per_model.items():
        tokens = [clean_token_display(t) for t in raw_tokens]
        tokens = [t for t in tokens if t]

        spans = ""
        for i, tok in enumerate(tokens):
            c = color_pool[i % len(color_pool)]
            spans += f"""
            <span style='background:{c}22; border:1.5px solid {c};
                         color:{c}; padding:3px 8px; border-radius:4px;
                         margin:2px; font-size:14px; font-weight:bold;
                         display:inline-block'>{tok}</span>"""

        count = len(tokens)
        label = f"({count} token{'s' if count != 1 else ''})"
        html += f"""
        <div style='margin:8px 0; display:flex; align-items:center; gap:12px;'>
          <span style='width:130px; font-weight:bold; font-size:13px;
                       color:#333;'>{model_name}</span>
          <span style='color:#999; font-size:12px; min-width:75px;'>{label}</span>
          <div style='flex:1;'>{spans}</div>
        </div>"""

    html += "</div>"
    return html


# Test on 3 complexity levels
# Simple: one morpheme, common word
# Medium: 2-3 morphemes, frequent verb form
# Complex: 4+ morphemes, heavily agglutinated verb
test_words = [
    ("Simple — மரம் (tree)",              "மரம்"),
    ("Medium — படிக்கிறாள் (she studies)", "படிக்கிறாள்"),
    ("Complex — வந்திருக்கிறான் (he has come)", "வந்திருக்கிறான்"),
]

for label, word in test_words:
    tokens_by_model = {
        name: tok.tokenize(word)
        for name, tok in tokenizers.items()
    }
    display(HTML(f"<h4>{label}</h4>"))
    display(HTML(render_token_spans(word, tokens_by_model)))
```

---

### CELL 4 — VIZ C1: Vocabulary Coverage Donut Charts

```python
# Donut charts: what % of Tamil words each model handles well
# known = best, fragmented = acceptable, unknown = worst
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for ax, (model_name, color) in zip(axes, MODEL_COLORS.items()):
    stats = vocab_stats[model_name]
    total = max(sum(stats.values()), 1)

    sizes  = [stats["known"]/total, stats["fragmented"]/total, stats["unknown"]/total]
    colors = [color, "#F18F01", "#C73E1D"]

    wedges, _, autotexts = ax.pie(
        sizes, colors=colors, autopct="%1.0f%%",
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax.set_title(model_name, fontsize=12, fontweight="bold", pad=10)

legend_elements = [
    mpatches.Patch(facecolor="#2E86AB", label="Known (1 token)"),
    mpatches.Patch(facecolor="#F18F01", label="Fragmented (split)"),
    mpatches.Patch(facecolor="#C73E1D", label="Unknown (UNK)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=3, fontsize=12,
           bbox_to_anchor=(0.5, -0.05))
fig.suptitle("Tamil Vocabulary Coverage by Model", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig("plots/partc_donut_coverage.png", bbox_inches="tight", dpi=150)
plt.show()
```

---

### CELL 5 — VIZ C2: Memory Footprint

```python
# Transformer attention memory scales quadratically with token count: O(n²)
# More tokens per sentence = more memory = harder to run on long texts
token_df["memory_score"] = token_df["target_token_count"] ** 2
mem_summary = token_df.groupby("model")["memory_score"].mean().sort_values()

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(
    mem_summary.index,
    mem_summary.values / 1000,
    color=[MODEL_COLORS[m] for m in mem_summary.index],
    edgecolor="white", linewidth=1.5, height=0.55,
)
for bar, val in zip(bars, mem_summary.values / 1000):
    ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val:.1f}k", va="center", fontsize=11, fontweight="bold")

ax.set_xlabel("Relative Attention Memory Score (token² / 1000)", fontsize=12)
ax.set_title("Transformer Memory Pressure by Model\n"
             "(More tokens = O(n²) attention cost — lower is more efficient)",
             fontsize=14, fontweight="bold")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig("plots/partc_memory_footprint.png", bbox_inches="tight", dpi=150)
plt.show()
```

---

### CELL 6 — VIZ C3: Characters Per Token

```python
chars_summary = token_df.groupby("model")["avg_word_length"].agg(["mean","std"])

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(chars_summary))
bars = ax.bar(
    x, chars_summary["mean"],
    yerr=chars_summary["std"],
    color=list(MODEL_COLORS.values()),
    edgecolor="white", linewidth=1.5,
    capsize=5, width=0.55,
)
ax.set_xticks(x)
ax.set_xticklabels(chars_summary.index, fontsize=12)
ax.set_ylabel("Avg Characters per Token", fontsize=12)
ax.set_title("Tamil Subword Quality: Characters per Token\n"
             "(Higher = less fragmentation = better Tamil tokenizer)",
             fontsize=14, fontweight="bold")

best_idx = chars_summary["mean"].argmax()
bars[best_idx].set_edgecolor("#2E86AB")
bars[best_idx].set_linewidth(3)
ax.text(
    best_idx,
    chars_summary["mean"].iloc[best_idx] + chars_summary["std"].iloc[best_idx] + 0.05,
    "★ Best", ha="center", color="#2E86AB", fontweight="bold",
)
plt.tight_layout()
plt.savefig("plots/partc_chars_per_token.png", bbox_inches="tight", dpi=150)
plt.show()
```

---

## 7. REQUIREMENTS.TXT

```
torch>=2.0.0
transformers>=4.38.0
sentencepiece>=0.1.99
sacrebleu>=2.3.1
datasets>=2.14.0
accelerate>=0.24.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
jupyter>=1.0.0
ipython>=8.0.0
indic-trans
```

Note: IndicTransToolkit is installed from GitHub in the notebook install cell:
```
git+https://github.com/VarunGumma/IndicTransToolkit.git
```

---

## 8. .gitignore

```
__pycache__/
*.pyc
.ipynb_checkpoints/
*.egg-info/
.env
*.pt
*.bin
*.safetensors
data/raw/
*.log
.DS_Store
```

---

## 9. LICENSE

Add a `LICENSE` file at repo root — required by submission guide.
Use MIT License (standard for academic/project submissions):
```
MIT License
Copyright (c) 2025 [Your Name]
Permission is hereby granted, free of charge, to any person obtaining a copy...
```
Generate full text at: https://choosealicense.com/licenses/mit/

---

## 10. observations.md TEMPLATE

Use this in all three part folders. Fill in actual numbers after running.

```markdown
# Observations — Part [A / B / C]

## Dataset
- FLoRes-200 devtest split (eng_Latn-tam_Taml), first 100 sentences
- Source: https://huggingface.co/datasets/facebook/flores
- Used in place of translation_dataset.csv (not provided) — same evaluation purpose

## Key Findings
1. [Most important numeric finding — e.g., "IndicTrans2 achieved corpus BLEU of XX.X,
   outperforming NLLB-200 by X.X points"]
2. [Second finding — e.g., "mT5 fragmented XX% of Tamil words into 3+ subwords"]
3. [Third finding — e.g., "Helsinki degraded most on sentences with >15 tokens"]

## Surprising Result
[One unexpected finding — e.g., "MADLAD-3B scored lower than NLLB-600M despite
being 5x larger, likely because it was not specifically optimized for Tamil"]

## Model Recommendation
Based on this analysis, IndicTrans2 (ai4bharat/indictrans2-en-indic-1B) is recommended
for English→Tamil translation because:
1. [Specific BLEU number]
2. [Specific tokenization metric]
3. [Training data advantage — Indic-specific corpus vs general multilingual]

## Limitations
- Only 100 sentences evaluated (full FLoRes devtest has 1011)
- mT5 excluded from BLEU scoring (not a translation model)
- Evaluation on FLoRes sentences only — domain generalization not tested
- All models run on Kaggle free tier P100 — larger batch sizes might give different results
```

---

## 11. EXECUTION ORDER FOR CLAUDE CODE

1. Create full directory structure including `LICENSE`
2. Write `requirements.txt` and `.gitignore`
3. Install: `pip install indic-trans git+https://github.com/VarunGumma/IndicTransToolkit.git`
4. Build Part A notebook — run translation loop, save `sacrebleu_results.csv` with ALL 5 models including mT5
5. Build Part B notebook — load from Part A CSV, compute metrics, save `token_counts.csv`
6. Build Part C notebook — load from Part B CSV, load tokenizers only, run analysis
7. Fill `observations.md` in each subfolder with actual numbers
8. Write `README.md`

**Data flow is strictly linear: A → B → C via saved CSVs. No kernel sharing.**

---

## 12. KEY LINKS SUMMARY (for Claude Code context)

| Model | HuggingFace | Paper |
|---|---|---|
| Helsinki MarianMT | https://huggingface.co/Helsinki-NLP/opus-mt-en-ta | https://aclanthology.org/W18-6311/ |
| mT5-base | https://huggingface.co/google/mt5-base | https://arxiv.org/abs/2010.11934 |
| NLLB-200 | https://huggingface.co/facebook/nllb-200-distilled-600M | https://arxiv.org/abs/2207.04672 |
| IndicTrans2 | https://huggingface.co/ai4bharat/indictrans2-en-indic-1B | https://arxiv.org/abs/2305.16307 |
| MADLAD-400 | https://huggingface.co/google/madlad400-3b-mt | https://arxiv.org/abs/2309.04662 |
| IndicTransToolkit | https://github.com/VarunGumma/IndicTransToolkit | — |
| FLoRes-200 dataset | https://huggingface.co/datasets/facebook/flores | — |
| sacreBLEU | https://github.com/mjpost/sacrebleu | — |
| HuggingFace Transformers | https://huggingface.co/docs/transformers | — |

---

*Blueprint v3.0 — All issues fixed, all models documented, Claude Code ready*
