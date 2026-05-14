# Part A — Batch Translation Evaluation Report

**Task:** Task 1 — Indic Translation & ASR Project  
**Date:** 2026-05-14  
**Dataset:** FLoRes-200 (`openlanguagedata/flores_plus`) — first 100 `devtest` sentences, English → Tamil  
**Execution environment:** Kaggle Notebook, Tesla T4 GPU (15.6 GB VRAM), PyTorch 2.10.0+cu128, Python 3.12.12

---

## 1. Objective

Evaluate five translation systems on English-to-Tamil translation using corpus-level and sentence-level BLEU scores against FLoRes-200 human reference translations. The five systems represent a range of model families, sizes, and training approaches.

---

## 2. Models Evaluated

| Model | Family | Parameters | Training Approach |
|---|---|---|---|
| Helsinki-NLP/opus-mt-en-dra | MarianMT | ~74M | Supervised, EN→Dravidian multilingual (ta/kn/ml/te) |
| mT5-base | T5 multilingual | ~580M | Tokenizer analysis only — no translation output |
| facebook/nllb-200-distilled-600M | NLLB-200 | ~600M | Supervised multilingual (200 languages) |
| ai4bharat/indictrans2-en-indic-1B | IndicTrans2 | ~1B | Supervised, Indic-specific (22 languages) |
| google/madlad400-3b-mt | MADLAD-400 | ~3B | Supervised multilingual (400 languages) |

**Note on mT5:** mT5-base was included in this pipeline solely for tokenizer analysis (vocabulary coverage, subword segmentation). Its `pred_mT5` column in the output CSV contains raw SentencePiece token sequences, not natural language translations. It is excluded from BLEU evaluation.

---

## 3. Corpus BLEU Results

Corpus BLEU was computed with SacreBLEU `corpus_bleu` over all 100 sentences (aggregated n-gram statistics, not sentence-average).

| Model | Corpus BLEU | Rank |
|---|---|---|
| **MADLAD-400** | **29.58** | 1 |
| IndicTrans2 | 27.75 | 2 |
| NLLB-200 | 24.17 | 3 |
| Helsinki | 8.26 | 4 |
| mT5 | — | — |

---

## 4. Per-Sentence BLEU Analysis

Per-sentence BLEU scores were computed individually (NLTK `sentence_bleu` with smoothing) and stored in `sacrebleu_results.csv`. These differ from corpus BLEU because sentence-level smoothing reduces scores for short or difficult sentences.

### Summary statistics

| Model | Mean | Median | Min | Max | Sentences ≥30 | Sentences ≥50 | Sentences <5 |
|---|---|---|---|---|---|---|---|
| IndicTrans2 | 21.28 | 14.21 | 2.33 | 85.79 | 27 | 11 | 19 |
| NLLB-200 | 15.61 | 10.01 | 2.32 | 58.50 | 15 | 3 | 27 |
| MADLAD | 11.78 | 8.62 | 2.12 | 58.31 | 7 | 1 | 26 |
| Helsinki | 6.14 | 4.29 | 0.51 | 38.34 | 2 | 0 | 57 |

### Per-sentence "wins" (which model scored highest)

| Model | Sentences where it ranked 1st |
|---|---|
| IndicTrans2 | **65** |
| NLLB-200 | 21 |
| MADLAD | 12 |
| Helsinki | 2 |

### Head-to-head: MADLAD vs IndicTrans2

| Outcome | Count |
|---|---|
| MADLAD sentence BLEU > IndicTrans2 | 20 |
| IndicTrans2 sentence BLEU > MADLAD | **78** |
| Tie | 2 |

**Interpretation:** Although MADLAD leads on corpus BLEU (29.58 vs 27.75), IndicTrans2 wins on 78 of 100 individual sentences. MADLAD's corpus BLEU advantage comes from better long n-gram precision on a subset of sentences where it performs well — not from consistent sentence-level superiority. IndicTrans2 is the more reliable per-sentence model.

---

## 5. Best and Worst Sentence Performance (IndicTrans2)

### Top 5 sentences (BLEU ≥ 62)

| Row | Sentence BLEU | Source excerpt |
|---|---|---|
| 54 | 85.8 | "The Report is highly critical of almost every aspect of the present policy…" |
| 23 | 75.9 | "The other nominations include Best Picture, Director, Cinematography…" |
| 24 | 68.8 | "Two songs from the movie, Audition (The Fools Who Dream) and City of Stars…" |
| 21 | 65.3 | "The movie, featuring Ryan Gosling and Emma Stone, received nominations…" |
| 19 | 62.0 | "During the 1976 selections he advised Carter on foreign policy…" |

High-BLEU sentences tend to be: award category lists, short named-entity-heavy descriptions, or sentences with common vocabulary that overlaps well with FLoRes-200 references.

### Bottom 3 sentences (BLEU < 3)

| Row | Sentence BLEU | Source excerpt |
|---|---|---|
| 93 | 2.3 | "This theory says that most dark matter around a galaxy is located…" |
| 75 | 2.4 | "But Prime Minister John Howard has said the act was only to safeguard…" |
| 95 | 2.9 | "Local authorities are warning residents in the vicinity of the plant…" |

Low-BLEU sentences are typically long, complex, or domain-specific (astrophysics, political statements), where the model produces a semantically plausible translation that diverges in wording from the reference.

---

## 6. Qualitative Translation Analysis

A manual inspection of sample rows reveals the following patterns:

### Helsinki (BLEU 8.26)
- Shortest model, weakest fluency. Translations are often truncated or miss nuance.
- Struggles with named entities (e.g., "Dalhousie University" rendered as "டால்சோசி பல்கலைக்கழகம்" — an approximation).
- Being a 74M bilingual-style model, it lacks the context capacity for long sentences.

### NLLB-200 (BLEU 24.17)
- Generally fluent Tamil output. Good at standard sentence structures.
- Occasional missed proper nouns (e.g., "QVC" retained correctly; "Sveriges Radio" romanised differently from reference).
- More conservative translations — closer paraphrases of source.

### IndicTrans2 (BLEU 27.75, best per-sentence)
- Consistently produces grammatically correct, fluent Tamil.
- Strong named-entity transliteration (e.g., "டல்ஹெளசி பல்கலைக்கழகம்", "நோவா ஸ்கோட்டியா").
- Indic-specific training gives it better coverage of Tamil script norms and honorifics.
- Occasionally verbose — adds contextual phrases not in the source.

### MADLAD-400 (BLEU 29.58, highest corpus BLEU)
- Best aggregate n-gram overlap due to strong common-phrase precision.
- Occasional hallucinations on ambiguous nouns (Row 2: "mice" → "சுறாக்கள்" (sharks)).
- Prefix-based language targeting (`<2ta>`) works correctly; no Tamil script artifacts.
- Benefits most from the `no_repeat_ngram_size=3` + `repetition_penalty=1.2` fix — without it, corpus BLEU was 5.34 (repetition loop artifact).

---

## 7. Key Technical Fix: MADLAD Repetition Artifact

MADLAD-400 without generation constraints produced degenerate output — the same Tamil phrase repeated across the entire output for longer sentences. This caused the initial corpus BLEU of **5.34**.

**Fix applied:**
```python
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    num_beams=4,
    no_repeat_ngram_size=3,   # prevents 3-gram repetition loops
    repetition_penalty=1.2,   # penalises repeated token sequences
)
```

**Result:** Corpus BLEU improved from **5.34 → 29.58** (+24.24 points), making MADLAD the highest-scoring model.

---

## 8. Model Size vs Performance

| Model | Parameters | Corpus BLEU | BLEU per billion params |
|---|---|---|---|
| Helsinki | ~74M | 8.26 | 111.6 |
| NLLB-200 | ~600M | 24.17 | 40.3 |
| IndicTrans2 | ~1B | 27.75 | 27.8 |
| MADLAD-400 | ~3B | 29.58 | 9.9 |

Helsinki achieves the highest BLEU per billion parameters (a rough efficiency metric), but its absolute quality is weakest. IndicTrans2 and NLLB-200 show the best quality-to-size ratio in absolute terms. MADLAD requires 3× the parameters of IndicTrans2 for a modest +1.83 corpus BLEU gain.

---

## 9. Output Files

| File | Description |
|---|---|
| `sacrebleu_results.csv` | 100 rows × 11 columns: source, reference, 5 model predictions, 4 per-sentence BLEU scores |
| `translation_outputs.csv` | Same predictions without BLEU scores |

---

## 10. Conclusions

1. **MADLAD-400 achieves the highest corpus BLEU (29.58)**, but only after fixing the repetition artifact with generation constraints.
2. **IndicTrans2 wins on 78/100 individual sentences** and is the most consistently reliable model sentence-by-sentence.
3. **NLLB-200 is the best mid-size option** — 600M parameters, BLEU 24.17, no special fixes required.
4. **Helsinki is unsuitable for production** at 8.26 BLEU; useful only for tokenizer analysis or resource-constrained offline use.
5. **mT5-base does not produce Tamil translations** from its standard fine-tuning — it serves only as a tokenization reference in this pipeline.
6. All five models handle the Tamil script correctly in terms of Unicode rendering; quality differences are in semantic accuracy and fluency, not character encoding.

---

## 11. Downstream Analysis

Parts B and C of this pipeline have been completed. Their findings extend and explain the Part A results:

**Part B (token analysis):** Using `sacrebleu_results.csv` as input, Part B computed four tokenization metrics (expansion_ratio, avg_word_length, subword_fragmentation, unknown_token_rate) for each model across all 100 sentences. Key finding: Helsinki's tokenizer maps 33.62% of Tamil tokens to `<unk>`, directly explaining its low BLEU. IndicTrans2 has the highest expansion ratio (5.15) but zero unknown tokens — its aggressive subword fragmentation is the cost of complete Tamil morphological coverage. See `part_b_token_analysis/REPORT.md` for the full analysis.

**Part C (Indic token behavior):** Using `engineered_features.csv` from Part B, Part C performs a deeper analysis of Tamil-specific vocabulary coverage, character-level token statistics, and cross-model comparison of Indic script handling. See `part_c_indic_token_behavior/observations.md` for findings.
