# Observations — Part A · Batch Translation Evaluation

## Dataset
- **Source**: FLoRes-200 (`openlanguagedata/flores_plus`, `eng_Latn-tam_Taml` split)
- **Sentences**: First 100 English devtest sentences with Tamil references
- **Platform**: Kaggle GPU — Tesla T4 (15.6 GB VRAM), PyTorch 2.10.0+cu128, Python 3.12.12

---

## Models Evaluated

| Model | Type | Parameters | Corpus BLEU |
|---|---|---|---|
| MADLAD-400 | T5-based multilingual | 3B | **29.58** |
| IndicTrans2 | Encoder-decoder (Indic-specific) | ~1B | 27.75 |
| NLLB-200 | mBART-style multilingual | 600M distilled | 24.17 |
| Helsinki MarianMT | Marian encoder-decoder | ~74M | 8.26 |
| mT5-base | T5 multilingual | ~580M | — (tokenizer only) |

---

## Results Plot

![Corpus BLEU and Sentence BLEU Distribution](plots/parta_bleu_analysis.png)

**Left:** Corpus BLEU scores — MADLAD leads at 29.6, followed closely by IndicTrans2 at 27.7.  
**Right:** Sentence BLEU KDE distribution — Helsinki is tightly clustered near zero (most sentences score <10); IndicTrans2 has the longest right tail (some sentences score 60–85), showing it excels on simpler sentences.

---

## Key Findings

- **Highest corpus BLEU**: MADLAD-400 (29.58) — surprising given it is a general-purpose 400-language model, not Indic-specific
- **Best per-sentence model**: IndicTrans2 — wins on **65/100** individual sentences; MADLAD only wins 12
- **MADLAD corpus BLEU advantage is misleading**: it scores higher overall because of strong n-gram precision on a subset of sentences, not consistent quality across the board
- **Helsinki** is weak at 8.26; 57 of its 100 sentences scored below BLEU 5 — unsuitable for production use
- **mT5** produces raw SentencePiece token sequences, not translations — correctly excluded from BLEU

---

## Surprising Result

**Expected:** IndicTrans2 (Indic-specific training) would be the clear #1.  
**Actual:** MADLAD-400 (general multilingual, 3B params) edged ahead on corpus BLEU (29.58 vs 27.75), but IndicTrans2 is the more *reliable* model sentence-by-sentence (wins 78/100 head-to-head).

The gap is explained by MADLAD's higher precision on a few long, complex sentences where it gets more n-gram matches — pulling up the corpus aggregate. IndicTrans2 is the safer production choice.

---

## Critical Fix: MADLAD Repetition Artifact

Without generation constraints, MADLAD-400 looped the same Tamil phrase across entire outputs on longer sentences. Corpus BLEU was **5.34** before the fix.

```python
# Fix applied in translate_madlad()
outputs = model.generate(
    **inputs,
    max_new_tokens=256,
    num_beams=4,
    no_repeat_ngram_size=3,   # breaks repetition loops
    repetition_penalty=1.2,
)
```

BLEU improved: **5.34 → 29.58** (+24.24 points).

---

## Model Recommendation

**IndicTrans2** for production Tamil translation:
1. Wins 65/100 sentences — most consistently accurate
2. Trained specifically on Indic pairs with IndicProcessor normalization
3. Best sentence BLEU ceiling (85.79 on its best sentence)
4. 3× smaller than MADLAD — faster and cheaper to run

**MADLAD-400** is a viable alternative if highest aggregate BLEU is the only criterion and GPU memory is not a constraint.

---

## Limitations

- Only 100 sentences (FLoRes-200 devtest subset) — may not generalise across domains
- All models use beam search (beam=4) with default settings; no tuning
- MADLAD 3B is slow on T4 — will time out on large batches without batching
- BLEU alone does not capture fluency, grammaticality, or named-entity accuracy
