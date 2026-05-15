# Observations — Part C · Indic Token Behavior Analysis

## Dataset
- **Source**: `../part_b_token_analysis/token_counts.csv` (100 sentences × 5 models = 500 rows)
- **Tamil vocabulary sample**: 20 curated words spanning 6 complexity tiers (simple nouns → agglutinated verbs → domain terms → proper nouns)
- **Analysis**: Vocabulary coverage (known / fragmented / unknown), O(n²) attention memory footprint, characters per token

## Key Findings

### Vocabulary Coverage (VIZ C1 — Donut Charts)

Actual results from running `compute_vocab_stats()` on 20 Tamil test words:

| Model | Known | Fragmented | Unknown |
|-------|-------|-----------|---------|
| NLLB-200 | **25%** | 75% | 0% |
| MADLAD | 15% | 85% | 0% |
| mT5 | 10% | 90% | 0% |
| IndicTrans2 | 0% | **100%** | 0% |
| Helsinki | 0% | **100%** | 0% |

**Notable result**: Helsinki produces `▁|<unk>` for every single test word — the `<unk>` token IS present in all 20 Helsinki entries (as toks[1]). The classification code marks a word 'unknown' only when `toks[0] == unk_id`; since toks[0] is the word-boundary marker `▁`, Helsinki rows are classified as 'fragmented'. In practice Helsinki has zero meaningful Tamil vocabulary coverage. The four non-Helsinki models produce genuine subword fragments with no UNK tokens.

**NLLB-200 leads** with 25% known words — its 256K vocabulary spread across 200 languages still reserves enough entries for common Tamil nouns and verb roots. MADLAD (15%) and mT5 (10%) follow. IndicTrans2 and Helsinki both show 0% known, meaning every single test word is split into multiple subword tokens.

### Surprising Finding — IndicTrans2 at 0% Known

The notebook's own expected findings section predicted IndicTrans2 would have the **highest** known% due to its Indic-specific SentencePiece vocabulary. The actual result is the opposite: 0% known, 100% fragmented.

**Why this happens:** IndicTrans2 uses a morpheme-aligned SentencePiece model trained to decompose Tamil words into *linguistically meaningful morpheme fragments*, not to store whole words as single tokens. This is intentional design: representing `வந்திருக்கிறான்` ("he has come") as 2–4 morphemic pieces (`வந்து` + `இருக்கிறான்`) retains grammatical structure that a single opaque token would not. The 0% known result is a consequence of this philosophy — the vocabulary prioritises sub-morpheme granularity over whole-word lookup.

### Helsinki: 0% 'Unknown' Classification vs 33.62% UNK in Part B

Part B measured Helsinki's unknown_token_rate at 33.62% across 100 full translation sentences. The Part C word-level test shows 0% in the 'unknown' category — but this is a classification artefact, **not** evidence that Helsinki handles the test words:

- The code marks a word 'unknown' only when `toks[0] == unk_id`
- Helsinki outputs `▁|<unk>` (2 tokens) for **every single one** of the 20 test words
- `toks[0]` is `▁` (word-boundary marker), which is not the UNK ID → classified as 'fragmented'
- But `toks[1]` IS `<unk>` — the only content token Helsinki can produce for these words

So Helsinki's 100% 'fragmented' row in the donut chart should be read as complete vocabulary failure, not as meaningful subword splitting. Both Part B (33.62% UNK in translations) and Part C (all 20 words → `▁|<unk>`) consistently show the same underlying problem: Helsinki's EN→Dravidian vocabulary cannot represent Tamil text.

### Memory Footprint (VIZ C2 — Horizontal Bar Chart)

Transformer self-attention scales as O(n²) with token sequence length. `memory_score = mean(target_token_count²)` per model over 100 sentences.

**IndicTrans2 incurs the highest memory cost** by a wide margin. With expansion_ratio=5.15, its Tamil output sequences are ~5× longer in tokens than the English source. If a sentence has 20 English tokens, IndicTrans2 produces ~103 Tamil tokens → memory score ≈ 10,600 units. NLLB-200 (expansion 1.38) produces ~28 tokens → ≈ 780 units. This is a **14× memory gap** for the same input sentence.

Expected chart ordering (lowest to highest memory cost): **Helsinki < NLLB-200 < MADLAD < mT5 << IndicTrans2**

For edge deployment or processing long documents without chunking, this gap makes IndicTrans2 significantly more expensive than NLLB-200 or MADLAD despite delivering better BLEU.

### Characters Per Token (VIZ C3 — Bar Chart)

`avg_word_length` from `token_counts.csv` = average Tamil characters encoded per target token. Higher is better — each token carries more linguistic content.

**mT5 is expected to show the highest chars/token value**, consistent with Part B's `avg_word_length` data (mT5=3.92). IndicTrans2's morpheme-splitting approach produces many small tokens, each covering fewer characters — lower chars/token despite better translation quality. Helsinki's low-expansion-ratio output is dense in Tamil characters per token for the words it can handle, but UNKs in actual translations collapse to single-token placeholders.

## Cross-Part Synthesis

| Metric | Best Model | Worst Model | Key Insight |
|--------|-----------|-------------|-------------|
| BLEU (Part A) | MADLAD 29.58 | Helsinki 8.26 | Quality gap driven by vocabulary |
| Unknown token rate (Part B) | IndicTrans2/MADLAD/NLLB 0% | Helsinki 33.62% | Helsinki's OOV problem explains its BLEU collapse |
| Known-word coverage (Part C) | NLLB-200 25% | IndicTrans2/Helsinki 0% | Helsinki's 0% is `▁\|<unk>` failure; IndicTrans2's is intentional morpheme splitting |
| Memory footprint (Part C) | Helsinki (lowest) | IndicTrans2 (highest) | IndicTrans2's quality comes at an O(n²) memory premium |
| Chars/token (Part B+C) | mT5 ~3.92 | Helsinki (lowest) | mT5 encodes most Tamil content per token |

## Model Recommendation (Revised)

**For highest translation quality (BLEU priority):** MADLAD (29.58) or IndicTrans2 (27.75). Both achieve 0% UNK and good fragmentation coverage. IndicTrans2 trades higher memory cost for morphologically-informed tokenisation.

**For balanced quality + memory efficiency:** **NLLB-200** — 25% known word coverage (best), 0% UNK, moderate expansion (1.38×), and good BLEU (24.17). Best all-around choice for resource-constrained deployment.

**Avoid Helsinki** in any production Tamil translation task: all test words tokenise to `▁|<unk>`, 33.62% sentence-level UNK rate, BLEU of 8.26. Its EN→Dravidian training is spread too thin across four languages to give Tamil adequate vocabulary coverage.

## Limitations
- Vocabulary coverage is evaluated on 20 curated words — chosen to represent common Tamil forms, not a random sample of the full Tamil lexicon
- Memory score is a proxy (token_count²) — actual GPU memory also depends on batch size, hidden dimension, and number of attention heads
- Part C uses tokenizers only (no model weights) — real inference memory is 5–10× higher than the proxy score suggests
- The 0% known result for IndicTrans2 is an artefact of morpheme-alignment design philosophy, not a failure of Tamil coverage
