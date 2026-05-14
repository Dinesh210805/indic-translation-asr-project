# Part B — Token Analysis & Tokenization Efficiency
## Detailed Report with Visualization Interpretation

**Notebook:** `part_b_token_eda.ipynb`  
**Dataset:** FLoRes-200 devtest, `eng_Latn-tam_Taml` split, first 100 sentences  
**Output artifacts:** `token_counts.csv`, `engineered_features.csv`  
**Prerequisite:** `sacrebleu_results.csv` from Part A  

---

## Table of Contents

1. [What Part B Does](#what-part-b-does)
2. [Why Tokenization Matters](#why-tokenization-matters)
3. [The Four Core Metrics](#the-four-core-metrics)
4. [Feature Engineering](#feature-engineering)
5. [Visualization 1 — Heatmap: Mean Metric Values per Model](#visualization-1--heatmap-mean-metric-values-per-model)
6. [Visualization 2 — Violin Plot: Expansion Ratio Distribution](#visualization-2--violin-plot-expansion-ratio-distribution)
7. [Visualization 3 — Bubble Chart: Source vs Target Token Count](#visualization-3--bubble-chart-source-vs-target-token-count)
8. [Visualization 4 — Radar Chart: Multi-Metric Profile per Model](#visualization-4--radar-chart-multi-metric-profile-per-model)
9. [Cross-Metric Synthesis](#cross-metric-synthesis)
10. [Connection to Part A Results](#connection-to-part-a-results)
11. [Key Findings Summary](#key-findings-summary)

---

## What Part B Does

Part A translated 100 English sentences into Tamil using five models and scored the translations with sacreBLEU. Part B takes those same sentences and translations and looks *inside the tokenizer* for each model.

The central question is: **how efficiently does each model's tokenizer represent Tamil text?**

We load each model's tokenizer (without loading the full model weights, to save memory and time), tokenize both the English source sentences and the Tamil target translations, then compute four numeric metrics per sentence per model. Those 100 × 5 = 500 data rows are saved to `token_counts.csv`, and an enriched version with engineered features is saved to `engineered_features.csv`.

---

## Why Tokenization Matters

All five models in this project are Transformer-based. Transformers compute attention between every pair of tokens in the input — a computation that scales as **O(n²)** in sequence length. This means:

- A sentence tokenized into 10 tokens requires 100 attention computations.
- The same sentence tokenized into 20 tokens requires 400 computations — 4× more work.

This is not a hypothetical concern. At inference time, your GPU memory and compute budget are finite. Models that fragment Tamil text into many tiny subwords will be slower and more memory-hungry than models that use coarser, more information-dense tokens.

Beyond speed, tokenization quality directly affects translation quality. A tokenizer that maps a large fraction of Tamil characters to the `<unk>` (unknown) token is essentially telling the model "I don't know what this is" — which means the model cannot learn meaningful representations for those sounds, morphemes, or words.

---

## The Four Core Metrics

All four metrics are computed per sentence. The heatmap shows their **mean across all 100 sentences**.

### 1. `expansion_ratio`

```
expansion_ratio = (number of target tokens) / (number of source tokens)
```

This measures how much longer the Tamil tokenized sequence is compared to the English one. A ratio of 1.0 means Tamil and English tokenize to the same length. A ratio of 5.0 means the Tamil translation produces 5× more tokens than the English source.

**Why it matters:** High expansion directly increases inference cost for any model that re-encodes or re-scores the target side. It also signals that the tokenizer uses many small subword pieces to represent Tamil text.

### 2. `avg_word_length`

```
avg_word_length = (number of Tamil unicode characters in the translation) 
                  / (number of target tokens)
```

This is the average number of Tamil characters packed into each token. A value of 3.92 means that, on average, each token represents about 4 Tamil characters. A value of 0.90 means most tokens represent less than one full character — which happens when the SentencePiece model fragments text at or below the character level.

**Why it matters:** Higher values mean each token carries more linguistic content. Lower values indicate extreme fragmentation.

### 3. `subword_fragmentation`

```
subword_fragmentation = 1 / avg_word_length
```

This is the mathematical inverse of `avg_word_length`. It was computed as a separate metric because it's more intuitive in certain comparisons: a higher fragmentation score means worse (more fragmented) tokenization.

**Why it matters:** This metric makes the ranking order flip compared to `avg_word_length`, which is sometimes easier to read in visualizations where higher = worse.

### 4. `unknown_token_rate`

```
unknown_token_rate = (number of UNK tokens) / (total target tokens) × 100
```

The `<unk>` token is the tokenizer's way of saying "I cannot represent this piece of text." If a tokenizer does not have a Tamil character, akshara, or subword in its vocabulary, it emits UNK. A rate of 0.0 means the tokenizer handles all Tamil text it encounters. A rate of 33.62 means one out of every three Tamil tokens is unknown.

**Why it matters:** UNK tokens are dead weight. Any Tamil subword mapped to UNK contributes nothing to the model's understanding of that word. High UNK rates often indicate that the tokenizer was primarily trained on non-Tamil data.

---

## Feature Engineering

In addition to the four raw metrics, Part B derives three engineered features:

| Feature | Formula | Purpose |
|---|---|---|
| `log_expansion` | `log(expansion_ratio)` | Compress the long right tail of expansion_ratio for visualization |
| `efficiency_score` | `avg_word_length / expansion_ratio` | Combined measure: high score = tokens are long AND there are few of them |
| `fragmentation_class` | Categorical bucketing of subword_fragmentation | Groups sentences into Low / Medium / High / Very High fragmentation for distribution analysis |

The `efficiency_score` is worth understanding. It rewards models that simultaneously:
- Keep tokens information-dense (`avg_word_length` in the numerator), AND
- Keep sequences short (`expansion_ratio` in the denominator).

A model that fragments Tamil into many tiny tokens loses on both axes: `avg_word_length` drops and `expansion_ratio` rises, so `efficiency_score` falls steeply.

---

## Visualization 1 — Heatmap: Mean Metric Values per Model

**File:** `plots/partb_heatmap.png`

The heatmap displays the mean value of each metric across all 100 sentences for each of the five models. Cells are color-coded independently per column so that relative rankings are immediately visible.

### Exact values from the heatmap

| Model | expansion_ratio | avg_word_length | subword_fragmentation | unknown_token_rate |
|---|---|---|---|---|
| Helsinki | 0.99 | 3.62 | 0.31 | **33.62** |
| IndicTrans2 | **5.15** | 0.90 | **1.11** | 0.00 |
| MADLAD | 1.61 | 2.74 | 0.37 | 0.00 |
| NLLB-200 | 1.38 | 3.15 | 0.32 | 0.00 |
| mT5 | 2.13 | **3.92** | 0.26 | 0.00 |

### Reading the heatmap row by row

**Helsinki (opus-mt-en-mul-taml):**  
The most deceptive row in the table. An `expansion_ratio` of 0.99 looks ideal — it means Tamil translations tokenize to almost the same length as English sources. But this is an illusion. The `unknown_token_rate` of 33.62 reveals that Helsinki's tokenizer cannot handle a third of all Tamil tokens. When a tokenizer produces UNK, those tokens are short (usually counted as 1 token each) and contribute no characters to the decoded output. The result is artificially compressed sequences. Helsinki appears "efficient" in token count only because it is silently discarding Tamil text.

**IndicTrans2 (indictrans2-en-indic-1B):**  
The most striking row. `expansion_ratio` of 5.15 is by far the highest — Tamil translations are five times as long in tokens as the English source. `avg_word_length` of 0.90 means each token represents less than one Tamil character on average, and `subword_fragmentation` at 1.11 is the highest among all models. This sounds like a failure, but Part A showed IndicTrans2 achieving the best BLEU score. The explanation is that IndicTrans2 uses a SentencePiece model trained specifically on Indic languages, with a fine-grained Tamil vocabulary. It fragments Tamil into morpheme-aligned subword units rather than character n-grams, enabling the model to learn rich Tamil morphology. The `unknown_token_rate` of 0.00 confirms that every Tamil character is recognized — zero UNKs across all 100 sentences.

**MADLAD-400 (madlad400-3b-mt):**  
A middle-ground model. `expansion_ratio` of 1.61 is moderate, and `avg_word_length` of 2.74 indicates tokens contain roughly 2–3 Tamil characters each. `unknown_token_rate` of 0.00 confirms full Tamil coverage. This model balances sequence length against token granularity reasonably well.

**NLLB-200 (nllb-200-distilled-600M):**  
Very similar to MADLAD in profile. `expansion_ratio` of 1.38 is the second-lowest (after Helsinki's artificial 0.99), `avg_word_length` of 3.15 is healthy, and `unknown_token_rate` of 0.00. NLLB-200's SentencePiece vocabulary covers Tamil well and leans toward coarser tokens, meaning more characters per token. This translates to more compact sequences.

**mT5 (mT5-base):**  
The surprise entry. `avg_word_length` of 3.92 is the highest of all five models — mT5 tokens are the most information-dense on average. `subword_fragmentation` of 0.26 is the lowest, confirming the least fragmented tokenization. `expansion_ratio` of 2.13 is moderate. mT5 was not included in translation scoring (it is a text-to-text model, not a dedicated MT system), but its tokenizer — trained on a 101-language mC4 corpus — handles Tamil with high vocabulary coverage and coarse, character-rich subword units. This makes it a useful upper-bound reference for tokenization quality.

### What the color coding tells you

Each column is normalized independently. The darkest cells in `expansion_ratio` and `subword_fragmentation` mark the highest (worst) values; the darkest cells in `avg_word_length` mark the highest (best) values. This cross-column normalization means you cannot compare cell shades across columns, only within them.

---

## Visualization 2 — Violin Plot: Expansion Ratio Distribution

**File:** `plots/partb_violin_expansion.png`

The violin plot shows the full distribution of `expansion_ratio` across all 100 sentences for each model. A box-and-whisker plot would only show the median and quartiles; the violin plot also reveals the shape of the distribution — whether it is symmetric, skewed, bimodal, or has long tails.

### Model-by-model distribution descriptions

**IndicTrans2:**  
The widest and tallest violin by a large margin. The body of the distribution spans roughly 4.5 to 7.3, with a median around 5.3. The tails extend from approximately 2.3 to 8.0. This means that even in the "best case" sentences, IndicTrans2 still tokenizes Tamil to more than twice the English token count. The distribution is relatively symmetric with a slight upper skew — longer English sentences produce proportionally more Tamil tokens.

**NLLB-200:**  
The most compact violin in the plot. Median is approximately 1.3, the body is tight from roughly 0.9 to 1.8, and the range extends from about 0.7 to 2.3. This is the model that most consistently keeps Tamil token counts close to English token counts (setting aside Helsinki's artificial result). The tightness of the distribution means sentence length has little effect on the ratio — NLLB-200 handles short and long sentences with similar tokenization density.

**mT5:**  
Moderate median around 1.9, but with a notably elongated upper tail reaching 7.3. This indicates that for certain sentence types — likely those with complex Tamil verb forms or long compound nouns — mT5's tokenizer expands dramatically. The lower body of the distribution (most sentences) is compact, but the tail warns that mT5's tokenization is less consistent than NLLB-200's.

**Helsinki:**  
Low median around 1.1, with a range from approximately 0.0 to 2.1, including values below 1.0. Ratios below 1.0 mean the Tamil output has *fewer* tokens than the English source — which is only possible when significant content is mapped to UNK (a single unknown token replacing multiple characters). The presence of sub-1.0 values is a strong signal of vocabulary failure, not genuine efficiency.

**MADLAD:**  
A narrow, tall violin with median around 1.5 and tight range from roughly 1.2 to 2.6. Very similar shape to NLLB-200 but slightly higher and slightly wider. The consistency is similar — MADLAD's tokenization ratio does not vary much across sentence lengths or complexity levels.

### What to take away from the violin plot

The violin plot adds two insights not visible in the heatmap:

1. **Consistency**: NLLB-200 and MADLAD are the most consistent models (narrow violins). IndicTrans2 and mT5 have wide distributions, meaning their tokenization efficiency varies considerably across sentences.

2. **Outliers and tails**: mT5's long upper tail indicates that specific sentence types trigger extreme fragmentation even when the average case is moderate. This matters in production: average-case metrics can obscure worst-case behavior.

---

## Visualization 3 — Bubble Chart: Source vs Target Token Count

**File:** `plots/partb_bubble_chart.png`

The bubble chart plots each sentence as a point with:
- **X-axis:** number of source (English) tokens
- **Y-axis:** number of target (Tamil) tokens
- **Bubble size:** `subword_fragmentation` score (larger = more fragmented)
- **Color:** model identity

A diagonal line (y = x) is drawn for reference. Points on the diagonal mean source and target token counts are equal. Points above the diagonal mean Tamil has more tokens than English.

### Pattern analysis

**IndicTrans2:**  
All 100 bubbles cluster far above the y = x diagonal, in a band from roughly y = 30 to y = 80 for typical English sentences of 10–20 tokens. The bubbles are large, confirming high fragmentation scores. The relationship is approximately linear — longer English sentences produce proportionally longer Tamil token sequences. This consistent upward displacement is the geometric representation of the 5.15 mean expansion ratio.

**NLLB-200, MADLAD, Helsinki:**  
These three models cluster near the diagonal. Their points are tightly packed in the lower-left of the chart (since short-to-medium English sentences stay short in Tamil too), with small bubble sizes. This visual clustering makes it hard to distinguish the three models from each other in the scatter portion of the chart.

**mT5:**  
A moderate scatter above the diagonal, with some notably larger bubbles for the longer English sentences. This is the bubble chart representation of mT5's elongated upper tail from the violin plot — a subset of sentences drives disproportionately high Tamil token counts.

### What the bubble size adds

Bubble size encodes `subword_fragmentation` independently of the position. A point could be above the diagonal (high expansion) but have a small bubble (low per-token fragmentation) if the expansion is due to many coarse tokens. IndicTrans2 scores high on both: it is far above the diagonal (expansion) and has large bubbles (fragmentation), confirming the heatmap values.

---

## Visualization 4 — Radar Chart: Multi-Metric Profile per Model

**File:** `plots/partb_radar_chart.png`

The radar chart (also called a spider chart) places each of the four metrics on its own axis radiating from a central point. Each model is drawn as a polygon connecting its scores on each axis. Models that score well on all metrics would be represented by a large, regular polygon.

### How to interpret the radar chart

Because the four metrics have different scales, they are normalized before plotting. The key is not the absolute radius on each axis but the shape of the polygon and how the models compare relative to each other on each axis.

**IndicTrans2 (blue polygon):**  
Pulls far outward on the `expansion_ratio` axis (meaning the highest expansion — this is detrimental). It also pulls outward on `subword_fragmentation`. However, on the `unknown_token_rate` axis, IndicTrans2 pulls *inward toward the center* — which, if the axis is scaled so that higher is worse, means IndicTrans2 has the best (0.00) UNK rate.

**Helsinki (orange/red polygon):**  
The inverse of IndicTrans2 on the `unknown_token_rate` axis. Helsinki pulls outward (worst) on this axis, representing the 33.62% UNK rate. On `expansion_ratio` it pulls inward (artificially good), but we now understand this is because unknown tokens suppress the count.

**mT5, NLLB-200, MADLAD:**  
These three form a cluster in the middle of the radar, with moderate and similar shapes. mT5 differentiates itself by pulling slightly further outward on `avg_word_length` (best token density) and inward on `subword_fragmentation` (least fragmented).

### What the radar chart adds

The radar chart is the only visualization that shows all four metrics simultaneously for all models in a single image. Its value is gestalt — you can see the "shape" of each model's tokenization profile at a glance:

- **IndicTrans2**: high expansion, high fragmentation, zero UNK → aggressive but complete Tamil coverage
- **Helsinki**: low expansion, moderate fragmentation, catastrophic UNK → partial Tamil coverage with gaps
- **mT5**: moderate expansion, lowest fragmentation, zero UNK → dense, complete Tamil coverage
- **NLLB-200 / MADLAD**: similar moderate profiles, complete coverage, balanced metrics

---

## Cross-Metric Synthesis

Reading across all four visualizations together, two counter-intuitive insights emerge:

### Insight 1: High expansion ratio is not a failure for IndicTrans2

A naive reading of the heatmap would rank IndicTrans2 as the worst tokenizer (highest expansion, highest fragmentation). But:

1. Its `unknown_token_rate` is 0.00 — perfect coverage
2. Part A showed it achieving the highest BLEU score among the four MT models
3. Its fine-grained SentencePiece vocabulary is *designed* to capture Tamil morphology at the subword level

Tamil is a highly agglutinative language. A single Tamil word like *பேசிக்கொண்டிருந்தார்கள்* encodes tense, aspect, mood, number, and person in one orthographic unit that might require 4–6 English words to express. To represent all these morphological distinctions, the tokenizer must segment this word into multiple subwords. High fragmentation is the price of morphological expressiveness.

The comparison to Helsinki makes this concrete: Helsinki's 0.99 expansion ratio looks better, but it achieves this by failing to encode 33.62% of Tamil content. IndicTrans2's 5.15 expansion ratio reflects the actual complexity of Tamil morphology being fully encoded.

### Insight 2: mT5 has the best tokenization quality among all five models

mT5 was excluded from translation scoring because it is not a dedicated MT model. But its tokenizer — trained on a 101-language mC4 corpus — achieves:
- Highest `avg_word_length` (3.92): most information-dense tokens
- Lowest `subword_fragmentation` (0.26): least fragmented
- Zero `unknown_token_rate`: complete coverage
- Moderate `expansion_ratio` (2.13): manageable sequence length growth

If you were designing a new Tamil NLP system from scratch, mT5's tokenizer would be a strong candidate for tokenization quality. The trade-off is that mT5's pre-training objectives are different from translation, so fine-tuning for MT requires more data and compute than starting from NLLB-200 or MADLAD.

---

## Connection to Part A Results

Relating the tokenization metrics back to the BLEU scores from Part A:

| Model | BLEU (Part A) | expansion_ratio | unknown_token_rate | Comment |
|---|---|---|---|---|
| Helsinki | Low | 0.99 | 33.62 | Poor BLEU explained by tokenizer gaps |
| IndicTrans2 | Highest | 5.15 | 0.00 | Best translation despite high expansion |
| MADLAD | Moderate | 1.61 | 0.00 | Balanced metrics, decent BLEU |
| NLLB-200 | Moderate-good | 1.38 | 0.00 | Compact tokenization, strong BLEU |
| mT5 | Not scored | 2.13 | 0.00 | Best tokenizer, non-MT model |

The pattern is clear: **zero unknown_token_rate is a necessary condition for good translation quality.** Helsinki's 33.62% UNK rate correlates directly with its low BLEU — the model cannot translate what it cannot tokenize. Among models with zero UNK, translation quality depends on model architecture, training data, and fine-tuning strategy (captured by BLEU in Part A) rather than tokenization efficiency alone.

---

## Key Findings Summary

1. **Helsinki has a tokenization failure:** 33.62% unknown_token_rate — one in three Tamil tokens is unknown. Its apparent efficiency (expansion_ratio 0.99) is an artifact of these UNK tokens, not genuine Tamil language coverage.

2. **IndicTrans2 uses aggressive but complete tokenization:** expansion_ratio 5.15 and subword_fragmentation 1.11 are the highest values, but unknown_token_rate 0.00 confirms every Tamil character is recognized. High fragmentation is the cost of fine-grained morphological coverage, which is what enables its top BLEU score in Part A.

3. **mT5 has the best Tamil tokenizer in this comparison:** avg_word_length 3.92 (most information-dense), subword_fragmentation 0.26 (lowest), unknown_token_rate 0.00, and moderate expansion_ratio 2.13. It is not a practical MT system, but its tokenizer represents a quality ceiling.

4. **NLLB-200 and MADLAD are the most practical balance:** Both have zero UNK rates, moderate expansion ratios (1.38 and 1.61), and healthy avg_word_length values (3.15 and 2.74). They achieve usable translation quality without the extreme sequence length cost of IndicTrans2.

5. **The violin plot reveals consistency risk in mT5:** Despite the best mean metrics, mT5's expansion_ratio distribution has a long upper tail reaching 7.3 — certain sentence types trigger extreme fragmentation. NLLB-200 and MADLAD have the tightest, most predictable distributions.

6. **Expansion ratio alone is misleading without UNK rate:** Any tokenizer can achieve a low expansion ratio by mapping unknown text to a single UNK token. Always read `expansion_ratio` and `unknown_token_rate` together.

---

*Report generated from visualizations in `plots/`, metrics computed in `part_b_token_eda.ipynb`.*  
*Downstream analysis in Part C uses `engineered_features.csv` produced by this notebook.*
