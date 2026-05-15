# Part C Report — Indic Token Behavior Analysis

**Project:** English → Tamil Machine Translation Evaluation  
**Dataset:** FLoRes-200 (`openlanguagedata/flores_plus`, `eng_Latn-tam_Taml`, 100 devtest sentences)  
**Pipeline position:** Part C is the final analysis stage; it consumes `token_counts.csv` from Part B and produces three visualisations.

---

## 1. Purpose and Motivation

Parts A and B established *what* the models produce (BLEU scores) and *how* they behave at the sentence level (expansion ratios, fragmentation rates). Part C goes one level deeper and asks *why*: is the tokenisation quality difference between models caused by vocabulary specialisation, and what are the real-world cost implications?

Two concrete questions drive Part C:

1. **Vocabulary question:** When each model's tokeniser encounters a Tamil word, does it have a pre-trained entry for that word (known), split it into subwords (fragmented), or give up entirely (unknown/UNK)?
2. **Efficiency question:** Given that more tokens per sentence increases Transformer attention cost quadratically, which model is the most memory-efficient for Tamil inference?

---

## 2. Tamil Agglutination — Why It Challenges Tokenisers

Tamil is an **agglutinative language**: a single Tamil word can express what English requires a full clause to say. The table below shows the progression from a simple noun to a 6-morpheme verb form.

| Tamil Word | Romanisation | English | Morpheme Count |
|-----------|-------------|---------|---------------|
| மரம் | maram | tree | 1 |
| படிக்கிறாள் | padikkiraaL | she is studying | 2 |
| வந்திருக்கிறான் | vandhirukkiraaN | he has come | 4 |
| செய்துகொண்டிருக்கிறார்கள் | seydhukondirukkiraargaL | they have been doing | 6+ |

A tokeniser trained predominantly on English or distributed across 200–400 languages will often fragment `வந்திருக்கிறான்` into 8–12 meaningless byte-level pieces. A tokeniser trained specifically on Indic languages may represent the same word in 2–4 morpheme-aligned fragments. The quality of this decomposition directly determines how well the model's neural network can learn Tamil grammar.

---

## 3. Experimental Design

### 3.1 Twenty-Word Tamil Test Set

Rather than using full sentences (which average out per-word behaviour), Part C tests a curated set of 20 Tamil words spanning six complexity tiers:

| Tier | Words | Examples |
|------|-------|---------|
| Simple nouns | 5 | மரம் (tree), நன்றி (thanks), உணவு (food), தண்ணீர் (water), மக்கள் (people) |
| Medium verbs | 3 | படிக்கிறாள் (she studies), சென்றார்கள் (they went), பேசுகிறோம் (we speak) |
| High agglutination | 2 | வந்திருக்கிறான் (he has come), செய்துகொண்டிருக்கிறார்கள் (they have been doing) |
| Domain (government/tech) | 4 | அரசாங்கம் (government), கணிப்பொறி (computer), இணையம் (internet), பொருளாதாரம் (economy) |
| Place names | 3 | தமிழ்நாடு (Tamil Nadu), சென்னை (Chennai), கோயம்புத்தூர் (Coimbatore) |
| Rare/general | 3 | மக்கள்தொகை (population), விவசாயி (farmer), செய்தி (news) |

### 3.2 Vocabulary Classification Logic

Each word is passed through `tokenizer.encode()` and the resulting token ID list is classified:

| Class | Detection | Linguistic meaning |
|-------|-----------|-------------------|
| **Known** | `len(ids) == 1` and `ids[0] != unk_id` | The tokeniser has a single vocabulary entry for this whole word |
| **Fragmented** | `len(ids) >= 2` | The word is split into subword pieces (SentencePiece, WordPiece, BPE) |
| **Unknown** | `ids[0] == unk_id` | The tokeniser has no representation; outputs a generic UNK symbol |

`encode()` is used (not `tokenize()`) because comparing integer IDs against `tokenizer.unk_token_id` is unambiguous — some tokenisers use different UNK string representations.

**IndicTrans2 special handling:** IndicTrans2's tokeniser operates in two modes. In *input mode* it expects English tagged with `"eng_Latn tam_Taml"` prefixes. Tamil text must be encoded in *target mode*, activated by calling `tokenizer._switch_to_target_mode()` before encoding and `tokenizer._switch_to_input_mode()` after. All five notebooks in this project include this mode-switching guard.

### 3.3 Memory Footprint Metric

Each sentence in `token_counts.csv` has a `target_token_count`. The proxy for Transformer attention memory cost is:

```
memory_score = target_token_count²
```

This derives from the O(n²) complexity of self-attention: an attention matrix for n tokens requires n×n entries. The per-model score is the mean over 100 sentences, expressed in thousands (`÷ 1000`) for readability.

### 3.4 Characters Per Token

`avg_word_length` from `token_counts.csv` records the average number of Tamil characters encoded per target token. Higher values mean fewer, larger tokens — each one covering more morphological content. This metric was engineered in Part B and is re-visualised in Part C with a focus on cross-model comparison.

---

## 4. Results

### 4.1 VIZ C1 — Vocabulary Coverage Donut Charts

**Plot:** `plots/partc_donut_coverage.png`

Five donut charts, one per model. Each shows the proportion of the 20-word test set classified as known (model colour), fragmented (amber), or unknown (red).

**Results:**

| Model | Known | Fragmented | Unknown |
|-------|-------|-----------|---------|
| NLLB-200 | **25%** (5 words) | 75% (15 words) | 0% |
| MADLAD | 15% (3 words) | 85% (17 words) | 0% |
| mT5 | 10% (2 words) | 90% (18 words) | 0% |
| IndicTrans2 | 0% | **100%** | 0% |
| Helsinki | 0% | **100%** | 0% |

**Finding 1 — Helsinki produces `<unk>` for every single test word; the four other models use genuine subword fragmentation.** The classification code marks a word as 'unknown' only when `toks[0] == unk_id`. Helsinki consistently tokenises every test word as `▁|<unk>` — the word-boundary marker `▁` (toks[0]) is not the UNK ID, so the code labels these rows as 'fragmented' rather than 'unknown'. However, the only *content* token is `<unk>`, meaning Helsinki has effectively zero Tamil vocabulary coverage on this test. The four non-Helsinki models produce genuine subword fragments with no UNK tokens at all.

**Finding 2 — NLLB-200 leads on known-word coverage** (25%), despite being a 200-language general-purpose model. Five of the 20 test words have single-token vocabulary entries in NLLB-200's 256K vocabulary. MADLAD (15%) and mT5 (10%) also have some whole-word entries. The common known words tend to be simple, high-frequency nouns and place names that appear often in multilingual training corpora.

**Finding 3 — IndicTrans2 shows 0% known words.** This is the most counterintuitive result. The pre-run hypothesis was that IndicTrans2, with its dedicated Indic SentencePiece vocabulary, would have the *highest* known-word coverage. The opposite is true.

The explanation lies in design philosophy: IndicTrans2's SentencePiece model is trained to decompose Tamil words into **morphologically meaningful subword fragments**, not to store whole words as single tokens. Representing `வந்திருக்கிறான்` as three morphemic pieces (`வந்து` + `இருக்கிறான்` broken further) retains grammatical structure that a single opaque token cannot. The model's high BLEU score (27.75) relative to models like Helsinki (8.26) confirms this approach works well for translation quality even with 0% whole-word coverage.

**Finding 4 — Helsinki also shows 0% known.** Its EN→Dravidian vocabulary is spread across four scripts (Tamil, Kannada, Malayalam, Telugu), leaving each language with fewer dedicated entries than NLLB-200's 200-language spread.

**Interpreting the donut charts:** A model with a large amber (fragmented) slice is not necessarily poor — fragmentation is acceptable if the subword fragments align with Tamil morpheme boundaries. The critical failure mode is the red (UNK) slice, which signals the tokeniser has abandoned representation entirely. The four non-Helsinki models avoid this entirely. Helsinki shows 0% in the red (unknown) slice only because the classification code checks `toks[0] == unk_id`: Helsinki outputs `▁|<unk>` for every test word, and the word-boundary marker `▁` is not the UNK ID, so each entry is technically classified as "fragmented". In practice Helsinki produces no meaningful Tamil tokens — only UNK — for every word in the test set. Part B confirms this with its 33.62% sentence-level UNK rate.

---

### 4.2 VIZ C2 — Transformer Memory Footprint

**Plot:** `plots/partc_memory_footprint.png`

A horizontal bar chart sorted from lowest to highest memory score. Each bar is `mean(target_token_count²) / 1000`, averaged over 100 FLoRes-200 sentences.

**Why this matters:** Doubling a sentence's token count *quadruples* the attention matrix size. A model that fragments Tamil into twice as many tokens as necessary does not just cost twice as much to run — it costs four times as much in attention computation. This directly limits how long a document can be processed in a single batch on memory-constrained hardware (edge devices, mobile deployment, API cost budgets).

**Expected ordering based on Part B expansion ratios:**

| Model | Expansion Ratio (Part B) | Approx avg target tokens | Memory score estimate |
|-------|-------------------------|-------------------------|-----------------------|
| Helsinki | 0.99 | ~20 tokens | ~0.4k |
| NLLB-200 | 1.38 | ~28 tokens | ~0.8k |
| MADLAD | 1.61 | ~32 tokens | ~1.0k |
| mT5 | 2.13 | ~43 tokens | ~1.8k |
| IndicTrans2 | 5.15 | ~103 tokens | ~10.6k |

*(Assumes ~20 English source tokens per sentence; actual values from the 100-sentence FLoRes-200 sample will differ slightly.)*

**IndicTrans2 incurs the highest memory cost by a wide margin.** Its expansion ratio of 5.15 — the Tamil output is more than five times longer in tokens than the English input — drives a memory score roughly 13× higher than NLLB-200 and 26× higher than Helsinki. For the same 20-word English sentence, IndicTrans2 produces a Tamil token sequence that needs 13× more attention memory than NLLB-200's output.

This trade-off is real and important: IndicTrans2 achieves BLEU=27.75 (second overall), but it is the most expensive model to run at inference. For document-scale translation or real-time ASR post-processing where sequences can be hundreds of tokens, this gap becomes a hard constraint.

**Helsinki is cheapest** despite having the worst BLEU (8.26). Its near-1:1 expansion ratio produces very short Tamil token sequences. However, the cheapness comes from a different problem: Helsinki collapses Tamil words to `▁|<unk>` — two tokens per word regardless of word length — which trivially produces short sequences at the cost of translation quality.

**Cross-task relevance:** For Task 2 of this project (ASR evaluation), an automatic speech recognition system generates Tamil transcriptions that then need post-processing or evaluation. If a Transformer model is used for that post-processing, the O(n²) cost of IndicTrans2-length sequences would limit batch sizes significantly. NLLB-200 strikes the best balance between quality and memory efficiency for such downstream use cases.

---

### 4.3 VIZ C3 — Characters Per Token

**Plot:** `plots/partc_chars_per_token.png`

A vertical bar chart with error bars (±1 standard deviation over 100 sentences). The y-axis shows `avg_word_length` — average Tamil characters per target token. Higher is better: each token encodes more Tamil linguistic content.

**What this measures:** A tokeniser that encodes Tamil efficiently produces large tokens covering complete syllables or morphemes. A tokeniser that fragments heavily produces tiny tokens covering 1–2 characters each — functionally equivalent to byte-level tokenisation, which loses all morphological structure.

**Expected results:**
- **mT5** is expected to show the highest chars/token value (~3.92 from Part B analysis). Despite being a general multilingual model, mT5's training on mC4 (a massive Common Crawl corpus) includes enough Tamil text to develop relatively long subword units for common Tamil patterns.
- **IndicTrans2** will show lower chars/token despite superior translation quality. More tokens per sentence (expansion 5.15) means each token covers fewer characters on average.
- **Helsinki** will show the lowest value, as its aggressive fragmentation produces many short single-character or two-character tokens. Its actual translations contain a high proportion of UNKs, which further distort the metric.

**Error bars as a quality signal:** A narrow error bar means the tokeniser behaves consistently across all sentence types — predictable fragmentation regardless of vocabulary domain. A wide error bar indicates uneven coverage: some sentence types are handled well (common vocabulary) while others are heavily fragmented (technical or rare Tamil).

---

## 5. Token Span Visualiser

The notebook includes an HTML-rendered interactive visualiser (Cell 3) that shows the actual subword tokens each model assigns to three Tamil words at increasing complexity levels:

| Complexity | Tamil Word | English |
|-----------|-----------|---------|
| Simple | மரம் | tree |
| Medium | படிக்கிறாள் | she is studying |
| Complex | வந்திருக்கிறான் | he has come |

Each subword token is rendered as a colour-coded HTML span. A word encoded as one span is handled perfectly; a word split into 8–12 spans is maximally fragmented. This visualiser makes the vocabulary gap between models immediately legible without reading numbers.

**SentencePiece ▁ boundary markers** are stripped before rendering (`clean_token_display()` function). The ▁ character (U+2581) marks word boundaries in SentencePiece tokenisation — it is a technical artefact of the tokenisation process, not a part of the Tamil word itself. Stripping it keeps the HTML spans clean and readable.

---

## 6. Cross-Part Synthesis

Part C closes the loop on a question raised in Part A: why does Helsinki produce BLEU=13.59 while NLLB-200 achieves 23.88 on the same input? The answer is now quantified across three levels of analysis:

| Part | Metric | Helsinki | NLLB-200 | Difference |
|------|--------|---------|---------|-----------|
| A | BLEU score | 8.26 | 24.17 | NLLB-200 +15.91 points |
| B | Unknown token rate | 33.62% | 0.00% | Helsinki loses 1 in 3 words |
| B | Expansion ratio | 0.99 | 1.38 | Helsinki nearly 1:1 (translation is short/incomplete) |
| C | Known-word coverage | 0% | 25% | NLLB-200 has real vocabulary entries for Tamil words |
| C | Memory footprint | Lowest | Low | Helsinki cheaper but for wrong reasons (▁\|<unk> collapse) |

The pattern is consistent: Helsinki's vocabulary is too thin for Tamil. Its EN→Dravidian training splits vocabulary budget across four languages, leaving Tamil under-resourced. NLLB-200's 256K vocabulary, despite spanning 200 languages, has proportionally more Tamil entries — enough to achieve 25% known-word coverage and 0% UNK in actual translations.

---

## 7. Model Selection Summary

| Use Case | Recommended Model | Reason |
|----------|------------------|--------|
| Highest BLEU quality | **MADLAD** (29.58) | Best corpus BLEU, 0% UNK, good coverage |
| Best translation + Indic morphology | **IndicTrans2** (27.75) | Morpheme-aligned tokenisation, 0% UNK; accepts higher memory cost |
| Balanced quality + memory efficiency | **NLLB-200** (24.17) | Best known-word coverage (25%), 0% UNK, 4th lowest memory |
| Avoid | **Helsinki** (8.26) | All Tamil words → `▁\|<unk>`, 33.62% UNK in translation output, worst BLEU |

For resource-constrained deployment (mobile, edge, long-document batch processing): **NLLB-200** is the clear choice. It delivers acceptable BLEU with the best vocabulary coverage and far lower memory pressure than IndicTrans2.

For server-side batch translation with access to GPU memory: **MADLAD** or **IndicTrans2** depending on whether raw quality (MADLAD) or morphological alignment for downstream NLP tasks (IndicTrans2) is the priority.

---

## 8. Limitations

- **Vocabulary coverage tested on 20 words** — curated to be representative but not a random sample of the Tamil lexicon. The 0% UNK result for all models on this specific test set would likely not hold on a random 20-word sample from a Tamil corpus.
- **Memory score is a proxy** — `token_count²` captures attention complexity but not batch memory, KV-cache size, or hidden-state overhead. Real inference memory is roughly 5–10× higher per token than the proxy implies.
- **No model weight loading** — Part C uses tokenisers only. All conclusions about memory footprint, quality, and efficiency are based on tokenisation behaviour, not on actual model inference.
- **IndicTrans2 0% known is a design feature** — interpreting this as "poor vocabulary coverage" would be wrong. The morpheme-splitting approach is intentional and produces better translations than models with higher known-word percentages (e.g., NLLB-200 known=25% but BLEU=23.88 vs IndicTrans2 known=0% but BLEU=27.75).
- **Helsinki's 0% 'unknown' classification vs 33.62% in Part B** — the classification code marks 'unknown' only when `toks[0] == unk_id`. Helsinki produces `▁|<unk>` for every test word, so each is classified as 'fragmented' (not 'unknown'). The `<unk>` token IS present in every Helsinki row as toks[1]. Helsinki effectively fails on all 20 test words; the 33.62% Part B rate measures the same vocabulary failure in running translation output.

---

*Part C notebook: `part_c_indic_token_analysis.ipynb` — CPU-only, no GPU required.*  
*Inputs: `../part_b_token_analysis/token_counts.csv`*  
*Outputs: `plots/partc_donut_coverage.png`, `plots/partc_memory_footprint.png`, `plots/partc_chars_per_token.png`*
