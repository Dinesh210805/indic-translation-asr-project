# Observations — Part B · Token-Level EDA

## Dataset
- **Source**: `../part_a_batch_translation/sacrebleu_results.csv` (100 sentence pairs)
- **Models tokenized**: IndicTrans2, NLLB-200, Helsinki, MADLAD-400, mT5-base
- **Metrics**: expansion_ratio, avg_word_length (chars/token), subword_fragmentation, unknown_token_rate

## Metric Summary (mean across 100 sentences)

| Model | expansion_ratio | avg_word_length | subword_fragmentation | unknown_token_rate |
|---|---|---|---|---|
| Helsinki | 0.99 | 3.62 | 0.31 | **33.62** |
| IndicTrans2 | **5.15** | 0.90 | **1.11** | 0.00 |
| MADLAD | 1.61 | 2.74 | 0.37 | 0.00 |
| NLLB-200 | 1.38 | 3.15 | 0.32 | 0.00 |
| mT5 | 2.13 | **3.92** | **0.26** | 0.00 |

## Key Findings

### Expansion Ratio
- **Actual ordering**: IndicTrans2 (5.15) >> mT5 (2.13) > MADLAD (1.61) > NLLB-200 (1.38) > Helsinki (0.99*)
- IndicTrans2 has the highest expansion ratio by a large margin — Tamil sequences are 5× longer than English sources
- Helsinki's apparent "efficiency" (0.99) is an artifact of its 33.62% unknown token rate suppressing sequence length
- NLLB-200 and MADLAD are the genuine low-expansion models with complete Tamil coverage

### Subword Fragmentation
- IndicTrans2 has the highest fragmentation (1.11) because its SentencePiece model segments Tamil at a fine morpheme level
- mT5 has the lowest fragmentation (0.26) — its 250K vocabulary produces the most information-dense tokens
- Helsinki's apparent moderate fragmentation (0.31) is again distorted by UNK tokens masking character-level gaps

### Unknown Token Rate
- **Only Helsinki has non-zero UNK rate**: 33.62% — one in three Tamil tokens is unknown to its tokenizer
- IndicTrans2, NLLB-200, MADLAD, mT5 all achieve 0.00 unknown_token_rate — complete Tamil script coverage
- This is the single most important metric for translation suitability: zero UNK is a necessary condition for good MT

## Surprising Results

**IndicTrans2 is the "worst" tokenizer by expansion metrics but the best MT model.** Its 5.15 expansion ratio and 1.11 fragmentation score are the highest in the group, yet Part A confirms it produces the highest BLEU translations. The high fragmentation reflects fine-grained morpheme-aligned subword units — expensive in token count, but rich in linguistic information. Tamil agglutination means a single word like *பேசிக்கொண்டிருந்தார்கள்* packs tense, aspect, mood, number, and person into one orthographic form. Capturing all of that requires many subword tokens.

**mT5 has the best tokenization quality overall despite not being an MT model.** avg_word_length 3.92 (highest), subword_fragmentation 0.26 (lowest), zero UNK, moderate expansion. A 101-language mC4 corpus produces a richer Tamil vocabulary than some dedicated MT systems. mT5's tokenizer would be a strong candidate for any Tamil NLP task even though its translation outputs were not scored.

**Helsinki's low expansion ratio is a false positive.** Reading only `expansion_ratio` would rank Helsinki as the most efficient tokenizer. Reading `unknown_token_rate` alongside it reveals the opposite — it is the only model that fails to encode Tamil text at all.

## Distribution Behavior (from violin plot)
- NLLB-200 has the tightest distribution (median ~1.3, range ~0.7–2.3) — most predictable behavior
- IndicTrans2 has a wide symmetric distribution (median ~5.3, range ~2.3–8.0) — consistent high expansion
- mT5 has a long upper tail (reaching 7.3) despite the best mean metrics — certain sentences trigger extreme fragmentation
- MADLAD is nearly as tight as NLLB-200 (range ~1.2–2.6) — second most consistent

## Model Recommendation
For downstream tasks requiring efficient Tamil tokenisation (e.g., summarisation, QA):
- **mT5 tokenizer** — best overall coverage and density (lowest fragmentation, highest avg_word_length)
- **NLLB-200 tokenizer** — most consistent distribution, practical for production workloads
- **IndicTrans2 tokenizer** — best for translation tasks specifically; high fragmentation is manageable given the quality payoff

## Limitations
- Token metrics reflect the tokenizer alone, not translation quality — a bad tokenizer can belong to a good MT model and vice versa
- `subword_fragmentation` metric (1 / avg_chars_per_tok) is a proxy; does not account for morpheme boundaries
- Results are corpus-level averages over 100 sentences — sentence-level variance is high
