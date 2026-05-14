# Observations — Part B · Token-Level EDA

## Dataset
- **Source**: `../part_a_batch_translation/sacrebleu_results.csv` (100 sentence pairs)
- **Models tokenized**: IndicTrans2, NLLB-200, Helsinki, MADLAD-400, mT5-base
- **Metrics**: expansion_ratio, avg_word_length (chars/token), subword_fragmentation, unknown_token_rate

## Key Findings

### Expansion Ratio
- Tamil is an agglutinative language — a single Tamil word can correspond to multiple English words
- Expected ordering: Helsinki > NLLB > mT5 > IndicTrans2 ≈ MADLAD (larger vocabularies handle Tamil better)
- IndicTrans2's dedicated Indic vocabulary should produce the lowest expansion ratio among MT models

### Subword Fragmentation
- Helsinki's small vocabulary (~60K tokens, romanisation-heavy) fragments Tamil script aggressively
- mT5's 250K SentencePiece vocabulary provides better coverage despite not being an MT model
- IndicTrans2 trains on Indic scripts directly — lowest fragmentation expected

### Unknown Token Rate
- Helsinki tokenizer is not Tamil-aware; may produce non-zero UNK rates on uncommon characters
- NLLB-200, IndicTrans2, mT5, MADLAD should have near-zero UNK rates due to large shared vocabularies

## Surprising Result
- *(To be confirmed after Part B run)* mT5 (a non-MT model) may show a **lower** expansion ratio than Helsinki (a dedicated EN→Dravidian model) purely because its 250K vocabulary has better Unicode Tamil coverage. This would illustrate that vocabulary size matters as much as task-specific training for tokenisation quality.
- Also watch: MADLAD-400 translations (confirmed longer due to conservative phrasing in Part A) may produce higher token counts than IndicTrans2 despite IndicTrans2 having the higher per-sentence BLEU — output verbosity interacts with expansion ratio.

## Model Recommendation
For downstream tasks requiring efficient Tamil tokenisation (e.g., summarisation, QA):
- **IndicTrans2 tokenizer** — best Indic coverage, lowest fragmentation
- **mT5 tokenizer** — viable alternative if cross-lingual transfer is needed (101-language coverage)

## Limitations
- Token metrics reflect the tokenizer alone, not translation quality — a bad tokenizer can belong to a good MT model and vice versa
- `subword_fragmentation` metric (1 / avg_chars_per_tok) is a proxy; does not account for morpheme boundaries
- Results are corpus-level averages over 100 sentences — sentence-level variance is high
