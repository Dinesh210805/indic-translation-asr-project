# Indic Translation & ASR Evaluation Suite

A two-task research and engineering submission covering:

1. **Evaluation of Indic translation models** — sacreBLEU benchmarking and tokenizer behavior analysis across five state-of-the-art models for English → Tamil.
2. **ASR + transliteration system** — a deployable Whisper-based Tamil speech recognition pipeline with five romanization schemes, exposed through an interactive Gradio interface.

---

## Live Demo

🚀 **Task 2 — Tamil ASR + Transliteration:**
[**huggingface.co/spaces/DINESH210805/tamil-asr-transliteration**](https://huggingface.co/spaces/DINESH210805/tamil-asr-transliteration)

The Space runs the full pipeline in your browser — pick from 50 Common Voice Tamil clips, choose between three Whisper backends (Small / Medium / Groq Cloud Large-V3-Turbo), and watch the seven-stage pipeline animate as it processes audio.

> The Groq Cloud backend (sub-second inference) is the recommended path on the free CPU Space.
> Local Whisper variants work but are CPU-bound on the public deployment.

---

## Repository Structure

```
indic-translation-asr-project/
├── task1_translation_evaluation/
│   ├── part_a_batch_translation/      # Batch MT + sacreBLEU scoring (Kaggle GPU)
│   ├── part_b_token_analysis/         # Token-level EDA and feature engineering
│   └── part_c_indic_token_behavior/   # Indic vocab coverage and memory analysis
├── task2_asr_transliteration/
│   ├── app/                           # ASR pipeline + Gradio UI
│   ├── models/                        # Model config
│   ├── evaluation/                    # WER evaluation scripts + results
│   ├── sample_inputs/                 # Common Voice clips + recording guide
│   ├── tests/                         # Pytest suite (no GPU / API keys needed)
│   ├── Dockerfile                     # Reproducible container build
│   ├── docker-compose.yml
│   └── README.md                      # Full Task 2 setup + deploy guide
├── data/raw/                          # Raw datasets (gitignored)
├── DEMO_SCRIPT_PROJECT1.md            # Task 1 video walkthrough script
├── DEMO_SCRIPT_PROJECT2.md            # Task 2 video walkthrough script
├── TASK1_RUNBOOK.md                   # Detailed Task 1 execution guide
├── kaggle.yml                         # Kaggle kernel push config
├── requirements.txt
└── LICENSE
```

---

## Task 1 — Indic Translation Evaluation

### Models compared

| Model | Parameters | Notes |
|-------|-----------|-------|
| IndicTrans2 (`ai4bharat/indictrans2-en-indic-1B`) | 1 B | Indic-specific; uses IndicProcessor |
| NLLB-200 (`facebook/nllb-200-distilled-600M`) | 600 M | Multilingual; explicit src/tgt language codes |
| Helsinki MarianMT (`Helsinki-NLP/opus-mt-en-dra`) | ~74 M | EN → Dravidian multilingual (ta/kn/ml/te) |
| MADLAD-400 (`google/madlad400-3b-mt`) | 3 B | Requires `<2ta>` task prefix |
| mT5-base (`google/mt5-base`) | 580 M | Tokenization analysis only — not a translation model |

### Dataset

**FLoRes-200** — `openlanguagedata/flores_plus`, split `eng_Latn-tam_Taml`, first 100 sentences (devtest split).

### How to run

The pipeline runs in three sequential steps. Part A requires a GPU and runs on Kaggle. Parts B and C are CPU-only and can run on Kaggle or locally.

#### Part A — Kaggle GPU (required)

GPU is needed to load and run the 600 M – 3 B parameter translation models.

```bash
kaggle kernels push -p task1_translation_evaluation/part_a_batch_translation
```

Download `sacrebleu_results.csv` from the kernel output before running Part B.

#### Part B — local or Kaggle CPU

```bash
# local
cd task1_translation_evaluation/part_b_token_analysis
jupyter notebook part_b_token_eda.ipynb

# or Kaggle
kaggle kernels push -p task1_translation_evaluation/part_b_token_analysis
```

Requires `sacrebleu_results.csv` in the `part_a_batch_translation/` directory.

#### Part C — local or Kaggle CPU

```bash
# local
cd task1_translation_evaluation/part_c_indic_token_behavior
jupyter notebook part_c_indic_token_analysis.ipynb

# or Kaggle
kaggle kernels push -p task1_translation_evaluation/part_c_indic_token_behavior
```

Requires `token_counts.csv` from Part B output.

You can also use the Kaggle Studio VS Code extension with `kaggle.yml` at the repo root for all three parts.

### Output artifacts

| File | Written by | Read by |
|------|-----------|---------|
| `part_a_batch_translation/sacrebleu_results.csv` | Part A | Part B |
| `part_b_token_analysis/token_counts.csv` | Part B | Part C |
| `part_b_token_analysis/engineered_features.csv` | Part B | — |
| `part_c_indic_token_behavior/tamil_token_patterns.csv` | Part C | — |
| `part_c_indic_token_behavior/tokenization_comparison.csv` | Part C | — |
| `part_*/plots/*.png` | each part | — |
| `part_*/REPORT.md` | — | human reference |
| `part_*/observations.md` | — | human reference |

---

## Task 2 — Tamil ASR + Transliteration

A deployable end-to-end pipeline:

```
Audio → Decode/Normalize → Buffer Queue → Whisper ASR → Tamil script → indic-transliteration → Roman script
```

### Highlights

- **Three ASR backends in one UI** — Whisper Small (244 M), Medium (769 M), and Large-V3-Turbo via Groq Cloud (1.5 B)
- **Five transliteration schemes** — ITRANS · ISO 15919 · IAST · Harvard-Kyoto · SLP1
- **Long-audio handling** — 30 s chunks with 5 s stride overlap to prevent boundary token loss
- **Visualized pipeline** — animated seven-stage flow showing exactly what happens to each chunk
- **50 Common Voice Tamil clips bundled** as a one-click sample picker
- **Reverse direction** — bonus playground for Roman → Tamil transliteration
- **Containerized** — runs identically locally (Docker), on Hugging Face Spaces, or via `python -m app.main`

### Quick start

```bash
cd task2_asr_transliteration
pip install -r requirements.txt
cp .env.example .env       # add GROQ_API_KEY if using the Cloud backend
python -m app.main         # opens http://localhost:7860
```

### Run with Docker

```bash
cd task2_asr_transliteration
docker build -t asr-system .
docker run -p 7860:7860 --env-file .env asr-system
# or
docker compose up --build
```

The container runs as a non-root user (UID 1000), so the same image works locally **and** on Hugging Face Spaces under the Docker SDK.

Full setup, model details, scheme comparison, and HF Spaces deploy instructions live in [`task2_asr_transliteration/README.md`](task2_asr_transliteration/README.md).

---

## Requirements

**Task 1** — see [`requirements.txt`](requirements.txt). Key packages: `transformers`, `sacrebleu`, `datasets`, `sentencepiece`, `torch`.

Install the IndicTrans2 toolkit separately:

```bash
pip install git+https://github.com/AI4Bharat/IndicTransToolkit.git
```

**Task 2** — see [`task2_asr_transliteration/requirements.txt`](task2_asr_transliteration/requirements.txt). Key packages: `torch`, `transformers`, `gradio`, `librosa`, `groq`, `indic-transliteration`.

---

## Demo Videos

Walkthrough scripts for screen recordings live at the repo root:

- [`DEMO_SCRIPT_PROJECT1.md`](DEMO_SCRIPT_PROJECT1.md) — cell-by-cell Task 1 notebook walkthrough
- [`DEMO_SCRIPT_PROJECT2.md`](DEMO_SCRIPT_PROJECT2.md) — Gradio app demo (under 5 minutes)

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Acknowledgements

- AI4Bharat — IndicTrans2 and IndicProcessor
- Meta AI — NLLB-200 and the FLoRes-200 evaluation set
- OpenAI — Whisper ASR family
- Groq — LPU inference platform
- Mozilla Common Voice — Tamil speech corpus
- `indic-transliteration` — rule-based Indic script conversion
