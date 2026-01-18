# The Linguistic Web of Luxembourg

Analyzing language availability on Luxembourg (.lu) websites from 2013 to 2024 using CommonCrawl archives.

📊 **[View the interactive visualization](https://juliogarbers.com/posts/website_languages_lux/)**

## Overview

This project detects language *availability* — which languages a website offers to visitors — rather than just language *content* (what language a page happens to be written in). This distinction matters because analyzing page content alone only reveals the language of that specific page, not the full set of languages a site offers. By detecting language switchers and hreflang tags, we can identify multilingual sites even when only one language version was archived.

**Key findings:**
- French dominates at ~75% of websites and has remained stable
- English (39%) now surpasses German (33%) — despite German being an official language
- Portuguese speakers make up 14.5% of the population but find only 2.4% of websites in their language
- Luxembourgish-only sites are rare (1.2%) — the national language almost always appears alongside French or German

## Pipeline

The analysis uses a three-tier detection approach, prioritizing the most reliable signals first:

```
┌─────────────────────────────────────────────────────────────────┐
│                     00_prepare_data_lux.py                      │
│         Filter .lu domains from CommonCrawl archives            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   01_extract_languages_lux.py                   │
│                                                                 │
│  1. Hreflang extraction (regex) ─── W3C standard, most reliable │
│                 │                                               │
│                 ▼                                               │
│  2. LLM detection (Magistral) ──── Detects language switchers   │
│                 │                   in navigation elements      │
│                 ▼                                               │
│  3. FastText fallback ──────────── Classifies monolingual sites │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       02_final_data.py                          │
│            Compute statistics and generate stats.json           │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
blog/
├── script/
│   ├── 00_prepare_data_lux.py          # Sample preparation
│   ├── 01_extract_languages_lux.py     # Language extraction (LLM + regex)
│   ├── 01_extract_languages_lux.sh     # SLURM job script for HPC
│   └── 02_final_data.py                # Statistics generation
├── data/                               # Intermediate data files
├── output/                             # Final output (stats.json)
├── pyproject.toml                      # Python dependencies
└── uv.lock                             # Dependency lock file
```

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) for dependency management (install: `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Access to CommonCrawl data
- GPU cluster with SLURM (for LLM inference)

### Dependencies

Main packages:
- `polars` — Fast DataFrame operations
- `vllm` — LLM inference server
- `fasttext` — Language classification fallback

Install dependencies:
```bash
uv sync
```

This creates a virtual environment and installs all dependencies from `pyproject.toml`.

## Usage

### 1. Prepare the sample

Filter .lu domains from CommonCrawl data:

```bash
uv run python website_languages_lux/script/00_prepare_data_lux.py
```

### 2. Extract languages

Run on an HPC cluster with GPUs:

```bash
sbatch website_languages_lux/script/01_extract_languages_lux.sh
```

This script:
- Launches a vLLM server with Magistral-Small-2506
- Processes websites through the three-tier detection pipeline
- Outputs language annotations per website-year

> **Note:** The SLURM script handles its own environment setup within the Apptainer container.

### 3. Generate statistics

Combine results and compute visualization data:

```bash
uv run python website_languages_lux/script/02_final_data.py
```

Outputs `stats.json` with yearly trends, multilingual patterns, and language combinations.

## Data

The analysis covers:
- **83,728** website-year observations
- **12 years** (2013–2024)
- **7 languages** tracked: French, German, English, Luxembourgish, Portuguese, Dutch, Other

### Detection breakdown

| Method | Website-years | Share |
|--------|---------------|-------|
| Hreflang (regex) | 15,808 | 19% |
| LLM (Magistral) | 67,774 | 81% |
| FastText fallback | 4,995 | 6% |

## Citation

```bibtex
@misc{garbers2025linguistic,
  author = {Garbers, Julio},
  title = {The Linguistic Web of Luxembourg},
  url = {https://juliogarbers.com/posts/website_languages_lux/},
  year = {2025}
}
```

## License

MIT

## Author

**Julio Garbers**  
[julio.garbers@liser.lu](mailto:julio.garbers@liser.lu)