"""
Luxembourg Language Analysis - Step 2: Generate Data for Visualization
=======================================================================
Combines LLM extraction results with FastText fallback and outputs
a JSON file with all statistics needed for the visualization.

Fallback logic:
- If regex or LLM found languages → use those
- If no languages found → use FastText detected language as single language

Author: Julio Garbers with contributions from Claude
Date: January 2026
"""

import json
from pathlib import Path

import polars as pl

# =============================================================================
# Configuration
# =============================================================================

# Input files
LLM_RESULTS_FILE = Path(
    "/project/home/p200812/blog/data/lux_sample_with_languages.parquet"
)
FASTTEXT_DATA_DIR = Path("/project/home/p201125/firm_websites/data/clean/luxembourg")

# Output
OUTPUT_DIR = Path("/project/home/p200812/blog/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "stats.json"

# Relevant languages
LANGUAGES = ["fr", "de", "en", "lb", "pt", "nl"]
LANGUAGE_LABELS = {
    "fr": "French",
    "de": "German",
    "en": "English",
    "lb": "Luxembourgish",
    "pt": "Portuguese",
    "nl": "Dutch",
    "other": "Other",
}


# =============================================================================
# Data Loading and Processing
# =============================================================================


def load_and_process_data() -> tuple[pl.DataFrame, dict]:
    """Load LLM results and merge with FastText fallback."""

    print("[LOAD] Loading LLM extraction results...")
    df_llm = pl.read_parquet(LLM_RESULTS_FILE)
    print(f"  LLM results: {len(df_llm):,} website-years")

    # Track data sources for methodology section
    methodology_stats = {
        "total_website_years": len(df_llm),
    }

    # Count regex vs LLM sources
    regex_count = df_llm.filter(pl.col("raw_response") == "REGEX_HREFLANG").height
    methodology_stats["regex_extracted"] = regex_count
    methodology_stats["llm_extracted"] = df_llm.filter(
        (pl.col("raw_response") != "REGEX_HREFLANG")
        & (pl.col("raw_response").is_not_null())
    ).height

    # Check if any language was detected (regex or LLM)
    df_llm = df_llm.with_columns(
        (
            pl.col("fr").fill_null(False)
            | pl.col("de").fill_null(False)
            | pl.col("en").fill_null(False)
            | pl.col("lb").fill_null(False)
            | pl.col("pt").fill_null(False)
            | pl.col("nl").fill_null(False)
            | pl.col("other").fill_null(False)
        ).alias("has_language_info")
    )

    has_info = df_llm.filter(pl.col("has_language_info")).height
    needs_fallback = df_llm.filter(~pl.col("has_language_info")).height
    print(f"  With language info: {has_info:,}")
    print(f"  Needs FastText fallback: {needs_fallback:,}")

    methodology_stats["needs_fallback"] = needs_fallback

    # Load FastText data for fallback
    print("\n[LOAD] Loading FastText data for fallback...")
    parquet_files = list(FASTTEXT_DATA_DIR.glob("*.parquet"))

    df_fasttext = (
        pl.scan_parquet(parquet_files)
        .filter(pl.col("website_url").str.ends_with(".lu"))
        .filter(pl.col("confidence_fasttext") >= 0.5)
        .group_by(["website_url", "year"])
        .agg(pl.col("language_fasttext").mode().first().alias("fasttext_lang"))
        .collect()
    )
    print(f"  FastText records: {len(df_fasttext):,}")

    # Join FastText data
    df = df_llm.join(df_fasttext, on=["website_url", "year"], how="left")

    # Apply fallback: if no language info, use FastText language
    for lang in LANGUAGES + ["other"]:
        df = df.with_columns(
            pl.when(~pl.col("has_language_info") & (pl.col("fasttext_lang") == lang))
            .then(True)
            .otherwise(pl.col(lang))
            .alias(lang)
        )

    # Handle FastText languages not in our main list → mark as "other"
    df = df.with_columns(
        pl.when(
            ~pl.col("has_language_info")
            & pl.col("fasttext_lang").is_not_null()
            & ~pl.col("fasttext_lang").is_in(LANGUAGES)
        )
        .then(True)
        .otherwise(pl.col("other"))
        .alias("other")
    )

    # Fill remaining nulls with False
    for lang in LANGUAGES + ["other"]:
        df = df.with_columns(pl.col(lang).fill_null(False))

    # Count how many got FastText fallback
    methodology_stats["fasttext_applied"] = df.filter(
        ~pl.col("has_language_info") & pl.col("fasttext_lang").is_not_null()
    ).height

    print(f"\n[PROCESS] Final dataset: {len(df):,} website-years")

    return df, methodology_stats


def compute_statistics(df: pl.DataFrame) -> dict:
    """Compute all statistics needed for visualization."""

    stats = {}

    # 1. Language availability over time (% of sites offering each language)
    print("[STATS] Computing language availability over time...")

    yearly_stats = []
    for year in sorted(df["year"].unique().to_list()):
        year_df = df.filter(pl.col("year") == year)
        n_sites = len(year_df)

        year_data = {"year": int(year), "n_sites": n_sites}
        for lang in LANGUAGES + ["other"]:
            count = year_df.filter(pl.col(lang)).height
            year_data[f"{lang}_count"] = count
            year_data[f"{lang}_pct"] = (
                round(count / n_sites * 100, 2) if n_sites > 0 else 0
            )

        yearly_stats.append(year_data)

    stats["yearly"] = yearly_stats

    # 2. Multilingual analysis
    print("[STATS] Computing multilingual statistics...")

    df = df.with_columns(
        (
            pl.col("fr").cast(pl.Int8)
            + pl.col("de").cast(pl.Int8)
            + pl.col("en").cast(pl.Int8)
            + pl.col("lb").cast(pl.Int8)
            + pl.col("pt").cast(pl.Int8)
            + pl.col("nl").cast(pl.Int8)
        ).alias("n_languages")
    )

    # Count excluded unknowns for methodology
    stats["excluded_unknown"] = df.filter(pl.col("n_languages") == 0).height

    multilingual_yearly = []
    for year in sorted(df["year"].unique().to_list()):
        year_df = df.filter(pl.col("year") == year)
        # Exclude unknowns (sites with no languages detected)
        year_df_known = year_df.filter(pl.col("n_languages") >= 1)
        n_sites = len(year_df_known)

        mono = year_df_known.filter(pl.col("n_languages") == 1).height
        bi = year_df_known.filter(pl.col("n_languages") == 2).height
        tri = year_df_known.filter(pl.col("n_languages") == 3).height
        quad_plus = year_df_known.filter(pl.col("n_languages") >= 4).height

        multilingual_yearly.append(
            {
                "year": int(year),
                "monolingual": round(mono / n_sites * 100, 2) if n_sites > 0 else 0,
                "bilingual": round(bi / n_sites * 100, 2) if n_sites > 0 else 0,
                "trilingual": round(tri / n_sites * 100, 2) if n_sites > 0 else 0,
                "quadlingual_plus": round(quad_plus / n_sites * 100, 2)
                if n_sites > 0
                else 0,
            }
        )

    stats["multilingual"] = multilingual_yearly

    # 3. Language combinations (most recent year)
    print("[STATS] Computing language combinations...")

    latest_year = int(df["year"].max())
    latest_df = df.filter(pl.col("year") == latest_year)

    def get_combo(row):
        langs = []
        for lang in ["fr", "de", "en", "lb"]:  # Main 4 languages
            if row[lang]:
                langs.append(LANGUAGE_LABELS[lang])
        if not langs:
            return None  # Will be filtered out
        return " + ".join(sorted(langs))

    combos = {}
    for row in latest_df.iter_rows(named=True):
        combo = get_combo(row)
        if combo is not None:  # Skip unknowns
            combos[combo] = combos.get(combo, 0) + 1

    # Sort by count and take top 12
    sorted_combos = sorted(combos.items(), key=lambda x: x[1], reverse=True)[:12]
    stats["combinations"] = [
        {"combo": c, "count": n, "pct": round(n / len(latest_df) * 100, 1)}
        for c, n in sorted_combos
    ]
    stats["combinations_year"] = latest_year

    # 4. Summary statistics
    print("[STATS] Computing summary statistics...")

    first_year = int(df["year"].min())
    last_year = int(df["year"].max())

    first_df = df.filter(pl.col("year") == first_year)
    last_df = df.filter(pl.col("year") == last_year)

    summary = {
        "first_year": first_year,
        "last_year": last_year,
        "total_website_years": len(df),
        "total_websites_first": len(first_df),
        "total_websites_latest": len(last_df),
        "years_covered": last_year - first_year + 1,
    }

    for lang in LANGUAGES + ["other"]:
        first_pct = (
            first_df.filter(pl.col(lang)).height / len(first_df) * 100
            if len(first_df) > 0
            else 0
        )
        last_pct = (
            last_df.filter(pl.col(lang)).height / len(last_df) * 100
            if len(last_df) > 0
            else 0
        )
        summary[f"{lang}_first"] = round(first_pct, 1)
        summary[f"{lang}_last"] = round(last_pct, 1)
        summary[f"{lang}_change"] = round(last_pct - first_pct, 1)

    # Multilingual stats
    first_multi = (
        first_df.filter(pl.col("n_languages") >= 2).height / len(first_df) * 100
        if len(first_df) > 0
        else 0
    )
    last_multi = (
        last_df.filter(pl.col("n_languages") >= 2).height / len(last_df) * 100
        if len(last_df) > 0
        else 0
    )
    summary["multilingual_first"] = round(first_multi, 1)
    summary["multilingual_last"] = round(last_multi, 1)
    summary["multilingual_change"] = round(last_multi - first_multi, 1)

    stats["summary"] = summary

    return stats


# =============================================================================
# Main
# =============================================================================


def main():
    print("=" * 70)
    print("Luxembourg Language Analysis — Data Generator")
    print("=" * 70)

    # Load and process data
    df, methodology_stats = load_and_process_data()

    # Compute statistics
    stats = compute_statistics(df)

    # Add methodology stats
    stats["methodology"] = methodology_stats
    stats["methodology"]["excluded_unknown"] = stats.pop("excluded_unknown")

    # Save as JSON
    print("\n[SAVE] Writing stats.json...")
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved to: {OUTPUT_JSON}")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    summary = stats["summary"]
    print(f"  Years: {summary['first_year']} - {summary['last_year']}")
    print(f"  Total website-years: {summary['total_website_years']:,}")
    print(f"  Websites in {summary['last_year']}: {summary['total_websites_latest']:,}")
    print()
    print("  Language availability changes:")
    for lang in LANGUAGES[:4]:
        change = summary[f"{lang}_change"]
        print(f"    {LANGUAGE_LABELS[lang]:15} {change:+.1f}pp")

    print("\n" + "=" * 70)
    print("Done! Now open visualization.html in a browser.")
    print("=" * 70)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
