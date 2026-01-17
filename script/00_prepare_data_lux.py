"""
Luxembourg Language Analysis - Step 0: Prepare Sample
======================================================
Prepares the sample of .lu websites for language detection via LLM.
Outputs a parquet file with unique (website_url, page_url, year) combinations
that will be joined with raw HTML data in the next step.

Author: Julio Garbers with contributions from Claude
Date: January 2026
"""

from pathlib import Path

import polars as pl

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("/project/home/p201125/firm_websites/data/clean/luxembourg")
OUTPUT_DIR = Path("/project/home/p200812/blog/data")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "lux_sample_for_llm.parquet"


# =============================================================================
# Data Loading
# =============================================================================


def load_data(data_dir: Path) -> pl.LazyFrame:
    """Load all parquet files from directory as a lazy frame."""
    parquet_files = list(data_dir.glob("*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    print(f"Found {len(parquet_files)} parquet files")

    # Scan all parquet files lazily
    lf = pl.scan_parquet(parquet_files)

    return lf


# =============================================================================
# Data Processing
# =============================================================================


def prepare_sample(lf: pl.LazyFrame) -> pl.DataFrame:
    """
    Prepare the sample for LLM language detection.

    1. Filter to .lu domains only
    2. Select relevant columns for joining with raw HTML
    3. Deduplicate on (page_url, year) to avoid processing duplicates
    """

    # Filter to .lu domains and select columns needed for joining
    df = (
        lf.filter(pl.col("website_url").str.ends_with(".lu"))
        .select(["website_url", "page_url", "year"])
        .unique(subset=["page_url", "year"])  # Deduplicate
        .collect(engine="streaming")
    )

    print(f"Total unique page-year combinations: {len(df):,}")

    return df


# =============================================================================
# Summary Statistics
# =============================================================================


def print_summary_stats(df: pl.DataFrame):
    """Print summary statistics about the sample."""

    print("\n" + "=" * 60)
    print("SAMPLE STATISTICS")
    print("=" * 60)

    # Total counts
    print(f"\nTotal pages in sample: {len(df):,}")
    print(f"Unique websites: {df['website_url'].n_unique():,}")

    # Pages by year
    pages_by_year = (
        df.group_by("year")
        .agg(
            [
                pl.len().alias("n_pages"),
                pl.col("website_url").n_unique().alias("n_websites"),
            ]
        )
        .sort("year")
    )

    print("\nPages and websites per year:")
    print(pages_by_year)


# =============================================================================
# Main
# =============================================================================


def main():
    print("Luxembourg Language Analysis - Prepare Sample")
    print("=" * 60)

    # Load data
    print("\n1. Loading data...")
    lf = load_data(DATA_DIR)

    # Prepare sample
    print("\n2. Preparing sample...")
    df_sample = prepare_sample(lf)

    # Print summary
    print_summary_stats(df_sample)

    # Save sample
    print("\n3. Saving sample...")
    df_sample.write_parquet(OUTPUT_FILE, compression="zstd", compression_level=10)
    print(f"Saved to: {OUTPUT_FILE}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    main()
