"""
Luxembourg Language Analysis - Step 1: Extract Available Languages via LLM
===========================================================================
Joins the sample with raw HTML data, extracts <head> and navigation,
then detects which languages each website offers.

OPTIMIZATIONS:
1. Regex-first: Extract hreflang tags via regex
2. Website-year deduplication: Only need one successful extraction per (website_url, year)
3. Homepage preference: For LLM fallback, prefer homepage over other pages
4. LLM fallback: Only use LLM for website-years where regex found nothing

Input:  Sample parquet + raw HTML gz files
Output: Parquet with detected languages per website-year

Author: Julio Garbers with contributions from Claude
Date: January 2026
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import aiohttp
import polars as pl
from tqdm.asyncio import tqdm_asyncio

# =============================================================================
# Configuration
# =============================================================================

# Silence noisy warnings
warnings.filterwarnings("ignore", category=UserWarning, module="aiohttp")

# Parse command-line arguments
cli = argparse.ArgumentParser(
    description="Extract available languages from Luxembourg websites"
)
cli.add_argument("--host")
cli.add_argument("--model")
cli.add_argument("--tensor-parallel-size")
cli.add_argument("--pipeline-parallel-size")
args, _ = cli.parse_known_args()

# Directories
SAMPLE_FILE = Path("/project/home/p200812/blog/data/lux_sample_for_llm.parquet")
RAW_DATA_DIR = Path("/project/home/p201125/firm_websites/data/raw/luxembourg")
OUTPUT_DIR = Path("/project/home/p200812/blog/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "lux_sample_with_languages.parquet"

# Years to process
YEARS = list(range(2013, 2025))

# Model and API configuration
API_URL = (
    os.getenv("VLLM_SERVER_URL", args.host or "http://localhost:8000").rstrip("/")
    + "/v1/chat/completions"
)
HF_MODEL = os.getenv("HF_MODEL", args.model or "mistralai/Magistral-Small-2506")

PIPELINE_STAGES = int(
    os.getenv("PIPELINE_PARALLEL_SIZE", args.pipeline_parallel_size or "1")
)
# Increased concurrency for better throughput
CONCURRENCY = PIPELINE_STAGES * 128
TIMEOUT_S = int(os.getenv("TIMEOUT_S", "300"))
MAX_RETRIES = 3


# =============================================================================
# Prompt Templates
# =============================================================================

SYS_PROMPT = """\
You are a web analyst. Given the HTML <head> section and navigation area of a Luxembourg website, \
identify which languages the website is available in.

Look for these signals (this is not an exhaustive list, there may be other ways languages are indicated):
1. <link rel="alternate" hreflang="..."> tags (most reliable)
2. Language switcher links: text like "FR", "DE", "EN", "LU", "Français", "Deutsch", "English", "Lëtzebuergesch"
3. URL patterns in links: /fr/, /de/, /en/, /lu/, ?lang=fr, etc.
4. <meta> language tags
5. Navigation menus with language options
6. Dropdown selectors or flags indicating language choices

Output a JSON object with boolean values for each language. Only mark true if you have clear evidence.

Languages to detect:
- fr: French
- de: German  
- en: English
- lb: Luxembourgish (also indicated by "lu", "LU", "Lëtzebuergesch", "Luxembourgish")
- pt: Portuguese
- nl: Dutch
- other: Any other language not listed above (e.g., Spanish, Italian, Chinese, etc.)

Output format (*exactly like this*):
{
  "fr": true/false,
  "de": true/false,
  "en": true/false,
  "lb": true/false,
  "pt": true/false,
  "nl": true/false,
  "other": true/false
}

*Do not provide any additional explanation.*\
"""

USER_PROMPT = "HTML content:\n{html}\n\nIdentify the available languages."


# =============================================================================
# Homepage Detection
# =============================================================================

HOMEPAGE_PATHS = {"", "/", "/index.html", "/index.php", "/index.htm"}


def is_homepage(page_url: str) -> bool:
    """
    Check if a URL is a homepage based on its path.
    Homepages typically have empty or minimal paths.
    """
    try:
        path = urlparse(page_url).path
        return path in HOMEPAGE_PATHS
    except Exception:
        return False


# =============================================================================
# Regex-based Language Extraction (Fast Path)
# =============================================================================

# Map hreflang codes to our standard codes
HREFLANG_MAP = {
    "fr": "fr",
    "de": "de",
    "en": "en",
    "lb": "lb",
    "lu": "lb",  # Luxembourg code often used for Luxembourgish
    "pt": "pt",
    "nl": "nl",
    # Common variants
    "fr-fr": "fr",
    "fr-lu": "fr",
    "fr-be": "fr",
    "de-de": "de",
    "de-lu": "de",
    "de-at": "de",
    "en-us": "en",
    "en-gb": "en",
    "en-lu": "en",
    "pt-pt": "pt",
    "pt-br": "pt",
    "nl-nl": "nl",
    "nl-be": "nl",
}

# Relevant languages
TRACKED_LANGUAGES = {"fr", "de", "en", "lb", "pt", "nl"}


def extract_languages_regex(html: str) -> str | None:
    """
    Extract languages from hreflang tags using regex.
    Returns a comma-separated string of found language codes, or None if no hreflang tags found.
    E.g., "fr,de,en" or None
    """
    if not html:
        return None

    # Find all hreflang values
    # Matches: hreflang="fr" or hreflang='fr-FR' etc.
    hreflang_matches = re.findall(
        r'hreflang=["\']([a-zA-Z]{2}(?:-[a-zA-Z]{2})?)["\']', html, re.IGNORECASE
    )

    if not hreflang_matches:
        return None

    # Track found languages
    found_langs = set()

    # Map found hreflang codes to our categories
    for code in hreflang_matches:
        code_lower = code.lower()

        if code_lower in HREFLANG_MAP:
            found_langs.add(HREFLANG_MAP[code_lower])
        elif code_lower.split("-")[0] in HREFLANG_MAP:
            # Try base language code (e.g., "fr" from "fr-CA")
            found_langs.add(HREFLANG_MAP[code_lower.split("-")[0]])
        elif code_lower not in ("x-default",):
            # Unknown language, mark as other
            found_langs.add("other")

    # Only return if at least one real language is found
    if found_langs:
        return ",".join(sorted(found_langs))

    return None


def parse_regex_result(regex_str: str | None) -> dict[str, bool]:
    """Convert regex result string back to dict of booleans."""
    result = {
        "fr": False,
        "de": False,
        "en": False,
        "lb": False,
        "pt": False,
        "nl": False,
        "other": False,
    }
    if regex_str:
        for lang in regex_str.split(","):
            if lang in result:
                result[lang] = True
    return result


# =============================================================================
# HTML Extraction for LLM
# =============================================================================


def extract_head_and_nav(
    html: str, head_limit: int = 10000, body_limit: int = 5000
) -> str:
    """
    Extract the <head> section and first part of <body> from HTML.
    These contain hreflang tags and navigation with language switchers.
    """
    if not html or not isinstance(html, str):
        return ""

    result_parts = []

    # Extract <head>...</head>
    head_match = re.search(r"<head[^>]*>(.*?)</head>", html, re.IGNORECASE | re.DOTALL)
    if head_match:
        head_content = head_match.group(0)
        if len(head_content) > head_limit:
            head_content = head_content[:head_limit] + "...</head>"
        result_parts.append(head_content)

    # Extract first part of <body> (contains navigation)
    body_match = re.search(r"<body[^>]*>(.*)", html, re.IGNORECASE | re.DOTALL)
    if body_match:
        body_content = body_match.group(0)
        if len(body_content) > body_limit:
            body_content = body_content[:body_limit] + "..."
        result_parts.append(body_content)

    if not result_parts:
        result_parts.append(html[: head_limit + body_limit])

    return "\n".join(result_parts)


# =============================================================================
# Statistics Tracking
# =============================================================================


@dataclass
class ExtractionStats:
    """Track statistics for the extraction process."""

    total_pages: int = 0
    total_website_years: int = 0

    # Regex extraction
    regex_success: int = 0
    regex_no_hreflang: int = 0

    # LLM extraction
    llm_needed: int = 0
    llm_success: int = 0
    llm_failed: int = 0

    # Homepage stats
    llm_with_homepage: int = 0
    llm_without_homepage: int = 0

    # Final results
    detected_fr: int = 0
    detected_de: int = 0
    detected_en: int = 0
    detected_lb: int = 0
    detected_pt: int = 0
    detected_nl: int = 0
    detected_other: int = 0

    def print_summary(self) -> None:
        """Print extraction statistics."""
        print(
            "===============================================================================",
            flush=True,
        )
        print("[EXTRACTION STATS] Summary:", flush=True)
        print("\n  Input:", flush=True)
        print(f"    Total pages processed:     {self.total_pages:,}", flush=True)
        print(
            f"    Unique website-years:      {self.total_website_years:,}", flush=True
        )

        print("\n  Regex extraction (hreflang):", flush=True)
        print(f"    Success (hreflang found):  {self.regex_success:,}", flush=True)
        print(f"    No hreflang tags:          {self.regex_no_hreflang:,}", flush=True)

        print("\n  LLM extraction (fallback):", flush=True)
        print(f"    Website-years sent to LLM: {self.llm_needed:,}", flush=True)
        print(f"    - with homepage available: {self.llm_with_homepage:,}", flush=True)
        print(
            f"    - without homepage:        {self.llm_without_homepage:,}", flush=True
        )
        print(f"    LLM parse success:         {self.llm_success:,}", flush=True)
        print(f"    LLM parse failed:          {self.llm_failed:,}", flush=True)

        if self.total_website_years > 0:
            print("\n  Languages detected (website-years offering each):", flush=True)
            print(
                f"    French (fr):        {self.detected_fr:,} ({self.detected_fr / self.total_website_years * 100:.1f}%)",
                flush=True,
            )
            print(
                f"    German (de):        {self.detected_de:,} ({self.detected_de / self.total_website_years * 100:.1f}%)",
                flush=True,
            )
            print(
                f"    English (en):       {self.detected_en:,} ({self.detected_en / self.total_website_years * 100:.1f}%)",
                flush=True,
            )
            print(
                f"    Luxembourgish (lb): {self.detected_lb:,} ({self.detected_lb / self.total_website_years * 100:.1f}%)",
                flush=True,
            )
            print(
                f"    Portuguese (pt):    {self.detected_pt:,} ({self.detected_pt / self.total_website_years * 100:.1f}%)",
                flush=True,
            )
            print(
                f"    Dutch (nl):         {self.detected_nl:,} ({self.detected_nl / self.total_website_years * 100:.1f}%)",
                flush=True,
            )
            print(
                f"    Other:              {self.detected_other:,} ({self.detected_other / self.total_website_years * 100:.1f}%)",
                flush=True,
            )

        print(
            "===============================================================================",
            flush=True,
        )


# Global stats tracker
stats = ExtractionStats()


# =============================================================================
# API Request Functions
# =============================================================================


def build_payload(prompt: str) -> dict[str, Any]:
    """Build JSON payload for vLLM server."""
    return {
        "model": HF_MODEL,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 150,
        "temperature": 0.0,
        "seed": 666,
    }


async def post_request(sess: aiohttp.ClientSession, prompt: str) -> str:
    """Send POST request to vLLM server and return response content."""
    async with sess.post(API_URL, json=build_payload(prompt)) as response:
        response.raise_for_status()
        data = await response.json()
        return data["choices"][0]["message"]["content"].strip()


async def retry_post(
    sess: aiohttp.ClientSession, prompt: str, sem: asyncio.Semaphore
) -> str | None:
    """Retry POST request with exponential backoff on failure."""
    attempt = 0
    while attempt <= MAX_RETRIES:
        try:
            async with sem:
                return await post_request(sess, prompt)
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt == MAX_RETRIES:
                return None
            await asyncio.sleep((2**attempt) + random.random())
            attempt += 1
    return None


async def run_inference(prompts: list[str]) -> list[str | None]:
    """Run batch inference on all prompts with concurrency control."""
    timeout = aiohttp.ClientTimeout(total=TIMEOUT_S)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [retry_post(session, prompt, semaphore) for prompt in prompts]
        return await tqdm_asyncio.gather(*tasks, desc="LLM inference")


# =============================================================================
# Response Parsing
# =============================================================================


def parse_llm_response(text: str | None) -> dict[str, bool] | None:
    """Parse JSON response from LLM."""
    global stats

    if text is None:
        stats.llm_failed += 1
        return None

    text = text.strip()

    # Handle markdown code blocks
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
        stats.llm_success += 1
    except json.JSONDecodeError:
        stats.llm_failed += 1
        return None

    # Normalize to our expected format
    result = {
        "fr": bool(data.get("fr", False)),
        "de": bool(data.get("de", False)),
        "en": bool(data.get("en", False)),
        "lb": bool(data.get("lb", False)),
        "pt": bool(data.get("pt", False)),
        "nl": bool(data.get("nl", False)),
        "other": bool(data.get("other", False)),
    }

    return result


# =============================================================================
# Main Pipeline
# =============================================================================


async def main_async() -> None:
    """Main extraction pipeline."""
    global stats

    print(
        "===============================================================================",
        flush=True,
    )
    print("[LOAD] Loading sample and raw data for all years...", flush=True)

    # Load full sample
    df_sample = pl.scan_parquet(SAMPLE_FILE).collect()
    print(f"  Total sample pages: {len(df_sample):,}", flush=True)

    if len(df_sample) == 0:
        print("  No pages in sample, exiting.", flush=True)
        return

    # Load raw HTML data for all years
    all_gz_files = []
    for year in YEARS:
        raw_dir = RAW_DATA_DIR / str(year) / "gz"
        gz_files = sorted(raw_dir.glob("*.gz"))
        all_gz_files.extend(gz_files)
        print(f"  Found {len(gz_files)} raw gz files for {year}", flush=True)

    print(f"  Total raw gz files: {len(all_gz_files):,}", flush=True)

    if not all_gz_files:
        print("  No raw files found, exiting.", flush=True)
        return

    # Load raw data and join with sample
    print("[JOIN] Joining sample with raw HTML...", flush=True)

    df_raw = (
        pl.scan_parquet(all_gz_files)
        .select(["url", "year", "html"])
        .rename({"url": "page_url"})
    )

    df = (
        df_sample.lazy()
        .join(df_raw, on=["page_url", "year"], how="inner")
        .collect(engine="streaming")
    )

    print(f"  Matched pages with HTML: {len(df):,}", flush=True)
    stats.total_pages = len(df)

    if len(df) == 0:
        print("  No matches found, exiting.", flush=True)
        return

    # =========================================================================
    # PHASE 1: Regex extraction (fast path)
    # =========================================================================
    print("\n[PHASE 1] Regex extraction of hreflang tags...", flush=True)

    # Extract languages via regex for all pages (returns comma-separated string or None)
    df = df.with_columns(
        pl.col("html")
        .map_elements(extract_languages_regex, return_dtype=pl.String)
        .alias("regex_result")
    )

    # Count successes
    regex_success_count = df.filter(pl.col("regex_result").is_not_null()).height
    stats.regex_success = regex_success_count
    stats.regex_no_hreflang = len(df) - regex_success_count

    print(f"  Pages with hreflang tags: {regex_success_count:,}", flush=True)
    print(f"  Pages without hreflang:   {stats.regex_no_hreflang:,}", flush=True)

    # =========================================================================
    # PHASE 2: Aggregate to website-year level (with homepage preference)
    # =========================================================================
    print("\n[PHASE 2] Aggregating to website-year level...", flush=True)

    # Add is_homepage flag
    df = df.with_columns(
        pl.col("page_url")
        .map_elements(is_homepage, return_dtype=pl.Boolean)
        .alias("is_homepage")
    )

    homepage_count = df.filter(pl.col("is_homepage")).height
    print(f"  Pages identified as homepages: {homepage_count:,}", flush=True)

    # For each (website_url, year), I need:
    # 1. If any page has regex_result → use that (prefer first non-null)
    # 2. If no page has regex_result → pick homepage HTML for LLM (if available), else first page

    # Sort by is_homepage descending so homepages come first within each group
    df_sorted = df.sort(
        ["website_url", "year", "is_homepage"], descending=[False, False, True]
    )

    # Group and aggregate
    website_year_groups = df_sorted.group_by(["website_url", "year"]).agg(
        [
            # Get first non-null regex result
            pl.col("regex_result").drop_nulls().first().alias("regex_result"),
            # Get first page's HTML for LLM fallback (will be homepage if available due to sorting)
            pl.col("html").first().alias("html_for_llm"),
            # Track if I have a homepage for this website-year
            pl.col("is_homepage").any().alias("has_homepage"),
            # Count pages per website-year
            pl.len().alias("n_pages"),
        ]
    )

    stats.total_website_years = len(website_year_groups)
    print(f"  Unique website-years: {stats.total_website_years:,}", flush=True)

    # Split into regex-solved and needs-LLM
    df_regex_solved = website_year_groups.filter(pl.col("regex_result").is_not_null())
    df_needs_llm = website_year_groups.filter(pl.col("regex_result").is_null())

    print(f"  Solved by regex:      {len(df_regex_solved):,}", flush=True)
    print(f"  Need LLM:             {len(df_needs_llm):,}", flush=True)

    stats.llm_needed = len(df_needs_llm)

    # Count how many LLM cases have homepage available
    if len(df_needs_llm) > 0:
        stats.llm_with_homepage = df_needs_llm.filter(pl.col("has_homepage")).height
        stats.llm_without_homepage = len(df_needs_llm) - stats.llm_with_homepage
        print(f"    - with homepage:    {stats.llm_with_homepage:,}", flush=True)
        print(f"    - without homepage: {stats.llm_without_homepage:,}", flush=True)

    # =========================================================================
    # PHASE 3: LLM extraction for remaining website-years
    # =========================================================================
    llm_results = {}

    if len(df_needs_llm) > 0:
        print(
            f"\n[PHASE 3] LLM extraction for {len(df_needs_llm):,} website-years...",
            flush=True,
        )

        # Extract head+nav for LLM pages
        df_needs_llm = df_needs_llm.with_columns(
            pl.col("html_for_llm")
            .map_elements(extract_head_and_nav, return_dtype=pl.String)
            .alias("html_extract")
        )

        # Filter out pages with empty extracts
        df_needs_llm = df_needs_llm.filter(pl.col("html_extract").str.len_chars() > 100)
        print(f"  Pages with valid HTML extract: {len(df_needs_llm):,}", flush=True)

        if len(df_needs_llm) > 0:
            # Build prompts
            prompts = [
                USER_PROMPT.format(html=row["html_extract"])
                for row in df_needs_llm.iter_rows(named=True)
            ]

            # Run inference
            raw_responses = await run_inference(prompts)

            # Parse responses and store with keys
            for i, row in enumerate(df_needs_llm.iter_rows(named=True)):
                key = (row["website_url"], row["year"])
                parsed = parse_llm_response(raw_responses[i])
                llm_results[key] = {
                    "result": parsed,
                    "raw_response": raw_responses[i],
                }

    # =========================================================================
    # PHASE 4: Combine results
    # =========================================================================
    print("\n[PHASE 4] Combining results...", flush=True)

    # Build final results list
    final_rows = []

    for row in website_year_groups.iter_rows(named=True):
        website_url = row["website_url"]
        year = row["year"]
        key = (website_url, year)

        # Determine source of language info
        if row["regex_result"] is not None:
            # Parse the comma-separated string back to dict
            langs = parse_regex_result(row["regex_result"])
            raw_response = "REGEX_HREFLANG"
        elif key in llm_results and llm_results[key]["result"] is not None:
            langs = llm_results[key]["result"]
            raw_response = llm_results[key]["raw_response"]
        else:
            # No data available
            langs = {
                "fr": None,
                "de": None,
                "en": None,
                "lb": None,
                "pt": None,
                "nl": None,
                "other": None,
            }
            raw_response = None

        final_rows.append(
            {
                "website_url": website_url,
                "year": year,
                "fr": langs.get("fr"),
                "de": langs.get("de"),
                "en": langs.get("en"),
                "lb": langs.get("lb"),
                "pt": langs.get("pt"),
                "nl": langs.get("nl"),
                "other": langs.get("other"),
                "raw_response": raw_response,
            }
        )

        # Update stats
        if langs.get("fr"):
            stats.detected_fr += 1
        if langs.get("de"):
            stats.detected_de += 1
        if langs.get("en"):
            stats.detected_en += 1
        if langs.get("lb"):
            stats.detected_lb += 1
        if langs.get("pt"):
            stats.detected_pt += 1
        if langs.get("nl"):
            stats.detected_nl += 1
        if langs.get("other"):
            stats.detected_other += 1

    # Create final DataFrame
    result_df = pl.DataFrame(
        final_rows,
        schema={
            "website_url": pl.String,
            "year": pl.Int64,
            "fr": pl.Boolean,
            "de": pl.Boolean,
            "en": pl.Boolean,
            "lb": pl.Boolean,
            "pt": pl.Boolean,
            "nl": pl.Boolean,
            "other": pl.Boolean,
            "raw_response": pl.String,
        },
    )

    # Print stats
    stats.print_summary()

    # Save results
    print("[SAVE] Writing results...", flush=True)
    result_df.write_parquet(OUTPUT_FILE, compression="zstd", compression_level=10)
    print(f"  Saved to: {OUTPUT_FILE}", flush=True)
    print(f"  Total rows: {len(result_df):,}", flush=True)

    print(
        "===============================================================================",
        flush=True,
    )
    print("Processing complete for all years.", flush=True)


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    asyncio.run(main_async())
