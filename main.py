#!/usr/bin/env python3
"""Extract amendments from bills and calculate similarity scores against USC diffs."""

import words_to_data as wtd
import json
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from tqdm import tqdm
import click

MAX_RETRIES = 5
INITIAL_BACKOFF = 5  # seconds


def load_system_prompt(prompt_path: str) -> str:
    """Load the system prompt from file."""
    with open(prompt_path, "r") as f:
        return f.read()


def log_error(output_dir: Path, amendment_id: str, error_type: str, error: Exception, extra: str = ""):
    """Log an error to the error log file."""
    error_log = output_dir / "errors.log"
    with open(error_log, "a") as f:
        f.write(f"{'='*60}\n")
        f.write(f"Amendment: {amendment_id}\n")
        f.write(f"Error type: {error_type}\n")
        f.write(f"Error: {error}\n")
        if extra:
            f.write(f"{extra}\n")
        f.write(f"{'='*60}\n\n")
    return error_log


def prompt_changes(
    amendment: wtd.BillAmendment,
    system_prompt: str,
    base_url: str,
    output_dir: Path,
) -> list[wtd.BillDiff]:
    """Query LLM to extract word-level changes from amendment text."""
    message = f"""
    Extract the word-level changes from this amendment text:

    <amendment>
    {amendment.amending_text}
    </amendment>
        """
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        "temperature": 1,
        "max_tokens": 64_000,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    # Retry loop with exponential backoff
    raw_answer = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                raw_answer = result["choices"][0]["message"]["content"]
                break
        except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError) as e:
            backoff = INITIAL_BACKOFF * (2 ** attempt)
            if attempt < MAX_RETRIES - 1:
                print(f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}. Retrying in {backoff}s...")
                time.sleep(backoff)
            else:
                error_log = log_error(output_dir, amendment.id, "NetworkError", e)
                print(f"ERROR: All retries failed for amendment {amendment.id}")
                print(f"Details logged to {error_log}")
                return []
        except (json.JSONDecodeError, KeyError) as e:
            error_log = log_error(output_dir, amendment.id, "ResponseParseError", e)
            print(f"ERROR: Failed to parse API response for amendment {amendment.id}")
            print(f"Details logged to {error_log}")
            return []

    if raw_answer is None:
        return []

    # Extract JSON from <response> tags
    match = re.search(r'<response>(.*?)</response>', raw_answer, re.DOTALL)
    if not match:
        error_log = log_error(
            output_dir, amendment.id, "NoResponseTags",
            ValueError("No <response> tags found"),
            f"Raw answer:\n{raw_answer}"
        )
        print(f"ERROR: No <response> tags found for amendment {amendment.id}")
        print(f"Details logged to {error_log}")
        return []

    changes: list[wtd.BillDiff] = []
    json_str = match.group(1).strip()
    try:
        diffs = json.loads(json_str)
    except json.JSONDecodeError as e:
        error_log = log_error(
            output_dir, amendment.id, "JSONDecodeError", e,
            f"Raw answer:\n{raw_answer}\nExtracted JSON string:\n{json_str}"
        )
        print(f"ERROR: Failed to parse JSON for amendment {amendment.id}")
        print(f"Details logged to {error_log}")
        return []

    for diff in diffs:
        changes.append(wtd.BillDiff(added=diff['added'], removed=diff['removed']))

    return changes


def save_amendments(cache_file: Path, amendment_data: wtd.AmendmentData):
    """Save amendment data to cache file."""
    with open(cache_file, "w") as f:
        f.write(json.dumps(json.loads(amendment_data.to_json()), indent=2))


def process_bill(
    bill_path: str,
    diff: wtd.TreeDiff,
    system_prompt: str,
    base_url: str,
    use_cache: bool,
    max_amendments: int | None,
    output_dir: Path,
):
    """Process a single bill and return amendment data with similarity scores."""
    bill_name = Path(bill_path).stem
    cache_file = output_dir / f"amendment_data_{bill_name}.json"

    amendment_data = wtd.parse_bill_amendments(bill_path)
    amendments_to_process = amendment_data.amendments
    if max_amendments is not None:
        amendments_to_process = amendments_to_process[:max_amendments]

    # Load existing progress if available
    processed_ids: set[str] = set()
    processed_amendments: list[wtd.BillAmendment] = []
    if cache_file.exists():
        click.echo(f"Loading cached progress for {bill_name}...")
        with open(cache_file, "r") as f:
            cached = wtd.AmendmentData.from_json(f.read())
            for amendment in cached.amendments:
                processed_ids.add(amendment.id)
                processed_amendments.append(amendment)

    # Check if already complete
    remaining = [a for a in amendments_to_process if a.id not in processed_ids]
    if use_cache and not remaining:
        click.echo(f"All amendments already cached for {bill_name}")
        scanned_amendments = wtd.AmendmentData(amendment_data.bill_id, processed_amendments)
    else:
        if processed_ids:
            click.echo(f"Resuming {bill_name}: {len(processed_ids)} cached, {len(remaining)} remaining")
        else:
            click.echo(f"Processing amendments for {bill_name}...")

        for amendment in tqdm(remaining, desc=f"Processing {bill_name}"):
            changes = prompt_changes(amendment, system_prompt, base_url, output_dir)
            processed_amendments.append(amendment.update_changes(changes=changes))

            # Save after each successful amendment
            scanned_amendments = wtd.AmendmentData(amendment_data.bill_id, processed_amendments)
            save_amendments(cache_file, scanned_amendments)

        scanned_amendments = wtd.AmendmentData(amendment_data.bill_id, processed_amendments)

    # Calculate similarity scores
    scores = diff.calculate_amendment_similarities(scanned_amendments)
    return scanned_amendments, scores


@click.command()
@click.option(
    "--old-usc",
    required=True,
    type=click.Path(exists=True),
    help="Path to the older USC XML file",
)
@click.option(
    "--new-usc",
    required=True,
    type=click.Path(exists=True),
    help="Path to the newer USC XML file",
)
@click.option(
    "--base-url",
    required=True,
    type=click.STRING,
    help="Base URL of chatGPT completions compatible api i.e. http://192.168.0.172:8080",
)
@click.option(
    "--old-date",
    required=True,
    help="Date string for the old USC version (e.g., 2025-07-18)",
)
@click.option(
    "--new-date",
    required=True,
    help="Date string for the new USC version (e.g., 2025-07-30)",
)
@click.option(
    "--bill",
    "-b",
    "bills",
    multiple=True,
    required=True,
    type=click.Path(exists=True),
    help="Path to bill XML file (can be specified multiple times)",
)
@click.option(
    "--use-cache/--no-cache",
    default=False,
    show_default=True,
    help="Use cached amendment data if available",
)
@click.option(
    "--similarity-cutoff",
    default=0.4,
    show_default=True,
    type=float,
    help="Minimum similarity score to include in output",
)
@click.option(
    "--max-amendments",
    default=None,
    type=int,
    help="Maximum number of amendments to process per bill (default: all)",
)
@click.option(
    "--output-dir",
    "-o",
    default="output",
    show_default=True,
    type=click.Path(),
    help="Directory for output files",
)
@click.option(
    "--prompt-file",
    default="prompts/extract_amendment_diffs.txt",
    show_default=True,
    type=click.Path(exists=True),
    help="Path to the system prompt file",
)
def main(
    old_usc: str,
    new_usc: str,
    old_date: str,
    new_date: str,
    bills: tuple[str, ...],
    base_url: str,
    use_cache: bool,
    similarity_cutoff: float,
    max_amendments: int | None,
    output_dir: str,
    prompt_file: str,
):
    """Extract amendments from bills and calculate similarity scores against USC diffs.

    Example usage:

        uv run main.py \\
            --old-usc samples/usc26-2025-07-18.xml \\
            --new-usc samples/usc26-2025-07-30.xml \\
            --old-date 2025-07-18 \\
            --new-date 2025-07-30 \\
            --bill samples/hr-119-21.xml
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load system prompt
    system_prompt = load_system_prompt(prompt_file)

    # Load two versions of USC
    click.echo(f"Loading old USC from {old_usc}...")
    old = wtd.parse_uslm_xml(old_usc, old_date)
    click.echo(f"Loading new USC from {new_usc}...")
    new = wtd.parse_uslm_xml(new_usc, new_date)

    # Compute word level diff
    click.echo("Computing diff between USC versions...")
    diff = wtd.compute_diff(old, new)

    # Process all bills and collect scores
    all_scores: list[wtd.AmendmentSimilarity] = []
    for bill_path in bills:
        _, scores = process_bill(
            bill_path=bill_path,
            diff=diff,
            system_prompt=system_prompt,
            base_url=base_url,
            use_cache=use_cache,
            max_amendments=max_amendments,
            output_dir=output_path,
        )
        all_scores.extend(scores)

    # Filter and sort combined scores
    filtered_scores = [s for s in all_scores if s.score > similarity_cutoff]
    scores_sorted = sorted(filtered_scores, key=lambda x: x.score, reverse=True)

    # Save results
    output_file = output_path / "similarity_scores.json"
    with open(output_file, "w") as f:
        f.write(json.dumps([json.loads(score.to_json()) for score in scores_sorted], indent=2))

    click.echo(f"Processed {len(bills)} bill(s)")
    click.echo(f"Found {len(scores_sorted)} amendments above similarity cutoff {similarity_cutoff}")
    click.echo(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
