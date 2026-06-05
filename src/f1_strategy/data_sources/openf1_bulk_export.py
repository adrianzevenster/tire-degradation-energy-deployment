"""Bulk export all races for given years and drivers from OpenF1 public API."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from pathlib import Path


def _sessions(year: int) -> list[dict]:
    url = f"https://api.openf1.org/v1/sessions?year={year}&session_name=Race"
    with urllib.request.urlopen(url, timeout=15) as resp:
        rows = json.loads(resp.read())
    return sorted([s for s in rows if not s.get("is_cancelled")], key=lambda s: s["date_start"])


def bulk_export(years: list[int], drivers: list[str]) -> None:
    jobs: list[tuple[int, str, str, str]] = []
    for year in years:
        for sess in _sessions(year):
            circuit = str(sess["circuit_short_name"]).lower().replace(" ", "-")
            for drv in drivers:
                out = Path(f"data/openf1-{year}-{circuit}-r-{drv}.csv")
                if not out.exists():
                    jobs.append((year, str(sess["country_name"]), drv, str(out)))

    print(f"Remaining exports: {len(jobs)}")
    done = errors = 0
    for year, country, drv, output_path in jobs:
        r = subprocess.run(
            [
                sys.executable,
                "-m",
                "f1_strategy.data_sources.openf1_export",
                "--year",
                str(year),
                "--event",
                country,
                "--session",
                "Race",
                "--driver",
                drv,
                "--output",
                output_path,
            ],
            capture_output=True, text=True, timeout=300,
        )
        if r.returncode == 0:
            done += 1
            print(f"  ✓ {year} {country}/{drv}")
        else:
            errors += 1
            msg = (r.stderr or r.stdout).strip().splitlines()[-1][:80] if (r.stderr or r.stdout).strip() else "?"
            print(f"  ✗ {year} {country}/{drv}: {msg}")

    total = len(list(Path("data").glob("openf1-*.csv")))
    print(f"\nDone={done}  Errors={errors}  Total openf1 files={total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk-export OpenF1 race sessions.")
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2023])
    parser.add_argument("--drivers", nargs="+", default=["VER", "HAM", "NOR", "LEC"])
    args = parser.parse_args()
    bulk_export(years=args.years, drivers=args.drivers)


if __name__ == "__main__":
    main()
