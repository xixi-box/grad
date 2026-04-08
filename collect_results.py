#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
import yaml

def latest_json(glob_iter):
    files = sorted(list(glob_iter), key=lambda x: x.name)
    return files[-1] if files else None

def load_json(path):
    if not path or not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def fmt(x):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments_quick_5k.yaml")
    ap.add_argument("--result-root", default="./results")
    ap.add_argument("--csv-out", default="results_summary_5k.csv")
    ap.add_argument("--md-out", default="results_summary_5k.md")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    root = Path(args.result_root)
    rows = []
    for exp in cfg["experiments"]:
        exp_dir = root / exp["name"]
        stats_dir = exp_dir / "stats"
        val_json = latest_json(stats_dir.glob("val_step*.json")) if stats_dir.exists() else None
        prune_json = latest_json(stats_dir.glob("prune_stats_*.json")) if stats_dir.exists() else None

        val = load_json(val_json)
        prune = load_json(prune_json)

        rows.append({
            "name": exp["name"],
            "description": exp.get("description", ""),
            "psnr": val.get("psnr") if val else None,
            "ssim": val.get("ssim") if val else None,
            "lpips": val.get("lpips") if val else None,
            "num_GS": val.get("num_GS") if val else None,
            "ellipse_time": val.get("ellipse_time") if val else None,
            "prune_step": prune.get("prune_step") if prune else None,
            "num_before": prune.get("num_before") if prune else None,
            "num_after": prune.get("num_after") if prune else None,
            "prune_ratio": prune.get("prune_ratio") if prune else None,
            "D_mean": prune.get("D_mean") if prune else None,
            "V_mean": prune.get("V_mean") if prune else None,
            "N_mean": prune.get("N_mean") if prune else None,
            "S_mean": prune.get("S_mean") if prune else None,
            "val_json": str(val_json) if val_json else "",
            "prune_json": str(prune_json) if prune_json else "",
        })

    if rows:
        fieldnames = list(rows[0].keys())
        with open(args.csv_out, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        headers = ["name","psnr","ssim","lpips","num_GS","num_before","num_after","prune_ratio","D_mean","V_mean","N_mean","S_mean"]
        with open(args.md_out, "w", encoding="utf-8") as f:
            f.write("| " + " | ".join(headers) + " |\n")
            f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
            for row in rows:
                f.write("| " + " | ".join(fmt(row[h]) for h in headers) + " |\n")

    print(f"[OK] CSV saved to {args.csv_out}")
    print(f"[OK] Markdown saved to {args.md_out}")

if __name__ == "__main__":
    main()
