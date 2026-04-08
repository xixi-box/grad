#!/usr/bin/env python3
import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import yaml

def fmt_steps(values):
    return [str(int(v)) for v in values]

def build_cmd(common, exp):
    trainer = common["trainer_script"]
    cmd = [sys.executable, trainer, common.get("subcommand", "default")]

    result_dir = Path(common["result_root"]) / exp["name"]

    cmd += ["--data-dir", common["data_dir"]]
    cmd += ["--result-dir", str(result_dir)]
    cmd += ["--data-factor", str(common["data_factor"])]
    cmd += ["--strategy.refine-stop-iter", str(common["refine_stop_iter"])]
    cmd += ["--max-steps", str(common["max_steps"])]
    cmd += ["--eval-steps", *fmt_steps(common["eval_steps"])]
    cmd += ["--save-steps", *fmt_steps(common["save_steps"])]
    cmd += ["--ply-steps", *fmt_steps(common["ply_steps"])]
    cmd += ["--opacity-reg", str(common.get("opacity_reg", 0.01))]

    if common.get("disable_viewer", False):
        cmd += ["--disable-viewer"]
    if common.get("disable_video", False):
        cmd += ["--disable-video"]
    if common.get("disable_tb", False):
        cmd += ["--disable-tb"]
    if common.get("save_eval_images", False):
        cmd += ["--save-eval-images"]

    if exp.get("depth_loss", False):
        cmd += ["--depth-loss"]
        cmd += ["--dense-depth-dir", common["dense_depth_dir"]]
        cmd += ["--depth-lambda", str(exp.get("depth_lambda", 0.01))]

    if exp.get("enable_prune", False):
        cmd += ["--enable-prune"]
        cmd += ["--dense-depth-dir", common["dense_depth_dir"]]
        cmd += ["--prune-step-ratio", str(common.get("prune_step_ratio", 0.7))]
        cmd += ["--prune-tau", str(exp["prune_tau"])]
        cmd += ["--prune-lambda-d", str(exp["prune_lambda_d"])]
        cmd += ["--prune-lambda-v", str(exp["prune_lambda_v"])]
        cmd += ["--prune-lambda-n", str(exp["prune_lambda_n"])]

    for item in exp.get("extra_args", []):
        cmd.append(str(item))

    return cmd, result_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="experiments_quick_5k.yaml")
    ap.add_argument("--only", nargs="*", default=None, help="只跑指定实验名")
    ap.add_argument("--overwrite-results", action="store_true", help="删除同名结果目录后重跑")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    common = cfg["common"]
    exps = cfg["experiments"]
    if args.only:
        only = set(args.only)
        exps = [e for e in exps if e["name"] in only]

    if not exps:
        print("No experiments selected.")
        return 1

    for exp in exps:
        cmd, result_dir = build_cmd(common, exp)

        if result_dir.exists() and args.overwrite_results:
            shutil.rmtree(result_dir)

        log_dir = result_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "run.log"

        if result_dir.exists() and not args.overwrite_results:
            stats_dir = result_dir / "stats"
            existing = list(stats_dir.glob("val_step*.json")) if stats_dir.exists() else []
            if existing:
                print(f"[SKIP] {exp['name']} already has results: {result_dir}")
                continue

        print("=" * 100)
        print(f"[RUN] {exp['name']} - {exp.get('description','')}")
        print(" ".join(cmd))
        print(f"[OUT] {result_dir}")
        print(f"[LOG] {log_file}")
        print("=" * 100)

        if args.dry_run:
            continue

        with log_file.open("w", encoding="utf-8") as f:
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            print(f"[FAIL] {exp['name']} failed. Check: {log_file}")
            return proc.returncode
        print(f"[OK] {exp['name']} finished.")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
