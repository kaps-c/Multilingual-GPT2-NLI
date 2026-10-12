"""Run the recovered core experiments independently, in source order."""

import argparse
import subprocess
import sys
from pathlib import Path

from .cli import write_json


def core_jobs():
    """PDF tasks 1–3; multilingual settings are on pages 25–32."""
    jobs = [
        ("pretrain", ["pretrain", "--epochs", "3"]),
        ("english", ["train"]),
        (
            "english-batch8",
            ["train", "--accumulation-steps", "8", "--lr", "3e-5"],
        ),
        ("english-epochs2", ["train", "--epochs", "2"]),
        ("english-lr1e5", ["train", "--lr", "1e-5"]),
    ]
    for language in ("es", "fr", "de", "sw"):
        jobs.append(
            (
                language,
                [
                    "train",
                    "--languages",
                    language,
                    "--epochs",
                    "2",
                    "--train-fraction",
                    "0.2",
                    "--dev-fraction",
                    "0.2",
                ],
            )
        )
    jobs.append(
        (
            "combined",
            [
                "train",
                "--languages",
                "en",
                "es",
                "fr",
                "de",
                "sw",
                "--epochs",
                "2",
                "--train-fraction",
                "0.2",
                "--dev-fraction",
                "0.2",
            ],
        )
    )
    return jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=["small", "full"], default="small"
    )
    parser.add_argument(
        "--only", nargs="+", choices=[name for name, _ in core_jobs()]
    )
    parser.add_argument("--data", default="data/xnli")
    parser.add_argument("--model", default="artifacts/gpt2")
    parser.add_argument("--output", default="results/experiments")
    parser.add_argument("--checkpoints", default="checkpoints/experiments")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    output = Path(args.output) / args.profile
    checkpoints = Path(args.checkpoints) / args.profile
    status = {}
    for name, options in core_jobs():
        if args.only and name not in args.only:
            continue
        target = output / name
        checkpoint = checkpoints / f"{name}.pt"
        if (target / "results.json").exists() and checkpoint.exists():
            status[name] = "existing_completed_run"
            continue
        command = [
            sys.executable,
            "-m",
            "multilingual_gpt2",
            options[0],
            "--data",
            args.data,
            "--model",
            args.model,
            "--device",
            args.device,
            "--batch-size",
            "1",
            "--accumulation-steps",
            "4",
            "--output",
            str(target),
            "--checkpoint",
            str(checkpoint),
            "--max-length",
            "128"
            if name == "pretrain"
            else "1024"
            if name.startswith("english")
            else "512",
            *options[1:],
        ]
        if args.device == "cuda":
            command.append("--mixed-precision")
        if args.profile == "small":
            command.extend(
                [
                    "--epochs",
                    "1",
                    "--train-limit",
                    "32",
                    "--dev-limit",
                    "12",
                    "--test-limit",
                    "12",
                    "--max-length",
                    "64",
                ]
            )
        target.mkdir(parents=True, exist_ok=True)
        write_json(target / "command.json", command)
        status[name] = "running"
        write_json(output / "status.json", status)
        with (target / "run.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT
            )
        status[name] = "completed" if result.returncode == 0 else "failed"
        write_json(output / "status.json", status)
        print(f"{name}: {status[name]}", flush=True)
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {target / 'run.log'}")


if __name__ == "__main__":
    main()
