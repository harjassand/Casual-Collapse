import subprocess
import sys


def run(cmd, label):
    print(f"==> {label}")
    subprocess.run(cmd, check=True)


def main() -> None:
    try:
        run(["python3", "-m", "compileall", "-q", "."], "compileall")
        run(["python3", "smoke_test.py"], "smoke_test")

        runs = [
            {
                "name": "hmm",
                "env": "hmm",
                "model": "hmm",
                "num_future_classes": 4,
                "run_dir": "runs/self_check_hmm",
            },
            {
                "name": "object",
                "env": "object",
                "model": "object",
                "num_future_classes": 2,
                "run_dir": "runs/self_check_object",
            },
            {
                "name": "mechanism",
                "env": "mechanism",
                "model": "mechanism",
                "num_future_classes": 2,
                "run_dir": "runs/self_check_mechanism",
            },
        ]

        for entry in runs:
            run([
                "python3",
                "training/train.py",
                f"env={entry['env']}",
                f"model={entry['model']}",
                f"train.run_dir={entry['run_dir']}",
                "train.num_steps=300",
                "train.checkpoint_interval=100",
                "train.log_interval=50",
                "train.buffer_size=500",
                "train.batch_size=32",
                "train.horizon=2",
                f"train.num_future_classes={entry['num_future_classes']}",
            ], f"train_{entry['name']}")

            run([
                "python3",
                "training/rollout_eval.py",
                f"env={entry['env']}",
                f"model={entry['model']}",
                f"eval.ckpt_path={entry['run_dir']}/checkpoint.pt",
                f"eval.output_path={entry['run_dir']}/eval_metrics.json",
                "eval.steps=30",
            ], f"eval_{entry['name']}")

            run([
                "python3",
                "analysis/causal_state_alignment.py",
                f"--config={entry['run_dir']}/config.json",
                f"--ckpt={entry['run_dir']}/checkpoint.pt",
                f"--output_path={entry['run_dir']}/alignment_metrics.json",
            ], f"alignment_{entry['name']}")

            run([
                "python3",
                "analysis/one_shot_test.py",
                f"--config={entry['run_dir']}/config.json",
                f"--ckpt={entry['run_dir']}/checkpoint.pt",
                f"--output_path={entry['run_dir']}/one_shot_result.json",
            ], f"one_shot_{entry['name']}")

            run(["python3", "validate_run_dir.py", entry["run_dir"]], f"validate_{entry['name']}")
    except subprocess.CalledProcessError as exc:
        print("FAIL")
        print(exc)
        sys.exit(1)

    print("PASS")


if __name__ == "__main__":
    main()
