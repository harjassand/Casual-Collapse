import subprocess
import sys


def run(cmd, label):
    print(f"==> {label}")
    subprocess.run(cmd, check=True)


def main() -> None:
    try:
        run(["python3", "-m", "compileall", "-q", "."], "compileall")
        run(["python3", "smoke_test.py"], "smoke_test")

        run_dir = "runs/self_check_hmm"
        run([
            "python3",
            "training/train.py",
            "env=hmm",
            "model=hmm",
            f"train.run_dir={run_dir}",
            "train.num_steps=300",
            "train.checkpoint_interval=100",
            "train.log_interval=50",
            "train.buffer_size=500",
            "train.batch_size=32",
            "train.horizon=2",
            "train.num_future_classes=4",
        ], "train_hmm")

        run([
            "python3",
            "training/rollout_eval.py",
            "env=hmm",
            "model=hmm",
            f"eval.ckpt_path={run_dir}/checkpoint.pt",
            f"eval.output_path={run_dir}/eval_metrics.json",
            "eval.steps=30",
        ], "eval_hmm")

        run([
            "python3",
            "analysis/causal_state_alignment.py",
            f"--config={run_dir}/config.json",
            f"--ckpt={run_dir}/checkpoint.pt",
            f"--output_path={run_dir}/alignment_metrics.json",
        ], "alignment_hmm")

        mech_dir = "runs/self_check_mech"
        run([
            "python3",
            "training/train.py",
            "env=mechanism",
            "model=mechanism",
            f"train.run_dir={mech_dir}",
            "train.num_steps=300",
            "train.checkpoint_interval=100",
            "train.log_interval=50",
            "train.buffer_size=500",
            "train.batch_size=32",
            "train.horizon=2",
            "train.num_future_classes=2",
        ], "train_mechanism")

        run([
            "python3",
            "analysis/one_shot_test.py",
            f"--config={mech_dir}/config.json",
            f"--ckpt={mech_dir}/checkpoint.pt",
            f"--output_path={run_dir}/one_shot_result.json",
        ], "one_shot")

        run(["python3", "tools/validate_run_dir.py", run_dir], "validate_run_dir")
    except subprocess.CalledProcessError as exc:
        print("FAIL")
        print(exc)
        sys.exit(1)

    print("PASS")


if __name__ == "__main__":
    main()
