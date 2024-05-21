#!/usr/bin/env python
import argparse
import shlex
import subprocess
import threading
import wandb
from codenamize import codenamize
from pathlib import Path


PRESETS = {
    "basic": [
        "--cpu",
        "1",
        "--gpu",
        "0.2",
        "--large-shm",
    ],
    "cpu": [
        "--cpu",
        "32",
        "--preemptible",
        "--node-type",
        "CPU",
    ],
    "cpu-small": [
        "--cpu",
        "8",
        "--preemptible",
        "--node-type",
        "CPU",
    ],
    "gpu": [
        "--cpu",
        "8",
        "--gpu",
        "1",
        "--large-shm",
        "--preemptible",
        "--node-type",
        "A100",
    ],
    "gpu-small": [
        "--cpu",
        "4",
        "--gpu",
        "0.2",
        "--large-shm",
        "--preemptible",
        "--node-type",
        "A100",
    ],
    "gpu-half": [
        "--cpu",
        "4",
        "--gpu",
        "0.5",
        "--large-shm",
        "--preemptible",
        "--node-type",
        "A100",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preset', type=str, required=True,
                        choices=PRESETS.keys(),)
    parser.add_argument('--tmux', action='store_true', help='Run the command in a tmux session')
    parser.add_argument('--job-type', type=str, )
    parser.add_argument("-e", "--env", type=str, nargs="+", help="Environment variables")
    parser.add_argument('--dry-run', action='store_true', help='Print the command instead of running it')
    parser.add_argument('--cmd', type=str, required=True, nargs=argparse.REMAINDER,
                        metavar="CMD", help='Command to run')
    args = parser.parse_args()
    return args


def get_job_type(args):
    if args.job_type:
        return args.job_type
    elif "split_data.py" in args.cmd[:2]:
        return "generate-split"
    elif "compute_medians.py" in args.cmd[:2]:
        return "compute-medians"
    elif "experiment.py" in args.cmd[:2]:
        parser = argparse.ArgumentParser()
        parser.add_argument('--job_type', type=str, default="train", required=False,
                            choices=['train', 'val'])
        args2, _ = parser.parse_known_args(args.cmd[1:])
        return args2.job_type
    elif args.cmd[:1] == ["wandb"]:
        return "cli"
    else:
        raise ValueError(f"Unknown job type")


def get_job_prefix(job_type):
    return {
        "generate-split": "split",
        "compute-medians": "med",
        "train": "t",
    }.get(job_type, job_type)


def get_job_increment():
    job_counter_file = 0
    job_counter_file_path = Path(__file__).parent / "job_counter.txt"
    if job_counter_file_path.exists():
        with open(job_counter_file_path, "r") as f:
            job_counter_file = int(f.read())

    api = wandb.Api()
    last_run = api.runs(
        path="jjurm/agri-strat",
        order="-created_at",
        per_page=1,
    )[0]
    job_counter_wandb = int(last_run.name.split("-")[-1])

    job_counter = max(job_counter_file, job_counter_wandb) + 1
    return job_counter


def write_job_increment(job_increment):
    job_counter_file_path = Path(__file__).parent / "job_counter.txt"
    with open(job_counter_file_path, "w") as f:
        f.write(str(job_increment))


def run_command(command):
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True  # This allows for text mode and consistent behavior across platforms
    )

    # Function to read from the subprocess's stdout and stderr and print it to Python's stdout/stderr
    def print_output(pipe):
        for line in iter(pipe.readline, ''):
            print(line, end='')  # Use 'end='' to prevent adding additional newline

    # Start two threads to print stdout and stderr in real-time
    stdout_thread = threading.Thread(target=print_output, args=(process.stdout,))
    stderr_thread = threading.Thread(target=print_output, args=(process.stderr,))
    stdout_thread.start()
    stderr_thread.start()

    # Forward input from Python to the subprocess
    try:
        process.wait()  # Wait for the subprocess to finish
    except KeyboardInterrupt:
        process.terminate()  # Terminate the subprocess if Ctrl+C is pressed
        process.wait()  # Wait for the subprocess to finish

    return process.returncode


def main():
    args = parse_args()

    job_type = get_job_type(args)
    job_prefix = get_job_prefix(job_type)
    job_increment = get_job_increment()
    job_codename = codenamize(
        job_increment,
        max_item_chars=5,
    )

    job_name = f"{job_prefix}-{job_codename}"
    job_name_id = f"{job_prefix}-{job_codename}-{job_increment}"

    job_command = [
        "tmux",
        "new-session",
        "-s",
        "job",
        shlex.quote(shlex.join(args.cmd)),
    ] if args.tmux else args.cmd
    runai_command = [
                        "runai",
                        "submit",
                        "--name",
                        job_name,
                        "--environment",
                        f"WANDB_NAME={job_name_id}",
                    ] + [
                        x for var in args.env or [] for x in ["--environment", var]
                    ] + [
                        "--image",
                        "jjurm/runai-job",
                        "--working-dir",
                        "/mydata/studentmichele/juraj/thesis-python/agri_strat",
                        "--backoff-limit",
                        "0",
                    ] + PRESETS[args.preset] + (
                        ["--dry-run"] if args.dry_run else []
                    ) + [
                        "--",
                    ] + job_command

    print(" ".join(runai_command))
    if not args.dry_run:
        return_code = run_command(runai_command)
        if return_code == 0:
            write_job_increment(job_increment)
            print(f"Wrote job increment {job_increment} to file")


if __name__ == '__main__':
    main()
