import argparse
from pathlib import Path
import subprocess
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child-pid-file", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    child_script = Path(__file__).with_name("fake_process_tree_child.py")
    child = subprocess.Popen([sys.executable, str(child_script)])
    Path(args.child_pid_file).write_text(f"{child.pid}\n", encoding="utf-8")

    try:
        while True:
            time.sleep(1)
    finally:
        child.terminate()


if __name__ == "__main__":
    main()
