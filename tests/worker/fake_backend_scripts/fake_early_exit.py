import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exit-code", type=int, default=17)
    parser.add_argument("--message", default="fake backend exited early")
    return parser.parse_args()


def main():
    args = parse_args()
    sys.stderr.write(args.message + "\n")
    sys.stderr.flush()
    raise SystemExit(args.exit_code)


if __name__ == "__main__":
    main()
