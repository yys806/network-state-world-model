"""Launch the frozen PI-JWM Formal Dataset v1 GPU training run."""
import argparse
from pathlib import Path

from pi_jwm.step5_6b_formal_runner_v1 import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--formal-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if len(args.source_git_sha) != 40 or any(c not in "0123456789abcdef" for c in args.source_git_sha):
        parser.error("--source-git-sha must be a full lowercase commit SHA")
    run(args.manifest, args.formal_config, args.run_dir, args.source_git_sha, args.resume)


if __name__ == "__main__":
    main()
