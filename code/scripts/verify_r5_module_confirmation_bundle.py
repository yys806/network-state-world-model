"""Verify a downloaded PI-JWM R5 module-confirmation result bundle."""

from __future__ import annotations

import argparse
import json

from run_r5_module_confirmation_training import verify_downloaded_bundle, verify_smoke_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    verifier = verify_smoke_bundle if args.smoke else verify_downloaded_bundle
    print(
        json.dumps(
            verifier(args.root),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
