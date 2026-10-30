"""Manual M0 five-head/own-weight reconstruction; no mail, orders or accounts."""
import argparse
from pathlib import Path

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--archive", type=Path, help="Original read-only native M1 frozen_bundle.json")
    modes.add_argument("--verify", type=Path, help="Previously published M0 model folder")
    parser.add_argument("--output-root", type=Path, help="New crawler research area, never original archive")
    args = parser.parse_args(argv)
    try:
        from domain.les_v7.contracts import canonical
        from infrastructure.les_v7_fr2_m0 import load_m0_bundle, publish_m0
        if args.verify:
            if args.output_root:
                raise ValueError("FR2_M0_VERIFY_REQUIRES_NO_OUTPUT")
            _, result = load_m0_bundle(args.verify)
        else:
            if args.output_root is None:
                raise ValueError("FR2_M0_OUTPUT_ROOT_REQUIRED")
            result = publish_m0(args.archive, args.output_root)
        print(canonical(result))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
