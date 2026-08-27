from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serving.app import serve  # noqa: E402
from serving.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the public/offline local recommendation service")
    parser.add_argument("--config", default=str(ROOT / "configs" / "serving-local.json"))
    arguments = parser.parse_args()
    serve(load_config(arguments.config))


if __name__ == "__main__":
    main()
