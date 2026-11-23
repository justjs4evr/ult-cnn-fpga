"""Webots entry point for the ULT-CNN user project."""

import sys
from pathlib import Path

USER_PROJECT = Path(__file__).resolve().parents[3] / "ULT_CNN"
sys.path.insert(0, str(USER_PROJECT))
from webots_cnn_controller import main  # noqa: E402

if __name__ == "__main__":
    main()
