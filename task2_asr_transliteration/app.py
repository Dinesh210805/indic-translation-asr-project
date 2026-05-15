"""HuggingFace Spaces entry point.

Spaces expects app.py at the repo root and manages routing itself —
do NOT pass server_name or server_port here.
"""
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

from app.interface import build_ui

demo = build_ui()
demo.queue()

if __name__ == "__main__":
    demo.launch()
