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

# Disable the auto-generated /gradio_api/info schema endpoint to dodge the
# gradio_client schema-introspection crash on certain component combos.
# (Still safe with __init__ — show_api on Blocks works in gradio 5.x.)
demo.show_api = False

if __name__ == "__main__":
    demo.launch(show_api=False)
