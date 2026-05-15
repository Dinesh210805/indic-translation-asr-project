import logging
import os

from dotenv import load_dotenv

load_dotenv()  # load .env before any module imports that read env vars

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

from app.interface import build_ui

if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",  # REQUIRED inside Docker
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=False,
    )
