import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from rag_core.config import load_settings
from workflow import run_workflow

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    with TemporaryDirectory() as folder:
        settings = load_settings().model_copy(update={"index_dir": Path(folder) / "missing-index"})
        print(json.dumps(run_workflow("What is the refund policy?", settings, offline=True), indent=2))
