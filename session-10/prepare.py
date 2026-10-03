import argparse
import json

from rag_core.config import load_settings
from rag_core.rag import build_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare the policy index")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(build_index(load_settings(), args.offline), indent=2))
    except Exception as exc:
        parser.exit(1, f"Preparation failed ({type(exc).__name__}). Check corpus, configuration, and provider access.\n")
