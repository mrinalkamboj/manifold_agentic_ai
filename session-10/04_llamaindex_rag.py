import argparse
import json

from llama_comparison import run_native
from rag_core.config import load_settings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: native LlamaIndex comparison")
    parser.add_argument("question")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--search-only", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(run_native(args.question, load_settings(), args.offline, args.search_only), indent=2, ensure_ascii=False))
    except Exception as exc:
        parser.exit(1, f"LlamaIndex comparison failed ({type(exc).__name__}). Check preparation, provider access, and the tested dependency lock.\n")
