"""Theis is the debug file for the MASX Global Signal Generator Agentic AI"""

import logging
from src.app.workflows.orchestrator import MASXOrchestrator
from src.app.core.state import MASXState
import sys
import json

# Configure logging for debug
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")


def main():
    print("[DEBUG] MASX Global Signal Generator Agentic AI Debug Entrypoint")
    orchestrator = MASXOrchestrator()

    # Optionally load input data from a file or stdin
    input_data = {}
    if len(sys.argv) > 1:
        try:
            with open(sys.argv[1], "r", encoding="utf-8") as f:
                input_data = json.load(f)
            print(f"[DEBUG] Loaded input data from {sys.argv[1]}")
        except Exception as e:
            print(f"[ERROR] Failed to load input data: {e}")
            sys.exit(1)
    else:
        print("[DEBUG] No input file provided. Using empty/default input data.")

    # Run the daily workflow
    try:
        print("[DEBUG] Starting orchestrator.run_daily_workflow()...")
        final_state: MASXState = orchestrator.run_daily_workflow(input_data)
        print("[DEBUG] Workflow completed.")
        print("\n===== FINAL STATE =====")
        print(json.dumps(final_state.model_dump(), indent=2, default=str))
        if final_state.errors:
            print("\n===== ERRORS =====")
            for err in final_state.errors:
                print(f"[ERROR] {err}")
        else:
            print("[DEBUG] No errors detected.")
    except Exception as e:
        print(f"[FATAL] Exception during workflow: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
