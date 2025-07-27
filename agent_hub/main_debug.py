"""Theis is the debug file for the MASX Global Signal Generator Agentic AI"""

# ┌───────────────────────────────────────────────────────────────┐
# │  Copyright (c) 2025 Ateet Vatan Bahmani                       │
# │  Project: MASX AI – Strategic Agentic AI System               │
# │  All rights reserved.                                         │
# └───────────────────────────────────────────────────────────────┘
#
# MASX AI is a proprietary software system developed and owned by Ateet Vatan Bahmani.
# The source code, documentation, workflows, designs, and naming (including "MASX AI")
# are protected by applicable copyright and trademark laws.
#
# Redistribution, modification, commercial use, or publication of any portion of this
# project without explicit written consent is strictly prohibited.
#
# This project is not open-source and is intended solely for internal, research,
# or demonstration use by the author.
#
# Contact: ab@masxai.com | MASXAI.com

import logging
from app.workflows.orchestrator import MASXOrchestrator
from app.core.state import MASXState
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
    except Exception as e:
        print(f"[FATAL] Exception during workflow: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
