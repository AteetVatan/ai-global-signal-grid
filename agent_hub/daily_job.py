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

# scheduler.py
from datetime import datetime
import logging
import time

from app.workflows.orchestrator import MASXOrchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def run_gsg_workflow():
    utc_now = datetime.utcnow().isoformat()
    logging.info(f"[{utc_now} UTC] Triggering MASX GSG workflow")

    try:
        MASXOrchestrator().run_daily_workflow({})
        logging.info("MASX GSG workflow completed successfully")
    except Exception as e:
        logging.error(f"MASX GSG workflow failed: {e}", exc_info=True)
        print(f"[FATAL] Exception during workflow: {e}")

        # Retry the workflow after 5 minutes
        time.sleep(300)
        #retry once in case of error
        try:
            MASXOrchestrator().run_daily_workflow({})
            logging.info("MASX GSG workflow completed successfully")
        except Exception as retry_error:
            logging.error(f"MASX GSG workflow failed: {retry_error}", exc_info=True)
            print(f"[FATAL] Exception during workflow: {retry_error}")
            
if __name__ == "__main__":
    run_gsg_workflow()