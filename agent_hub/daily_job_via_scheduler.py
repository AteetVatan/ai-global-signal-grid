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

import logging
import time
import signal
import sys
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.date import DateTrigger

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

def graceful_shutdown(signum, frame):
    logging.info("Shutdown signal received. Stopping scheduler.")
    scheduler.shutdown(wait=False)
    sys.exit(0)


if __name__ == "__main__":
    jobstores = {"default": SQLAlchemyJobStore(url="sqlite:///jobs.db")}
    scheduler = BackgroundScheduler(jobstores=jobstores, timezone="UTC")

    # Daily cron job: UTC midnight
    scheduler.add_job(
        run_gsg_workflow,
        trigger='cron',
        hour=0,
        minute=0,
        id='masx_gsg_daily',
        replace_existing=True
    )

    # One-time test job one minute from now
    scheduler.add_job(
        run_gsg_workflow,
        trigger="date",
        run_date=datetime.utcnow() + timedelta(minutes=1),
        id="masx_gsg_test_once",
        replace_existing=True,
    )

    scheduler.start()
    logging.info("Scheduler started; daily job set for midnight UTC.")

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        graceful_shutdown(None, None)
