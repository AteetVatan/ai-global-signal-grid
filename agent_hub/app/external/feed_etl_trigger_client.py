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


#from tarfile import LinkFallbackError
#from psycopg2.errors import IdleSessionTimeout
import requests
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from itertools import islice
from collections import defaultdict
import time
import random
import asyncio
import aiohttp
from ..config.settings import get_settings
from ..services.flashpoint_db_service import FlashpointDatabaseService


class FeedETLTriggerClient:
    settings = get_settings()    
    BASE_URL = settings.etl_base_url
    ETL_BASE_URL_PREFIX=settings.etl_base_url_prefix
    ETL_BASE_URL_SUFFIX=settings.etl_base_url_suffix
    ETL_SERVICES_DEPLOYED=settings.etl_services_deployed
    
    debug_mode = settings.debug
    
    if debug_mode:
        ETL_BASE_URLS = [
            "https://masxaietlcpupipeline1-production.up.railway.app/feed/process/batch_articles",
            "https://masxaietlcpupipeline2-production.up.railway.app/feed/process/batch_articles",
            "https://masxaietlcpupipeline3-production.up.railway.app/feed/process/batch_articles",
            "https://masxaietlcpupipeline4-production.up.railway.app/feed/process/batch_articles",
            "https://masxaietlcpupipeline5-production.up.railway.app/feed/process/batch_articles",
            "https://masxaietlcpupipeline6-production.up.railway.app/feed/process/batch_articles",
        ]
    else:
        ETL_BASE_URLS = [
            "https://masxaietlcpupipeline1-production.railway.internal/feed/process/batch_articles",
            "https://masxaietlcpupipeline2-production.railway.internal/feed/process/batch_articles",
            "https://masxaietlcpupipeline3-production.railway.internal/feed/process/batch_articles",
            "https://masxaietlcpupipeline4-production.railway.internal/feed/process/batch_articles",
            "https://masxaietlcpupipeline5-production.railway.internal/feed/process/batch_articles",
            "https://masxaietlcpupipeline6-production.railway.internal/feed/process/batch_articles",
        ]
    

    END_POINT_ALL = settings.etl_end_point_all
    END_POINT_BY_ARTICLE_IDS = settings.etl_end_point_by_article_ids
    API_KEY = settings.gsg_api_key


    @staticmethod
    def trigger_feed_etl(date: Optional[datetime] = None, trigger: str = "masxai") -> dict:
        """
        Trigger MASX AI ETL Feed Process endpoint via POST.

        Args:
            date (str): The date to process.
            trigger (str): The trigger identifier.

        Returns:
            dict: Parsed JSON response or error message.
        """
        if not date:
            date = datetime.utcnow()
            
        date_str = date.strftime("%Y-%m-%d")    
            
            
        url = f"{FeedETLTriggerClient.BASE_URL}{FeedETLTriggerClient.END_POINT_ALL}"
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": FeedETLTriggerClient.API_KEY,
        }
        payload = {"date": date_str, "trigger": trigger}

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            logging.info(f"Feed triggered successfully: {payload}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Feed trigger failed: {e}")
            return {"error": str(e)}
 
        
    @staticmethod
    async def trigger_feed_etl_by_article_ids(
        date: Optional[datetime],
        flashpoint_db_service: "FlashpointDatabaseService" = None,
        trigger: str = "masxai",
    ) -> Dict[str, Any]:
        """
        Trigger MASX AI ETL Feed Process endpoint via POST,
        sending feed article IDs in batches of 50 to deployed ETL services.
        Executes only once — no retries.
        """
        try:
            if not date:
                raise ValueError("Date is required")
            
            date_str = date.strftime("%Y-%m-%d")
            logging.info(f"🚀 Starting ETL trigger for {date_str}")

            # Initialize DB service if not provided
            if not flashpoint_db_service:
                flashpoint_db_service = FlashpointDatabaseService(date, create_tables=False)
            if not flashpoint_db_service.client:
                await flashpoint_db_service.connect()
                
            
            # --- Warm-up phase ---
            async def warmup_service(session, base_url: str, retries: int = 5):
                health_url = base_url.replace("/feed/process/batch_articles", "/health")
                headers = {"X-API-Key": FeedETLTriggerClient.API_KEY}
                for attempt in range(1, retries + 1):
                    try:
                        async with session.get(health_url, headers=headers, timeout=60) as resp:
                            if resp.status == 200:
                                logging.info(f"[{base_url}] Warmup OK (attempt {attempt})")
                                return True
                            else:
                                logging.warning(f"[{base_url}] Warmup bad status {resp.status}")
                    except Exception as e:
                        logging.warning(f"[{base_url}] Warmup attempt {attempt} failed: {type(e).__name__} ({e})")
                    await asyncio.sleep(10 * attempt)
                return False
               
                
            async with aiohttp.ClientSession() as session:
                await asyncio.gather(*(warmup_service(session, url) for url in FeedETLTriggerClient.ETL_BASE_URLS))
            logging.info("Warm-up complete. Waiting 30s for replicas to boot…")
            await asyncio.sleep(30)  

            # --- Fetch all unprocessed feed entries ---
            entry_records: List[Dict[str, str]] = await flashpoint_db_service.read_feed_entry_ids_with_flashpoint(date=date)
            total_articles = len(entry_records)
            logging.info(f"Found {total_articles} articles to process")
            

            if total_articles == 0:
                return {"status": "no_records", "message": "No feed entries found"}

            # --- Extract valid IDs ---
            article_ids = [r["id"] for r in entry_records if r.get("id")]
            if not article_ids:
                logging.warning(" No valid article IDs found.")
                return {"status": "no_ids", "message": "No valid article IDs found"}

            logging.info(f" Valid article IDs: {len(article_ids)}")

            # --- Send all articles to ETL Batch Client ---
            logging.info(f"📡 Sending {len(article_ids)} article IDs to ETL services...")
            batch_result = await FeedETLTriggerClient.trigger_feed_etl_batch(date, article_ids, trigger)

            # --- Return structured result ---
            return {
                "status": "success",
                "date": date_str,
                "trigger": trigger,
                "total_articles": len(article_ids),
                "etl_result": batch_result,
            }

        except Exception as e:
            logging.error(f"trigger_feed_etl_by_article_ids() failed: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    
    
    @staticmethod
    async def trigger_feed_etl_batch(
        date: Optional[datetime],
        articles_ids: List[str],
        trigger: str = "masxai",
        max_concurrent: int = 30,          
        batch_size: int = 5,              
        per_batch_timeout: int = 3000,      
    ) -> Dict[str, Any]:
        if not date:
            raise ValueError("Date is required")
        date_str = date.strftime("%Y-%m-%d")

        trigger = ""  # no background

        def chunk_list(lst, size):
            for i in range(0, len(lst), size):
                yield lst[i:i + size]

        article_batches = list(chunk_list(articles_ids, batch_size))
        total_batches = len(article_batches)
        print(f"[INFO] Dispatching {total_batches} total batches with max {max_concurrent} concurrent requests")

        queue: asyncio.Queue = asyncio.Queue()
        retry_counter: Dict[int, int] = defaultdict(int)
        results: List[Dict[str, Any]] = []

        # preload all batches
        for idx, batch in enumerate(article_batches):
            await queue.put((idx, batch))

        # -----------------------------
        # Helper: exponential backoff
        # -----------------------------
        def compute_backoff(attempt: int) -> float:
            base = 2 ** (attempt - 1)
            jitter = random.uniform(0.5, 1.5)
            return min(base * jitter, 30)

        # -----------------------------
        # POST helper with retry count
        # -----------------------------
        async def post_payload(session, url, payload, batch_index, attempt=1):
            headers = {"X-API-Key": FeedETLTriggerClient.API_KEY}
            start = time.perf_counter()
            await asyncio.sleep(random.uniform(1, 3))  # small stagger
            try:
                async with session.post(url, json=payload, headers=headers, timeout=per_batch_timeout) as resp:
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {"text": await resp.text()}
                    elapsed = round(time.perf_counter() - start, 2)
                    status = "completed" if resp.status == 200 else "failed"

                    print(f"[{attempt}/4] Batch {batch_index+1}/{total_batches} → {status.upper()} ({resp.status}) "
                        f"in {elapsed}s | {url}")

                    return {
                        "url": url,
                        "batch_index": batch_index,
                        "status": status,
                        "status_code": resp.status,
                        "response": data,
                        "processing_time": elapsed,
                        "attempt": attempt,
                    }

            except Exception as e:
                elapsed = round(time.perf_counter() - start, 2)
                print(f"[ERROR] Batch {batch_index+1}/{total_batches} failed at attempt {attempt} "
                    f"after {elapsed}s | {type(e).__name__}: {e}")
                return {
                    "url": url,
                    "batch_index": batch_index,
                    "status": "failed",
                    "error": str(e),
                    "processing_time": elapsed,
                    "attempt": attempt,
                }

        # -----------------------------
        # Worker logic with requeue
        # -----------------------------
        async def worker(worker_id: int):
            connector = aiohttp.TCPConnector(force_close=True, ttl_dns_cache=60)
            async with aiohttp.ClientSession(connector=connector) as session:
                while True:
                    item = await queue.get()
                    if item is None:
                        queue.task_done()
                        return

                    batch_index, batch = item
                    retry_counter[batch_index] += 1
                    attempt = retry_counter[batch_index]
                    url = FeedETLTriggerClient.ETL_BASE_URLS[batch_index % len(FeedETLTriggerClient.ETL_BASE_URLS)]
                    payload = {"date": date_str, "articles_ids": batch, "trigger": trigger}

                    res = await post_payload(session, url, payload, batch_index, attempt)
                    results.append(res)

                    # Evaluate success condition
                    resp = res.get("response", {})
                    is_successful = (
                        res["status"] == "completed"
                        and isinstance(resp, dict)
                        and resp.get("status") == "completed"
                        and resp.get("successful", 0) > 2
                    )

                    if is_successful:
                        print(f"[WORKER {worker_id}] ✅ Batch {batch_index+1} succeeded on attempt {attempt}.")
                        queue.task_done()
                    else:
                        if attempt < 4:
                            backoff = compute_backoff(attempt)
                            print(f"[WORKER {worker_id}] 🔁⚠️ Batch {batch_index+1} failed (attempt {attempt}) — requeuing in {backoff:.1f}s...")
                            queue.task_done()
                            await asyncio.sleep(backoff)
                            await queue.put((batch_index, batch))  # requeue the same batch
                        else:
                            print(f"[WORKER {worker_id}] ❌ Batch {batch_index+1} failed after 4 attempts. Giving up.")
                            queue.task_done()

        # -----------------------------
        # Launch workers
        # -----------------------------
        workers = [asyncio.create_task(worker(i + 1)) for i in range(max_concurrent)]
        await queue.join()
        print("[DEBUG] All batches processed; stopping workers...")

        for _ in range(max_concurrent):
            await queue.put(None)
        await asyncio.gather(*workers, return_exceptions=True)

        successful = sum(1 for r in results if r.get("status") == "completed")
        failed = sum(1 for r in results if r.get("status") == "failed")

        summary = {
            "date": date_str,
            "total_batches": total_batches,
            "total_articles": len(articles_ids),
            "successful": successful,
            "failed": failed,
            "timestamp": datetime.utcnow().isoformat(),
            "details": results,
        }

        print(f"[SUMMARY] {successful}/{total_batches} batches completed successfully.")
        return summary






    
if __name__ == "__main__":
    result = FeedETLTriggerClient.trigger_feed_etl("2025-07-01", "masxai")
    print(result)