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
    
    
    ETL_BASE_URLS = [
        "https://masxaietlcpupipeline1-production.up.railway.app/feed/process/batch_articles",
        "https://masxaietlcpupipeline2-production.up.railway.app/feed/process/batch_articles",
        "https://masxaietlcpupipeline3-production.up.railway.app/feed/process/batch_articles",
        "https://masxaietlcpupipeline4-production.up.railway.app/feed/process/batch_articles",
        "https://masxaietlcpupipeline5-production.up.railway.app/feed/process/batch_articles",
        "https://masxaietlcpupipeline6-production.up.railway.app/feed/process/batch_articles",
    ]
    

    END_POINT_ALL = settings.etl_end_point_all
    END_POINT_BY_ARTICLE_IDS = settings.etl_end_point_by_article_ids
    API_KEY = settings.gsg_api_key
#read_daily_feed_entry_ids

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
                flashpoint_db_service = FlashpointDatabaseService(date)

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
        max_concurrent: int = 20,          # 20 total active workers (4 services × 5 replicas)
        batch_size: int = 100,               # each batch ~1–2 min
        per_batch_timeout: int = 300,      # 5 min hard timeout
    ) -> Dict[str, Any]:
        if not date:
            raise ValueError("Date is required")
        date_str = date.strftime("%Y-%m-%d")

        # Split into batches of N
        def chunk_list(lst, size):
            for i in range(0, len(lst), size):
                yield lst[i:i + size]

        article_batches = list(chunk_list(articles_ids, batch_size))
        total_batches = len(article_batches)
        results: List[Dict[str, Any]] = []
        queue: asyncio.Queue = asyncio.Queue()
        for idx, batch in enumerate(article_batches):
            await queue.put((idx, batch))

        # retry helper
        def next_backoff(attempt: int):
            return min(1.5 ** attempt + random.uniform(0, 1.5), 15)

        async def post_payload(session, url, payload, batch_index, attempt=1):
            headers = {"X-API-Key": FeedETLTriggerClient.API_KEY}
            start = time.perf_counter()            
            try:
                #await asyncio.sleep(random.uniform(0, 1))
                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=3600,
                ) as resp:
                    try:
                        data = await resp.json()
                    except Exception:
                        data = {"text": await resp.text()}
                    elapsed = round(time.perf_counter() - start, 3)
                    status = "completed" if resp.status == 200 else "failed"
                    print(f"[INFO] Batch {batch_index+1}/{total_batches} → {status.upper()} "
                          f"({resp.status}) in {elapsed}s | {url}")
                    return {
                        "url": url,
                        "batch_index": batch_index,
                        "status": status,
                        "status_code": resp.status,
                        "response": data,
                        "processing_time": elapsed,
                    }
            except Exception as e:
                if attempt < 3:
                    wait = next_backoff(attempt)
                    print(f"[WARN] {url} retry {attempt+1} for batch {batch_index+1}: {e} "
                          f"(sleep {wait:.1f}s)")
                    await asyncio.sleep(wait)
                    return await post_payload(session, url, payload, batch_index, attempt + 1)
                print(f"[ERROR] {url} failed after 3 retries: {e}")
                return {
                    "url": url,
                    "batch_index": batch_index,
                    "status": "failed",
                    "error": str(e),
                }

        async def worker(worker_id: int):
            connector = aiohttp.TCPConnector(force_close=True, ttl_dns_cache=60)
            async with aiohttp.ClientSession(connector=connector) as session:
                while True:
                    item = await queue.get()
                    try:
                        if item is None:
                            return
                        batch_index, batch = item
                        # Select service URL in round-robin manner
                        url = FeedETLTriggerClient.ETL_BASE_URLS[batch_index % len(FeedETLTriggerClient.ETL_BASE_URLS)]
                        payload = {"date": date_str, "articles_ids": batch, "trigger": trigger}

                        try:
                            res = await asyncio.wait_for(
                                post_payload(session, url, payload, batch_index),
                                timeout=per_batch_timeout,
                            )
                        except asyncio.TimeoutError:
                            print(f"[TIMEOUT] Batch {batch_index+1} > {per_batch_timeout}s. Marking failed.")
                            res = {
                                "url": url,
                                "batch_index": batch_index,
                                "status": "failed",
                                "error": f"Hard timeout ({per_batch_timeout}s)",
                            }
                        results.append(res)
                    finally:
                        queue.task_done()

        # Launch 20 workers
        workers = [asyncio.create_task(worker(i + 1)) for i in range(max_concurrent)]

        print(f"[DEBUG] Dispatching {total_batches} batches across 4 services × 5 replicas...")
        await queue.join()
        print("[DEBUG] All batches processed; stopping workers...")

        for _ in workers:
            await queue.put(None)
        await asyncio.gather(*workers)

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