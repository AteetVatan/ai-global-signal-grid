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
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, asdict
import re
import json
import asyncpg
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
import psycopg2
from contextlib import closing

from ..core.utils import measure_execution_time
from ..config.logging_config import get_service_logger
from ..config.settings import get_settings
from ..core.exceptions import DatabaseException, ConfigurationException
from itertools import islice

# Replace with your actual database client imports
# from your_project.database import supabase_client, pg_pool
BATCH_SIZE = 100


@dataclass
class FlashpointRecord:
    title: str
    description: str
    entities: Optional[List[str]] = None
    domains: Optional[List[str]] = None
    run_id: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class FeedRecord:
    flashpoint_id: str
    url: str
    title: str
    seendate: Optional[str] = None
    domain: Optional[str] = None
    language: Optional[str] = None
    sourcecountry: Optional[str] = None
    description: Optional[str] = None
    image: Optional[str] = None
    title_en: Optional[str] = None #new
    images: Optional[List[str]] = None #new
    hostname: Optional[str] = None #new
    content: Optional[str] = None #new
    compressed_content: Optional[str] = None #new
    summary: Optional[str] = None #new
    entities: Optional[List[str]] = None #new
    geo_entities: Optional[List[str]] = None #new    
 


class FlashpointDatabaseService:
    
    FLASHPOINT_TABLE_PREFIX = "flash_point"
    FEED_TABLE_PREFIX = "feed_entries"

    
    def __init__(self, date: Optional[datetime] = None, create_tables: bool = True):
        self.date = date or datetime.utcnow()
        self.settings = get_settings()
        self.logger = get_service_logger("FlashpointDatabaseService")
        self.client: Optional[Client] = None  # Supabase client
        self.pool: Optional[asyncpg.Pool] = None  # asyncpg pool
        self._connection_params = {}
        self._initialize_connection()        
        if create_tables:
            self._create_required_tables()       

    def _initialize_connection(self):
        """Initialize database connection parameters."""
        try:
            # Supabase configuration
            if not self.settings.supabase_url or not self.settings.supabase_anon_key:
                raise ConfigurationException("Supabase URL and key are required")

            # Database Configuration (Supabase)
            # SUPABASE_URL=https://your-project.supabase.co
            # SUPABASE_ANON_KEY=your_supabase_anon_key_here
            # SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key_here
            # SUPABASE_DB_PASSWORD=your_database_password_here
            # SUPABASE_DB_URL=your_database_connection_url

            self._connection_params = {
                "supabase_url": self.settings.supabase_url,
                "supabase_key": self.settings.supabase_anon_key,
                "database_url": self.settings.supabase_db_url,
                "max_connections": self.settings.database_max_connections or 10,
                "min_connections": self.settings.database_min_connections or 1,
            }

            # test connection
            # import asyncio
            # asyncio.run(self.connect())
            # asyncio.run(self.disconnect())
            self.logger.info("Database connection parameters initialized")

        except Exception as e:
            self.logger.error(f"Failed to initialize database connection: {e}")
            raise DatabaseException(f"Database initialization failed: {str(e)}")
        
    def _create_required_tables(self):
        """Create required tables."""
        try:            
            #get daliy table name
            flashpoint_table_name = self.get_daily_table_name(self.FLASHPOINT_TABLE_PREFIX)            
            #flashpoint_table_name ='flash_point_20250807'
            self.ensure_flashpoint_table_exists(flashpoint_table_name)            
            feeds_table_name = self.get_daily_table_name(self.FEED_TABLE_PREFIX)
            #feeds_table_name ='feed_entries_20250807'
            self.ensure_feed_table_exists(feeds_table_name, flashpoint_table_name)
        except Exception as e:
            self.logger.error(f"Failed to create required tables: {e}")
            raise DatabaseException(f"Table creation failed: {str(e)}")

    async def connect(self):
        """Establish database connections."""
        try:
            # Initialize Supabase client
            options = ClientOptions(
                schema="public", headers={"X-Client-Info": "masx-ai-system"}
            )

            self.client = create_client(
                self._connection_params["supabase_url"],
                self._connection_params["supabase_key"],
                options=options,
            )

            # Validate and establish PostgreSQL pool
            database_url = self._connection_params.get("database_url")
            if database_url:
                if not self.is_valid_postgres_url(database_url):
                    raise ConfigurationException(
                        f"Invalid PostgreSQL connection URL: {database_url}"
                    )

                # Initialize connection pool for direct PostgreSQL access
                self.pool = await asyncpg.create_pool(
                    database_url,
                    min_size=self._connection_params["min_connections"],
                    max_size=self._connection_params["max_connections"],
                    command_timeout=60,
                    server_settings={"application_name": "masx_ai_system"},
                )

            self.logger.info("Database connections established successfully")

        except Exception as e:
            self.logger.error(f"Failed to establish database connections: {e}")
            raise DatabaseException(f"Database connection failed: {str(e)}")

    async def disconnect(self):
        """Close database connections."""
        try:
            if self.pool:
                await self.pool.close()
                self.pool = None

            if self.client:
                self.client = None

            self.logger.info("Database connections closed")

        except Exception as e:
            self.logger.error(f"Error closing database connections: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    def get_daily_table_name(self, base: str, date: Optional[datetime] = None) -> str:
        date = date or self.date
        return f"{base}_{date.strftime('%Y%m%d')}"
    
    def ensure_flashpoint_table_exists(self, table_name: str) -> None:
        """
        Ensure that the daily flashpoint table exists in the database.

        Args:
            table_name (str): Name of the flashpoint table to create
        """
        conn_str = self._connection_params["database_url"]

        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title TEXT NOT NULL,
            description TEXT NOT NULL,
            entities JSONB,
            domains JSONB,
            run_id TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """
        rls_policies = self.__get_all_rls_policies_cmd(table_name)
        with measure_execution_time(f"ensure_table:{table_name}"):
            try:
                with closing(psycopg2.connect(conn_str)) as conn:
                    with conn.cursor() as cur:
                        cur.execute(create_table_query)
                        for policy in rls_policies:
                            cur.execute(policy)
                    conn.commit()

                self.logger.info(f"Table ensured: {table_name}")

            except Exception as e:
                self.logger.error(f"Failed to ensure table {table_name}: {e}")
                raise DatabaseException(f"Table creation failed: {str(e)}")


    async def ensure_flashpoint_table_exists_async(self, table_name: str) -> None:
        """
        Ensure that the daily flashpoint table exists in the database.

        Args:
            table_name (str): Name of the table to create
        """
        with measure_execution_time(f"ensure_table:{table_name}"):
            try:
                if not self.pool:
                    raise DatabaseException("PostgreSQL pool not initialized")

                query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    entities JSONB,
                    domains JSONB,
                    run_id TEXT,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                """              
                rls_policies = self.__get_all_rls_policies_cmd(table_name)
                async with self.pool.acquire() as conn:
                    await conn.execute(query)                  
                    for policy in rls_policies:
                        await conn.execute(policy)
                    self.logger.info(f"Table ensured: {table_name}")

            except Exception as e:
                self.logger.error(f"Failed to ensure table {table_name}: {e}")
                raise DatabaseException(f"Table creation failed: {str(e)}")

    def ensure_feed_table_exists(self, feed_table: str, flashpoint_table: str):
        conn_str = self._connection_params["database_url"]

        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {feed_table} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            flashpoint_id UUID,
            url TEXT,
            title TEXT,
            seendate TEXT,
            domain TEXT,
            language TEXT,
            sourcecountry TEXT,
            description TEXT,
            image TEXT,
            title_en TEXT,
            images TEXT[] DEFAULT '{{}}',
            hostname TEXT,
            content TEXT,
            compressed_content TEXT,
            summary TEXT,
            entities JSONB,
            geo_entities JSONB,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """

        create_index_query = f"""
        CREATE INDEX IF NOT EXISTS idx_{feed_table}_flashpoint_id ON {feed_table}(flashpoint_id);
        """

        alter_fk_if_needed_query = f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM information_schema.table_constraints
                WHERE constraint_type = 'FOREIGN KEY'
                AND table_name = '{feed_table}'
                AND constraint_name = 'fk_flashpoint'
            ) THEN
                ALTER TABLE {feed_table}
                ADD CONSTRAINT fk_flashpoint
                FOREIGN KEY (flashpoint_id)
                REFERENCES {flashpoint_table}(id)
                ON DELETE CASCADE;
            END IF;
        END
        $$;
        """
        rls_policies = self.__get_all_rls_policies_cmd(feed_table)
        with closing(psycopg2.connect(conn_str)) as conn:
            with conn.cursor() as cur:
                cur.execute(create_table_query)
                cur.execute(create_index_query)
                cur.execute(alter_fk_if_needed_query)
                for policy in rls_policies:
                    cur.execute(policy)
            conn.commit()

    async def ensure_feed_table_exists_async(
        self, feed_table: str, flashpoint_table: str
    ):
        """
        Ensures the feed table for the given day exists.
        Adds foreign key index on flashpoint_id and required RLS policies.
        """
        with measure_execution_time(f"ensure_feed_table_exists_async: {feed_table}"):
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {feed_table} (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                flashpoint_id UUID,
                url TEXT,
                title TEXT,
                seendate TEXT,
                domain TEXT,
                language TEXT,
                sourcecountry TEXT,
                description TEXT,
                image TEXT,
                title_en TEXT,
                images TEXT[] DEFAULT '{{}}',
                hostname TEXT,
                content TEXT,
                compressed_content TEXT,
                summary TEXT,
                entities JSONB,
                geo_entities JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
            """

            create_index_query = f"""
            CREATE INDEX IF NOT EXISTS idx_{feed_table}_flashpoint_id ON {feed_table}(flashpoint_id);
            """

            # Safe FK check: only add if it doesn't already exist
            alter_fk_if_needed_query = f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM information_schema.table_constraints
                    WHERE constraint_type = 'FOREIGN KEY'
                    AND table_name = '{feed_table}'
                    AND constraint_name = 'fk_flashpoint'
                ) THEN
                    ALTER TABLE {feed_table}
                    ADD CONSTRAINT fk_flashpoint
                    FOREIGN KEY (flashpoint_id)
                    REFERENCES {flashpoint_table}(id)
                    ON DELETE CASCADE;
                END IF;
            END
            $$;
            """

            rls_policies = self.__get_all_rls_policies_cmd(feed_table)

            async with self.pool.acquire() as conn:
                await conn.execute(create_table_query)
                await conn.execute(create_index_query)
                await conn.execute(alter_fk_if_needed_query)
                for policy in rls_policies:
                    await conn.execute(policy)

            self.logger.info(f"Feed table ensured: {feed_table}")
       
        

    async def store_flashpoint(self, fp: FlashpointRecord, run_id: str) -> str:
        """
        Stores a flashpoint record in the Supabase daily flashpoint table.

        Args:
            fp (FlashpointItem): Flashpoint item to store.
            run_id (str): Identifier of the ETL run.

        Returns:
            str: ID of the stored flashpoint.
        """
        with measure_execution_time("store_flashpoint"):
            try:
                if not self.client:
                    raise DatabaseException("Supabase client not connected")

                # Step 1: Determine daily table name
                table_name = self.get_daily_table_name(self.FLASHPOINT_TABLE_PREFIX)

                # Step 2: Ensure the table exists (Postgres side)
                await self.ensure_flashpoint_table_exists_async(table_name)
                now = datetime.utcnow().isoformat()
                # Step 3: Convert to record and clean up
                record = FlashpointRecord(
                    title=fp.title.strip(),
                    description=fp.description.strip(),
                    entities=fp.entities or [],
                    domains=fp.domains or [],
                    run_id=run_id or "unknown_run",
                    created_at=now,
                    updated_at=now,
                )
                data = {k: v for k, v in asdict(record).items() if v is not None}

                # Fix types: Supabase often needs JSON-safe formats
                if isinstance(data["entities"], list):
                    data["entities"] = json.dumps(data["entities"])
                if isinstance(data["domains"], list):
                    data["domains"] = json.dumps(data["domains"])

                self.logger.debug(
                    f"[store_flashpoint] Payload for {table_name}: {json.dumps(data, indent=2)}"
                )
                # Step 4: Insert using Supabase client
                
                result = self.client.table(table_name).insert(data).execute()

                if result.data:
                    flashpoint_id = result.data[0].get("id")
                    self.logger.info(
                        f"[store_flashpoint] Stored in {table_name} → ID: {flashpoint_id}"
                    )
                    return flashpoint_id or "unknown_id"
                else:
                    self.logger.error(f"[store_flashpoint] No data returned: {result}")
                    raise DatabaseException("No data returned from flashpoint insert")

            except Exception as e:
                self.logger.error(f"store_flashpoint() failed: {e}")
                raise DatabaseException(f"store_flashpoint failed: {str(e)}")

    async def store_feed_entries(self, flashpoint_id: str, feeds: List) -> int:
        try:
            if not self.client:
                raise DatabaseException("Supabase client not connected")


            feed_table = self.get_daily_table_name(self.FEED_TABLE_PREFIX)
            flashpoint_table = self.get_daily_table_name(self.FLASHPOINT_TABLE_PREFIX)
            await self.ensure_feed_table_exists_async(feed_table, flashpoint_table)

            # Run sync DDL
            # self.ensure_feed_table_exists(feed_table, flashpoint_table)

            def batch_iterator(iterable, size):
                it = iter(iterable)
                while True:
                    batch = list(islice(it, size))
                    if not batch:
                        break
                    yield batch

            payload = []
            for feed in feeds:
                record = FeedRecord(
                    flashpoint_id=flashpoint_id,
                    url=feed.url,
                    title=feed.title,
                    seendate=feed.seendate,
                    domain=feed.domain,
                    language=feed.language,
                    sourcecountry=feed.sourcecountry,
                    description=feed.description,
                    image=feed.image,
                )
                data = {k: v for k, v in asdict(record).items() if v is not None}
                payload.append(data)

            # from asyncio import to_thread
            # def insert_batch(feed_table, client, batch):
            #     return client.table(feed_table).insert(batch).execute()

            total_inserted = 0
            # for batch in batch_iterator(payload, BATCH_SIZE):
            #     result = await to_thread(insert_batch, feed_table, self.client, batch)
            #     #result = await to_thread(lambda: insert_func().execute())
            #     inserted = len(result.data or [])
            #     total_inserted += inserted

            for batch in batch_iterator(payload, BATCH_SIZE):
                try:
                    self.logger.info(f"Inserting batch of size: {len(batch)}")
                    result = self.client.table(feed_table).insert(batch).execute()

                    # Log errors clearly
                    if not result.data:
                        self.logger.warning(
                            f"[Insert Warning] No data inserted into {feed_table}"
                        )
                    else:
                        inserted = len(result.data)
                        total_inserted += inserted
                        self.logger.info(
                            f"[Insert Success] {inserted} entries added to {feed_table}"
                        )

                except Exception as e:
                    self.logger.exception(f"[Insert Exception] {e}")

            return total_inserted

        except Exception as e:
            self.logger.error(f"store_feed_entries() failed: {e}")
            raise DatabaseException(str(e))
        
    async def read_feed_entry_ids_with_flashpoint(self, date: Optional[datetime] = None) -> List[dict]:
        """
        Reads 'id' and 'flashpoint_id' columns from today's feed table, handling >1000 rows via pagination.

        Returns:
            List[dict]: A list of records, each containing {'id': ..., 'flashpoint_id': ...}
        """
        try:
            if not self.client:
                raise DatabaseException("Supabase client not connected")

            feed_table = self.get_daily_table_name(self.FEED_TABLE_PREFIX, date)
            self.logger.info(f"Reading feed entry IDs from: {feed_table}")

            records: List[dict] = []
            batch_size = 1000
            offset = 0

            while True:
                # Paginated query
                result = (
                    self.client.table(feed_table)
                    .select("id, flashpoint_id")                   
                    .or_("content.is.null,content.eq.''") #only get records that are not processed
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )

                data = result.data or []
                if not data:
                    break

                for row in data:
                    if row.get("id") and row.get("flashpoint_id"):
                        records.append({
                            "id": row["id"],
                            "flashpoint_id": row["flashpoint_id"]
                        })

                self.logger.info(f"Fetched batch: {len(data)} (offset {offset})")

                # Stop when last batch smaller than full page
                if len(data) < batch_size:
                    break

                offset += batch_size

            self.logger.info(f"Total retrieved {len(records)} records from {feed_table}")
            return records

        except Exception as e:
            self.logger.error(f"read_feed_entry_ids_with_flashpoint() failed: {e}")
            raise DatabaseException(str(e))
   
        

    async def drop_daily_tables(self) -> None:
        """
        Drops the daily flashpoint and feed tables for the given date if they exist.
        Used for cleanup or test resets.

        Args:
            date (datetime, optional): The date to target. Defaults to today (UTC).
        """
        try:
            if not self.pool:
                raise DatabaseException("PostgreSQL pool not initialized")

            flashpoint_table = self.get_daily_table_name(self.FLASHPOINT_TABLE_PREFIX)
            feed_table = self.get_daily_table_name(self.FEED_TABLE_PREFIX)

            async with self.pool.acquire() as conn:
                # Drop feed table first due to FK constraint
                drop_feed = f"DROP TABLE IF EXISTS {feed_table} CASCADE;"
                drop_flashpoint = f"DROP TABLE IF EXISTS {flashpoint_table} CASCADE;"

                await conn.execute(drop_feed)
                await conn.execute(drop_flashpoint)

            self.logger.info(f"Dropped tables: {feed_table}, {flashpoint_table}")

        except Exception as e:
            self.logger.error(f"Failed to drop daily tables: {e}")
            raise DatabaseException(f"Drop table failed: {str(e)}")

    def is_valid_postgres_url(self, url: str) -> bool:
        return bool(re.match(r"^postgres(?:ql)?://.+:.+@.+:\d+/.+", url))
    
    def __get_all_rls_policies_cmd(self, table_name: str) -> List[str]:
        """
        Generate all RLS-related SQL commands for the given table:
        - Enables and forces RLS
        - Creates SELECT, INSERT, UPDATE, DELETE policies for 'authenticated' role
        """

        enable_rls_query = f"ALTER TABLE {table_name} ENABLE ROW LEVEL SECURITY;"
        force_rls_query = f"ALTER TABLE {table_name} FORCE ROW LEVEL SECURITY;"

        create_select_policy_query = f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies WHERE policyname = 'allow_select' AND tablename = '{table_name}'
            ) THEN
                EXECUTE format($sql$
                    CREATE POLICY allow_select ON {table_name}
                    FOR SELECT TO anon, authenticated USING (true);
                $sql$);
            END IF;
        END $$;
        """

        create_insert_policy_query = f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies WHERE policyname = 'allow_insert' AND tablename = '{table_name}'
            ) THEN
                EXECUTE format($sql$
                    CREATE POLICY allow_insert ON {table_name}
                    FOR INSERT TO anon, authenticated WITH CHECK (true);
                $sql$);
            END IF;
        END $$;
        """

        create_update_policy_query = f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies WHERE policyname = 'allow_update' AND tablename = '{table_name}'
            ) THEN
                EXECUTE format($sql$
                    CREATE POLICY allow_update ON {table_name}
                    FOR UPDATE TO anon, authenticated USING (true) WITH CHECK (true);
                $sql$);
            END IF;
        END $$;
        """

        create_delete_policy_query = f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies WHERE policyname = 'allow_delete' AND tablename = '{table_name}'
            ) THEN
                EXECUTE format($sql$
                    CREATE POLICY allow_delete ON {table_name}
                    FOR DELETE TO anon, authenticated USING (true);
                $sql$);
            END IF;
        END $$;
        """

        return [
            enable_rls_query,
            force_rls_query,
            create_select_policy_query,
            create_insert_policy_query,
            create_update_policy_query,
            create_delete_policy_query
        ]

