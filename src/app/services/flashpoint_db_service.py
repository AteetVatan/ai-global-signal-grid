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

from ..core.utils import measure_execution_time
from ..config.logging_config import get_service_logger
from ..config.settings import get_settings
from ..core.exceptions import DatabaseException, ConfigurationException

# Replace with your actual database client imports
# from your_project.database import supabase_client, pg_pool


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
    


class FlashpointDatabaseService:
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_service_logger("FlashpointDatabaseService")
        self.client: Optional[Client] = None # Supabase client
        self.pool: Optional[asyncpg.Pool] = None # asyncpg pool
        self._connection_params = {}
        self._initialize_connection()
        
        
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
        date = date or datetime.utcnow()
        return f"{base}_{date.strftime('%Y%m%d')}"

    async def ensure_flashpoint_table_exists(self, table_name: str) -> None:
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

                async with self.pool.acquire() as conn:
                    await conn.execute(query)
                    self.logger.info(f"✅ Table ensured: {table_name}")

            except Exception as e:
                self.logger.error(f"❌ Failed to ensure table {table_name}: {e}")
                raise DatabaseException(f"Table creation failed: {str(e)}")

    async def ensure_feed_table_exists(self, feed_table: str, flashpoint_table: str):
        """
        Ensures the feed table for the given day exists. Also creates a foreign key index on flashpoint_id.
        """
        with measure_execution_time(f"ensure_feed_table_exists: {feed_table}"):
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

            async with self.pool.acquire() as conn:
                await conn.execute(create_table_query)
                await conn.execute(create_index_query)
                await conn.execute(alter_fk_if_needed_query)

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
                table_name = self.get_daily_table_name("flash_point")

                # Step 2: Ensure the table exists (Postgres side)
                await self.ensure_flashpoint_table_exists(table_name)
                now = datetime.utcnow().isoformat()
                # Step 3: Convert to record and clean up
                record = FlashpointRecord(
                    title=fp.title.strip(),
                    description=fp.description.strip(),
                    entities=fp.entities,
                    domains=fp.domains,
                    run_id=run_id,
                    created_at=now,
                    updated_at=now,
                )
                data = {k: v for k, v in asdict(record).items() if v is not None}


                 # Fix types: Supabase often needs JSON-safe formats
                if isinstance(data["entities"], list):
                    data["entities"] = json.dumps(data["entities"])
                if isinstance(data["domains"], list):
                    data["domains"] = json.dumps(data["domains"])
                    
                self.logger.debug(f"[store_flashpoint] Payload for {table_name}: {json.dumps(data, indent=2)}")
      # Step 4: Insert using Supabase client
                result = self.client.table(table_name).insert(data).execute()

                if result.data:
                    flashpoint_id = result.data[0].get("id")
                    self.logger.info(f"[store_flashpoint] Stored in {table_name} → ID: {flashpoint_id}")
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

            feed_table = self.get_daily_table_name("feed_entries")
            flashpoint_table = self.get_daily_table_name("flash_point")
            await self.ensure_feed_table_exists(feed_table, flashpoint_table)

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

            result = self.client.table(feed_table).insert(payload).execute()
            return len(result.data or [])
        except Exception as e:
            self.logger.error(f"store_feed_entries() failed: {e}")
            raise DatabaseException(str(e))
        
    async def drop_daily_tables(self, date: Optional[datetime] = None) -> None:
        """
        Drops the daily flashpoint and feed tables for the given date if they exist.
        Used for cleanup or test resets.

        Args:
            date (datetime, optional): The date to target. Defaults to today (UTC).
        """
        try:
            if not self.pool:
                raise DatabaseException("PostgreSQL pool not initialized")

            flashpoint_table = self.get_daily_table_name("flash_point", date)
            feed_table = self.get_daily_table_name("feed_entries", date)

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