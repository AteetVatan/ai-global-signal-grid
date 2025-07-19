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
from datetime import datetime

class DateUtils:
    
    @staticmethod
    def convert_iso_to_date(iso_str: str) -> str:
        """
        Convert ISO 8601 date string to YYYY-MM-DD format.
        """
        dt = datetime.strptime(iso_str, "%Y%m%dT%H%M%SZ")
        return dt.strftime("%Y-%m-%d")

    @staticmethod    
    def convert_rfc822_to_iso_date(date_str: str) -> str:
        """
        Convert RFC 822 date string to YYYY-MM-DD format.
        """
        dt = datetime.strptime(date_str, "%a, %d %b %Y %H:%M:%S %Z")
        return dt.strftime("%Y-%m-%d")