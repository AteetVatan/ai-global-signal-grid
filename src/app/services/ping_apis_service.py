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

import requests
import threading
from ..config.logging_config import get_logger
from ..config.settings import get_settings


class PingApisService:
    """
    Service to ping MASX API servers (e.g., Render-hosted) to keep it alive or wake it up.
    Automatically pings on instantiation unless disabled.
    """

    def __init__(
        self,
        timeout: int = 60,
        auto_ping: bool = True
    ):
        self.settings = get_settings()
        self.ping_url = self.settings.GDELT_API_URL
        self.timeout = timeout
        self.logger = get_logger("PingApisService")

        if auto_ping:
            thread = threading.Thread(target=self._safe_ping, daemon=True)
            thread.start()


    def _safe_ping(self):
        try:
            self.logger.info(f"Pinging MASX server at {self.ping_url}")
            response = requests.get(self.ping_url, timeout=self.timeout)
            self.logger.info(f"Ping response: {response.status_code}")
        except Exception as e:
            self.logger.warning(f"MASX server ping failed: {e}")

    def ping(self) -> bool:
        """
        Optional manual synchronous ping. Returns True if 200 OK.
        """
        try:
            response = requests.get(self.ping_url, timeout=self.timeout)
            return response.status_code == 200
        except:
            return False