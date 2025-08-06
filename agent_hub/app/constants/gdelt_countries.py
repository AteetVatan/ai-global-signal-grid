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

import json

class GdeltCountries:
    def __init__(self):
        self.countries = []
        self.__load_gdelt_countries()
        
    def __load_gdelt_countries(self):
        with open("app/constants/lookup-gkg-countries.json", "r") as f:
            self.countries = json.load(f)

    def get_countries(self):
        return self.countries

    def get_country_code_by_name(self, country_name):
        for country in self.countries:
            if country["name"] == country_name:
                return country["code"]
        return None

    def get_country_name_by_code(self, country_code):
        for country in self.countries:
            if country["code"] == country_code:
                return country["name"]
        return None

    def get_country_code(self, country_name):
        for country in self.countries:
            if country["name"] == country_name:
                return country["code"]