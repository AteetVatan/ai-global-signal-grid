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
from typing import Dict

from .filters import Filters
from .helpers import load_json
from .errors import raise_response_error


class GdeltDoc:
    """
    API client for the GDELT 2.0 Doc API

    ```
    from gdeltdoc import GdeltDoc, Filters

    f = Filters(
        keyword = "climate change",
        start_date = "2020-05-10",
        end_date = "2020-05-11"
    )

    gd = GdeltDoc()

    # Search for articles matching the filters
    articles = gd.article_search(f)

    # Get a timeline of the number of articles matching the filters
    timeline = gd.timeline_search("timelinevol", f)
    ```

    ### Article List
    The article list mode of the API generates a list of news articles that match the filters.
    The client returns this as a pandas DataFrame with columns `url`, `url_mobile`, `title`,
    `seendate`, `socialimage`, `domain`, `language`, `sourcecountry`.

    ### Timeline Search
    There are 5 available modes when making a timeline search:
    * `timelinevol` - a timeline of the volume of news coverage matching the filters,
        represented as a percentage of the total news articles monitored by GDELT.
    * `timelinevolraw` - similar to `timelinevol`, but has the actual number of articles
        and a total rather than a percentage
    * `timelinelang` - similar to `timelinevol` but breaks the total articles down by published language.
        Each language is returned as a separate column in the DataFrame.
    * `timelinesourcecountry` - similar to `timelinevol` but breaks the total articles down by the country
        they were published in. Each country is returned as a separate column in the DataFrame.
    * `timelinetone` - a timeline of the average tone of the news coverage matching the filters.
        See [GDELT's documentation](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/)
        for more information about the tone metric.
    """

    def __init__(self, json_parsing_max_depth: int = 100) -> None:
        """
        Params
        ------
        json_parsing_max_depth
            A parameter for the json parsing function that removes illegal character. If 100 it will remove at max
            100 characters before exiting with an exception
        """
        self.max_depth_json_parsing = json_parsing_max_depth

    def article_search(self, filters: Filters):
        """
        Make a query against the `ArtList` API to return a DataFrame of news articles that
        match the supplied filters.

        Params
        ------
        filters
            A `gdelt-doc.Filters` object containing the filter parameters for this query.

        Returns
        -------
        json
            A json of the articles returned from the API.
        """
        articles = self._query("artlist", filters.query_string)
        if "articles" in articles:
            return articles["articles"]
        else:
            return []

    def timeline_search(self, mode: str, filters: Filters):
        """
        Make a query using one of the API's timeline modes.

        Params
        ------
        mode
            The API mode to call. Must be one of "timelinevol", "timelinevolraw",
            "timelinetone", "timelinelang", "timelinesourcecountry".

            See https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/ for a
            longer description of each mode.

        filters
            A `gdelt-doc.Filters` object containing the filter parameters for this query.

        Returns
        -------
        json
            A json of the articles returned from the API.
        """
        timeline = self._query(mode, filters.query_string)

        # If no results
        if (timeline == {}) or (len(timeline["timeline"]) == 0):
            return {}

        results = {
            "datetime": [entry["date"] for entry in timeline["timeline"][0]["data"]]
        }

        for series in timeline["timeline"]:
            results[series["series"]] = [entry["value"] for entry in series["data"]]

        if mode == "timelinevolraw":
            results["All Articles"] = [
                entry["norm"] for entry in timeline["timeline"][0]["data"]
            ]

        # formatted = pd.DataFrame(results)
        # formatted["datetime"] = pd.to_datetime(formatted["datetime"])

        return results

    def _query(self, mode: str, query_string: str) -> Dict:
        """
        Submit a query to the GDELT API and return the results as a parsed JSON object.

        Params
        ------
        mode
            The API mode to call. Must be one of "artlist", "timelinevol",
            "timelinevolraw", "timelinetone", "timelinelang", "timelinesourcecountry".

        query_string
            The query parameters and date range to call the API with.

        Returns
        -------
        Dict
            The parsed JSON response from the API.
        """
        if mode not in [
            "artlist",
            "timelinevol",
            "timelinevolraw",
            "timelinetone",
            "timelinelang",
            "timelinesourcecountry",
        ]:
            raise ValueError(f"Mode {mode} not in supported API modes")

        headers = {
            "User-Agent": f"GDELT DOC Python API client- https://github.com/AteetVatan"
        }

        url = f"https://api.gdeltproject.org/api/v2/doc/doc?query={query_string}&mode={mode}&format=json"

        response = requests.get(
            url,
            headers=headers,
        )

        raise_response_error(response=response)

        # Sometimes the API responds to an invalid request with a 200 status code
        # and a text/html content type. I can't figure out a pattern for when that happens so
        # this raises a ValueError with the response content instead of one of the library's
        # custom error types.
        if "text/html" in response.headers["content-type"]:
            raise ValueError(
                f"The query was not valid. The API error message was: {response.text.strip()}"
            )

        return load_json(response.content, self.max_depth_json_parsing)
