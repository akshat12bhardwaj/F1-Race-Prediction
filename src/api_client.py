"""
Cached client for the Jolpica F1 API (Ergast-compatible).

What it does
------------
1. Disk cache: every request is saved as one JSON file whose path mirrors
   the URL, e.g.
       /2024/1/results.json?limit=100&offset=0
       -> data/raw/jolpica/2024/1/results__limit-100__offset-0.json
   An identical request later is read from that file (no network call).
2. Polite requests: a minimum gap between network calls (config.yaml).
3. Retries with exponential back-off for temporary problems
   (HTTP 429 rate limit, HTTP 5xx, connection errors, timeouts).
4. Pagination: fetches every page of an endpoint using limit=100 and
   MRData.total, then merges the pages race by race.

Behaviour found in Step 2 (docs/DECISIONS_LOG.md, F-001) that shaped this code:
- limit > 100 is silently capped and bad limit/offset values are silently
  replaced, so we reject them ourselves before sending the request.
- A round/season that does not exist returns HTTP 200 with total = 0.
  That is a valid (empty) answer, not an error.
- HTTP 400 (bad endpoint) and 404 (non-numeric round, HTML body) are not
  temporary, so they are raised immediately without retrying.
- One race can be split across two pages, so pages are merged by (season, round).
- All values arrive as strings. This client never converts types;
  that happens in the cleaning step.
"""

import csv
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from src.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

MAX_PAGE_SIZE = 100                           # Jolpica maximum (verified in Step 2)
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")   # allowed characters in cache path parts


class JolpicaAPIError(Exception):
    """Raised when a request fails permanently or returns unusable data."""

    def __init__(self, message, status_code=None, url=None):
        super().__init__(message)
        self.status_code = status_code
        self.url = url


class JolpicaClient:
    """Small cached client for the Jolpica F1 API."""

    def __init__(self, config=None, cache_dir=None, session=None, sleep=time.sleep):
        # config, cache_dir, session and sleep can be replaced in tests.
        config = config or load_config()
        api = config["api"]
        jolpica = api["jolpica"]

        self.base_url = jolpica["base_url"].rstrip("/")
        self.page_limit = min(int(jolpica["page_limit"]), MAX_PAGE_SIZE)
        self.min_interval = float(jolpica["min_seconds_between_requests"])
        self.timeout = api["timeout_seconds"]
        self.max_retries = int(api["max_retries"])
        self.backoff_base = float(api["backoff_base_seconds"])

        self.cache_dir = Path(cache_dir) if cache_dir else PROJECT_ROOT / config["paths"]["data_raw_jolpica"]
        # The manifest and failure log live next to the cache folder (data/raw/)
        self.manifest_path = self.cache_dir.parent / Path(config["paths"]["raw_manifest"]).name
        self.failed_path = self.cache_dir.parent / Path(config["paths"]["failed_requests"]).name

        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": api["user_agent"]})
        self._sleep = sleep
        self._last_request_time = None

        # Counters so we can see (and test) what the client actually did
        self.stats = {"cache_hits": 0, "network_requests": 0, "retries": 0}

    # ------------------------------------------------------------------
    # Public: one request
    # ------------------------------------------------------------------
    def get_json(self, path, params=None, refresh=False):
        """
        Return the parsed JSON for one API request.

        path    : endpoint path, e.g. "/2024/1/results.json"
        params  : query parameters, e.g. {"limit": 100, "offset": 0}
        refresh : True = ignore the cache, download again and overwrite the file
        """
        params = self._validate_params(params)
        cache_file = self.cache_path(path, params)

        if refresh:
            logger.info("Cache bypassed (refresh=True): %s", self._display(cache_file))
        else:
            cached = self._read_cache(cache_file)
            if cached is not None:
                self.stats["cache_hits"] += 1
                logger.info("Cache hit: %s", self._display(cache_file))
                return cached
            logger.info("Cache miss: %s", self._display(cache_file))

        response = self._request_with_retries(path, params)
        data = self._parse_json(response, path)
        self._write_cache(cache_file, response.content)
        self._append_manifest(path, params, cache_file, response, data)
        return data

    # ------------------------------------------------------------------
    # Public: pagination
    # ------------------------------------------------------------------
    def get_paginated(self, path, params=None, refresh=False):
        """
        Fetch every page of an endpoint and return the list of page JSONs.

        Uses limit=100 and increases offset by 100 until offset >= MRData.total.
        Works when total = 0 (one page, no records) and when total is not a
        multiple of 100. Each page is cached on its own.
        """
        base_params = dict(params or {})
        base_params.pop("offset", None)
        base_params["limit"] = self.page_limit

        pages, total, offset = [], None, 0
        while True:
            page = self.get_json(path, {**base_params, "offset": offset}, refresh=refresh)
            page_total = int(page["MRData"]["total"])
            if total is None:
                total = page_total
            elif page_total != total:
                raise JolpicaAPIError(f"MRData.total changed during pagination ({total} -> {page_total})",
                                      url=self.base_url + path)
            pages.append(page)
            offset += self.page_limit
            if offset >= total:
                return pages

    # ------------------------------------------------------------------
    # Public: endpoint helpers (thin wrappers)
    # ------------------------------------------------------------------
    def get_race_results(self, season, round=None, refresh=False):
        """Race results for one round, or the whole season if round is None."""
        return self._get_races(season, round, "results", "Results", refresh)

    def get_qualifying(self, season, round=None, refresh=False):
        """Qualifying results for one round, or the whole season if round is None."""
        return self._get_races(season, round, "qualifying", "QualifyingResults", refresh)

    def get_sprint_results(self, season, round=None, refresh=False):
        """Sprint results for one round, or the whole season if round is None."""
        return self._get_races(season, round, "sprint", "SprintResults", refresh)

    def get_schedule(self, season, refresh=False):
        """The season's race calendar (dates, circuits, sessions). Needed in Steps 4-5."""
        return self._get_races(season, None, "races", None, refresh)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def cache_path(self, path, params=None):
        """
        Deterministic cache file for a request: the URL path becomes folders,
        and the parameters (sorted by name) are appended to the file name.
        """
        parts = [p for p in path.strip("/").split("/") if p]
        if not parts or not parts[-1].endswith(".json"):
            raise ValueError(f"Path must end with .json: {path!r}")
        *folders, filename = parts
        name = filename[: -len(".json")]
        for key in sorted(params or {}):
            name += f"__{key}-{params[key]}"
        for part in folders + [name]:
            if not SAFE_NAME.match(part):
                raise ValueError(f"Unsafe characters in request path/params: {path!r} {params!r}")
        return self.cache_dir.joinpath(*folders, name + ".json")

    def _read_cache(self, cache_file):
        """Return cached JSON, or None if missing or unreadable (then it is re-downloaded)."""
        if not cache_file.exists():
            return None
        try:
            data = json.loads(cache_file.read_bytes())
            if "MRData" in data:
                return data
        except ValueError:
            pass
        logger.warning("Ignoring unreadable cache file, will re-download: %s", self._display(cache_file))
        return None

    def _write_cache(self, cache_file, content):
        """Save the exact response bytes. Write to a temp file first, then rename,
        so an interrupted run never leaves a half-written cache file."""
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = cache_file.with_suffix(".json.tmp")
        temp_file.write_bytes(content)
        os.replace(temp_file, cache_file)
        logger.info("Saved to cache: %s (%d bytes)", self._display(cache_file), len(content))

    # ------------------------------------------------------------------
    # Network helpers
    # ------------------------------------------------------------------
    def _validate_params(self, params):
        """Reject values the API would silently change (Step 2 findings)."""
        params = dict(params or {})
        if "limit" in params:
            limit = int(params["limit"])
            if not 1 <= limit <= MAX_PAGE_SIZE:
                raise ValueError(f"limit must be 1..{MAX_PAGE_SIZE} (API would silently change it), got {limit}")
            params["limit"] = limit
        if "offset" in params:
            offset = int(params["offset"])
            if offset < 0:
                raise ValueError(f"offset must be >= 0 (API would silently use 0), got {offset}")
            params["offset"] = offset
        return params

    def _wait_for_slot(self):
        """Keep at least min_interval seconds between network requests."""
        if self._last_request_time is not None:
            elapsed = time.monotonic() - self._last_request_time
            if elapsed < self.min_interval:
                self._sleep(self.min_interval - elapsed)
        self._last_request_time = time.monotonic()

    def _request_with_retries(self, path, params):
        """Send the request; retry temporary failures with waits of 2, 4, 8, ... seconds."""
        url = self.base_url + path
        total_attempts = self.max_retries + 1
        last_error = None

        for attempt in range(1, total_attempts + 1):
            self._wait_for_slot()
            logger.info("Request attempt %d/%d: %s %s", attempt, total_attempts, path, params)
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                self.stats["network_requests"] += 1
            except (requests.ConnectionError, requests.Timeout) as error:
                last_error = JolpicaAPIError(f"{type(error).__name__}: {error}", url=url)
            else:
                if response.status_code == 200:
                    return response
                message = f"HTTP {response.status_code} for {path} {params}"
                if response.status_code not in RETRY_STATUS_CODES:
                    # 400, 404, ... will not fix themselves: fail immediately
                    logger.error("%s (not retried): %s", message, response.text[:120].replace("\n", " "))
                    self._log_failure(path, params, message)
                    raise JolpicaAPIError(message, status_code=response.status_code, url=url)
                last_error = JolpicaAPIError(message, status_code=response.status_code, url=url)

            if attempt < total_attempts:
                wait = self.backoff_base * 2 ** (attempt - 1)
                self.stats["retries"] += 1
                logger.warning("Retry %d/%d in %.0f s after: %s", attempt, self.max_retries, wait, last_error)
                self._sleep(wait)

        logger.error("Giving up after %d attempts: %s", total_attempts, last_error)
        self._log_failure(path, params, str(last_error))
        raise last_error

    def _parse_json(self, response, path):
        """A 200 response must be JSON with an MRData object; otherwise it is not cached."""
        try:
            data = json.loads(response.content)
        except ValueError:
            raise JolpicaAPIError(f"Response for {path} is not valid JSON", status_code=200) from None
        if "MRData" not in data:
            raise JolpicaAPIError(f"Response for {path} has no MRData", status_code=200)
        return data

    # ------------------------------------------------------------------
    # Race-level helper
    # ------------------------------------------------------------------
    def _get_races(self, season, round, endpoint, record_key, refresh):
        """Fetch all pages for a race-table endpoint and merge them race by race."""
        season = int(season)  # int() rejects "abc", which the API would answer with a 404 HTML page
        path = f"/{season}/{int(round)}/{endpoint}.json" if round is not None else f"/{season}/{endpoint}.json"
        pages = self.get_paginated(path, refresh=refresh)
        races = merge_race_pages(pages, record_key)

        total = int(pages[0]["MRData"]["total"])
        found = len(races) if record_key is None else sum(len(r[record_key]) for r in races)
        if found != total:
            raise JolpicaAPIError(f"{path}: expected {total} records (MRData.total), got {found}")
        return races

    # ------------------------------------------------------------------
    # Logging to CSV files
    # ------------------------------------------------------------------
    def _append_manifest(self, path, params, cache_file, response, data):
        self._append_csv(self.manifest_path,
                         ["fetched_at_utc", "path", "params", "status", "n_bytes", "mrdata_total", "cache_file"],
                         [_now(), path, json.dumps(params, sort_keys=True), response.status_code,
                          len(response.content), data["MRData"].get("total"), self._display(cache_file)])

    def _log_failure(self, path, params, error):
        self._append_csv(self.failed_path, ["failed_at_utc", "path", "params", "error"],
                         [_now(), path, json.dumps(params, sort_keys=True), error])

    @staticmethod
    def _append_csv(csv_path, header, row):
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not csv_path.exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(header)
            writer.writerow(row)

    @staticmethod
    def _display(file_path):
        """Short project-relative path for log messages."""
        try:
            return file_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(file_path)


def merge_race_pages(pages, record_key):
    """
    Combine paginated responses into one list of races.

    Pages are cut by record count, not by race, so the same race can appear on
    two pages holding different records. Races are merged by (season, round) and
    their records concatenated in API order. A record repeated exactly (same
    race, position and driverId) is dropped as a duplicate. Position is part of
    the key because in the 1950s one driver could have two entries in one race.

    record_key : "Results", "QualifyingResults", "SprintResults",
                 or None for endpoints without per-driver records (schedule).
    Values are returned unchanged (strings).
    """
    races = {}
    seen_records = set()
    duplicates = 0

    for page in pages:
        for race in page["MRData"]["RaceTable"]["Races"]:
            race_key = (race["season"], race["round"])
            if race_key not in races:
                races[race_key] = {k: v for k, v in race.items() if k != record_key}
                if record_key is not None:
                    races[race_key][record_key] = []
            if record_key is None:
                continue
            for record in race.get(record_key, []):
                record_id = race_key + (record.get("position"), record["Driver"]["driverId"])
                if record_id in seen_records:
                    duplicates += 1
                    continue
                seen_records.add(record_id)
                races[race_key][record_key].append(record)

    if duplicates:
        logger.warning("Dropped %d duplicate record(s) while merging pages", duplicates)
    return list(races.values())


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
