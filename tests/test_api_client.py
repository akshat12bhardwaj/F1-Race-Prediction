"""
Tests for src/api_client.py.

Almost all tests use a FakeSession instead of the internet: it records every
request and returns responses we prepare. A fake sleep() records wait times
instead of actually waiting, so the whole file runs in about a second.
Only the test marked @pytest.mark.live talks to the real API.

Run:  python -m pytest tests/test_api_client.py -v
Skip the live test:  python -m pytest tests/ -v -m "not live"
"""

import copy
import csv
import json
import logging

import pytest
import requests

from src.api_client import JolpicaAPIError, JolpicaClient, merge_race_pages
from src.config import load_config


# ----------------------------------------------------------------------
# Test helpers
# ----------------------------------------------------------------------
class FakeResponse:
    """Looks enough like requests.Response for our client."""

    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self.content = json.dumps(body).encode() if body is not None else text.encode()
        self.text = self.content.decode()


class FakeSession:
    """Records calls. Returns queued responses/exceptions, or asks a handler function."""

    def __init__(self, responses=None, handler=None):
        self.headers = {}
        self.calls = []
        self._responses = list(responses or [])
        self._handler = handler

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {})})
        if self._handler:
            return self._handler(url, dict(params or {}))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def mrdata(races, total, limit=100, offset=0):
    """Build a minimal Jolpica-style body. All values are strings, like the real API."""
    return {"MRData": {"limit": str(limit), "offset": str(offset), "total": str(total),
                       "RaceTable": {"Races": races}}}


def make_record(position, driver_id):
    return {"number": "1", "position": str(position), "positionText": str(position),
            "points": "0", "Driver": {"driverId": driver_id}, "Constructor": {"constructorId": "team"},
            "grid": "1", "laps": "50", "status": "Finished"}


def one_race_body(n_drivers=20, round_="1"):
    records = [make_record(p, f"driver_{p}") for p in range(1, n_drivers + 1)]
    return mrdata([{"season": "2024", "round": round_, "raceName": "Test GP", "Results": records}], n_drivers)


@pytest.fixture
def config():
    """Real config.yaml, but no gap between requests so tests stay fast and deterministic."""
    cfg = copy.deepcopy(load_config())
    cfg["api"]["jolpica"]["min_seconds_between_requests"] = 0
    return cfg


@pytest.fixture
def sleeps():
    return []


def make_client(config, tmp_path, session, sleeps):
    return JolpicaClient(config=config, cache_dir=tmp_path / "jolpica", session=session, sleep=sleeps.append)


# ----------------------------------------------------------------------
# A. Initialisation
# ----------------------------------------------------------------------
def test_client_initialises_from_config(config, tmp_path, sleeps):
    session = FakeSession()
    client = make_client(config, tmp_path, session, sleeps)
    assert client.base_url == "https://api.jolpi.ca/ergast/f1"
    assert client.page_limit == 100
    assert client.timeout == config["api"]["timeout_seconds"]
    assert client.max_retries == config["api"]["max_retries"]
    assert session.headers["User-Agent"] == config["api"]["user_agent"]
    assert client.cache_dir == tmp_path / "jolpica"


def test_default_client_uses_project_cache_folder():
    client = JolpicaClient()
    assert client.cache_dir.as_posix().endswith("data/raw/jolpica")
    assert client.manifest_path.name == "_manifest.csv"


# ----------------------------------------------------------------------
# B. Real API request (live)
# ----------------------------------------------------------------------
@pytest.mark.live
def test_live_request_and_cache(tmp_path):
    client = JolpicaClient(cache_dir=tmp_path / "jolpica")
    try:
        races = client.get_race_results(2024, 1)
    except JolpicaAPIError as error:
        pytest.skip(f"Live API not reachable: {error}")
    assert len(races) == 1
    assert races[0]["raceName"] == "Bahrain Grand Prix"
    assert len(races[0]["Results"]) == 20
    assert races[0]["Results"][0]["position"] == "1"  # still a string
    assert client.stats["network_requests"] == 1

    client.get_race_results(2024, 1)  # identical request
    assert client.stats["network_requests"] == 1  # no new network call
    assert client.stats["cache_hits"] == 1


# ----------------------------------------------------------------------
# C. Cache behaviour
# ----------------------------------------------------------------------
def test_second_identical_request_uses_cache(config, tmp_path, sleeps):
    session = FakeSession([FakeResponse(200, one_race_body())])
    client = make_client(config, tmp_path, session, sleeps)

    first = client.get_json("/2024/1/results.json", {"limit": 100, "offset": 0})
    second = client.get_json("/2024/1/results.json", {"offset": 0, "limit": 100})  # same params, other order

    assert len(session.calls) == 1              # only the first request used the network
    assert first == second
    assert client.stats == {"cache_hits": 1, "network_requests": 1, "retries": 0}
    cache_file = tmp_path / "jolpica" / "2024" / "1" / "results__limit-100__offset-0.json"
    assert cache_file.exists()
    assert cache_file.read_bytes() == json.dumps(one_race_body()).encode()  # exact bytes


def test_cache_key_depends_on_parameters(config, tmp_path, sleeps):
    client = make_client(config, tmp_path, FakeSession(), sleeps)
    a = client.cache_path("/2024/results.json", {"limit": 100, "offset": 0})
    b = client.cache_path("/2024/results.json", {"limit": 100, "offset": 100})
    c = client.cache_path("/2024/results.json", {"offset": 0, "limit": 100})
    assert a != b
    assert a == c  # deterministic: parameter order does not matter
    assert a.name == "results__limit-100__offset-0.json"


def test_refresh_bypasses_and_overwrites_cache(config, tmp_path, sleeps):
    session = FakeSession([FakeResponse(200, one_race_body(20)), FakeResponse(200, one_race_body(19))])
    client = make_client(config, tmp_path, session, sleeps)

    client.get_json("/2024/1/results.json")
    refreshed = client.get_json("/2024/1/results.json", refresh=True)
    assert len(session.calls) == 2
    assert refreshed["MRData"]["total"] == "19"
    assert client.get_json("/2024/1/results.json")["MRData"]["total"] == "19"  # cache now holds new copy


def test_corrupt_cache_file_is_downloaded_again(config, tmp_path, sleeps):
    session = FakeSession([FakeResponse(200, one_race_body())])
    client = make_client(config, tmp_path, session, sleeps)
    cache_file = client.cache_path("/2024/1/results.json")
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text("{ not json")

    data = client.get_json("/2024/1/results.json")
    assert len(session.calls) == 1
    assert data["MRData"]["total"] == "20"


def test_manifest_records_each_download(config, tmp_path, sleeps):
    session = FakeSession([FakeResponse(200, one_race_body())])
    client = make_client(config, tmp_path, session, sleeps)
    client.get_json("/2024/1/results.json", {"limit": 100})
    client.get_json("/2024/1/results.json", {"limit": 100})  # cache hit: not logged again

    with open(tmp_path / "_manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["path"] == "/2024/1/results.json"
    assert rows[0]["status"] == "200"
    assert rows[0]["mrdata_total"] == "20"


def test_cache_hit_and_miss_are_logged(config, tmp_path, sleeps, caplog):
    session = FakeSession([FakeResponse(200, one_race_body())])
    client = make_client(config, tmp_path, session, sleeps)
    with caplog.at_level(logging.INFO, logger="src.api_client"):
        client.get_json("/2024/1/results.json")
        client.get_json("/2024/1/results.json")
    messages = " | ".join(r.getMessage() for r in caplog.records)
    assert "Cache miss" in messages
    assert "Saved to cache" in messages
    assert "Cache hit" in messages


# ----------------------------------------------------------------------
# D. Pagination
# ----------------------------------------------------------------------
def season_handler(n_records=250, per_race=17):
    """
    Fake a season endpoint with n_records records, per_race drivers per race.
    With 17 per race, page boundaries (100, 200) fall in the middle of races,
    like the real API (Step 2: 2024 rounds 6, 11, 16, 21 were split).
    """
    records = [(1 + i // per_race, make_record(1 + i % per_race, f"driver_{i % per_race}"))
               for i in range(n_records)]

    def handler(url, params):
        offset, limit = params["offset"], params["limit"]
        page = records[offset: offset + limit]
        races = []
        for round_number, record in page:
            if not races or races[-1]["round"] != str(round_number):
                races.append({"season": "2024", "round": str(round_number), "raceName": f"GP {round_number}",
                              "Results": []})
            races[-1]["Results"].append(record)
        return FakeResponse(200, mrdata(races, n_records, limit, offset))

    return handler


def test_pagination_collects_all_records_across_pages(config, tmp_path, sleeps):
    session = FakeSession(handler=season_handler(n_records=250, per_race=17))
    client = make_client(config, tmp_path, session, sleeps)

    races = client.get_race_results(2024)

    assert [c["params"]["offset"] for c in session.calls] == [0, 100, 200]
    assert all(c["params"]["limit"] == 100 for c in session.calls)
    assert sum(len(r["Results"]) for r in races) == 250          # 250 is not a multiple of 100
    assert len(races) == 15                                        # 14 full races of 17 + one of 12
    assert [len(r["Results"]) for r in races] == [17] * 14 + [12]  # split races were merged back
    ids = [(r["round"], rec["position"], rec["Driver"]["driverId"]) for r in races for rec in r["Results"]]
    assert len(ids) == len(set(ids))                               # no duplicates
    assert races[0]["Results"][0]["position"] == "1"               # values remain strings


def test_pagination_exact_multiple_of_100(config, tmp_path, sleeps):
    session = FakeSession(handler=season_handler(n_records=200, per_race=20))
    client = make_client(config, tmp_path, session, sleeps)
    races = client.get_race_results(2024)
    assert [c["params"]["offset"] for c in session.calls] == [0, 100]  # no extra empty page
    assert sum(len(r["Results"]) for r in races) == 200


def test_paginated_pages_are_cached(config, tmp_path, sleeps):
    session = FakeSession(handler=season_handler(n_records=250))
    client = make_client(config, tmp_path, session, sleeps)
    client.get_race_results(2024)
    client.get_race_results(2024)
    assert len(session.calls) == 3          # second run entirely from cache
    assert client.stats["cache_hits"] == 3


def test_merge_drops_exact_duplicate_records():
    rec_a, rec_b = make_record(1, "a"), make_record(2, "b")
    page1 = mrdata([{"season": "2024", "round": "1", "Results": [rec_a, rec_b]}], 3)
    page2 = mrdata([{"season": "2024", "round": "1", "Results": [rec_b, make_record(3, "c")]}], 3)
    races = merge_race_pages([page1, page2], "Results")
    assert len(races) == 1
    assert [r["Driver"]["driverId"] for r in races[0]["Results"]] == ["a", "b", "c"]


def test_merge_keeps_same_driver_with_two_entries():
    # 1950s shared drives: one driver, two cars, two different positions in one race
    page = mrdata([{"season": "1955", "round": "1",
                    "Results": [make_record(1, "fangio"), make_record(4, "fangio")]}], 2)
    races = merge_race_pages([page], "Results")
    assert len(races[0]["Results"]) == 2


def test_record_count_must_match_total(config, tmp_path, sleeps):
    body = one_race_body(20)
    body["MRData"]["total"] = "21"  # API claims one more record than it sends
    client = make_client(config, tmp_path, FakeSession([FakeResponse(200, body)]), sleeps)
    with pytest.raises(JolpicaAPIError, match="expected 21 records"):
        client.get_race_results(2024, 1)


# ----------------------------------------------------------------------
# E. Empty result (total = 0)
# ----------------------------------------------------------------------
def test_total_zero_returns_empty_list(config, tmp_path, sleeps):
    session = FakeSession([FakeResponse(200, mrdata([], 0))])
    client = make_client(config, tmp_path, session, sleeps)
    races = client.get_race_results(2024, 99)  # Step 2: nonexistent round -> HTTP 200, total 0
    assert races == []
    assert len(session.calls) == 1
    assert client.cache_path("/2024/99/results.json", {"limit": 100, "offset": 0}).exists()


# ----------------------------------------------------------------------
# F. HTTP errors
# ----------------------------------------------------------------------
def test_http_400_raises_without_retry(config, tmp_path, sleeps):
    body = {"detail": "Bad Request: Endpoint does not support this request"}
    session = FakeSession([FakeResponse(400, body)])
    client = make_client(config, tmp_path, session, sleeps)
    with pytest.raises(JolpicaAPIError) as info:
        client.get_json("/2024/1/resultz.json")
    assert info.value.status_code == 400
    assert len(session.calls) == 1 and sleeps == []
    assert not client.cache_path("/2024/1/resultz.json").exists()  # errors are never cached
    assert (tmp_path / "_failed_requests.csv").exists()


def test_http_404_html_raises_without_retry(config, tmp_path, sleeps):
    session = FakeSession([FakeResponse(404, text="<!doctype html><html>Not Found</html>")])
    client = make_client(config, tmp_path, session, sleeps)
    with pytest.raises(JolpicaAPIError) as info:
        client.get_json("/2024/abc/results.json")
    assert info.value.status_code == 404
    assert len(session.calls) == 1


def test_http_429_is_retried_then_raises(config, tmp_path, sleeps):
    retries = config["api"]["max_retries"]
    session = FakeSession([FakeResponse(429, text="Request was throttled") for _ in range(retries + 1)])
    client = make_client(config, tmp_path, session, sleeps)
    with pytest.raises(JolpicaAPIError) as info:
        client.get_json("/2024/1/results.json")
    assert info.value.status_code == 429
    assert len(session.calls) == retries + 1
    assert sleeps == [2, 4, 8, 16, 32]  # exponential back-off from config (base 2 s)


def test_non_json_200_is_rejected_and_not_cached(config, tmp_path, sleeps):
    session = FakeSession([FakeResponse(200, text="<html>maintenance</html>")])
    client = make_client(config, tmp_path, session, sleeps)
    with pytest.raises(JolpicaAPIError, match="not valid JSON"):
        client.get_json("/2024/1/results.json")
    assert not client.cache_path("/2024/1/results.json").exists()


@pytest.mark.parametrize("params", [{"limit": 1000}, {"limit": 0}, {"limit": -5}, {"offset": -1}])
def test_values_the_api_would_silently_change_are_rejected(config, tmp_path, sleeps, params):
    session = FakeSession()
    client = make_client(config, tmp_path, session, sleeps)
    with pytest.raises(ValueError):
        client.get_json("/2024/1/results.json", params)
    assert session.calls == []  # rejected before any request


def test_non_numeric_round_is_rejected(config, tmp_path, sleeps):
    client = make_client(config, tmp_path, FakeSession(), sleeps)
    with pytest.raises(ValueError):
        client.get_race_results(2024, "abc")


# ----------------------------------------------------------------------
# G. Retries
# ----------------------------------------------------------------------
def test_transient_failures_are_retried_until_success(config, tmp_path, sleeps):
    session = FakeSession([
        requests.ConnectionError("connection reset"),
        FakeResponse(503, text="Service Unavailable"),
        FakeResponse(200, one_race_body()),
    ])
    client = make_client(config, tmp_path, session, sleeps)
    data = client.get_json("/2024/1/results.json")
    assert data["MRData"]["total"] == "20"
    assert len(session.calls) == 3
    assert client.stats["retries"] == 2
    assert sleeps == [2, 4]


def test_timeouts_exhaust_retries_and_raise(config, tmp_path, sleeps):
    retries = config["api"]["max_retries"]
    session = FakeSession([requests.Timeout("read timed out") for _ in range(retries + 1)])
    client = make_client(config, tmp_path, session, sleeps)
    with pytest.raises(JolpicaAPIError, match="Timeout"):
        client.get_json("/2024/1/results.json")
    assert len(session.calls) == retries + 1
    assert client.stats["retries"] == retries


def test_minimum_gap_between_network_requests(config, tmp_path, sleeps):
    config["api"]["jolpica"]["min_seconds_between_requests"] = 1.0
    session = FakeSession([FakeResponse(200, one_race_body()), FakeResponse(200, one_race_body(round_="2"))])
    client = make_client(config, tmp_path, session, sleeps)
    client.get_json("/2024/1/results.json")
    client.get_json("/2024/2/results.json")
    assert len(sleeps) == 1 and 0 < sleeps[0] <= 1.0  # waited before the second request
