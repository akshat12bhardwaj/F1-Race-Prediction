"""
Offline tests for src/weather.py (Step 6). No network: a FakeSession answers
like the Open-Meteo archive API, generating UTC hourly data from the request.
Temperature at each hour = UTC hour of day, so alignment errors are visible.
"""

import copy
import hashlib
import json

import pandas as pd
import pytest

from src.config import load_config
from src.weather import (OpenMeteoClient, align, hourly_frame, race_start, run, window_rows, window_summary)

VARS = ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "cloud_cover", "wind_speed_10m",
        "wind_direction_10m", "weather_code"]
UNITS = {"time": "iso8601", "temperature_2m": "°C", "relative_humidity_2m": "%", "precipitation": "mm", "rain": "mm",
         "cloud_cover": "%", "wind_speed_10m": "km/h", "wind_direction_10m": "°", "weather_code": "wmo code"}
TIMEZONES = {"36.1147": "America/Los_Angeles", "52.0786": "Europe/London", "39.795": "America/Indiana/Indianapolis",
             "45.6156": "Europe/Rome"}


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self.content = json.dumps(body).encode()
        self.text = self.content.decode()


class FakeSession:
    def __init__(self, fail_lat=None, bad_values_lat=None, gap_lat=None):
        self.headers, self.calls = {}, []
        self.fail_lat, self.bad_values_lat, self.gap_lat = fail_lat, bad_values_lat, gap_lat

    def get(self, url, params=None, timeout=None):
        self.calls.append(dict(params))
        lat = params["latitude"]
        if params["timezone"] == "auto":
            return FakeResponse(200, {"timezone": TIMEZONES[lat], "hourly": {"time": ["2020-01-01T00:00"]}})
        if lat == self.fail_lat:
            return FakeResponse(400, {"error": True, "reason": "Parameter out of range"})
        times = pd.date_range(params["start_date"], pd.Timestamp(params["end_date"]) + pd.Timedelta(hours=23), freq="h")
        hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in times],
                  "temperature_2m": [float(t.hour) for t in times], "relative_humidity_2m": [50] * len(times),
                  "precipitation": [1.0 if t.hour in (14, 15, 16) else 0.0 for t in times],
                  "rain": [1.0 if t.hour in (14, 15, 16) else 0.0 for t in times], "cloud_cover": [40] * len(times),
                  "wind_speed_10m": [10.0] * len(times), "wind_direction_10m": [350, 10] * (len(times) // 2),
                  "weather_code": [61 if t.hour in (14, 15, 16) else 0 for t in times]}
        if lat == self.bad_values_lat:
            hourly["temperature_2m"][5] = 80.0        # impossible temperature
            hourly["wind_speed_10m"][6] = -3.0        # impossible wind
        if lat == self.gap_lat:
            hourly["temperature_2m"] = [None] * len(times)
        return FakeResponse(200, {"latitude": float(lat) + 0.02, "longitude": float(params["longitude"]),
                                  "elevation": 100.0, "utc_offset_seconds": 0, "timezone": "GMT",
                                  "hourly_units": UNITS, "hourly": hourly})


def races_frame():
    rows = [
        # Las Vegas 2023: API date is the UTC date; local start is 22:00 on 18 Nov
        ("2023", "21", "Las Vegas Grand Prix", "2023-11-19", "06:00:00Z", "vegas", "36.1147", "-115.173", "USA", "False"),
        ("2004", "11", "British Grand Prix", "2004-07-11", "", "silverstone", "52.0786", "-1.01694", "UK", "False"),
        ("1955", "3", "Indianapolis 500", "1955-05-30", "", "indianapolis", "39.795", "-86.2347", "USA", "True"),
        ("2005", "9", "United States Grand Prix", "2005-06-19", "18:00:00Z", "indianapolis", "39.795", "-86.2347",
         "USA", "False"),
        ("2012", "13", "Italian Grand Prix", "2012-09-09", "12:00:00Z", "monza", "45.6156", "9.28111", "Italy", "False"),
    ]
    cols = ["season", "round", "race_name", "race_date", "race_time", "circuit_id", "circuit_lat", "circuit_lon",
            "country", "is_indianapolis_500"]
    df = pd.DataFrame(rows, columns=cols).replace({"": None})
    df["race_id"] = df["season"] + "_" + df["round"].str.zfill(2)
    return df


@pytest.fixture
def config():
    cfg = copy.deepcopy(load_config())
    cfg["api"]["openmeteo"]["min_seconds_between_requests"] = 0
    return cfg


def make_client(config, tmp_path, session):
    return OpenMeteoClient(config=config, cache_dir=tmp_path / "openmeteo", session=session, sleep=lambda s: None)


def run_all(config, tmp_path, session, name="out"):
    interim, tables = tmp_path / name / "interim", tmp_path / name / "tables"
    interim.mkdir(parents=True), tables.mkdir(parents=True)
    client = make_client(config, tmp_path, session)
    outputs = run(races_frame(), client, config, interim, tables)
    return client, {p.name: df for p, df in outputs.items()}, interim, tables


# ---------------- cache ----------------
def test_cache_naming_is_deterministic(config, tmp_path):
    client = make_client(config, tmp_path, FakeSession())
    a = client.cache_path("weather", "2023_21", {"latitude": "36.1", "hourly": VARS, "timezone": "GMT"})
    b = client.cache_path("weather", "2023_21", {"timezone": "GMT", "hourly": VARS, "latitude": "36.1"})
    c = client.cache_path("weather", "2023_21", {"latitude": "36.1", "hourly": VARS[:3], "timezone": "GMT"})
    assert a == b                    # parameter order does not matter
    assert a != c                    # different variables -> different file
    assert a.parent.name == "weather" and a.name.startswith("2023_21__") and a.suffix == ".json"


def test_cache_reuse_no_second_request(config, tmp_path):
    session = FakeSession()
    client = make_client(config, tmp_path, session)
    params = {"latitude": "36.1147", "longitude": "-115.173", "start_date": "2023-11-18",
              "end_date": "2023-11-20", "hourly": VARS, "timezone": "GMT"}
    first = client.get_json("weather", "2023_21", params)
    second = client.get_json("weather", "2023_21", params)
    assert first == second and len(session.calls) == 1
    assert client.stats["cache_hits"] == 1 and client.stats["network_requests"] == 1
    assert (tmp_path / "openmeteo" / "_manifest.csv").exists()


# ---------------- time alignment ----------------
def test_official_start_is_utc_and_local_uses_historical_dst():
    start_utc, start_local, source = race_start("2023-11-19", "06:00:00Z", "America/Los_Angeles", 14)
    assert source == "official_utc"
    assert start_utc == pd.Timestamp("2023-11-19 06:00", tz="UTC")
    assert start_local.strftime("%Y-%m-%d %H:%M") == "2023-11-18 22:00"   # PST = UTC-8 in November
    assert start_local.utcoffset() == pd.Timedelta(hours=-8)


def test_pre_2005_start_is_imputed_local_14h():
    start_utc, start_local, source = race_start("2004-07-11", None, "Europe/London", 14)
    assert source == "imputed_local_14h"
    assert start_local.strftime("%H:%M") == "14:00"
    assert start_utc == pd.Timestamp("2004-07-11 13:00", tz="UTC")       # BST = UTC+1 in July


def test_window_alignment_instantaneous_and_accumulated():
    start = pd.Timestamp("2012-09-09 12:30", tz="UTC")                    # H0 = 12:00
    times = pd.date_range("2012-09-09 06:00", "2012-09-09 18:00", freq="h", tz="UTC")
    hourly = pd.DataFrame({"time_utc": times, "temperature_2m": [float(t.hour) for t in times],
                           "precipitation": [float(t.hour) for t in times]})
    aligned = align(hourly, start, "Europe/Rome")
    assert window_rows(aligned, (0, 2), "temperature_2m")["time_utc"].dt.hour.tolist() == [12, 13, 14]
    assert window_rows(aligned, (0, 2), "precipitation")["time_utc"].dt.hour.tolist() == [13, 14]   # preceding-hour sums
    assert window_rows(aligned, (-3, -1), "temperature_2m")["time_utc"].dt.hour.tolist() == [9, 10, 11]


def test_race_window_values_match_the_right_utc_hours(config, tmp_path):
    _, out, _, _ = run_all(config, tmp_path, FakeSession())
    wr = out["weather_race.csv"].set_index("race_id")
    # Monza 2012 official start 12:00Z -> race window temps at 12,13,14 UTC -> mean 13
    assert wr.loc["2012_13", "wx_race_temp_mean_c"] == 13.0
    # precipitation falls at 14,15,16 UTC; race-window accumulated rows are 13:00 and 14:00 -> 1.0 mm
    assert wr.loc["2012_13", "wx_race_precip_sum_mm"] == 1.0
    # 2004 British GP: 14:00 BST = 13:00 UTC -> temps 13,14,15 -> mean 14
    assert wr.loc["2004_11", "start_time_source"] == "imputed_local_14h"
    assert wr.loc["2004_11", "wx_race_temp_mean_c"] == 14.0
    # wind directions 350, 10, 350 -> circular mean ~356.7 (north), not the arithmetic 236.7 (south-west)
    direction = wr.loc["2012_13", "wx_race_wind_dir_mean_deg"]
    assert direction > 350 or direction < 10


# ---------------- missing and suspicious data ----------------
def test_failed_request_is_reported_and_race_kept(config, tmp_path):
    client, out, _, _ = run_all(config, tmp_path, FakeSession(fail_lat="45.6156"))
    wr = out["weather_race.csv"].set_index("race_id")
    assert wr.loc["2012_13", "weather_status"] == "failed"
    assert not wr.loc["2012_13", "model_use_allowed"]
    assert len(wr) == 5                                                   # race row kept
    assert "request_failed" in out["weather_quality_report.csv"]["check"].tolist()
    assert (tmp_path / "openmeteo" / "_failed_requests.csv").exists()
    cov = out["weather_coverage_summary.csv"].set_index("group")
    assert cov.loc["all 1950-2025", "races_without_weather"] == 1


def test_missing_values_in_window_are_flagged(config, tmp_path):
    _, out, _, _ = run_all(config, tmp_path, FakeSession(gap_lat="52.0786"))
    q = out["weather_quality_report.csv"]
    assert ((q["race_id"] == "2004_11") & (q["check"] == "missing_values_in_window")).any()
    assert out["weather_race.csv"].set_index("race_id").loc["2004_11", "weather_status"] == "ok_with_errors"


def test_implausible_values_are_flagged_not_removed(config, tmp_path):
    _, out, interim, _ = run_all(config, tmp_path, FakeSession(bad_values_lat="36.1147"))
    checks = set(out["weather_quality_report.csv"].query("race_id == '2023_21'")["check"])
    assert {"implausible_temperature_2m", "implausible_wind_speed_10m"} <= checks
    # the raw cached response still contains the 80 °C value (nothing silently removed)
    cached = next((tmp_path / "openmeteo" / "weather").glob("2023_21__*.json"))
    assert 80.0 in json.loads(cached.read_bytes())["hourly"]["temperature_2m"]


def test_local_date_difference_is_info_not_error(config, tmp_path):
    _, out, _, _ = run_all(config, tmp_path, FakeSession())
    q = out["weather_quality_report.csv"].query("race_id == '2023_21'")
    row = q[q["check"] == "local_date_differs_from_api_date"]
    assert len(row) == 1 and row["severity"].iloc[0] == "info"
    assert out["weather_race.csv"].set_index("race_id").loc["2023_21", "weather_status"] == "ok"


# ---------------- Indianapolis ----------------
def test_indianapolis_500_excluded_us_gp_included(config, tmp_path):
    client, out, _, _ = run_all(config, tmp_path, FakeSession())
    wr = out["weather_race.csv"].set_index("race_id")
    assert wr.loc["1955_03", "weather_status"] == "excluded_indianapolis_500"
    assert not wr.loc["1955_03", "model_use_allowed"]
    assert wr.loc["2005_09", "weather_status"] == "ok" and wr.loc["2005_09", "model_use_allowed"]
    assert "1955_03" not in set(out["weather_hourly.csv"]["race_id"])
    assert not list((tmp_path / "openmeteo" / "weather").glob("1955_03__*"))   # never requested


# ---------------- reproducibility ----------------
def test_rerun_uses_cache_and_outputs_are_identical(config, tmp_path):
    session = FakeSession()
    _, _, interim1, tables1 = run_all(config, tmp_path, session, "run1")
    calls_after_first = len(session.calls)
    client2, _, interim2, tables2 = run_all(config, tmp_path, session, "run2")
    assert len(session.calls) == calls_after_first            # second run: 0 new requests
    assert client2.stats["network_requests"] == 0
    digest = lambda folder: {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob("*.csv")}
    assert digest(interim1) == digest(interim2) and digest(tables1) == digest(tables2)


def test_local_clock_values_labelled_utc_are_flagged_not_changed():
    """2005-style season: the same '14:00:00Z' everywhere -> local starts scattered -> suspected."""
    from src.weather import apply_start_time_check, race_start
    rows = []
    for season, entries in {"2005": [("Australia/Melbourne", "14:00:00Z"), ("Asia/Tokyo", "14:00:00Z"),
                                      ("America/Toronto", "14:00:00Z"), ("Europe/Rome", "14:00:00Z")],
                            "2008": [("Australia/Melbourne", "04:30:00Z"), ("Asia/Tokyo", "04:30:00Z"),
                                     ("America/Toronto", "17:00:00Z"), ("Europe/Rome", "12:00:00Z")]}.items():
        for i, (tz, t) in enumerate(entries, 1):
            _, local, source = race_start(f"{season}-06-0{i}", t, tz, 14)
            rows.append({"season": season, "race_id": f"{season}_0{i}", "race_time": t, "start_local": local.isoformat(),
                         "start_time_source": source, "weather_status": "ok", "model_use_allowed": True})
    wr = pd.DataFrame(rows)
    flags = pd.DataFrame(columns=["race_id", "check", "severity", "detail"])
    wr, flags, check = apply_start_time_check(wr, flags)
    assert check.set_index("season")["suspected_local_clock"].to_dict() == {"2005": True, "2008": False}
    assert wr.loc[wr["season"] == "2005", "race_time_suspected_local"].all()
    assert not wr.loc[wr["season"] == "2005", "model_use_allowed"].any()
    assert wr.loc[wr["season"] == "2008", "model_use_allowed"].all()
    assert (wr["race_time"].str.endswith("Z")).all()          # original values unchanged
    assert set(flags["race_id"]) == {"2005_01", "2005_02", "2005_03", "2005_04"}


def test_hourly_frame_keeps_values_unchanged():
    response = {"hourly": {"time": ["2020-01-01T00:00", "2020-01-01T01:00"], "temperature_2m": [1.5, None]}}
    df = hourly_frame(response, ["temperature_2m"])
    assert df["temperature_2m"].iloc[0] == 1.5 and pd.isna(df["temperature_2m"].iloc[1])
    assert str(df["time_utc"].dt.tz) == "UTC"
    assert window_summary(align(df, pd.Timestamp("2020-01-01 00:00", tz="UTC"), "UTC"), (0, 1), "x_")["x_temp_mean_c"] == 1.5
