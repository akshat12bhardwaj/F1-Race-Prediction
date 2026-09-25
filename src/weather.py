"""
Step 6: historical race weather from the Open-Meteo Historical Weather (archive) API.

IMPORTANT (blueprint 19.6): these values are HISTORICAL REANALYSIS (ERA5-based
model reconstruction on a grid of about 9-25 km), not observations at the track
and NOT a forecast that was available before the race. Race-window values are
an "oracle-forecast proxy" only.

Time handling
-------------
- Weather is requested in UTC (timezone=GMT), so timestamps are unambiguous.
- Open-Meteo's "timezone=auto" applies ONE fixed, current UTC offset to every
  date (verified Step 6, D-013), so it is used ONLY to look up each circuit's
  IANA timezone name (one small cached request per circuit). Local times are
  computed with Python's zoneinfo, which has historical daylight-saving rules.
- 2005-2025: official race start = race_date + race_time (UTC) from Jolpica.
- 1950-2004: no official start time exists in the data; the start is ASSUMED to
  be 14:00 local time (config) and flagged start_time_source = "imputed_local_14h".
- Windows are hour offsets relative to H0 = start rounded down to the full hour.
  Instantaneous variables use values at H0+a..H0+b; accumulated variables
  (precipitation, rain = total of the PRECEDING hour) use H0+a+1..H0+b.

Indianapolis 500 races (P-11) are not requested and are never model-usable.

Run from the project root:   python -m src.weather
"""

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.config import PROJECT_ROOT, load_config

logger = logging.getLogger(__name__)

RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
ACCUMULATED = {"precipitation", "rain"}
EXPECTED_UNITS = {"temperature_2m": "°C", "relative_humidity_2m": "%", "precipitation": "mm", "rain": "mm",
                  "cloud_cover": "%", "wind_speed_10m": "km/h", "wind_direction_10m": "°", "weather_code": "wmo code"}
# Physically plausible hourly ranges: values outside are FLAGGED, never removed
PLAUSIBLE = {"temperature_2m": (-30, 50), "relative_humidity_2m": (0, 100), "precipitation": (0, 100),
             "rain": (0, 100), "cloud_cover": (0, 100), "wind_speed_10m": (0, 150),
             "wind_direction_10m": (0, 360), "weather_code": (0, 99)}
VALID_WMO_CODES = {0, 1, 2, 3, 45, 48, 51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 71, 73, 75, 77, 80, 81, 82,
                   85, 86, 95, 96, 99}
ERAS = [(1950, 1969), (1970, 1989), (1990, 2009), (2010, 2025)]


class WeatherAPIError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


# ----------------------------------------------------------------------
# Cached client (same behaviour as the Jolpica client: cache, spacing, retries)
# ----------------------------------------------------------------------
class OpenMeteoClient:
    def __init__(self, config=None, cache_dir=None, session=None, sleep=time.sleep):
        config = config or load_config()
        api, om = config["api"], config["api"]["openmeteo"]
        self.url = om["archive_url"]
        self.min_interval = float(om["min_seconds_between_requests"])
        self.timeout = api["timeout_seconds"]
        self.max_retries = int(api["max_retries"])
        self.backoff_base = float(api["backoff_base_seconds"])
        self.cache_dir = Path(cache_dir) if cache_dir else PROJECT_ROOT / config["paths"]["data_raw_openmeteo"]
        self.manifest_path = self.cache_dir / "_manifest.csv"        # separate from the Jolpica manifest
        self.failed_path = self.cache_dir / "_failed_requests.csv"
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": api["user_agent"]})
        self._sleep = sleep
        self._last_request_time = None
        self.stats = {"cache_hits": 0, "network_requests": 0, "retries": 0, "failures": 0}

    @staticmethod
    def canonical_params(params):
        """Sorted, text-only parameters: identical requests always give identical keys."""
        return {k: (",".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v))
                for k, v in sorted(params.items())}

    def cache_path(self, kind, name, params):
        """data/raw/openmeteo/{kind}/{name}__{10-char hash of all parameters}.json (deterministic)."""
        digest = hashlib.sha1(json.dumps(self.canonical_params(params), sort_keys=True).encode()).hexdigest()[:10]
        return self.cache_dir / kind / f"{name}__{digest}.json"

    def get_json(self, kind, name, params, refresh=False):
        params = self.canonical_params(params)
        path = self.cache_path(kind, name, params)
        if path.exists() and not refresh:
            try:
                data = json.loads(path.read_bytes())
                if "hourly" in data:
                    self.stats["cache_hits"] += 1
                    logger.debug("Cache hit: %s", path.name)
                    return data
            except ValueError:
                pass
            logger.warning("Unreadable cache file, downloading again: %s", path.name)
        response = self._request(params)
        try:
            data = json.loads(response.content)
        except ValueError:
            data = None
        if not isinstance(data, dict) or "hourly" not in data or "time" not in data["hourly"]:
            self._fail(params, "response without hourly data")
            raise WeatherAPIError("response without hourly data", 200)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_bytes(response.content)
        os.replace(tmp, path)
        self._log_csv(self.manifest_path, ["fetched_at_utc", "kind", "name", "params", "status", "n_bytes", "cache_file"],
                      [_now(), kind, name, json.dumps(params, sort_keys=True), response.status_code,
                       len(response.content), path.relative_to(self.cache_dir).as_posix()])
        return data

    def _request(self, params):
        attempts = self.max_retries + 1
        last = None
        for attempt in range(1, attempts + 1):
            if self._last_request_time is not None:
                wait = self.min_interval - (time.monotonic() - self._last_request_time)
                if wait > 0:
                    self._sleep(wait)
            self._last_request_time = time.monotonic()
            try:
                response = self.session.get(self.url, params=params, timeout=self.timeout)
                self.stats["network_requests"] += 1
            except (requests.ConnectionError, requests.Timeout) as error:
                last = WeatherAPIError(f"{type(error).__name__}: {error}")
            else:
                if response.status_code == 200:
                    return response
                reason = response.text[:150].replace("\n", " ")
                if response.status_code not in RETRY_STATUS_CODES:
                    self._fail(params, f"HTTP {response.status_code}: {reason}")
                    raise WeatherAPIError(f"HTTP {response.status_code}: {reason}", response.status_code)
                last = WeatherAPIError(f"HTTP {response.status_code}: {reason}", response.status_code)
            if attempt < attempts:
                self.stats["retries"] += 1
                delay = self.backoff_base * 2 ** (attempt - 1)
                logger.warning("Retry %d/%d in %.0f s after: %s", attempt, self.max_retries, delay, last)
                self._sleep(delay)
        self._fail(params, str(last))
        raise last

    def _fail(self, params, error):
        self.stats["failures"] += 1
        logger.error("Weather request failed: %s", error)
        self._log_csv(self.failed_path, ["failed_at_utc", "params", "error"],
                      [_now(), json.dumps(params, sort_keys=True), error])

    @staticmethod
    def _log_csv(path, header, row):
        path.parent.mkdir(parents=True, exist_ok=True)
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(header)
            writer.writerow(row)


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ----------------------------------------------------------------------
# Time alignment
# ----------------------------------------------------------------------
def circuit_timezone(client, circuit_id, lat, lon):
    """IANA timezone name for a circuit (Open-Meteo auto lookup; only the NAME is used)."""
    params = {"latitude": lat, "longitude": lon, "start_date": "2020-01-01", "end_date": "2020-01-01",
              "hourly": "temperature_2m", "timezone": "auto"}
    return client.get_json("timezones", circuit_id, params)["timezone"]


def race_start(race_date, race_time, tz_name, assumed_local_hour):
    """
    Return (start_utc, start_local, start_time_source).
    Official: race_date + race_time is a UTC moment (e.g. Las Vegas 2023: 2023-11-19 06:00Z
    = 22:00 local on 18 Nov). Before 2005 there is no time: assume assumed_local_hour local.
    """
    if isinstance(race_time, str) and race_time:
        start_utc = pd.Timestamp(f"{race_date}T{race_time.rstrip('Z')}", tz="UTC")
        source = "official_utc"
    else:
        start_utc = pd.Timestamp(f"{race_date} {int(assumed_local_hour):02d}:00").tz_localize(tz_name).tz_convert("UTC")
        source = f"imputed_local_{int(assumed_local_hour)}h"
    return start_utc, start_utc.tz_convert(tz_name), source


def weather_request_params(lat, lon, start_utc, variables, days, tz_param="GMT"):
    return {"latitude": lat, "longitude": lon,
            "start_date": (start_utc - pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
            "end_date": (start_utc + pd.Timedelta(days=days)).strftime("%Y-%m-%d"),
            "hourly": list(variables), "timezone": tz_param}


def hourly_frame(response, variables):
    """All hourly rows of a UTC response, values unchanged."""
    hourly = response["hourly"]
    df = pd.DataFrame({"time_utc": pd.to_datetime(hourly["time"]).tz_localize("UTC")})
    for v in variables:
        df[v] = pd.to_numeric(pd.Series(hourly.get(v, [None] * len(df))), errors="coerce")
    return df


def align(hourly, start_utc, tz_name):
    """Add hour_offset relative to H0 (start rounded down to the hour) and local time."""
    h0 = start_utc.floor("h")
    out = hourly.copy()
    out["hour_offset"] = ((out["time_utc"] - h0) / pd.Timedelta(hours=1)).round().astype(int)
    out["time_local"] = out["time_utc"].dt.tz_convert(tz_name)
    return out


def window_rows(aligned, window, variable):
    """Rows used for one variable in window (a, b): instantaneous a..b, accumulated a+1..b."""
    a, b = window
    lo = a + 1 if variable in ACCUMULATED else a
    return aligned[(aligned["hour_offset"] >= lo) & (aligned["hour_offset"] <= b)]


def circular_mean_deg(degrees):
    d = pd.Series(degrees).dropna()
    if d.empty:
        return np.nan
    rad = np.deg2rad(d.to_numpy())
    return float(np.round((np.rad2deg(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) + 360) % 360, 1))


def window_summary(aligned, window, prefix):
    """Simple summaries of the aligned hourly values in one window (no thresholds, no wet flag)."""
    def vals(v):
        if v not in aligned.columns:          # variable absent from the response -> treated as missing
            return pd.Series(dtype=float)
        return window_rows(aligned, window, v)[v]
    a, b = window
    expected_instant = b - a + 1
    s = {
        f"{prefix}temp_mean_c": vals("temperature_2m").mean(),
        f"{prefix}temp_min_c": vals("temperature_2m").min(),
        f"{prefix}temp_max_c": vals("temperature_2m").max(),
        f"{prefix}humidity_mean_pct": vals("relative_humidity_2m").mean(),
        f"{prefix}precip_sum_mm": vals("precipitation").sum(min_count=1),
        f"{prefix}precip_max_hour_mm": vals("precipitation").max(),
        f"{prefix}rain_sum_mm": vals("rain").sum(min_count=1),
        f"{prefix}cloud_mean_pct": vals("cloud_cover").mean(),
        f"{prefix}wind_speed_mean_kmh": vals("wind_speed_10m").mean(),
        f"{prefix}wind_speed_max_kmh": vals("wind_speed_10m").max(),
        f"{prefix}wind_dir_mean_deg": circular_mean_deg(vals("wind_direction_10m")),
        f"{prefix}weather_code_max": vals("weather_code").max(),
        f"{prefix}hours_expected": expected_instant,
        f"{prefix}hours_found": int(window_rows(aligned, window, "temperature_2m").shape[0]),
    }
    missing = sum(int(window_rows(aligned, window, v)[v].isna().sum()) if v in aligned.columns
                  else s[f"{prefix}hours_expected"] for v in EXPECTED_UNITS)
    s[f"{prefix}missing_values"] = missing
    return {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) and not pd.isna(v) else v)
            for k, v in s.items()}


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


# ----------------------------------------------------------------------
# Sanity checks (flags only)
# ----------------------------------------------------------------------
def quality_flags(race, response, hourly, aligned, config, grid_km):
    """List of {check, severity, detail} for one race. Nothing is removed or changed."""
    w = config["weather"]
    flags = []

    def add(check, severity, detail):
        flags.append({"race_id": race["race_id"], "check": check, "severity": severity, "detail": detail})

    for var, unit in EXPECTED_UNITS.items():
        got = response.get("hourly_units", {}).get(var)
        if got != unit:
            add("unexpected_unit", "error", f"{var}: expected {unit!r}, got {got!r}")
    for var, (lo, hi) in PLAUSIBLE.items():
        bad = hourly[(hourly[var] < lo) | (hourly[var] > hi)]
        if len(bad):
            add(f"implausible_{var}", "warning", f"{len(bad)} hourly values outside [{lo}, {hi}], e.g. {bad[var].iloc[0]}")
    codes = set(hourly["weather_code"].dropna().astype(int)) - VALID_WMO_CODES
    if codes:
        add("unknown_weather_code", "warning", f"codes {sorted(codes)}")
    rain_gt_precip = hourly[hourly["rain"] > hourly["precipitation"] + 1e-9]
    if len(rain_gt_precip):
        add("rain_exceeds_precipitation", "warning", f"{len(rain_gt_precip)} hours")
    if hourly["time_utc"].duplicated().any():
        add("duplicate_timestamps", "error", f"{int(hourly['time_utc'].duplicated().sum())} duplicates")
    steps = hourly["time_utc"].diff().dropna()
    if (steps != pd.Timedelta(hours=1)).any():
        add("irregular_timestamps", "error", "hourly timestamps are not evenly spaced")
    lo = min(w["prestart_window_hours"][0], w["race_window_hours"][0])
    hi = max(w["prestart_window_hours"][1], w["race_window_hours"][1])
    have = set(aligned["hour_offset"])
    missing_hours = [o for o in range(lo, hi + 1) if o not in have]
    if missing_hours:
        add("missing_window_timestamps", "error", f"offsets {missing_hours} not in response")
    in_window = aligned[(aligned["hour_offset"] >= lo) & (aligned["hour_offset"] <= hi)]
    nulls = {v: int(in_window[v].isna().sum()) for v in EXPECTED_UNITS if in_window[v].isna().any()}
    if nulls:
        add("missing_values_in_window", "error", str(nulls))
    if race["start_time_source"] == "official_utc":
        local_hour = race["start_local"].hour
        if not 10 <= local_hour <= 23:
            add("unusual_local_start_hour", "warning", f"official start at {race['start_local']:%H:%M} local")
        if race["start_local"].strftime("%Y-%m-%d") != race["race_date"]:
            add("local_date_differs_from_api_date", "info",
                f"API date {race['race_date']} (UTC), local start {race['start_local']:%Y-%m-%d %H:%M}")
    else:
        add("imputed_start_time", "info", "no official start time before 2005; 14:00 local assumed")
    if grid_km > w["max_grid_distance_km"]:
        add("grid_cell_far_from_circuit", "warning", f"{grid_km:.1f} km")
    return flags


# ----------------------------------------------------------------------
# One race / all races
# ----------------------------------------------------------------------
RACE_META = ["season", "round", "race_id", "race_name", "race_date", "race_time", "circuit_id", "circuit_lat",
             "circuit_lon", "country", "is_indianapolis_500"]


def process_race(race, tz_name, client, config):
    """Returns (race_row dict, aligned hourly DataFrame or None, list of flags)."""
    w = config["weather"]
    variables = config["api"]["openmeteo"]["hourly_variables"]
    row = {k: race[k] for k in RACE_META}
    row["timezone"] = tz_name
    flags = []
    lat, lon = float(race["circuit_lat"]), float(race["circuit_lon"])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat == 0 and lon == 0):
        flags.append({"race_id": race["race_id"], "check": "invalid_circuit_coordinates", "severity": "error",
                      "detail": f"{lat}, {lon}"})
    start_utc, start_local, source = race_start(race["race_date"], race["race_time"], tz_name,
                                                w["assumed_local_start_hour_before_2005"])
    row.update({"start_time_source": source, "start_utc": start_utc.isoformat(),
                "start_local": start_local.isoformat(), "utc_offset_hours": start_local.utcoffset().total_seconds() / 3600})
    if race["is_indianapolis_500"] == "True":
        row.update({"weather_status": "excluded_indianapolis_500", "model_use_allowed": False})
        return row, None, flags
    params = weather_request_params(race["circuit_lat"], race["circuit_lon"], start_utc, variables,
                                    w["request_days_before_after"], config["api"]["openmeteo"]["timezone"])
    try:
        response = client.get_json("weather", race["race_id"], params)
    except WeatherAPIError as error:
        row.update({"weather_status": "failed", "failure_reason": str(error), "model_use_allowed": False})
        flags.append({"race_id": race["race_id"], "check": "request_failed", "severity": "error", "detail": str(error)})
        return row, None, flags

    hourly = hourly_frame(response, variables)
    aligned = align(hourly, start_utc, tz_name)
    grid_km = haversine_km(lat, lon, response["latitude"], response["longitude"])
    row.update({"grid_lat": response["latitude"], "grid_lon": response["longitude"],
                "grid_distance_km": round(grid_km, 2), "grid_elevation_m": response.get("elevation")})
    row.update(window_summary(aligned, w["prestart_window_hours"], "wx_prestart_"))
    row.update(window_summary(aligned, w["race_window_hours"], "wx_race_"))
    race_ctx = {**race, "start_time_source": source, "start_local": start_local}
    flags += quality_flags(race_ctx, response, hourly, aligned, config, grid_km)
    has_error = any(f["severity"] == "error" for f in flags)
    row.update({"weather_status": "ok_with_errors" if has_error else "ok", "model_use_allowed": not has_error})

    lo, hi = w["hourly_table_offsets"]
    keep = aligned[(aligned["hour_offset"] >= lo) & (aligned["hour_offset"] <= hi)].copy()
    pre, rw = w["prestart_window_hours"], w["race_window_hours"]
    keep["in_prestart_window"] = keep["hour_offset"].between(pre[0], pre[1])
    keep["in_race_window"] = keep["hour_offset"].between(rw[0], rw[1])
    for col in ["season", "round", "race_id"]:
        keep[col] = race[col]
    keep["time_utc"] = keep["time_utc"].dt.strftime("%Y-%m-%dT%H:%MZ")
    keep["time_local"] = keep["time_local"].map(lambda t: t.isoformat())
    return row, keep, flags


def _clock_hours(values):
    """'HH:MM...' text -> decimal hours."""
    s = pd.Series(values).astype(str)
    return s.str[:2].astype(int) + s.str[3:5].astype(int) / 60


def _resultant_length(hours):
    """How tightly hours cluster on a 24-h circle: 1 = all identical, 0 = spread evenly."""
    angles = 2 * np.pi * np.asarray(hours, dtype=float) / 24
    return float(np.hypot(np.cos(angles).mean(), np.sin(angles).mean())) if len(angles) else np.nan


# If official times were real UTC, the resulting LOCAL start hours of a season cluster
# (R >= 0.83 in every season 2007-2025). If a season's local hours are this scattered
# while the raw clock values are MORE clustered, the "UTC" values are really local clock times.
LOCAL_CLOCK_R_THRESHOLD = 0.75


def start_time_consistency(weather_race):
    """Season-level check of the official race_time values (flags only; times are not changed)."""
    official = weather_race[weather_race["start_time_source"] == "official_utc"]
    rows = []
    for season, g in official.groupby("season", sort=False):
        r_local = _resultant_length(_clock_hours(g["start_local"].str[11:16]))
        r_clock = _resultant_length(_clock_hours(g["race_time"]))
        rows.append({"season": season, "official_races": len(g), "R_local_if_utc": round(r_local, 3),
                     "R_clock_values": round(r_clock, 3),
                     "suspected_local_clock": bool(r_local < LOCAL_CLOCK_R_THRESHOLD and r_clock > r_local)})
    return pd.DataFrame(rows, columns=["season", "official_races", "R_local_if_utc", "R_clock_values",
                                       "suspected_local_clock"])


def apply_start_time_check(weather_race, flags):
    """Flag races of suspected seasons (error severity -> not model-usable). Values are not changed."""
    check = start_time_consistency(weather_race)
    suspect = set(check.loc[check["suspected_local_clock"], "season"])
    mask = weather_race["season"].isin(suspect) & (weather_race["start_time_source"] == "official_utc")
    weather_race["race_time_suspected_local"] = mask
    extra = [{"race_id": r["race_id"], "check": "race_time_suspected_local_clock", "severity": "error",
              "detail": f"race_time {r['race_time']} treated as UTC gives local start {r['start_local'][11:16]}; "
                        f"season {r['season']} values look like local clock times"}
             for _, r in weather_race[mask].iterrows()]
    has_weather = weather_race["weather_status"] == "ok"
    weather_race.loc[mask & has_weather, "weather_status"] = "ok_with_errors"
    weather_race.loc[mask, "model_use_allowed"] = False
    return weather_race, pd.concat([flags, pd.DataFrame(extra, columns=flags.columns)], ignore_index=True), check


def collect_weather(races, client, config):
    """Process every race (in the given order). Returns (weather_race, weather_hourly, flags, timezones)."""
    tz_rows = []
    tz_by_circuit = {}
    for circuit_id, g in races.groupby("circuit_id", sort=True):
        first = g.iloc[0]
        try:
            tz_by_circuit[circuit_id] = circuit_timezone(client, circuit_id, first["circuit_lat"], first["circuit_lon"])
        except WeatherAPIError as error:
            tz_by_circuit[circuit_id] = "UTC"
            logger.error("Timezone lookup failed for %s (%s); UTC used", circuit_id, error)
        tz_rows.append({"circuit_id": circuit_id, "country": first["country"], "circuit_lat": first["circuit_lat"],
                        "circuit_lon": first["circuit_lon"], "timezone": tz_by_circuit[circuit_id],
                        "races": len(g)})

    race_rows, hourly_parts, flags = [], [], []
    for i, (_, race) in enumerate(races.iterrows(), 1):
        row, hourly, race_flags = process_race(race, tz_by_circuit[race["circuit_id"]], client, config)
        race_rows.append(row)
        flags += race_flags
        if hourly is not None:
            hourly_parts.append(hourly)
        if i % 100 == 0:
            logger.info("%d/%d races | %s", i, len(races), client.stats)
    hourly_cols = ["season", "round", "race_id", "time_utc", "time_local", "hour_offset", "in_prestart_window",
                   "in_race_window"] + list(config["api"]["openmeteo"]["hourly_variables"])
    weather_hourly = pd.concat(hourly_parts, ignore_index=True)[hourly_cols] if hourly_parts else \
        pd.DataFrame(columns=hourly_cols)
    weather_race = pd.DataFrame(race_rows)
    flags = pd.DataFrame(flags, columns=["race_id", "check", "severity", "detail"])
    return weather_race, weather_hourly, flags, pd.DataFrame(tz_rows)


# ----------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------
def era_of(season):
    s = int(season)
    return next(f"{a}-{b}" for a, b in ERAS if a <= s <= b)


def coverage_reports(weather_race, weather_hourly, variables):
    wr = weather_race.copy()
    wr["has_weather"] = wr["weather_status"].isin(["ok", "ok_with_errors"])
    wr["era"] = wr["season"].map(era_of)

    def summarise(g):
        eligible = g[g["weather_status"] != "excluded_indianapolis_500"]
        return pd.Series({
            "races": len(g), "excluded_indianapolis_500": int((g["weather_status"] == "excluded_indianapolis_500").sum()),
            "eligible_races": len(eligible), "races_with_weather": int(eligible["has_weather"].sum()),
            "races_without_weather": int((~eligible["has_weather"]).sum()),
            "failed_requests": int((eligible["weather_status"] == "failed").sum()),
            "races_with_quality_errors": int((eligible["weather_status"] == "ok_with_errors").sum()),
            "model_use_allowed": int(eligible["model_use_allowed"].astype(bool).sum()),
            "coverage_pct": round(100 * eligible["has_weather"].mean(), 2) if len(eligible) else np.nan,
            "official_start_times": int((g["start_time_source"] == "official_utc").sum()),
            "imputed_start_times": int((g["start_time_source"] != "official_utc").sum()),
        })
    by_season = wr.groupby("season", sort=False).apply(summarise, include_groups=False).reset_index()
    by_era = wr.groupby("era").apply(summarise, include_groups=False).reset_index().rename(columns={"era": "group"})
    overall = summarise(wr).to_frame().T.assign(group="all 1950-2025")
    summary = pd.concat([overall, by_era], ignore_index=True)[["group"] + list(overall.columns.drop("group"))]
    missing_vars = pd.DataFrame([{"variable": v,
                                  "hourly_values": len(weather_hourly),
                                  "missing_hourly_values": int(weather_hourly[v].isna().sum()) if len(weather_hourly) else 0,
                                  "races_with_missing_in_window": int(
                                      weather_hourly[(weather_hourly["in_race_window"] | weather_hourly["in_prestart_window"])
                                                     & weather_hourly[v].isna()]["race_id"].nunique()) if len(weather_hourly) else 0}
                                 for v in variables])
    return by_season, summary, missing_vars


# Races widely reported as wet (general knowledge, NOT verified from an official source). Used only
# as a plausibility check of the reanalysis precipitation; no threshold is decided here.
REFERENCE_WET_RACES = [(1996, "Spanish Grand Prix"), (1998, "Belgian Grand Prix"), (2007, "Japanese Grand Prix"),
                       (2008, "British Grand Prix"), (2008, "Monaco Grand Prix"), (2011, "Canadian Grand Prix"),
                       (2016, "Brazilian Grand Prix"), (2021, "Belgian Grand Prix"), (2024, "São Paulo Grand Prix")]


def wet_race_check(weather_race):
    ref = pd.DataFrame(REFERENCE_WET_RACES, columns=["season", "race_name"]).astype({"season": str})
    cols = ["season", "race_name", "race_id", "start_time_source", "start_local", "wx_prestart_precip_sum_mm",
            "wx_race_precip_sum_mm", "wx_race_precip_max_hour_mm", "wx_race_weather_code_max", "weather_status"]
    out = ref.merge(weather_race.reindex(columns=cols), on=["season", "race_name"], how="left")
    out["reanalysis_shows_precipitation"] = (out["wx_race_precip_sum_mm"].fillna(0) > 0) | \
                                            (out["wx_prestart_precip_sum_mm"].fillna(0) > 0)
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def run(races, client, config, interim_dir, tables_dir, prefix=""):
    variables = config["api"]["openmeteo"]["hourly_variables"]
    weather_race, weather_hourly, flags, timezones = collect_weather(races, client, config)
    weather_race, flags, start_check = apply_start_time_check(weather_race, flags)
    by_season, summary, missing_vars = coverage_reports(weather_race, weather_hourly, variables)
    outputs = {
        interim_dir / f"{prefix}weather_race.csv": weather_race,
        interim_dir / f"{prefix}weather_hourly.csv": weather_hourly,
        tables_dir / f"{prefix}weather_coverage_by_season.csv": by_season,
        tables_dir / f"{prefix}weather_coverage_summary.csv": summary,
        tables_dir / f"{prefix}weather_missing_variables.csv": missing_vars,
        tables_dir / f"{prefix}weather_quality_report.csv": flags,
        tables_dir / f"{prefix}weather_circuit_timezones.csv": timezones,
        tables_dir / f"{prefix}weather_wet_race_check.csv": wet_race_check(weather_race),
        tables_dir / f"{prefix}weather_start_time_check.csv": start_check,
    }
    for path, df in outputs.items():
        df.to_csv(path, index=False, encoding="utf-8")
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Collect historical race weather (Open-Meteo archive, cached).")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--output-prefix", default="", help="e.g. 'subset_' for a test run")
    args = parser.parse_args(argv)

    config = load_config()
    interim = PROJECT_ROOT / config["paths"]["data_interim"]
    tables = PROJECT_ROOT / config["paths"]["tables"]
    logs = PROJECT_ROOT / config["paths"]["logs"]
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(logs / f"weather_{stamp}.log", encoding="utf-8")])

    races = pd.read_csv(interim / "races_policy.csv", dtype=str, keep_default_na=False, na_values=[""])
    if args.start or args.end:
        s = races["season"].astype(int)
        races = races[(s >= (args.start or 1950)) & (s <= (args.end or 2025))]
    client = OpenMeteoClient(config=config)
    began = time.monotonic()
    outputs = run(races, client, config, interim, tables, prefix=args.output_prefix)
    runtime = time.monotonic() - began
    logger.info("Weather stats: %s | runtime %.1f min", client.stats, runtime / 60)
    run_log = logs / "weather_runs.csv"
    pd.DataFrame([{"finished_at_utc": _now(), "races": len(races), **{f"stat_{k}": v for k, v in client.stats.items()},
                   "runtime_seconds": round(runtime, 1)}]).to_csv(run_log, mode="a", header=not run_log.exists(), index=False)
    return outputs


if __name__ == "__main__":
    main()
