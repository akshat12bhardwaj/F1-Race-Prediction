"""
One-off asset collection from Wikimedia Commons: driver portraits and constructor logos for the app's pool.

Identity: the lead (infobox) image of the Wikipedia article that the F1 data source (Jolpica) itself links for that
driver or constructor id, so the image belongs to the right person / team entity (rebrands such as Toro Rosso,
AlphaTauri and RB are separate ids with separate articles). Licence: kept ONLY if Commons lists a free licence
(Creative Commons, CC0, public domain, UK OGL). Non-free logos (fair use) are never downloaded; the app shows its
own neutral badge instead. Thumbnails are stored locally; authors and licences go to
frontend/src/{portraits,logos}.json and the public page frontend/public/credits.html.

Run:  python -m src.fetch_portraits [teams] [drivers]     (needs internet; not part of the model pipeline)
"""

import html
import hashlib
import io
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from src.config import PROJECT_ROOT

FRONT = PROJECT_ROOT / "frontend"
UA = {"User-Agent": "apex-f1-predictor/1.0 (local coursework app; asset fetch script)"}
API = "https://en.wikipedia.org/w/api.php?"
FREE = re.compile(r"^(cc[ -]|cc0|public domain|pd|ogl)", re.I)   # OGL = UK Open Government Licence
KINDS = {"drivers": ("drivers", "portraits.json", "Driver", 250), "teams": ("constructors", "logos.json", "Constructor", 160)}
# reviewed by eye: lead image is a car photo, not a logo (dallara, eurobrun, lambo, lola, spirit, team_lotus), or it carries 2026 sponsor
# branding that no 1981-2025 car wore (haas: TGR, mclaren: Mastercard). These teams get the app's own badge instead.
REJECT = {"dallara", "eurobrun", "lambo", "lola", "spirit", "team_lotus", "haas", "mclaren",
          # drivers whose lead image is a car or a helmet, not a face
          "fabi", "henton", "modena", "ghinzani", "caffi", "gugelmin"}
# era of a logo that carries a period title sponsor: shown only on cars of those seasons (others get the badge)
LOGO_SEASONS = {"williams": [2025, 2025], "alpine": [2022, 2025], "racing_point": [2019, 2020], "simtek": [1994, 1995],
                "mercedes": [2010, 2025]}


def get(url, raw=False):
    for wait in (1.5, 10, 30, 60, 120, 240):                 # polite pacing; back off when the server says 429
        time.sleep(wait)
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read() if raw else json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
        except (TimeoutError, urllib.error.URLError):   # slow or dropped connection: wait and retry
            pass
    raise RuntimeError(f"rate limited: {url}")


def source_urls(kind):
    """{id: wikipedia url} from the raw Jolpica pages (latest season wins)."""
    out = {}
    for f in sorted((PROJECT_ROOT / "data" / "raw" / "jolpica").glob(f"*/{kind}__*.json")):
        table = json.loads(f.read_text(encoding="utf-8"))["MRData"]
        rows = table.get("DriverTable", {}).get("Drivers") or table.get("ConstructorTable", {}).get("Constructors") or []
        for r in rows:
            out[r.get("driverId") or r.get("constructorId")] = r.get("url")
    return out


def credits_page():
    rows = []
    for name, label in (("portraits.json", "Driver photographs"), ("logos.json", "Constructor logos")):
        p = FRONT / "src" / name
        items = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
        rows.append(f"<h2>{label} ({sum(len(it.get('variants', [it])) for it in items.values())} images)</h2><ul>" + "".join(
            f'<li><b>{html.escape(v["article"])}</b>: {html.escape(v["author"])}, '
            f'<a href="{html.escape(v["source"])}">{html.escape(v["licence"])}</a>, via Wikimedia Commons'
            + (f' · <a href="{html.escape(v["licence_url"])}">Licence terms</a>' if v.get("licence_url") else '')
            + (f' · {html.escape(v["changes"])}' if v.get("changes") else '')
            + (f' · {html.escape(v["usage_note"])}' if v.get("usage_note") else '') + '</li>'
            for v in sorted((x for it in items.values() for x in it.get("variants", [it])), key=lambda v: v["article"])) + "</ul>")
    (FRONT / "public" / "credits.html").write_text(
        "<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>APEX image credits</title><style>body{background:#0b0c0d;color:#f4f5f2;font:14px/1.5 system-ui,sans-serif;"
        "max-width:900px;margin:0 auto;padding:24px}a{color:#d2f03c}h1{font-style:italic}li{margin:2px 0}</style>"
        "<h1>APEX · image credits and licences</h1><p>Images are used under the licences listed. Constructor names and "
        "logos are trademarks of their owners and are shown only to identify the teams. Teams without a freely "
        "licensed logo are shown with an original badge.</p>"
        '<p><a href="/models/ATTRIBUTION.html">3D car model credits, licenses and historical limitations</a></p>'
        + "".join(rows) + "</html>", encoding="utf-8")


def crop_portrait(raw, box):
    """Apply a manually reviewed face crop; never stretch or silently pad a bad box."""
    from PIL import Image, ImageOps
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    left, top, right, bottom = box
    if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height
            and right - left == bottom - top):
        raise ValueError("Portrait crop must be a square inside the source image")
    out = io.BytesIO()
    image.crop(box).resize((256, 256), Image.Resampling.LANCZOS).save(
        out, "JPEG", quality=84, optimize=True, progressive=True)
    return out.getvalue()


def import_reviewed(cache):
    """Reproduce approved crops from downloaded originals, checking their exact identity."""
    manifest = FRONT / "src" / "portraits.json"
    credits = json.loads(manifest.read_text(encoding="utf-8"))
    reviewed = json.loads((Path(__file__).with_name("reviewed_portraits.json")).read_text(encoding="utf-8"))
    for driver_id, item in reviewed.items():
        if driver_id in credits:
            continue  # preserve all existing photographs
        raw = (Path(cache) / item["original_file"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["original_sha256"]:
            raise ValueError(f"Unreviewed source bytes for {driver_id}")
        output = crop_portrait(raw, item["crop"])
        (FRONT / "public" / "drivers" / f"{driver_id}.jpg").write_bytes(output)
        credits[driver_id] = {k: v for k, v in item.items() if k != "original_file"}
    manifest.write_text(json.dumps(credits, indent=1, ensure_ascii=False), encoding="utf-8")
    credits_page()


def fetch(kind, ids):
    folder, manifest, label, width = KINDS[kind]
    out_dir = FRONT / "public" / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    path = FRONT / "src" / manifest
    credits = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    urls = source_urls("drivers" if kind == "drivers" else "constructors")
    done = skipped = 0
    for i in ids:
        if i in credits or i in REJECT or not urls.get(i):
            continue
        title = urllib.parse.unquote(urls[i].rsplit("/wiki/", 1)[-1]).replace("_", " ")
        q = get(API + urllib.parse.urlencode({"action": "query", "format": "json", "redirects": 1, "titles": title,
                                              "prop": "pageimages", "piprop": "name"}))
        page = next(iter(q["query"]["pages"].values()))
        name = page.get("pageimage")
        if not name:
            skipped += 1; continue
        r = get(API + urllib.parse.urlencode({"action": "query", "format": "json", "titles": f"File:{name}",
                                              "prop": "imageinfo", "iiprop": "url|extmetadata", "iiurlwidth": width}))
        info = next(iter(r["query"]["pages"].values())).get("imageinfo", [{}])[0]
        m = info.get("extmetadata", {})
        lic = m.get("LicenseShortName", {}).get("value", "")
        if not FREE.match(lic) or "thumburl" not in info:
            print(f"{label} {i}: '{lic}' not free, skipped"); skipped += 1; continue
        ext = Path(urllib.parse.urlparse(info["thumburl"]).path).suffix.lower() or ".png"
        (out_dir / f"{i}{ext}").write_bytes(get(info["thumburl"], raw=True))
        artist = html.unescape(re.sub(r"<[^>]+>", "", m.get("Artist", {}).get("value", ""))).strip() or "Unknown"
        credits[i] = {"article": page["title"], "file": name, "author": artist[:120], "licence": lic,
                      "source": info["descriptionurl"], "path": f"/{folder}/{i}{ext}"} | (
                         {"seasons": LOGO_SEASONS[i]} if kind == "teams" and i in LOGO_SEASONS else {})
        path.write_text(json.dumps(credits, indent=1, ensure_ascii=False), encoding="utf-8")
        done += 1
        print(f"{label} {i}: {lic}")
    credits_page()
    print(f"{kind}: {done} new, {skipped} skipped, {len(credits)} stored")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "reviewed":
        import_reviewed(sys.argv[2])
        return
    from src import simulate
    P = simulate.history_pool()
    kinds = sys.argv[1:] or ["teams", "drivers"]
    if "teams" in kinds:
        fetch("teams", sorted({c["constructor_id"] for c in P["cars"]}))
    if "drivers" in kinds:
        fetch("drivers", [d["driver_id"] for d in sorted(P["drivers"], key=lambda d: -d["net_race_skill"])])


if __name__ == "__main__":
    main()
