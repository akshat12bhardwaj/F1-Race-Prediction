"""
Second asset pass (D-034): driver portraits and era-specific constructor logos via Wikidata.

Identity chain: Jolpica id -> the Wikipedia article Jolpica links -> its Wikidata item -> that item's
  P18  (image of the person; for drivers) and
  P154 (logo image, often several with P580 start / P582 end qualifiers = era variants; for constructors).
Every candidate is a Wikimedia Commons file; its licence is still checked (free licences only, src.fetch_portraits.FREE).
Portraits are cropped to one square format (upper-centre crop, 256 px JPEG); logos are kept as transparent PNG (200 px).
Licences and authors go to frontend/src/{portraits,logos}.json and the credits page.

Run:  python -m src.fetch_wikidata_assets [drivers] [teams]      (needs internet; not part of the model pipeline)
"""

import io
import json
import sys
import urllib.parse

from PIL import Image, ImageChops

from src.config import PROJECT_ROOT
from src.fetch_portraits import FREE, FRONT, credits_page, get, html, re, source_urls

WIKI = "https://en.wikipedia.org/w/api.php?"
WD = "https://www.wikidata.org/w/api.php?"
COMMONS = "https://commons.wikimedia.org/w/api.php?"
SIZE = 256
# reviewed by eye after a run: Wikidata image is not a clear face (car, helmet, group, far away) -> monogram instead
BAD_PORTRAIT = {
 "baumgartner": "ChampCar_2007_Baumgartner.jpg",
 "bernard": "Cadillac_Northstar_LMP_02_-_Emmanuel_Collard,_Eric_Bernard_&_J.J._Lehto_at_the_2002_Le_Mans_(54294864286).jpg",
 "campos": "Ferrar1_333SP_-_Michel_Ferte,_Adrian_Campos_&_Charles_Nearburg_entering_Dunlop_Chicane_at_the_1997_Le_Mans_(51635711376).jpg",
 "inoue": "Taki_Inoue_1995_British_GP.jpg",
 "giacomelli": "Bruno_Giacomelli_1982.jpg",
 "beretta": "Oliver_Gavin,_Olivier_Bereta_&_Jan_Magnussen_celebrate_their_GTS_class_win_on_the_podium_at_the_2004_Le_Mans_(50884349326).jpg",
 "dalmas": "Yannick_DALMAS.JPG",
 "martini": "Pierluigi_Martini_celebrates_on_the_podium_at_the_1999_Le_Mans_(51889440091).jpg",
 "nannini": "Alessandro_Nannini_-_Alfa_Corse_-_Alfa_Romeo_155_V6_TI_94_(46745279511).jpg",
 "rebaque": "1981_Argentine_Grand_Prix,_Rebaque.jpg",
 "rosberg": "Nico_Rosberg_obtuvo_el_segundo_lugar.jpg",
 "pizzonia": "Fale_F1_Monza_2004_13.jpg",
 "satoru_nakajima": "Satoru_Nakajima_2008_Motorsport_Japan_(cropped).jpg",
 "tarquini": "こうしてみるとただの爺さん_(15506042327).jpg"
}   # id -> lead image reviewed as unusable
# reviewed by eye: logo file that is not this team's mark, or (season ranges) where Wikidata has no dates
BAD_LOGO = set()
LOGO_SEASONS = {   # reviewed by eye: the team's actual F1 years for logos Wikidata gives no (or a wider) date range
    ('alpine', 'Alpine F1 Team Logo.svg'): [2021, 2025],
    ('ats', 'Ats logo 2009.svg'): [1977, 1984],
    ('caterham', 'Caterham F1 Team logo (2014).svg'): [2012, 2014],
    ('dallara', 'Dallara logo.svg'): [1988, 1992],
    ('haas', 'Haas F1 Team Logo.svg'): [2016, 2025],
    ('haas', 'MoneyGram Haas F1 Team Logo.svg'): [2023, 2025],
    ('lotus_f1', 'Lotus F1 Team logo.jpg'): [2012, 2015],
    ('mclaren', 'McLaren Racing logo.png'): [2017, 2025],
    ('sauber', 'Logo Sauber F1.png'): [2016, 2018],
    ('sauber', 'Logo_sauber_2023.jpg'): [2023, 2025],
    ('stewart', 'Stewart-logo.jpg'): [1997, 1999],
    ('williams', 'Atlassian Williams F1 Team logo.svg'): [2025, 2025],
    ('williams', 'Atlassian_Williams_F1_Team_logo.svg'): [2025, 2025],
    ('racing_point', 'BWT Racing Point Logo.svg'): [2019, 2020],
}
# Third pass (logo completion): Commons files picked by eye from contact sheets of Commons search hits (free licences
# only). Seasons = the years the file's own description / design dates it to, clipped to that constructor id's seasons.
# Rejected on review: car-nose photos (brabham, marussia), user-drawn approximations (Minardi .gif marks, "Mini Free
# Logo" set), other entities' marks (Brun Motorsport for eurobrun, Formula E Jaguar/Virgin), strips too thin to read
# in a tile (Vodafone McLaren Mercedes, Williams Martini Racing), and Ligier's modern "Automobiles" marks.
LOGO_PICKS = {
 'alfa': [('Alfa Romeo F1 Team Stake Logo.svg', [2023, 2023])],   # 1982-85 roundel: only a photo of two badges
 'aston_martin': [('Aston Martin F1 Team logo 2024.jpg', [2024, 2025])],
 'benetton': [('United Colors of Benetton - logo (Italy, 1989-1996).svg', [1989, 1995]),
              ('United Colors of Benetton - logo (Italy, 1996-2011).svg', [1996, 2001])],
 'bmw_sauber': [('BMW Sauber F1 Team logo.png', [2006, 2009])],
 'brawn': [('Brawn GP logo.svg', [2009, 2009])],
 'coloni': [('Enzo coloni racing logo.svg', [1987, 1991])],
 'ferrari': [('Ferrari wordmark.svg', [1950, 2100])],          # wordmark: no freely licensed Scuderia shield on Commons
 'fondmetal': [('Fondmetal Logo.png', [1991, 1992])],
 'force_india': [('Force India.svg', [2011, 2018])],           # "Sahara Force India" identity (2011-2018)
 'honda': [('Logo Honda F1 Racing.svg', [2006, 2008])],
 'lambo': [('Lamborghini - logo wordmark (italy, 1963-).svg', [1991, 1991])],
 'lola': [('Lola Cars logo.png', [1950, 2100])],
 'lotus_racing': [('Lotus F1 Racing Logo.jpg', [2010, 2010])],
 'manor': [('Manor F1 Team logo.png', [2015, 2015]), ('Manor Racing MRT.jpg', [2016, 2016])],
 'mercedes': [('Logo Mercedes GP Petronas 2011.svg', [2010, 2011]), ('Mercedes AMG Petronas F1 Logo.svg', [2012, 2025])],
 'minardi': [('MinardiF1(1994) logo ico.gif', [1994, 1994]), ('Minardi F1 Team.svg', [2001, 2005])],
 'pacific': [('Logo Pacific Grand Prix.svg', [1994, 1995])],
 'red_bull': [('Red Bull Racing - 2005 Logo.png', [2005, 2012])],
 'renault': [('Renault Vasarely 1981-1992 Logo.png', [1981, 1985]), ('Renault Sport F1 logo as of 2016.png', [2016, 2018]),
             ('Renault F1 Team logo 2019.png', [2019, 2020])],
 'rial': [('Logo Rial.svg', [1988, 1989])],
 'sauber': [('Logo of Stake F1 Team Kick Sauber.png', [2024, 2025])],
 'spyker': [('Logo Spyker.svg', [2007, 2007])],
 'super_aguri': [('Super Aguri logo.svg', [2006, 2007])],
 'theodore': [('Theodore logo.png', [1981, 1983])],
 'toleman': [('Toleman badge, Race Retro Stoneleigh 2018.jpg', [1981, 1985])],
 'toyota': [('Panasonic Toyota Racing logo.svg', [2002, 2009])],
 'tyrrell': [('Tyrrell.svg', [1970, 1998])],
 'virgin': [('Virginracing.png', [2010, 2011])],
 'williams': [('Williams Racing Logo 2020.webp', [2020, 2023]), ('Williams Racing Logo 2024.webp', [2024, 2024])],
}
# replaced by a period mark above: Honda's 2024 "new H" (shown on 2006-08 cars), the generic BMW roundel, and
# Mercedes' 2026 logo (was shown on 2010-2025 cars)
DROP_LOGO = {('honda', 'Honda_new_h_mark.svg'), ('bmw_sauber', 'BMW.svg'),
             ('mercedes', 'Mercedes-AMG_Petronas_F1_Team_logo_(2026).svg')}
CHOICE = {
 "sirotkin": "Sergey Sirotkin SMP Racing.jpg",
 "rosberg": "Nico Rosberg 2016.jpg",
 "beretta": "Olivier Beretta 2012 WEC Fuji.jpg",
 "tarquini": "Gabriele tarquini spafrancorchamps2014.JPG",
 "martini": "Pierluigi Martini in 2016.jpg",
 "dalmas": "Yannick DALMAS.JPG",
 "henton": "Brian Henton (1971).jpg"
}   # id -> Commons file picked by eye from outputs/asset_review (phase 2)


def batched(xs, n=50):
    xs = list(xs)
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def qids(titles):
    """{title: Q-id} through the en.wikipedia page props (redirects followed)."""
    out = {}
    for b in batched(titles):
        r = get(WIKI + urllib.parse.urlencode({"action": "query", "format": "json", "redirects": 1, "prop": "pageprops",
                                               "ppprop": "wikibase_item", "titles": "|".join(b)}))["query"]
        back = {x["to"]: x["from"] for x in r.get("normalized", []) + r.get("redirects", [])}
        for p in r["pages"].values():
            t = p.get("title")
            while t in back:
                t = back[t]
            if "pageprops" in p:
                out[t] = p["pageprops"]["wikibase_item"]
    return out


def claims(ids):
    out = {}
    for b in batched(ids):
        r = get(WD + urllib.parse.urlencode({"action": "wbgetentities", "format": "json", "props": "claims", "ids": "|".join(b)}))
        out |= {k: v.get("claims", {}) for k, v in r["entities"].items()}
    return out


def files(names, width):
    """{file name: imageinfo} for Commons files (url, thumb, licence, author)."""
    out = {}
    for b in batched(names):
        r = get(COMMONS + urllib.parse.urlencode({"action": "query", "format": "json", "prop": "imageinfo",
                                                  "iiprop": "url|extmetadata|size", "iiurlwidth": width,
                                                  "titles": "|".join(f"File:{n}" for n in b)}))["query"]
        norm = {x["to"]: x["from"] for x in r.get("normalized", [])}
        for p in r["pages"].values():
            if "imageinfo" in p:
                out[norm.get(p["title"], p["title"]).removeprefix("File:")] = p["imageinfo"][0]
    return out


def values(cl, prop):
    """[(file, start_year, end_year)] for a property, preferred rank first."""
    out = []
    for c in sorted(cl.get(prop, []), key=lambda c: c.get("rank") != "preferred"):
        v = c["mainsnak"].get("datavalue", {}).get("value")
        if not v or c.get("rank") == "deprecated":
            continue
        q = c.get("qualifiers", {})
        yr = lambda p: int(q[p][0]["datavalue"]["value"]["time"][1:5]) if p in q and "datavalue" in q[p][0] else None
        out.append((v, yr("P580"), yr("P582")))
    return out


def credit(info, title, name):
    m = info.get("extmetadata", {})
    lic = m.get("LicenseShortName", {}).get("value", "")
    if not FREE.match(lic):
        return None
    artist = html.unescape(re.sub(r"<[^>]+>", "", m.get("Artist", {}).get("value", ""))).strip() or "Unknown"
    return {"article": title, "file": name, "author": artist[:120], "licence": lic, "source": info["descriptionurl"]}


def square(raw):
    """One framing for every portrait: square crop, horizontally centred, from just below the top (faces sit in the
    upper part of portrait photos), 256 px."""
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    s = min(w, h)
    top = int(min(h - s, 0.06 * h)) if h > w else 0
    im = im.crop(((w - s) // 2, top, (w - s) // 2 + s, top + s)).resize((SIZE, SIZE), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "JPEG", quality=84, optimize=True, progressive=True)
    return b.getvalue()


def category_files(cat, n=12):
    r = get(COMMONS + urllib.parse.urlencode({"action": "query", "format": "json", "list": "categorymembers", "cmtype": "file",
                                              "cmlimit": n, "cmtitle": f"Category:{cat}"}))
    return [m["title"].removeprefix("File:") for m in r["query"]["categorymembers"] if m["title"].lower().endswith((".jpg", ".jpeg", ".png"))]


def drivers(ids, choose=None):
    """Phase 1 (choose=None): for drivers without a portrait, collect candidates (Wikidata P18 + the driver's Commons
    category, P373), save small previews to outputs/asset_review/ for a person to look at. Phase 2: CHOICE {id: file}
    picked by eye -> licence check, square crop, manifest."""
    path = FRONT / "src" / "portraits.json"
    man = json.loads(path.read_text(encoding="utf-8"))
    urls = source_urls("drivers")
    todo = {i: urllib.parse.unquote(urls[i].rsplit("/wiki/", 1)[-1]).replace("_", " ") for i in ids if i not in man and urls.get(i)}
    if choose:
        info = files(set(choose.values()), 330)
        for i, f in choose.items():
            c = info.get(f) and credit(info[f], todo.get(i, i), f)
            if not c:
                print("not free / missing", i, f); continue
            (FRONT / "public" / "drivers" / f"{i}.jpg").write_bytes(square(get(info[f]["thumburl"], raw=True)))
            man[i] = c | {"path": f"/drivers/{i}.jpg", "via": "wikidata/commons, chosen by eye"}
            path.write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
            print("portrait", i, c["licence"])
        return
    q = qids(todo.values())
    cl = claims({q[t] for t in todo.values() if t in q})
    review = PROJECT_ROOT / "outputs" / "asset_review"
    review.mkdir(parents=True, exist_ok=True)
    cands = {}
    for i, t in todo.items():
        c = cl.get(q.get(t), {})
        fs = [f for f, _, _ in values(c, "P18")]
        for cat, _, _ in values(c, "P373"):
            fs += category_files(cat)
        cands[i] = [f for f in dict.fromkeys(fs) if f != BAD_PORTRAIT.get(i)][:10]
    info = files({f for v in cands.values() for f in v}, 120)
    out = {}
    for i, fs in cands.items():
        ok = [f for f in fs if info.get(f) and credit(info[f], todo[i], f)]
        out[i] = ok
        for k, f in enumerate(ok):
            (review / f"{i}__{k}.jpg").write_bytes(get(info[f]["thumburl"], raw=True))
    (review / "candidates.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print({i: len(v) for i, v in out.items()})


def teams(ids):
    path = FRONT / "src" / "logos.json"
    man = json.loads(path.read_text(encoding="utf-8"))
    urls = source_urls("constructors")
    todo = {i: urllib.parse.unquote(urls[i].rsplit("/wiki/", 1)[-1]).replace("_", " ") for i in ids
            if urls.get(i) and "variants" not in man.get(i, {})}            # resume: skip teams already done
    q = qids(todo.values())
    cl = claims({q[t] for t in todo.values() if t in q})
    cand = {i: values(cl.get(q.get(t), {}), "P154") for i, t in todo.items()}
    info = files({f for v in cand.values() for f, _, _ in v}, 200)
    n = 0
    for i, vs in cand.items():
        variants = []
        for k, (f, a, b) in enumerate(vs):
            c = info.get(f) and credit(info[f], todo[i], f)
            if not c or (i, f) in BAD_LOGO:
                continue
            im = Image.open(io.BytesIO(get(info[f]["thumburl"], raw=True))).convert("RGBA")
            im.thumbnail((200, 200))
            name = f"{i}-{k}.png"
            im.save(FRONT / "public" / "constructors" / name, optimize=True)
            variants.append(c | {"path": f"/constructors/{name}", "seasons": LOGO_SEASONS.get((i, f), [a or 1950, b or 2100])})
        if variants:
            old = man.get(i)
            man[i] = {"variants": variants + ([old] if old and "path" in old and not any(v["file"] == old["file"] for v in variants) else [])}
            path.write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")   # resumable
            n += 1
            print("logo", i, [(v["file"][:40], v["seasons"]) for v in variants])
    path.write_text(json.dumps(man, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"logos: {n} constructors with Wikidata variants, {len(man)} total")


def fit_logo(raw):
    """Trim the file's plain margin (so every mark fills its tile the same way), fit 240x120, transparent PNG."""
    im = Image.open(io.BytesIO(raw)).convert("RGBA")
    r, g, b_, a = ImageChops.difference(im, Image.new("RGBA", im.size, im.getpixel((0, 0)))).split()
    diff = ImageChops.lighter(ImageChops.lighter(r, g), ImageChops.lighter(b_, a))   # alpha too: black ink on transparent
    box = diff.point(lambda p: 255 if p > 24 else 0).getbbox()        # tolerance: JPEG noise in a white margin
    if box:
        m = 2
        im = im.crop((max(box[0] - m, 0), max(box[1] - m, 0), min(box[2] + m, im.width), min(box[3] + m, im.height)))
    im.thumbnail((240, 120), Image.LANCZOS)
    b = io.BytesIO(); im.save(b, "PNG", optimize=True)
    return b.getvalue()


def picked_logos():
    """LOGO_PICKS -> licence check, trim/fit, manifest variants (era ranges); DROP_LOGO variants removed."""
    path = FRONT / "src" / "logos.json"
    man = json.loads(path.read_text(encoding="utf-8"))
    urls = source_urls("constructors")
    info = files({f for v in LOGO_PICKS.values() for f, _ in v}, 480)
    for i, picks in LOGO_PICKS.items():
        e = man.get(i)
        names = {f for f, _ in picks}
        keep = [v for v in (e.get("variants", [e]) if e else []) if (i, v["file"]) not in DROP_LOGO and v["file"] not in names]
        title = urllib.parse.unquote(urls.get(i, i).rsplit("/wiki/", 1)[-1]).replace("_", " ")
        new = []
        for k, (f, yrs) in enumerate(picks):
            c = info.get(f) and credit(info[f], title, f)
            if not c:
                print("not free / missing", i, f); continue
            (FRONT / "public" / "constructors" / f"{i}-p{k}.png").write_bytes(fit_logo(get(info[f]["thumburl"], raw=True)))
            new.append(c | {"path": f"/constructors/{i}-p{k}.png", "seasons": yrs, "via": "commons search, chosen by eye"})
        man[i] = {"variants": keep + new}
        print("logo", i, [(v["file"][:40], v["seasons"]) for v in new])
    path.write_text(json.dumps(dict(sorted(man.items())), indent=1, ensure_ascii=False), encoding="utf-8")


def main():
    if sys.argv[1:] == ["picked"]:
        picked_logos(); credits_page(); return
    from src import simulate
    P = simulate.history_pool()
    kinds = sys.argv[1:] or ["drivers", "teams"]
    if "drivers" in kinds:
        drivers([d["driver_id"] for d in P["drivers"]], CHOICE or None)
    if "teams" in kinds:
        teams(sorted({c["constructor_id"] for c in P["cars"]}))
    credits_page()


if __name__ == "__main__":
    main()
