#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
  * Hanya ulasan publik. Nama pengguna TIDAK disimpan (ID ulasan di-hash).
  * Korelasi/lead-lag yang dihasilkan = ASOSIASI & DAYA PREDIKSI, bukan sebab-akibat.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------
TZ = "Asia/Jakarta"
START_DATE = "2025-04-10"        # tanggal rilis New BIONS di toko aplikasi
LAUNCH_EVENT_DATE = "2025-05-05"  # peluncuran resmi di BEI (liputan media) -> uji sensitivitas
MIN_REVIEWS_PER_APP = 100
TARGET_KEYWORDS = ["saham", "IHSG", "investasi saham", "BIONS", "BNI Sekuritas"]  # maks 5
MARKET_KEYWORDS = ["saham", "IHSG", "investasi saham"]

# Hari ketika IHSG kena trading halt (diverifikasi dari liputan media).
EVENT_DATES = ["2026-01-28", "2026-01-29", "2026-10-16"]

# Jam perdagangan BEI (menit sejak 00:00, WIB). VERIFIKASI ke pengumuman BEI terbaru.
SESSIONS_MON_THU = [(9 * 60, 12 * 60), (13 * 60 + 30, 15 * 60 + 50)]
SESSIONS_FRI = [(9 * 60, 11 * 60 + 30), (14 * 60, 15 * 60 + 50)]

# Model sentimen Indonesia
SENTIMENT_MODEL = "w11wo/indonesian-roberta-base-sentiment-classifier"

DEFAULT_APPS = [
    {"name": "BIONS (BNIS)", "play": "id.bions.bnis.android", "appstore": "6736508566",
     "is_target": True},
    {"name": "Stockbit", "play": "", "appstore": "", "query": "Stockbit Sekuritas", "is_target": False},
    {"name": "Ajaib", "play": "", "appstore": "", "query": "Ajaib saham reksadana", "is_target": False},
    {"name": "IPOT", "play": "", "appstore": "", "query": "IPOT Indo Premier", "is_target": False},
    {"name": "MOST", "play": "", "appstore": "", "query": "MOST Mirae Asset Sekuritas", "is_target": False},
]

DATA_DIR = Path("data")
OUT_DIR = Path("output")
DEMO = False
RNG = np.random.default_rng(42)

# ----------------------------------------------------------------------------
# ASPEK & KAMUS (Bahasa Indonesia + Inggris + slang)
# ----------------------------------------------------------------------------
ASPECTS = {
    "latency": ("Kecepatan & latency", [
        "lemot", "lambat", "lelet", "lag", "lagging", "laggy", "ngelag", "loading", "muter",
        "muter muter", "berat", "delay", "telat", "freeze", "freezing", "hang", "ngehang",
        "slow", "sluggish", "lama banget", "lama sekali", "lama bgt", "loading lama",
        "waiting", "stuck", "nyangkut", "macet"]),
    "stability": ("Stabilitas & error", [
        "error", "eror", "crash", "force close", "fc", "keluar sendiri", "bug", "buggy",
        "bugging", "gagal", "failed", "tidak bisa dibuka", "gak bisa dibuka", "ga bisa dibuka",
        "nggak bisa dibuka", "cant open", "cannot open", "sometimes cant be opened", "down",
        "maintenance", "server", "restart", "not working", "tidak berfungsi", "tidak bisa dibuka"]),
    "login": ("Login & akses", [
        "login", "log in", "masuk", "otp", "pin", "password", "sandi", "verifikasi",
        "verification", "biometrik", "biometric", "face", "wajah", "liveness", "ekyc", "kyc",
        "registrasi", "daftar", "terkunci", "locked", "sign in"]),
    "data": ("Akurasi data & saldo", [
        "saldo", "portofolio", "portfolio", "total asset", "zero", "nol", "kosong",
        "tidak sesuai", "gak sesuai", "tidak update", "not updated", "harga berubah",
        "real time", "realtime", "delay harga", "selisih", "salah hitung", "cash", "kas",
        "order tidak masuk", "order pending", "order gagal", "harga tidak", "tick"]),
    "security": ("Keamanan & kepercayaan", [
        "aman", "keamanan", "hack", "diretas", "bobol", "phishing", "penipuan", "scam",
        "data pribadi", "privasi", "curiga", "tidak percaya", "dana hilang", "uang hilang"]),
    "ux": ("Tampilan & fitur", [
        "tampilan", "ui", "ux", "desain", "fitur", "menu", "ribet", "susah", "bingung",
        "friendly", "simple", "simpel", "mudah", "gampang", "lengkap", "chart", "grafik",
        "notifikasi", "watchlist", "favorit"]),
    "service": ("Layanan pelanggan", [
        "cs", "customer service", "customer care", "hubungi", "dihubungi", "respon",
        "tanggapan", "komplain", "bantuan", "call center"]),
    "fees": ("Biaya", ["fee", "biaya", "komisi", "mahal", "potongan", "charge", "pajak"]),
}
INFRA_KEYS = ["latency", "stability", "login", "data"]
# Kata yang secara inheren bermakna keluhan bila muncul tanpa negasi
INHERENT_NEG_KEYS = {"latency", "stability", "data"}

POS_CUES = ["bagus", "mantap", "cepat", "lancar", "stabil", "mudah", "gampang", "keren", "puas",
            "suka", "recommended", "good", "great", "fast", "smooth", "nice", "love", "top"]
NEG_CUES = ["jelek", "buruk", "parah", "kecewa", "payah", "bad", "worst", "terrible", "hate",
            "sampah", "bobrok", "tidak recommended", "not recommended", "ga recommended"]
NEGATIONS = {"tidak", "gak", "ga", "nggak", "ngga", "bukan", "no", "not", "never", "tanpa", "gk", "tak"}

CLAUSE_SPLIT = re.compile(r"[.!?\n;,]+|\b(?:tapi|namun|but|however|sayangnya|cuma|sedangkan)\b")


def collapse(s: str) -> str:
    """Rapatkan huruf berulang: 'lemooot' -> 'lemot' (dipakai pada teks & kamus)."""
    return re.sub(r"(.)\1+", r"\1", s)


def norm(s: str) -> str:
    s = s.lower().replace("-", "")
    s = re.sub(r"https?://\S+", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    return re.sub(r"\s+", " ", collapse(s)).strip()


def _compile(words):
    alts = sorted({re.escape(collapse(w.replace("-", ""))) for w in words}, key=len, reverse=True)
    return re.compile(r"(?<![a-z0-9])(?:" + "|".join(alts) + r")(?![a-z0-9])")


ASPECT_RX = {k: _compile(v[1]) for k, v in ASPECTS.items()}
POS_RX, NEG_RX = _compile(POS_CUES), _compile(NEG_CUES)
INHERENT_RX = _compile(sum((ASPECTS[k][1] for k in INHERENT_NEG_KEYS), []))


# ----------------------------------------------------------------------------
# UTIL
# ----------------------------------------------------------------------------
def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_apps():
    p = DATA_DIR / "apps.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))["apps"]
    return DEFAULT_APPS


def hid(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:12]


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


# ----------------------------------------------------------------------------
# 0. DISCOVER
# ----------------------------------------------------------------------------
def cmd_discover(_a):
    ensure_dirs()
    import requests
    from google_play_scraper import search
    apps = load_apps()
    cands = {}
    for app in apps:
        q = app.get("query") or app["name"]
        play = search(q, lang="id", country="id", n_hits=5)
        ios = requests.get("https://itunes.apple.com/search",
                           params={"term": q, "country": "id", "entity": "software", "limit": 5},
                           timeout=30).json().get("results", [])
        cands[app["name"]] = {
            "play": [{"appId": x.get("appId"), "title": x.get("title"), "developer": x.get("developer"),
                      "score": x.get("score")} for x in play],
            "appstore": [{"trackId": x.get("trackId"), "name": x.get("trackName"),
                          "seller": x.get("sellerName"), "rating": x.get("averageUserRating"),
                          "n": x.get("userRatingCount")} for x in ios]}
        print(f"\n=== {app['name']}  (query: {q}) ===")
        for x in cands[app["name"]]["play"]:
            print("  PLAY ", x)
        for x in cands[app["name"]]["appstore"]:
            print("  IOS  ", x)
    (DATA_DIR / "apps_candidates.json").write_text(json.dumps(cands, ensure_ascii=False, indent=2),
                                                   encoding="utf-8")
    if not (DATA_DIR / "apps.json").exists():
        (DATA_DIR / "apps.json").write_text(json.dumps({"apps": apps}, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print("\nSalin appId (Play) & trackId (App Store) yang BENAR ke data/apps.json, lalu jalankan 'scrape'.")
    print("Pastikan yang dipilih aplikasi trading sekuritas yang tepat, bukan aplikasi lain dengan nama mirip.")


# ----------------------------------------------------------------------------
# 1. SCRAPE
# ----------------------------------------------------------------------------
def scrape_play(app_id, since, max_reviews=8000):
    from google_play_scraper import Sort, reviews
    out, token = [], None
    while len(out) < max_reviews:
        if token is None:
            batch, token = reviews(app_id, lang="id", country="id", sort=Sort.NEWEST, count=200)
        else:
            batch, token = reviews(app_id, continuation_token=token)
        if not batch:
            break
        out.extend(batch)
        if min(r["at"] for r in batch) < since or token is None:
            break
        time.sleep(1.0)
    rows = []
    for r in out:
        # 'at' dari google-play-scraper = waktu lokal SISTEM Anda (naive). Diubah ke WIB.
        ts = pd.Timestamp(r["at"].astimezone() if r["at"].tzinfo is None else r["at"]).tz_convert(TZ)
        rows.append({"platform": "play", "review_id": hid("play", app_id, r["reviewId"]),
                     "text": r.get("content") or "", "rating": r.get("score"), "ts": ts,
                     "thumbs": r.get("thumbsUpCount") or 0,
                     "version": r.get("appVersion") or r.get("reviewCreatedVersion"),
                     "dev_reply": r.get("replyContent") is not None})
    return rows


def scrape_appstore(app_id, pages=10):
    import requests
    rows = []
    for p in range(1, pages + 1):
        url = f"https://itunes.apple.com/id/rss/customerreviews/page={p}/id={app_id}/sortby=mostrecent/json"
        r = requests.get(url, timeout=30, headers={"User-Agent": "bions-research-script/1.0"})
        if r.status_code != 200:
            break
        entries = r.json().get("feed", {}).get("entry", [])
        if isinstance(entries, dict):
            entries = [entries]
        if not entries:
            break
        for e in entries:
            if "im:rating" not in e:
                continue
            ts = pd.to_datetime(e["updated"]["label"], utc=True).tz_convert(TZ)
            rows.append({"platform": "appstore", "review_id": hid("ios", app_id, e["id"]["label"]),
                         "text": ((e.get("title") or {}).get("label", "") + ". " +
                                  (e.get("content") or {}).get("label", "")),
                         "rating": int(e["im:rating"]["label"]), "ts": ts,
                         "thumbs": int((e.get("im:voteSum") or {}).get("label", 0) or 0),
                         "version": (e.get("im:version") or {}).get("label"), "dev_reply": False})
        time.sleep(1.0)
    return rows  # RSS Apple dibatasi 500 ulasan terbaru -> tidak bisa menjangkau terlalu jauh ke belakang


def cmd_scrape(a):
    ensure_dirs()
    since = datetime.fromisoformat(a.since)
    frames = []
    for app in load_apps():
        for plat, key, fn in (("play", "play", lambda i: scrape_play(i, since)),
                              ("appstore", "appstore", scrape_appstore)):
            ident = str(app.get(key) or "").strip()
            if not ident:
                print(f"[skip] {app['name']} ({plat}): ID kosong, isi data/apps.json")
                continue
            try:
                rows = fn(ident)
            except Exception as e:  # noqa: BLE001
                print(f"[gagal] {app['name']} ({plat}): {e}")
                continue
            df = pd.DataFrame(rows)
            if df.empty:
                continue
            df["app"], df["is_target"] = app["name"], bool(app.get("is_target"))
            frames.append(df)
            print(f"[ok] {app['name']:<20} {plat:<9} {len(df):>5} ulasan")
    if not frames:
        sys.exit("Tidak ada data. Periksa koneksi & data/apps.json")
    df = pd.concat(frames, ignore_index=True).drop_duplicates("review_id")
    df = df[df["ts"] >= pd.Timestamp(a.since, tz=TZ)]
    df.to_csv(DATA_DIR / "reviews_raw.csv", index=False)
    print(f"\nTersimpan: {DATA_DIR/'reviews_raw.csv'} ({len(df)} baris)")
    print(df.groupby(["app", "platform"]).size())


# ----------------------------------------------------------------------------
# 2. ABSA (aspek berbasis kamus + sentimen per klausa)
# ----------------------------------------------------------------------------
def lexicon_polarity(clause_norm: str, rating) -> int:
    """Cadangan tanpa model: kamus + negasi sederhana, jatuh ke rating bintang bila ambigu."""
    toks = clause_norm.split()
    score = 0
    for m in INHERENT_RX.finditer(clause_norm):
        before = clause_norm[:m.start()].split()[-2:]
        score += 1 if any(t in NEGATIONS for t in before) else -1
    for m in POS_RX.finditer(clause_norm):
        before = clause_norm[:m.start()].split()[-2:]
        score += -1 if any(t in NEGATIONS for t in before) else 1
    score -= len(NEG_RX.findall(clause_norm))
    if score == 0 and rating is not None and not pd.isna(rating):
        return -1 if rating <= 2 else (1 if rating >= 4 else 0)
    return int(np.sign(score))


class SentimentEngine:
    def __init__(self, engine="auto"):
        self.mode, self.pipe = "lexicon", None
        if engine in ("auto", "transformer"):
            try:
                from transformers import pipeline
                self.pipe = pipeline("text-classification", model=SENTIMENT_MODEL,
                                     truncation=True, max_length=128)
                self.mode = "transformer"
            except Exception as e:  # noqa: BLE001
                if engine == "transformer":
                    raise
                print(f"[peringatan] model transformer tidak tersedia ({e}). Memakai mode kamus.")
        print(f"[ABSA] mesin sentimen: {self.mode}")

    @staticmethod
    def _map(label: str) -> int:
        l = label.lower()
        if "pos" in l:
            return 1
        if "neg" in l:
            return -1
        if "neu" in l:
            return 0
        return {"label_0": 1, "label_1": 0, "label_2": -1}.get(l, 0)

    def polarity(self, items):
        """items: list of (clause_norm, rating) -> list[int]"""
        lex = [lexicon_polarity(c, r) for c, r in items]
        if self.mode == "lexicon":
            return lex
        outs = self.pipe([c for c, _ in items], batch_size=32)
        res = [self._map(o["label"]) for o in outs]
        # Aturan koreksi: model sering menilai "loading lama" netral -> paksa negatif
        return [(-1 if (m == 0 and l == -1) else m) for m, l in zip(res, lex)]


def cmd_absa(a):
    ensure_dirs()
    raw = pd.read_csv(DATA_DIR / "reviews_raw.csv")
    absa_from_df(raw, a.engine)


def absa_from_df(raw: pd.DataFrame, engine="auto"):
    se = SentimentEngine(engine)
    buf = []
    for r in raw.itertuples():
        for c in CLAUSE_SPLIT.split(str(r.text or "").lower()):
            if not c or not c.strip():
                continue
            nc = norm(c)
            for key, rx in ASPECT_RX.items():
                if nc and rx.search(nc):
                    buf.append((r.review_id, key, nc, r.rating))
    if not buf:
        sys.exit("Tidak ada aspek terdeteksi. Periksa data.")
    pol = se.polarity([(c, rt) for _, _, c, rt in buf])
    long = pd.DataFrame(buf, columns=["review_id", "aspect", "clause", "rating"])
    long["polarity"] = pol
    long.to_csv(DATA_DIR / "aspects_long.csv", index=False)

    agg = long.groupby(["review_id", "aspect"])["polarity"].mean().reset_index()
    agg["neg"] = (agg["polarity"] <= -0.34).astype(int)
    wide = agg.pivot(index="review_id", columns="aspect", values="neg")
    wide = wide.reindex(columns=list(ASPECTS)).fillna(0).astype(int)
    wide.columns = [f"neg_{c}" for c in wide.columns]
    wide["infra_neg"] = (wide[[f"neg_{k}" for k in INFRA_KEYS]].sum(axis=1) > 0).astype(int)
    out = raw.merge(wide, left_on="review_id", right_index=True, how="left")
    for c in wide.columns:
        out[c] = out[c].fillna(0).astype(int)
    out.to_csv(DATA_DIR / "reviews_absa.csv", index=False)
    print(f"Tersimpan: {DATA_DIR/'reviews_absa.csv'} ({len(out)} ulasan, "
          f"{out['infra_neg'].mean():.1%} berkeluhan infrastruktur)")


# ----------------------------------------------------------------------------
# 3. PASAR & SINYAL
# ----------------------------------------------------------------------------
def load_market(csv=None, start=START_DATE, end=None):
    if csv:
        df = pd.read_csv(csv)
        df.columns = [c.lower() for c in df.columns]
        df["date"] = pd.to_datetime(df["date"])
    else:
        import yfinance as yf
        df = yf.download("^JKSE", start=start, end=end, auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df.columns = [str(c).lower() for c in df.columns]
    df = df[["date", "open", "high", "low", "close", "volume"]].dropna(subset=["close"]).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize().astype("datetime64[ns]")
    df = df.sort_values("date").reset_index(drop=True)
    df["ret"] = df["close"].pct_change()
    df["abs_ret"] = df["ret"].abs()
    df["range"] = (df["high"] - df["low"]) / df["close"]
    lv = np.log(df["volume"].replace(0, np.nan))
    df["vol_z"] = (lv - lv.rolling(60, min_periods=20).mean()) / lv.rolling(60, min_periods=20).std()
    q_r, q_g = df["abs_ret"].quantile(0.90), df["range"].quantile(0.90)
    df["stress"] = ((df["abs_ret"] >= q_r) | (df["range"] >= q_g)).astype(int)
    if (df["volume"] == 0).mean() > 0.3:
        print("[peringatan] volume IHSG di Yahoo banyak bernilai 0. Gunakan --market_csv "
              "(nilai transaksi harian dari idx.co.id) untuk hasil lebih andal.")
    return df


def spearman_perm(x, y, n_perm=3000, min_shift=5):
    """Spearman + uji permutasi pergeseran sirkular (menjaga autokorelasi)."""
    from scipy.stats import rankdata
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y))
    x, y = x[m], y[m]
    n = len(x)
    if n < 30 or np.std(x) == 0 or np.std(y) == 0:
        return dict(rho=np.nan, p=np.nan, n=n)
    rx, ry = rankdata(x), rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    shifts = RNG.integers(min_shift, max(min_shift + 1, n - min_shift), size=n_perm)
    null = np.array([np.corrcoef(rx, np.roll(ry, s))[0, 1] for s in shifts])
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1))
    return dict(rho=rho, p=p, n=n)


def attribute_to_trading_day(rev, mk):
    """Ulasan akhir pekan/libur dipetakan ke hari bursa terakhir sebelumnya."""
    rv = rev[["review_id", "ts", "infra_neg"]].copy()
    rv["d"] = pd.to_datetime(rv["ts"], utc=True).dt.tz_convert(TZ).dt.tz_localize(None) \
        .dt.normalize().astype("datetime64[ns]")
    rv = rv.sort_values("d")
    t = mk[["date"]].rename(columns={"date": "tday"}).astype({"tday": "datetime64[ns]"})
    rv = pd.merge_asof(rv, t, left_on="d", right_on="tday", direction="backward").dropna(subset=["tday"])
    daily = rv.groupby("tday").agg(total=("review_id", "size"), infra_neg=("infra_neg", "sum"))
    daily = mk.set_index("date").join(daily).fillna({"total": 0, "infra_neg": 0})
    daily["rate"] = np.where(daily["total"] >= 3, daily["infra_neg"] / daily["total"].replace(0, np.nan), np.nan)
    return daily


def in_market_hours(ts_local: pd.Series, trading_days: set):
    mins = ts_local.dt.hour * 60 + ts_local.dt.minute
    wd = ts_local.dt.weekday
    day = ts_local.dt.tz_localize(None).dt.normalize()
    is_td = day.isin(trading_days)
    def within(sess):
        return np.any([(mins >= a) & (mins < b) for a, b in sess], axis=0)
    mh = np.where(wd <= 3, within(SESSIONS_MON_THU), np.where(wd == 4, within(SESSIONS_FRI), False))
    return pd.Series(mh & is_td.to_numpy(), index=ts_local.index)


def boot_ratio(a, b, B=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return (np.nan, np.nan, np.nan)
    r = a.mean() / b.mean() if b.mean() > 0 else np.nan
    bs = [RNG.choice(a, len(a)).mean() / max(RNG.choice(b, len(b)).mean(), 1e-9) for _ in range(B)]
    return (r, *np.percentile(bs, [2.5, 97.5]))


def event_study(daily, events, window=(0, 2)):
    s = daily["infra_neg"].to_numpy(float)
    idx = {d: i for i, d in enumerate(daily.index)}
    ev_pos = sorted({idx[pd.Timestamp(d)] for d in events if pd.Timestamp(d) in idx})
    w0, w1 = window
    n = len(s)
    used = set()
    ev_vals = []
    for i in ev_pos:
        lo, hi = i + w0, i + w1 + 1
        if hi <= n:
            ev_vals.append(s[lo:hi].mean())
            used.update(range(i - 2, i + w1 + 3))
    base = [s[i + w0:i + w1 + 1].mean() for i in range(n - w1 - 1) if i not in used]
    r, lo, hi = boot_ratio(ev_vals, base)
    return dict(n_events=len(ev_vals), ratio=r, ci_lo=lo, ci_hi=hi,
                event_mean=float(np.mean(ev_vals)) if ev_vals else np.nan,
                base_mean=float(np.mean(base)) if base else np.nan)


def week_start(d: pd.Series) -> pd.Series:
    return (d - pd.to_timedelta((d.dt.weekday + 1) % 7, unit="D")).astype("datetime64[ns]")


def fetch_trends(start, end, cache=None):
    if cache and Path(cache).exists():
        df = pd.read_csv(cache, parse_dates=["date"]).set_index("date")
        return df
    from pytrends.request import TrendReq
    py = TrendReq(hl="id-ID", tz=420, retries=3, backoff_factor=1.5)
    py.build_payload(TARGET_KEYWORDS[:5], timeframe=f"{start} {end}", geo="ID")
    df = py.interest_over_time()
    df = df.drop(columns=[c for c in ["isPartial"] if c in df.columns])
    df.index = pd.to_datetime(df.index).tz_localize(None).astype("datetime64[ns]")
    df.index.name = "date"
    if cache:
        df.reset_index().to_csv(cache, index=False)
    return df


def lead_lag(a, b, ks=range(-4, 5)):
    """corr(diff a_t, diff b_{t+k}); k>0 artinya a MENDAHULUI b sebanyak k periode."""
    da, db = a.diff(), b.diff()
    rows = []
    for k in ks:
        r = spearman_perm(da.values, db.shift(-k).values, n_perm=2000, min_shift=3)
        rows.append(dict(k=k, **r))
    return pd.DataFrame(rows)


def granger(a, b, maxlag=4):
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except Exception:  # noqa: BLE001
        return None
    d = pd.concat([b.diff(), a.diff()], axis=1).dropna()
    if len(d) < 30:
        return None
    res = grangercausalitytests(d.values, maxlag=maxlag, verbose=False)
    ps = [res[l][0]["ssr_ftest"][1] for l in range(1, maxlag + 1)]
    return dict(min_p=float(min(ps)), bonferroni_p=float(min(1.0, min(ps) * maxlag)))


def signal_lift(mk, weekly_trend_up=None):
    """Seberapa sering hari bursa 'stres' besok didahului sinyal hari ini? (presisi & lift)."""
    d = mk.copy()
    d["target"] = d["stress"].shift(-1)
    sig = {"|return| kemarin >= P90": d["abs_ret"] >= d["abs_ret"].quantile(0.90),
           "volume kemarin >= +1 SD": d["vol_z"] >= 1}
    if weekly_trend_up is not None:
        sig["Tren pencarian minggu ini naik & tinggi"] = weekly_trend_up.reindex(d["date"]).fillna(False).values
    base = d["target"].mean()
    rows = []
    for name, s in sig.items():
        s = pd.Series(np.asarray(s, bool), index=d.index) & d["target"].notna()
        k, n = int(d.loc[s, "target"].sum()), int(s.sum())
        lo, hi = wilson(k, n)
        rows.append(dict(sinyal=name, n_sinyal=n, presisi=k / n if n else np.nan, ci_lo=lo, ci_hi=hi,
                         base_rate=base, lift=(k / n) / base if n and base else np.nan))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 4. ANALISIS
# ----------------------------------------------------------------------------
def aspect_gap(rev, B=1000):
    target = rev.loc[rev["is_target"], "app"].iloc[0]
    cols = [f"neg_{k}" for k in ASPECTS]
    mats = {app: g[cols].to_numpy(float) for app, g in rev.groupby("app")}
    peers = [a for a in mats if a != target]
    pt = mats[target].mean(0)
    pm = np.median(np.vstack([mats[a].mean(0) for a in peers]), axis=0)
    diffs = np.empty((B, len(cols)))
    for b in range(B):
        v = {a: m[RNG.integers(0, len(m), len(m))].mean(0) for a, m in mats.items()}
        diffs[b] = v[target] - np.median(np.vstack([v[a] for a in peers]), axis=0)
    lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0)
    out = pd.DataFrame({"aspect": [ASPECTS[k][0] for k in ASPECTS], "target_share": pt,
                        "peer_median_share": pm, "gap": pt - pm, "ci_lo": lo, "ci_hi": hi})
    for a in peers:
        out[f"peer::{a}"] = mats[a].mean(0)
    return out.sort_values("gap", ascending=False).reset_index(drop=True), target


def overview(rev):
    g = rev.groupby("app")
    return pd.DataFrame({"n": g.size(), "rating_mean": g["rating"].mean(),
                         "share_1_2_star": g["rating"].apply(lambda s: (s <= 2).mean()),
                         "share_infra_neg": g["infra_neg"].mean(),
                         "first": g["ts"].min(), "last": g["ts"].max()}).reset_index()


def cmd_analyze(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ensure_dirs()
    rev = pd.read_csv(DATA_DIR / "reviews_absa.csv")
    rev["ts"] = pd.to_datetime(rev["ts"], utc=True).dt.tz_convert(TZ)
    rev = rev[rev["ts"] >= pd.Timestamp(START_DATE, tz=TZ)]
    rev["is_target"] = rev["is_target"].astype(bool)
    if not rev["is_target"].any():
        sys.exit("Tidak ada ulasan aplikasi target. Periksa is_target di apps.json.")
    K, notes = {}, []
    wm = "DATA SINTETIS - HANYA UJI KODE" if DEMO else None

    def mark(ax):
        if wm:
            ax.text(0.5, 0.5, wm, transform=ax.transAxes, ha="center", va="center", alpha=0.18,
                    fontsize=22, rotation=20, color="red")

    ov = overview(rev)
    ov.to_csv(OUT_DIR / "t0_overview.csv", index=False)
    small = ov[ov["n"] < MIN_REVIEWS_PER_APP]["app"].tolist()
    if small:
        notes.append(f"PERINGATAN: ulasan < {MIN_REVIEWS_PER_APP} untuk {small}; hasil tidak andal.")
    K["A0_overview"] = ov

    # (1) Aspect gap vs kompetitor
    gap, target = aspect_gap(rev)
    gap.to_csv(OUT_DIR / "t1_aspect_gap.csv", index=False)
    K["A1_gap"] = gap
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(gap))
    ax.barh(y + 0.2, gap["target_share"] * 100, 0.4, label=target, color="#0E8F7F")
    ax.barh(y - 0.2, gap["peer_median_share"] * 100, 0.4, label="Median kompetitor", color="#B8C2CC")
    ax.set_yticks(y)
    ax.set_yticklabels(gap["aspect"])
    ax.invert_yaxis()
    ax.set_xlabel("% ulasan dengan keluhan pada aspek tsb")
    ax.legend()
    ax.set_title("Keluhan per aspek: BIONS vs kompetitor")
    mark(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "c1_aspect_gap.png", dpi=160)
    plt.close(fig)

    # bobot 'dirasakan banyak orang' (suara membantu)
    tg = rev[rev["is_target"]]
    K["A1b_helpful_share"] = float(tg.loc[tg["infra_neg"] == 1, "thumbs"].sum() / max(tg["thumbs"].sum(), 1))

    # --- (2) Pasar -----------------------------------------------------------
    mk = load_market(a.market_csv)
    mk.to_csv(DATA_DIR / "market_daily.csv", index=False)
    daily = attribute_to_trading_day(tg, mk)
    daily.to_csv(OUT_DIR / "t2_daily_target_vs_market.csv")
    tdays = set(mk["date"])

    # 2a. jam perdagangan
    tl = tg["ts"].dt.tz_convert(TZ)
    mh = in_market_hours(tl, tdays)
    inf = mh[tg["infra_neg"] == 1]
    oth = mh[tg["infra_neg"] == 0]
    r, lo, hi = boot_ratio(inf.astype(float).to_numpy(), oth.astype(float).to_numpy())
    K["A2a_hours"] = dict(share_infra=float(inf.mean()), share_other=float(oth.mean()),
                          lift=r, ci_lo=lo, ci_hi=hi, n_infra=int(len(inf)))
    hh = pd.DataFrame({"hour": tl.dt.hour, "infra": tg["infra_neg"]})
    prof = hh.groupby("hour")["infra"].agg(["sum", "count"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(prof.index, prof["sum"] / prof["sum"].sum() * 100, color="#0E8F7F", label="Ulasan keluhan infrastruktur")
    ax.plot(prof.index, prof["count"] / prof["count"].sum() * 100, color="#555", marker="o", label="Semua ulasan")
    ax.axvspan(9, 12, alpha=0.08)
    ax.axvspan(13.5, 15.83, alpha=0.08)
    ax.set_xlabel("Jam (WIB) - area abu = sesi perdagangan")
    ax.set_ylabel("% ulasan")
    ax.legend()
    ax.set_title("Waktu posting ulasan: keluhan infrastruktur vs semua ulasan")
    mark(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "c3_hour_profile.png", dpi=160)
    plt.close(fig)

    # 2b. korelasi harian, lag 0..2 (ulasan terjadi SETELAH hari bursa)
    cor = []
    for metric in ["abs_ret", "range", "vol_z"]:
        for k in (0, 1, 2):
            for kind, col in (("jumlah", "infra_neg"), ("rasio", "rate")):
                res = spearman_perm(daily[col].shift(-k).values, daily[metric].values)
                cor.append(dict(aktivitas=metric, lag_hari=k, ukuran=kind, **res))
    cor = pd.DataFrame(cor)
    cor.to_csv(OUT_DIR / "t3_daily_correlation.csv", index=False)
    K["A2b_corr"] = cor.sort_values("p").head(5)
    notes.append(f"Uji korelasi dijalankan {len(cor)}x; gunakan koreksi Bonferroni (p < {0.05/len(cor):.4f}).")

    # 2c. event study
    auto_ev = mk.nlargest(max(5, int(len(mk) * 0.05)), "abs_ret")["date"].tolist()
    es_h = event_study(daily, EVENT_DATES)
    es_a = event_study(daily, auto_ev)
    K["A2c_event_halt"], K["A2c_event_top5pct"] = es_h, es_a
    fig, ax = plt.subplots(figsize=(5.5, 4))
    labs, vals = ["Hari trading halt", "Top 5% |return|"], [es_h, es_a]
    ax.bar(labs, [v["ratio"] for v in vals], color=["#0E8F7F", "#7FB8B0"],
           yerr=[[v["ratio"] - v["ci_lo"] for v in vals], [v["ci_hi"] - v["ratio"] for v in vals]], capsize=6)
    ax.axhline(1, color="k", lw=1)
    ax.set_ylabel("Rasio keluhan/hari vs hari biasa")
    ax.set_title("Event study (jendela hari 0 s.d. +2)")
    mark(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "c5_event_study.png", dpi=160)
    plt.close(fig)

    # sensitivitas: hanya setelah peluncuran resmi
    post = tg[tg["ts"] >= pd.Timestamp(LAUNCH_EVENT_DATE, tz=TZ)]
    K["A2d_sensitivity_post_launch_n"] = int(len(post))

    # grafik runtun waktu mingguan
    wk = daily.reset_index().rename(columns={"index": "date"})
    wk["wk"] = week_start(wk["date"])
    w = wk.groupby("wk").agg(infra=("infra_neg", "sum"), total=("total", "sum"), absret=("abs_ret", "mean"))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(w.index, w["infra"], width=5, color="#0E8F7F", label="Ulasan keluhan infrastruktur / minggu")
    ax2 = ax.twinx()
    ax2.plot(w.index, w["absret"] * 100, color="#E07B00", label="Rata-rata |return| harian IHSG (%)")
    ax.set_title("Keluhan infrastruktur BIONS vs volatilitas IHSG (mingguan)")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    mark(ax)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "c2_weekly_vs_market.png", dpi=160)
    plt.close(fig)

    # (3) Sinyal dini: Google Trends
    trend_up = None
    if not a.no_trends:
        try:
            end = datetime.now().strftime("%Y-%m-%d")
            tr = pd.read_csv(a.trends_csv, parse_dates=["date"]).set_index("date") if a.trends_csv else \
                fetch_trends(START_DATE, end, DATA_DIR / "trends_cache.csv")
            tr.index = pd.to_datetime(tr.index).astype("datetime64[ns]")
            cols = [c for c in MARKET_KEYWORDS if c in tr.columns]
            tr["trend_mkt"] = ((tr[cols] - tr[cols].mean()) / tr[cols].std()).mean(axis=1)
            m = mk.copy()
            m["wk"] = week_start(m["date"])
            wa = m.groupby("wk").agg(absret=("abs_ret", "mean"), volz=("vol_z", "mean"))
            wj = wa.join(tr[["trend_mkt"]], how="inner").join(w[["infra"]], how="left").dropna(subset=["trend_mkt"])
            ll_a = lead_lag(wj["trend_mkt"], wj["absret"])
            ll_v = lead_lag(wj["trend_mkt"], wj["infra"])
            ll_a.to_csv(OUT_DIR / "t4_leadlag_trend_vs_marketvol.csv", index=False)
            ll_v.to_csv(OUT_DIR / "t4_leadlag_trend_vs_complaints.csv", index=False)
            K["A3_leadlag_market"] = ll_a.loc[ll_a["p"].idxmin()].to_dict() if ll_a["p"].notna().any() else None
            K["A3_leadlag_complaints"] = ll_v.loc[ll_v["p"].idxmin()].to_dict() if ll_v["p"].notna().any() else None
            K["A3_granger_trend_to_market"] = granger(wj["trend_mkt"], wj["absret"])
            K["A3_granger_market_to_trend"] = granger(wj["absret"], wj["trend_mkt"])
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(ll_a["k"], ll_a["rho"], color=["#0E8F7F" if k > 0 else "#B8C2CC" for k in ll_a["k"]])
            ax.axhline(0, color="k", lw=1)
            ax.set_xlabel("k (minggu): k>0 = tren pencarian MENDAHULUI volatilitas pasar")
            ax.set_ylabel("Spearman rho (selisih mingguan)")
            ax.set_title("Lead-lag: tren pencarian vs volatilitas IHSG")
            mark(ax)
            fig.tight_layout()
            fig.savefig(OUT_DIR / "c4_leadlag.png", dpi=160)
            plt.close(fig)
            up = (wj["trend_mkt"].diff() > 0) & (wj["trend_mkt"] >= wj["trend_mkt"].quantile(0.75))
            wk_map = week_start(mk["date"])
            trend_up = pd.Series(wk_map.map(up).fillna(False).astype(bool).values, index=mk["date"])
        except Exception as e:  # noqa: BLE001
            notes.append(f"Analisis Google Trends dilewati: {e}")
    lift = signal_lift(mk, trend_up)
    lift.to_csv(OUT_DIR / "t5_signal_lift.csv", index=False)
    K["A4_signal_lift"] = lift

    write_summary(K, notes, target)
    print(f"\nSelesai. Lihat folder: {OUT_DIR.resolve()}")


def fmt(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def write_summary(K, notes, target):
    L = ["# RINGKASAN HASIL - isi placeholder slide dari angka di bawah", ""]
    if DEMO:
        L += ["> **DATA SINTETIS - ANGKA INI BUKAN HASIL NYATA. JANGAN DIMASUKKAN KE SLIDE.**", ""]
    L += ["## [A0] Gambaran data", K["A0_overview"].to_markdown(index=False) if hasattr(pd.DataFrame, "to_markdown")
          and _has_tabulate() else K["A0_overview"].to_string(index=False), ""]
    g = K["A1_gap"]
    L += ["## [A1] Aspek mana yang paling menjauhkan BIONS dari kompetitor?",
          "Gap = (% ulasan BIONS berkeluhan aspek X) - (median kompetitor). CI = bootstrap 95%.", ""]
    for r in g.itertuples():
        sig = "SIGNIFIKAN (CI tidak memuat 0)" if (r.ci_lo > 0 or r.ci_hi < 0) else "tidak signifikan"
        L.append(f"- {r.aspect}: BIONS {r.target_share:.1%} vs median {r.peer_median_share:.1%} | gap "
                 f"{r.gap*100:+.1f} pp [{r.ci_lo*100:+.1f}, {r.ci_hi*100:+.1f}] -> {sig}")
    infra_names = {ASPECTS[k][0] for k in INFRA_KEYS}
    top = g.iloc[0]
    L += ["", f"Aspek dengan gap terbesar: **{top['aspect']}** "
              f"({'infrastruktur' if top['aspect'] in infra_names else 'BUKAN infrastruktur -> ubah narasi slide 1'}).",
          f"[A1b] Porsi 'suara membantu' pada ulasan berkeluhan infrastruktur: {K['A1b_helpful_share']:.1%}", ""]
    h = K["A2a_hours"]
    L += ["## [A2a] Apakah keluhan infrastruktur menumpuk pada jam perdagangan?",
          f"Porsi ulasan keluhan infra yang diposting saat jam bursa: {h['share_infra']:.1%} "
          f"vs ulasan lain {h['share_other']:.1%} | lift {fmt(h['lift'])}x "
          f"[{fmt(h['ci_lo'])}, {fmt(h['ci_hi'])}] (n={h['n_infra']})",
          "Catatan: waktu ulasan = waktu menulis, bukan waktu kejadian gangguan.", ""]
    L += ["## [A2b] Korelasi harian (5 hasil terkuat)", K["A2b_corr"].to_string(index=False), ""]
    for nm, key in (("trading halt", "A2c_event_halt"), ("top 5% |return|", "A2c_event_top5pct")):
        e = K[key]
        L.append(f"## [A2c] Event study - {nm}: n={e['n_events']}, rasio keluhan/hari {fmt(e['ratio'])}x "
                 f"[{fmt(e['ci_lo'])}, {fmt(e['ci_hi'])}] (hari acara {fmt(e['event_mean'])} vs biasa {fmt(e['base_mean'])})")
    L += ["", f"[A2d] Ulasan BIONS setelah peluncuran resmi ({LAUNCH_EVENT_DATE}): {K['A2d_sensitivity_post_launch_n']}", ""]
    if "A3_leadlag_market" in K:
        L += ["## [A3] Sinyal dini (Google Trends)"]
        for nm, key in (("volatilitas pasar", "A3_leadlag_market"), ("keluhan BIONS", "A3_leadlag_complaints")):
            v = K.get(key)
            if v:
                L.append(f"- Tren pencarian vs {nm}: puncak di k={int(v['k'])} minggu, rho={v['rho']:.2f}, p={v['p']:.3f} "
                         f"(k>0 = tren mendahului)")
        for nm, key in (("tren -> pasar", "A3_granger_trend_to_market"), ("pasar -> tren", "A3_granger_market_to_trend")):
            v = K.get(key)
            L.append(f"- Granger {nm}: " + (f"p min={v['min_p']:.3f}, Bonferroni={v['bonferroni_p']:.3f}" if v else "tidak tersedia (pasang statsmodels)"))
        L.append("")
    L += ["## [A4] Apakah sinyal hari ini memprediksi hari stres besok? (lift > 1 = lebih baik dari menebak)",
          K["A4_signal_lift"].to_string(index=False), ""]
    L += ["## Catatan & batasan wajib dicantumkan di slide/Q&A"]
    L += [f"- {n}" for n in notes]
    L += ["- Korelasi bukan kausalitas. Ulasan hanya mewakili pengguna yang menulis ulasan (bias seleksi).",
          "- Volume Yahoo untuk ^JKSE adalah proksi; gunakan nilai/frekuensi transaksi BEI bila tersedia.",
          "- Google Trends bernilai relatif (0-100) & bisa sedikit berubah antar penarikan; cache disimpan.",
          "- Kampanye BIONS Cuanpionship #4 (8 Apr-27 Jun 2025) bisa mengacaukan lonjakan awal; lihat sensitivitas [A2d].",
          "- Aspek diekstrak dengan kamus + sentimen per klausa (bukan model ABSA terlatih); validasi manual 100 ulasan "
          "(hitung akurasi) sebelum mengklaim hasil."]
    (OUT_DIR / "summary.md").write_text("\n".join(L), encoding="utf-8")
    (OUT_DIR / "results.json").write_text(json.dumps(_jsonable(K), ensure_ascii=False, indent=2, default=str),
                                          encoding="utf-8")


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _jsonable(o):
    if isinstance(o, pd.DataFrame):
        return json.loads(o.to_json(orient="records", date_format="iso"))
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


if __name__ == "__main__":
    main()
