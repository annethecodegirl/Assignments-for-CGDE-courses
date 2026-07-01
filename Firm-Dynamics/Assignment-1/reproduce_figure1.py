#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproduce_figure1.py
--------------------
Reproduce Figure 1 of Haltiwanger, Jarmin & Miranda (2013),
"Who Creates Jobs? Small versus Large versus Young"
(Review of Economics and Statistics 95(2): 347-361).

Figure 1 reports, for the 1992-2005 period, the share of employment,
gross job creation and gross job destruction accounted for by six firm
groups: the cross of firm size {Small (<500), Large (>=500)} with firm
type {Startups (age 0), Young (age 1-9), Mature (age 10+)}.

Data: Census Bureau Business Dynamics Statistics (BDS), national
"Firm Age by Firm Size" table. The Census site blocks automated
sandbox access, so run this on a machine with normal internet access.

Usage:
    python reproduce_figure1.py                # download + build figure
    python reproduce_figure1.py --file my.csv  # use a local BDS csv
    python reproduce_figure1.py --y0 1992 --y1 2005

Outputs (written to the working directory):
    figure1.pdf, figure1.png   -- the figure
    figure1_shares.csv         -- the plotted numbers (18 shares)
"""

import argparse
import io
import re
import urllib.request

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The first source is the classic "release" file of the paper's era
# (1977-2014); the second is the current 2023 vintage. Either one
# reproduces the qualitative figure.
SOURCES = [
    "https://www2.census.gov/ces/bds/firm/bds_f_agesz_release.csv",
    "https://www2.census.gov/programs-surveys/bds/tables/time-series/2023/bds2023_fa_fz.csv",
]
SUPPRESSION_FLAGS = {"D", "N", "S", "X"}     # BDS disclosure / quality flags


def load_bds(path_or_none):
    if path_or_none:
        return pd.read_csv(path_or_none, dtype=str)
    last_err = None
    for url in SOURCES:
        try:
            print(f"[info] downloading {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "research/1.0"})
            raw = urllib.request.urlopen(req, timeout=60).read()
            return pd.read_csv(io.BytesIO(raw), dtype=str)
        except Exception as e:                # noqa: BLE001
            print(f"[warn] {url} failed: {e}")
            last_err = e
    raise SystemExit(
        "Could not download the BDS file. Download 'Firm Age by Firm Size' "
        "from census.gov/data/datasets/time-series/econ/bds/bds-datasets.html "
        f"and pass it with --file.\nLast error: {last_err}"
    )


def normalise(df):
    """Standardise names across BDS vintages; blank out suppressed cells."""
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    df = df.rename(columns={"year2": "year", "fage4": "fage", "fsize4": "fsize"})

    age_col = next(c for c in df.columns if c in ("fage", "fagecoarse"))
    size_col = next(c for c in df.columns if c in ("fsize", "fsizecoarse"))
    for src in ("emp", "job_creation", "job_destruction"):
        if src not in df.columns:
            raise SystemExit(f"expected column '{src}' not found")
    df = df.rename(columns={age_col: "age", size_col: "size",
                            "job_creation": "jc", "job_destruction": "jd"})

    flagcol = {"emp": "emp_f", "jc": "job_creation_f", "jd": "job_destruction_f"}
    for m in ("emp", "jc", "jd"):
        vals = pd.to_numeric(df[m], errors="coerce")
        f = flagcol[m]
        if f in df.columns:
            vals = vals.where(~df[f].isin(SUPPRESSION_FLAGS), np.nan)
        vals = vals.where(~df[m].astype(str).str.contains(r"[DNSX]", na=False), np.nan)
        df[m] = vals

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df[["year", "age", "size", "emp", "jc", "jd"]]


def size_class(label):
    s = str(label).lower()
    if "left censored" in s:
        return np.nan
    nums = re.findall(r"\d+", s)
    return np.nan if not nums else ("Large" if int(nums[0]) >= 500 else "Small")


def age_class(label):
    s = str(label).lower()
    if "left censored" in s:
        return "Mature"
    nums = re.findall(r"\d+", s)
    if not nums:
        return np.nan
    lo = int(nums[0])
    if lo == 0:
        return "Startups"
    return "Young" if lo <= 9 else "Mature"


GROUP_ORDER = [("Small", "Startups"), ("Small", "Young"), ("Small", "Mature"),
               ("Large", "Startups"), ("Large", "Young"), ("Large", "Mature")]
GROUP_LABELS = ["Small\nStartups", "Small\nYoung", "Small\nMature",
                "Large\nStartups", "Large\nYoung", "Large\nMature"]


def build_shares(df, y0, y1):
    d = df[(df["year"] >= y0) & (df["year"] <= y1)].copy()
    d["size_g"] = d["size"].map(size_class)
    d["age_g"] = d["age"].map(age_class)
    d = d.dropna(subset=["size_g", "age_g"])

    lev = (d.groupby(["size_g", "age_g"])[["emp", "jc", "jd"]]
             .sum(min_count=1).reset_index())
    tot = lev[["emp", "jc", "jd"]].sum()
    for m in ("emp", "jc", "jd"):
        lev[f"{m}_share"] = 100 * lev[m] / tot[m]

    key = {g: i for i, g in enumerate(GROUP_ORDER)}
    lev["order"] = list(zip(lev["size_g"], lev["age_g"]))
    lev = lev[lev["order"].isin(key)].copy()
    lev["order"] = lev["order"].map(key)
    return lev.sort_values("order").reset_index(drop=True)


def make_figure(lev, y0, y1, stem="figure1"):
    g = lambda c: lev.set_index("order")[c].reindex(range(6)).fillna(0).values
    emp, jc, jd = g("emp_share"), g("jc_share"), g("jd_share")
    x, w = np.arange(6), 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w, emp, w, label="Share of employment", color="#3b4b6b")
    ax.bar(x,     jc,  w, label="Share of job creation", color="#7a9e7e")
    ax.bar(x + w, jd,  w, label="Share of job destruction", color="#b5651d")
    ax.set_xticks(x); ax.set_xticklabels(GROUP_LABELS)
    ax.set_ylabel("Percent")
    ax.set_title(f"Figure 1. Shares of employment, job creation and job "
                 f"destruction\nby firm size and firm age, {y0}\u2013{y1} (BDS)")
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    for xi, vals in zip(x, zip(emp, jc, jd)):
        for dx, v in zip((-w, 0, w), vals):
            ax.text(xi + dx, v + 0.4, f"{v:.1f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{stem}.pdf"); fig.savefig(f"{stem}.png", dpi=200)
    print(f"[info] wrote {stem}.pdf and {stem}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None)
    ap.add_argument("--y0", type=int, default=1992)
    ap.add_argument("--y1", type=int, default=2005)
    args = ap.parse_args()

    df = normalise(load_bds(args.file))
    lev = build_shares(df, args.y0, args.y1)

    out = lev[["size_g", "age_g", "emp_share", "jc_share", "jd_share"]]
    out.to_csv("figure1_shares.csv", index=False)
    pd.set_option("display.float_format", lambda v: f"{v:6.2f}")
    print(f"\nShares (%) pooled over {args.y0}-{args.y1}:\n{out.to_string(index=False)}\n")

    startup_emp = out.loc[out.age_g == "Startups", "emp_share"].sum()
    lg_mature = out[(out.size_g == "Large") & (out.age_g == "Mature")]["emp_share"].sum()
    print(f"[check] startup employment share      = {startup_emp:4.2f}%  (paper ~2.8%)")
    print(f"[check] large & mature employment share = {lg_mature:4.2f}%  (paper ~45%)")

    make_figure(lev, args.y0, args.y1)


if __name__ == "__main__":
    main()
