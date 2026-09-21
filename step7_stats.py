"""
step7_stats.py  --  the Step 7 calculations on a rated sample.
Author: Hrishin Debnath

Primary metric: CometKiwi-22 (0..1, higher = better). Secondary: MetricX-24 (error 0..25, lower = better),
stored negated in item_scores.csv as metricx_mean / metricx_min so that higher = better everywhere.
Changed 15 September 2026 (research document), not yet confirmed by Azad.

Input: one row per rated item with columns
  part (random | targeted_worst | targeted_disagree), cometkiwi_mean, cometkiwi_min, metricx_mean, roundtrip_chrf,
  option_swap_n, tier (D0..D3), fluency (1..5), tags (semicolon-separated, may be empty),
  tier_rater2 (optional), fluency_rater2 (optional).

"bad" everywhere means the human rater called the item D2 or D3.
"flagged" means the code's CometKiwi item minimum is at or below the line.

Run on a merged CSV:  python step7_stats.py rated.csv
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

TIER = {"D0": 0, "D1": 1, "D2": 2, "D3": 3}
SCORE_MEAN = "cometkiwi_mean"      # correlation and rank gap
SCORE_MIN = "cometkiwi_min"        # threshold, AUROC, recall per tag
RNG = np.random.default_rng(42)


def bad_mask(df):
    return (df["tier"].map(TIER) >= 2).astype(int)


def bootstrap_ci(x, y, fn, n=2000):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    vals = []
    for _ in range(n):
        idx = RNG.integers(0, len(x), len(x))          # draw with replacement
        if len(set(y[idx])) > 1 and len(set(x[idx])) > 1:
            vals.append(fn(x[idx], y[idx]))
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])] if vals else [float("nan"), float("nan")]


def correlation(df, score_col=SCORE_MEAN):
    """Random part only. Spearman and Kendall tau-b between the item mean score and the human tier.
    A good metric gives a negative value here: a higher score goes with a lower (better) tier."""
    r = df[df["part"] == "random"].dropna(subset=[score_col])
    x, y = r[score_col], r["tier"].map(TIER)
    sp = spearmanr(x, y).correlation
    kt = kendalltau(x, y, variant="b").correlation
    return {"score": score_col, "n": int(len(r)),
            "spearman": float(sp), "spearman_ci": bootstrap_ci(x, y, lambda a, b: spearmanr(a, b).correlation),
            "kendall_b": float(kt), "kendall_ci": bootstrap_ci(x, y, lambda a, b: kendalltau(a, b, variant="b").correlation)}


def auroc(df, score_col=SCORE_MIN):
    """AUROC of the item minimum for telling D2/D3 from the rest. Headline: random part. Enriched: all rated
    items, inflated because 25 of them were picked by the scorer under test. Lower score = more likely bad,
    so the score is negated before scoring."""
    out = {"score": score_col}
    for name, d in (("random", df[df["part"] == "random"]), ("all_enriched", df)):
        d = d.dropna(subset=[score_col])
        bad = bad_mask(d)
        out[name] = {"n": int(len(d)), "n_bad": int(bad.sum()),
                     "auroc": float(roc_auc_score(bad, -d[score_col])) if 0 < bad.sum() < len(d) else float("nan")}
    return out


def place_line(scores, bad, target_recall):
    """The highest line that still catches target_recall of the bad items: flagged = score <= line.
    Sorted ascending, so the first line that meets the target is the tightest one."""
    scores, bad = np.asarray(scores, dtype=float), np.asarray(bad, dtype=int)
    if bad.sum() == 0:
        return None
    for t in np.sort(np.unique(scores)):
        flagged = scores <= t
        recall = (flagged & (bad == 1)).sum() / bad.sum()
        if recall >= target_recall:
            precision = (flagged & (bad == 1)).sum() / max(flagged.sum(), 1)
            return {"threshold": float(t), "recall": float(recall), "precision": float(precision),
                    "flag_share": float(flagged.mean())}
    return None


def threshold(df, target_recall=0.90, score_col=SCORE_MIN):
    """All rated items. The line is applied to the full item set afterwards (run_all.py decide)."""
    d = df.dropna(subset=[score_col])
    bad = bad_mask(d)
    best = place_line(d[score_col], bad, target_recall) or {"threshold": float("nan"), "recall": float("nan"),
                                                             "precision": float("nan"), "flag_share": float("nan")}
    return {"score": score_col, "n": int(len(d)), "n_bad": int(bad.sum()), "target_recall": target_recall, **best}


def cv_recall_precision(df, target_recall=0.90, folds=5, n_boot=200, score_col=SCORE_MIN, seed=42):
    """Five folds with the same share of bad items each. Place the line on four folds, measure recall and
    precision on the fifth, repeat per fold, report the mean. Bootstrap intervals: resample the rated items
    with replacement and repeat the whole five-fold run n_boot times."""
    d = df.dropna(subset=[score_col]).reset_index(drop=True)
    scores, bad = d[score_col].to_numpy(dtype=float), bad_mask(d).to_numpy()

    def one_run(s, b, rs):
        if b.sum() < folds:
            return float("nan"), float("nan")
        rec, prec = [], []
        for tr, te in StratifiedKFold(n_splits=folds, shuffle=True, random_state=rs).split(s, b):
            line = place_line(s[tr], b[tr], target_recall)
            if line is None or b[te].sum() == 0:
                continue
            flagged = s[te] <= line["threshold"]
            rec.append((flagged & (b[te] == 1)).sum() / b[te].sum())
            prec.append((flagged & (b[te] == 1)).sum() / max(flagged.sum(), 1))
        return float(np.mean(rec)) if rec else float("nan"), float(np.mean(prec)) if prec else float("nan")

    rec, prec = one_run(scores, bad, seed)
    rng = np.random.default_rng(seed)
    boots = []
    for k in range(n_boot):
        idx = rng.integers(0, len(scores), len(scores))
        boots.append(one_run(scores[idx], bad[idx], seed + k))
    boots = np.array([b for b in boots if not np.isnan(b).any()])
    ci = lambda col: [float(v) for v in np.percentile(boots[:, col], [2.5, 97.5])] if len(boots) else [float("nan")] * 2
    return {"score": score_col, "folds": folds, "n": int(len(d)), "n_bad": int(bad.sum()),
            "recall_mean": rec, "recall_ci": ci(0), "precision_mean": prec, "precision_ci": ci(1), "n_boot_used": int(len(boots))}


def rank_gap(df):
    """Percentile ranks of the CometKiwi mean and the MetricX mean (both stored so that higher = better),
    and their absolute difference. The cut is the median gap among items the human called D2 or D3."""
    d = df.dropna(subset=[SCORE_MEAN, "metricx_mean"])
    gap = (d[SCORE_MEAN].rank(pct=True) - d["metricx_mean"].rank(pct=True)).abs()
    bad = bad_mask(d) == 1
    return {"n": int(len(d)), "rank_gap": float(gap[bad].median()) if bad.any() else float("nan"),
            "gap_median_all": float(gap.median()) if len(d) else float("nan")}


def chrf_cut(df):
    """The round-trip chrF value below which 90 percent of the D2 and D3 items fall."""
    bad = df[(bad_mask(df) == 1) & df["roundtrip_chrf"].notna()]
    return {"n_bad_with_chrf": int(len(bad)),
            "roundtrip_chrf_cut": float(bad["roundtrip_chrf"].quantile(0.9)) if len(bad) else float("nan")}


def agreement(df):
    """Quadratic-weighted Cohen's kappa on the tier and on fluency, over the double-rated items."""
    out = {"n_double_rated": 0}
    if "tier_rater2" not in df:
        return out
    r2 = df.dropna(subset=["tier_rater2"])
    out["n_double_rated"] = int(len(r2))
    if len(r2):
        out["kappa_tier_quadratic"] = float(cohen_kappa_score(r2["tier"].map(TIER), r2["tier_rater2"].map(TIER), weights="quadratic"))
        if "fluency_rater2" in r2 and r2["fluency_rater2"].notna().any():
            f = r2.dropna(subset=["fluency_rater2"])
            out["kappa_fluency_quadratic"] = float(cohen_kappa_score(f["fluency"].astype(int), f["fluency_rater2"].astype(int), weights="quadratic"))
    return out


def recall_per_tag(df, thr, score_col=SCORE_MIN):
    """Share of the items carrying each error tag that the line catches."""
    d = df.dropna(subset=[score_col])
    flagged = d[score_col] <= thr
    tags = d["tags"].fillna("").astype(str).str.split(";").explode().str.strip()
    tags = tags[tags != ""]
    rows = []
    for tag in sorted(tags.unique()):
        idx = tags[tags == tag].index
        rows.append({"tag": tag, "items": int(len(idx)), "caught": int(flagged.loc[idx].sum()),
                     "recall": float(flagged.loc[idx].mean())})
    return pd.DataFrame(rows, columns=["tag", "items", "caught", "recall"])


def option_swap_check(df):
    """How often the option_swap flag fired on an item the human called fine (D0 or D1), and how often it
    fired on an item the human tagged option_changed."""
    if "option_swap_n" not in df:
        return {}
    fired = df[df["option_swap_n"].fillna(0) > 0]
    tags = fired["tags"].fillna("").astype(str)
    return {"fired_in_sample": int(len(fired)),
            "fired_on_D0_D1": int((fired["tier"].map(TIER) <= 1).sum()),
            "fired_with_option_changed_tag": int(tags.str.contains("option_changed").sum())}


def full_report(df, target_recall=0.90):
    thr = threshold(df, target_recall)
    rep = {"correlation_primary": correlation(df, SCORE_MEAN),
           "correlation_secondary": correlation(df, "metricx_mean") if "metricx_mean" in df else None,
           "auroc": auroc(df), "threshold": thr, "cv": cv_recall_precision(df, target_recall),
           "rank_gap": rank_gap(df) if "metricx_mean" in df else None, "chrf_cut": chrf_cut(df),
           "agreement": agreement(df), "option_swap": option_swap_check(df)}
    per_tag = recall_per_tag(df, thr["threshold"]) if not np.isnan(thr["threshold"]) else pd.DataFrame()
    return rep, per_tag


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    rep, per_tag = full_report(df)
    print(json.dumps(rep, indent=2, default=float))
    print(per_tag.to_string(index=False))
