"""
run_all.py  --  one driver for Steps 1 to 8 of the translation quality scoring workflow.
Author: Hrishin Debnath

Needs bangla_qe_pipeline.py and step7_stats.py in the same folder.
Primary metric CometKiwi-22, secondary MetricX-24 (research document, 15 September 2026, not yet confirmed by Azad).

Commands (run in this order):

  python run_all.py score --json 500_translated_gptoss.json --bt backtranslations_gptoss120b_pass1_500.csv --out results500 [--gpu] [--allow latin_allow_list_pass2.txt]
        Steps 1 to 6 on every item and translator. Input is the GUI JSON (fields item_id, english, bangla, model)
        or a Panel workbook (--xlsx WORKBOOK --sheet NAME, columns in TRANSLATORS below). --bt is the IndicTrans2
        back-translation CSV (columns item_id, translator, back_en) from Back_Translation_500.ipynb; without it
        round-trip chrF is empty. Writes OUTDIR/long_scores.csv (one row per item, translator, segment) and
        OUTDIR/item_scores.csv (one row per item and translator). Without --gpu only the CPU checks and chrF run;
        with --gpu CometKiwi, MetricX, LaBSE and the LaBSE option_swap check also run, checkpointed to OUTDIR/ckpt/.
        --bengali-only scores under the D12 policy (20 Sep 2026): the translation must be Bengali script only, so every
        Latin run of two or more letters and every Greek letter is a hard flag. Score Pass 1 and Pass 2 with the same flag.
        --allow (a Latin allow list) is the older mixed-script policy, kept for the record.
        Also writes OUTDIR/throughput.json (minutes per step and a projection for 12,000 items).

  python run_all.py summary --out OUTDIR
        The Panel 1 style defect table, one column per translator. Writes OUTDIR/summary_by_translator.csv.

  python run_all.py decide --out OUTDIR [--thresholds OUTDIR/thresholds.json]
        Step 8. Adds decision and decision_reason to item_scores.csv. Without a thresholds file only the hard
        flags (Step 3 and option_swap) fire, which is enough to run before Step 7.

  python run_all.py sample --out OUTDIR --n-random 150 --n-targeted 50
        Step 7, the draw. Writes rating_sheet.csv (shuffled, scores and translator hidden, columns for the rater),
        rating_sheet_rater2.csv (half of it, for the second rater), anchor_sheet.csv (10 items outside the sample,
        both raters practise on them first and those ratings are discarded), rating_guide.csv, and the keys
        rating_key.csv and anchor_key.csv. Do not open the keys while rating.

  python run_all.py queue --out OUTDIR --json 500_translated_gptoss_pass2.json
        Step 7, the queues for the GUI Rate tab: rating_queue.json (200), rating_queue_rater2.json (100), anchor_queue.json (10).
        Load them with "Load translations JSON" in the Rate tab; it writes ratings_<rater>_<stamp>.csv next to them.

  python run_all.py stats --out OUTDIR --ratings RATED.csv [--ratings2 RATED2.csv] [--recall 0.90]
        RATED.csv is either a filled rating_sheet.csv or a GUI ratings_<rater>_<stamp>.csv (matched by item_id).
        Step 7, the calculations. Joins the filled sheets with the key and the scores, checks the ratings against
        the rating guide, prints correlation, AUROC, threshold, five-fold recall and precision, rank gap, chrF cut,
        kappa and recall per tag. Writes thresholds.json (for decide) and stats_report.json.
"""
import argparse
import json
import pathlib
import random
import time

import numpy as np
import pandas as pd

import bangla_qe_pipeline as p
import step7_stats as s7

# ---------------------------------------------------------------- configuration
SRC_COL = "B"                                   # Panel workbook layout only
TRANSLATORS = {                                 # name: (bangla_column_letter, backtranslation_column_letter or None)
    "Gemini 3.5 Flash Lite": ("C", "E"),
    "ChatGPT GPT-5.6 Luna": ("F", "H"),
    "Claude Sonnet 4.6": ("I", "J"),
    "Google Translate": ("K", "L"),
    "Qwen 3.3 27B": ("M", "N"),
    "Gemma 4 31B": ("O", "P"),
}
HARD_FLAG_COLS = ["seg_mismatch", "other_script_n", "devanagari_n", "assamese_n", "latin_word_n",
                  "num_hard_n", "dup_option_n", "option_swap_n",
                  "bn_option_labels_n"]      # any value above 0 is a hard flag; Bengali option labels added 20 Sep 2026 (labels stay A, B, C)
SOFT_FLAG_COLS = ["num_soft_n"]
LEN_Z_LIMIT = 3.5
SEED = 42
CSV_ENC = "utf-8-sig"                           # BOM so Excel opens the Bangla correctly

MEANING_TAGS = ["number", "negation", "drug_or_disease", "clause_dropped", "option_changed", "other_fact"]
FORM_TAGS = ["untranslated", "gloss", "script", "option_missing"]
TIERS = ["D0", "D1", "D2", "D3"]
RATING_GUIDE = [
    ("tier", "D0", "Wording differs, clinical content identical."),
    ("tier", "D1", "Register or word order changed, no clinical fact changed."),
    ("tier", "D2", "A clinical fact changed (number, negation, drug or disease, dropped clause, changed option)."),
    ("tier", "D3", "The Bangla supports a different option than the answer key."),
    ("fluency", "5", "Natural Bangla. Nothing marks it as translated."),
    ("fluency", "4", "Slightly stiff or over-formal, but correct. A native speaker would not stumble."),
    ("fluency", "3", "Clearly translated. Word order or phrasing follows the English. Still understood on one reading."),
    ("fluency", "2", "Hard to read. It needs a second pass, or a sentence has to be guessed at."),
    ("fluency", "1", "Not usable Bangla. Broken grammar, wrong script, or English in place of Bangla."),
    ("acceptable", "yes/no", "Whether the item is acceptable as it is."),
    ("tag (meaning changed)", "number", "A number, dose, lab value, unit, age, sex, side or duration is missing, changed or misplaced."),
    ("tag (meaning changed)", "negation", "A negative was dropped or added. No fever becomes fever."),
    ("tag (meaning changed)", "drug_or_disease", "A drug or condition name became a different drug or condition."),
    ("tag (meaning changed)", "clause_dropped", "A clinical clause in the English is missing from the Bangla."),
    ("tag (meaning changed)", "option_changed", "An option now reads like a different option, or points at a different answer."),
    ("tag (meaning changed)", "other_fact", "The meaning changed in a way no other tag covers. Describe it in the note column."),
    ("tag (translator problem)", "untranslated", "English left in the Bangla outside parentheses."),
    ("tag (translator problem)", "gloss", "The English term added in parentheses after the Bangla term."),
    ("tag (translator problem)", "script", "Letters from another script, or a visible encoding defect."),
    ("tag (translator problem)", "option_missing", "The Bangla has fewer options than the English."),
    ("how to rate", "order", "Rate fluency with the English out of view. Then read the English, then set the tier and the tags."),
    ("how to rate", "tags", "Separate several tags with a semicolon, for example number;clause_dropped. Leave empty when nothing applies."),
]


def col_index(letter):
    import openpyxl
    return openpyxl.utils.column_index_from_string(letter) - 1


# ---------------------------------------------------------------- input readers
def read_json_records(path):
    """The GUI output: a JSON list. One translator per file (the model field)."""
    recs = json.load(open(path, encoding="utf-8"))
    rows, skipped = [], 0
    for r in recs:
        if r.get("error") or not str(r.get("bangla") or "").strip():
            skipped += 1
            continue
        rows.append({"item_id": r["item_id"], "translator": r.get("model", "json"),
                     "src_raw": str(r["english"]), "hyp_raw": str(r["bangla"]), "bt_raw": None})
    if skipped:
        print(f"skipped {skipped} records with an error or an empty output")
    return rows


def read_xlsx_records(path, sheet):
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    rows = []
    for r in [r for r in list(wb[sheet].iter_rows(values_only=True))[1:] if r[0]]:
        src = str(r[col_index(SRC_COL)])
        for t, (bc, btc) in TRANSLATORS.items():
            raw = r[col_index(bc)]
            if not raw:
                continue
            bt = r[col_index(btc)] if btc else None
            rows.append({"item_id": r[0], "translator": t, "src_raw": src, "hyp_raw": str(raw),
                         "bt_raw": str(bt) if bt else None})
    return rows


def attach_backtranslations(rows, bt_path):
    """Merge the IndicTrans2 CSV (item_id, translator, back_en). A file with no translator column is
    matched on item_id only, which is right when the file holds one translator."""
    bt = pd.read_csv(bt_path, encoding=CSV_ENC)
    keyed = {}
    for _, r in bt.iterrows():
        key = (str(r["item_id"]), str(r["translator"])) if "translator" in bt else (str(r["item_id"]), None)
        keyed[key] = str(r["back_en"]) if pd.notna(r["back_en"]) else None
    n = 0
    for row in rows:
        v = keyed.get((str(row["item_id"]), row["translator"])) or keyed.get((str(row["item_id"]), None))
        if v:
            row["bt_raw"] = v
            n += 1
    print(f"back-translations attached: {n} of {len(rows)} records")
    return rows


# ---------------------------------------------------------------- Steps 1 to 6
def score(args):
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    timing = {}
    rows = read_json_records(args.json) if args.json else read_xlsx_records(args.xlsx, args.sheet)
    if args.bt:
        rows = attach_backtranslations(rows, args.bt)
    extra_allow = p.load_allow_list(args.allow) if args.allow else None
    if extra_allow:
        print(f"Latin allow list: {len(extra_allow)} tokens from {args.allow}")

    # Steps 1 and 2: count the encoding defect, normalise, segment. Step 3: structural checks.
    # Step 4: round-trip chrF (needs the back-translation). Step 5: consensus chrF (needs 3+ translators).
    by_item = {}
    for row in rows:
        by_item.setdefault(row["item_id"], []).append(row)
    seg_rows, item_rows = [], []
    for item_id, group in by_item.items():
        src = p.normalise_symbols(group[0]["src_raw"])
        src_segs = p.segment(src)
        cleaned = {}
        for row in group:
            hyp_nfc, n_pre = p.nfc(row["hyp_raw"])
            cleaned[row["translator"]] = (p.clean_bangla(hyp_nfc), n_pre)
        cons = p.consensus_chrf({t: h for t, (h, _) in cleaned.items()}) if len(cleaned) >= 3 else {}
        for row in group:
            t = row["translator"]
            hyp, n_pre = cleaned[t]
            hyp_segs = p.segment(hyp)
            mismatch = int(len(hyp_segs) != len(src_segs))
            pairs = list(zip(src_segs, hyp_segs)) if not mismatch else [(src, hyp)]
            for i, (s, h) in enumerate(pairs):
                seg_rows.append({"item_id": item_id, "translator": t, "seg_id": i, "src_en": s, "hyp_bn": h,
                                 "seg_mismatch": mismatch})
            d = {"item_id": item_id, "translator": t, "precomposed_n": n_pre, "seg_mismatch": mismatch,
                 "n_seg_src": len(src_segs), "n_seg_hyp": len(hyp_segs),
                 "len_ratio": p.length_ratio(src, hyp), "consensus_chrf": cons.get(t, np.nan)}
            d.update(p.script_checks(hyp, src, bengali_only=args.bengali_only))
            d.update(p.untranslated_english(src, hyp, extra_allow, bengali_only=args.bengali_only))
            d.update(p.numbers_preserved(src, hyp))
            d.update(p.duplicate_options(hyp_segs))
            d["roundtrip_chrf"] = p.roundtrip_chrf([src], [p.normalise_symbols(row["bt_raw"])])[0] if row["bt_raw"] else np.nan
            item_rows.append(d)
    long_df = pd.DataFrame(seg_rows)
    items = pd.DataFrame(item_rows)
    items["len_z"] = items.groupby("translator")["len_ratio"].transform(p.robust_z)
    timing["cpu_steps_1_to_5_min"] = round((time.time() - t0) / 60, 2)
    print(f"Steps 1 to 5 done: {len(items)} records, {len(long_df)} segments, {timing['cpu_steps_1_to_5_min']} min")

    # Step 6: learned scores per segment, checkpointed, then rolled up to item mean and min.
    if args.gpu:
        scorable = long_df[long_df["seg_mismatch"] == 0]      # seg_mismatch cells are FLAG already; a one-block score would be truncated
        metrics = {"cometkiwi": (p.cometkiwi, 64),
                   "metricx": (lambda s, h: p.metricx_qe(s, h, metricx_dir=args.metricx_dir), 64),
                   "labse": (p.labse_cosine, 256)}
        for name, (fn, batch) in metrics.items():
            t1 = time.time()
            print(f"scoring {name} on {len(scorable)} segments ({len(long_df) - len(scorable)} seg_mismatch rows skipped)")
            res = p.run_metric(scorable, name, fn, out / "ckpt", batch=batch)
            long_df = long_df.merge(res, on=["item_id", "translator", "seg_id"], how="left")
            timing[f"{name}_min"] = round((time.time() - t1) / 60, 2)
            print(f"  {name}: {timing[f'{name}_min']} min")
        long_df["metricx_err"] = long_df["metricx"]           # raw error score, lower = better
        long_df["metricx"] = -long_df["metricx"]              # negated so that higher = better, like CometKiwi
        agg = long_df.groupby(["item_id", "translator"]).agg(
            cometkiwi_mean=("cometkiwi", "mean"), cometkiwi_min=("cometkiwi", "min"),
            metricx_mean=("metricx", "mean"), metricx_min=("metricx", "min"),
            metricx_err_mean=("metricx_err", "mean"), metricx_err_max=("metricx_err", "max"),
            labse_mean=("labse", "mean"), labse_min=("labse", "min"),
            n_seg_scored=("cometkiwi", "count")).reset_index()
        items = items.merge(agg, on=["item_id", "translator"], how="left")
        t1 = time.time()
        swaps = p.option_swap(long_df)
        items = items.merge(swaps, on=["item_id", "translator"], how="left")
        items["option_swap_n"] = items["option_swap_n"].fillna(0).astype(int)
        timing["option_swap_min"] = round((time.time() - t1) / 60, 2)
        print(f"  option_swap: {timing['option_swap_min']} min, fired on {(items['option_swap_n'] > 0).sum()} records")
    else:
        items["option_swap_n"] = 0                            # not computed; the column keeps decide() simple
        items["option_swaps"] = ""

    n_items = items["item_id"].nunique()
    timing["records"] = int(len(items))
    timing["projection_hours_per_12000_items_one_translator"] = {
        k: round(v / n_items * 12000 / 60, 1) for k, v in timing.items() if k.endswith("_min")}
    json.dump(timing, open(out / "throughput.json", "w"), indent=2)
    long_df.to_csv(out / "long_scores.csv", index=False, encoding=CSV_ENC)
    items.to_csv(out / "item_scores.csv", index=False, encoding=CSV_ENC)
    print(f"wrote {out / 'item_scores.csv'}: {len(items)} rows, {n_items} items, {items.translator.nunique()} translators")
    print("throughput:", json.dumps(timing))


# ---------------------------------------------------------------- summary table (Panel 1 style)
def summary(args):
    out = pathlib.Path(args.out)
    items = pd.read_csv(out / "item_scores.csv", encoding=CSV_ENC)
    rows = {}
    for t, g in items.groupby("translator"):
        r = {"records": len(g),
             "precomposed chars (items)": f"{int(g.precomposed_n.sum())} ({int((g.precomposed_n > 0).sum())})",
             "option count mismatch (items)": int(g.seg_mismatch.sum()),
             "items with other-script letters": int((g.other_script_n + g.devanagari_n + g.assamese_n > 0).sum()),
             "items with untranslated English (words)": f"{int((g.latin_word_n > 0).sum())} ({int(g.latin_word_n.sum())})",
             "English gloss words": int(g.gloss_n.sum()),
             "items with a hard number miss": int((g.num_hard_n > 0).sum()),
             "items with a soft number miss": int((g.num_soft_n > 0).sum()),
             "items with duplicate options": int((g.dup_option_n > 0).sum()),
             "items with length z beyond 3.5": int((g.len_z.abs() > LEN_Z_LIMIT).sum()),
             "items with option_swap": int((g.option_swap_n > 0).sum()) if "option_swap_n" in g else "not run",
             "items with any hard flag": int(hard_flag(g).sum()),
             "consensus chrF mean": round(g.consensus_chrf.mean(), 1) if g.consensus_chrf.notna().any() else "n/a (1 translator)",
             "round-trip chrF mean (min)": f"{g.roundtrip_chrf.mean():.1f} ({g.roundtrip_chrf.min():.1f})" if g.roundtrip_chrf.notna().any() else "no back-translation"}
        for c in ["cometkiwi_mean", "cometkiwi_min", "metricx_err_mean", "metricx_err_max", "labse_mean", "labse_min"]:
            if c in g and g[c].notna().any():
                r[f"{c} (mean over items)"] = round(g[c].mean(), 3)
        if "decision" in g:
            for k, v in g.decision.value_counts().items():
                r[f"decision {k}"] = int(v)
        rows[t] = r
    table = pd.DataFrame(rows)
    table.index.name = "check"
    table.to_csv(out / "summary_by_translator.csv", encoding=CSV_ENC)
    print(table.to_string())


# ---------------------------------------------------------------- Step 8: one decision per item
def hard_flag(items):
    cols = [c for c in HARD_FLAG_COLS if c in items]
    return (items[cols].fillna(0).sum(axis=1) > 0) | (items["len_z"].abs() > LEN_Z_LIMIT)


def decide(args):
    out = pathlib.Path(args.out)
    items = pd.read_csv(out / "item_scores.csv", encoding=CSV_ENC)
    thr = json.load(open(args.thresholds)) if args.thresholds else {}
    decision = pd.Series("PASS", index=items.index)
    reason = pd.Series("", index=items.index)

    def add(mask, label):
        reason[mask] = reason[mask].where(reason[mask] == "", reason[mask] + ";") + label

    for c in [c for c in HARD_FLAG_COLS if c in items]:
        m = items[c].fillna(0) > 0
        add(m, f"hard:{c}")
    add(items["len_z"].abs() > LEN_Z_LIMIT, "hard:len_z")
    hard = hard_flag(items)
    if "cometkiwi_min_threshold" in thr and "cometkiwi_min" in items:
        m = items["cometkiwi_min"] <= thr["cometkiwi_min_threshold"]
        add(m & ~hard, "flag:cometkiwi_min")
        decision[m] = "FLAG"
    review = pd.Series(False, index=items.index)
    for c in [c for c in SOFT_FLAG_COLS if c in items]:
        m = items[c].fillna(0) > 0
        add(m & ~hard, f"soft:{c}")
        review |= m
    if "rank_gap" in thr and {"metricx_mean", "cometkiwi_mean"} <= set(items):
        gap = (items["cometkiwi_mean"].rank(pct=True) - items["metricx_mean"].rank(pct=True)).abs()
        m = gap > thr["rank_gap"]
        add(m & ~hard, "review:rank_gap")
        review |= m
    if "roundtrip_chrf_cut" in thr and "roundtrip_chrf" in items:
        m = items["roundtrip_chrf"] < thr["roundtrip_chrf_cut"]
        add(m & ~hard, "review:roundtrip_chrf")
        review |= m
    decision[review & (decision == "PASS")] = "REVIEW"
    decision[hard] = "FLAG"                                    # hard flags win over everything
    items["decision"] = decision
    items["decision_reason"] = reason
    items.to_csv(out / "item_scores.csv", index=False, encoding=CSV_ENC)
    print("thresholds used:", {k: thr[k] for k in ("cometkiwi_min_threshold", "rank_gap", "roundtrip_chrf_cut") if k in thr} or "hard flags only")
    print(items.groupby("translator")["decision"].value_counts().unstack(fill_value=0))


# ---------------------------------------------------------------- Step 7: draw the sample, write the blind sheets
def _item_text(out):
    long_df = pd.read_csv(out / "long_scores.csv", encoding=CSV_ENC)
    long_df = long_df.sort_values(["item_id", "translator", "seg_id"])
    labels = lambda g: [g["src_en"].iloc[0]] + [f"{p.OPTION_LABELS[i - 1]}. {v}" for i, v in enumerate(g["src_en"].iloc[1:], 1)]
    labels_bn = lambda g: [g["hyp_bn"].iloc[0]] + [f"{p.OPTION_LABELS[i - 1]}. {v}" for i, v in enumerate(g["hyp_bn"].iloc[1:], 1)]
    rows = []
    for (item, t), g in long_df.groupby(["item_id", "translator"]):
        if int(g["seg_mismatch"].iloc[0]) == 1:
            rows.append({"item_id": item, "translator": t, "src_en": g["src_en"].iloc[0], "hyp_bn": g["hyp_bn"].iloc[0]})
        else:
            rows.append({"item_id": item, "translator": t, "src_en": "\n".join(labels(g)), "hyp_bn": "\n".join(labels_bn(g))})
    return pd.DataFrame(rows)


def _write_sheet(df, key_cols, path, key_path, prefix):
    df = df.reset_index(drop=True)
    df["row_id"] = [f"{prefix}{i + 1:04d}" for i in range(len(df))]
    df[["row_id"] + key_cols].to_csv(key_path, index=False, encoding=CSV_ENC)
    sheet = df[["row_id", "src_en", "hyp_bn"]].copy()
    for col in ["tier", "fluency", "acceptable", "tags", "note"]:
        sheet[col] = ""
    sheet.to_csv(path, index=False, encoding=CSV_ENC)
    return df


def sample(args):
    out = pathlib.Path(args.out)
    items = pd.read_csv(out / "item_scores.csv", encoding=CSV_ENC)
    if "cometkiwi_min" not in items:
        raise SystemExit("item_scores.csv has no learned scores. Run: score --gpu")
    items = items[items["cometkiwi_min"].notna()]             # the sample is drawn from the scored set
    rng = random.Random(SEED)
    per_t = args.n_random // items.translator.nunique()
    rand = pd.concat([g.sample(min(per_t, len(g)), random_state=SEED) for _, g in items.groupby("translator")]).assign(part="random")
    rest = items.drop(rand.index)
    half = args.n_targeted // 2
    pool = rest.nsmallest(max(half, int(0.1 * len(rest))), "cometkiwi_min")    # the worst ten percent by CometKiwi item minimum
    worst = pool.sample(min(half, len(pool)), random_state=SEED).assign(part="targeted_worst")   # a random draw from them, not the very worst
    rest2 = rest.drop(worst.index)
    if "metricx_mean" in rest2 and rest2["metricx_mean"].notna().any():
        gap = (rest2["cometkiwi_mean"].rank(pct=True) - rest2["metricx_mean"].rank(pct=True)).abs()
        disagree = rest2.loc[gap.nlargest(args.n_targeted - half).index].assign(part="targeted_disagree")
    else:
        disagree = rest2.nsmallest(args.n_targeted - half, "roundtrip_chrf").assign(part="targeted_disagree")
    chosen = pd.concat([rand, worst, disagree]).reset_index(drop=True)
    order = list(range(len(chosen)))
    rng.shuffle(order)
    chosen = chosen.iloc[order].reset_index(drop=True)
    text = _item_text(out)
    chosen = chosen.merge(text, on=["item_id", "translator"])
    chosen = _write_sheet(chosen, ["item_id", "translator", "part"], out / "rating_sheet.csv", out / "rating_key.csv", "R")

    # second rater: half of the sheet, same share of random and targeted rows, same row_ids
    r2 = pd.concat([g.sample(len(g) // 2, random_state=SEED) for _, g in chosen.groupby(chosen["part"] == "random")])
    r2 = r2.sort_values("row_id")
    sheet2 = r2[["row_id", "src_en", "hyp_bn"]].copy()
    for col in ["tier", "fluency", "acceptable", "tags", "note"]:
        sheet2[col] = ""
    sheet2.to_csv(out / "rating_sheet_rater2.csv", index=False, encoding=CSV_ENC)

    # anchor set: 10 items outside the sample, rated by both raters first, ratings discarded
    left = items[~items["item_id"].isin(chosen["item_id"])]
    anchor = left.sample(min(10, len(left)), random_state=SEED).merge(text, on=["item_id", "translator"])
    _write_sheet(anchor, ["item_id", "translator"], out / "anchor_sheet.csv", out / "anchor_key.csv", "A")

    pd.DataFrame(RATING_GUIDE, columns=["field", "value", "meaning"]).to_csv(out / "rating_guide.csv", index=False, encoding=CSV_ENC)
    print(f"wrote rating_sheet.csv ({len(chosen)} rows: {len(rand)} random, {len(worst)} worst ten percent by CometKiwi min, "
          f"{len(disagree)} largest CometKiwi vs MetricX disagreement), rating_sheet_rater2.csv ({len(sheet2)} rows), "
          f"anchor_sheet.csv ({len(anchor)} rows), rating_guide.csv, rating_key.csv, anchor_key.csv. Do not open the keys while rating.")


# ---------------------------------------------------------------- Step 7: rating queues for the GUI Rate tab
def queue(args):
    """Cuts the sampled records out of the translations JSON so the GUI Rate tab can load them.
    Writes rating_queue.json (the 200), rating_queue_rater2.json (the 100) and anchor_queue.json (the 10),
    records unchanged. The Rate tab shuffles them itself and writes ratings_<rater>_<stamp>.csv keyed by item_id."""
    out = pathlib.Path(args.out)
    recs = {r["item_id"]: r for r in json.load(open(args.json, encoding="utf-8"))}
    key = pd.read_csv(out / "rating_key.csv", encoding=CSV_ENC)
    r2 = pd.read_csv(out / "rating_sheet_rater2.csv", encoding=CSV_ENC)
    anchor = pd.read_csv(out / "anchor_key.csv", encoding=CSV_ENC)
    sets = {"rating_queue.json": key["item_id"].tolist(),
            "rating_queue_rater2.json": key[key["row_id"].isin(r2["row_id"])]["item_id"].tolist(),
            "anchor_queue.json": anchor["item_id"].tolist()}
    for name, ids in sets.items():
        missing = [i for i in ids if i not in recs]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} sampled items are not in {args.json}, for example {missing[:3]}")
        json.dump([recs[i] for i in ids], open(out / name, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"wrote {out / name}: {len(ids)} records")


# ---------------------------------------------------------------- Step 7: statistics and thresholds
def check_ratings(df, who):
    """Warn on anything the rating guide does not allow. Warnings only; nothing is changed."""
    n = 0
    tiers, flu = df["tier"].astype(str).str.strip(), pd.to_numeric(df["fluency"], errors="coerce")
    for rid in df.loc[~tiers.isin(TIERS), "row_id"]:
        print(f"  {who}: {rid} tier is not D0..D3"); n += 1
    for rid in df.loc[flu.isna() | (flu < 1) | (flu > 5), "row_id"]:
        print(f"  {who}: {rid} fluency is not 1..5"); n += 1
    tags = df["tags"].fillna("").astype(str)
    for rid, tg, tier in zip(df["row_id"], tags, tiers):
        parts = [x.strip() for x in tg.split(";") if x.strip()]
        for x in parts:
            if x not in MEANING_TAGS + FORM_TAGS:
                print(f"  {who}: {rid} unknown tag '{x}'"); n += 1
        if tier in ("D2", "D3") and not any(x in MEANING_TAGS for x in parts):
            print(f"  {who}: {rid} is {tier} but carries no meaning-changed tag"); n += 1
    acc = df["acceptable"].fillna("").astype(str).str.strip().str.lower()
    for rid in df.loc[~acc.isin(["yes", "no"]), "row_id"]:
        print(f"  {who}: {rid} acceptable is not yes/no"); n += 1
    print(f"{who}: {len(df)} rows checked, {n} warnings")


def stats(args):
    out = pathlib.Path(args.out)
    key = pd.read_csv(out / "rating_key.csv", encoding=CSV_ENC)

    def load_ratings(path, who):
        """A filled rating_sheet.csv (row_id) or a GUI Rate-tab file ratings_<rater>_<stamp>.csv (item_id)."""
        rt = pd.read_csv(path, encoding=CSV_ENC)
        rt = rt[rt["tier"].notna() & (rt["tier"].astype(str).str.strip() != "")]
        if "row_id" not in rt and "item_id" in rt:
            rt = rt.drop_duplicates("item_id", keep="last").merge(key[["row_id", "item_id"]], on="item_id", how="inner")
            print(f"{who}: GUI ratings file, {len(rt)} rated items matched to the sample")
        for c in ["acceptable", "tags", "note"]:
            if c not in rt:
                rt[c] = ""
        check_ratings(rt, who)
        return rt

    ratings = load_ratings(args.ratings, "rater 1")
    items = pd.read_csv(out / "item_scores.csv", encoding=CSV_ENC)
    df = ratings[["row_id", "tier", "fluency", "acceptable", "tags"]].merge(key, on="row_id").merge(items, on=["item_id", "translator"])
    if args.ratings2:
        r2 = load_ratings(args.ratings2, "rater 2")
        df = df.merge(r2[["row_id", "tier", "fluency"]].rename(columns={"tier": "tier_rater2", "fluency": "fluency_rater2"}),
                      on="row_id", how="left")
    if "cometkiwi_min" not in df:
        raise SystemExit("item_scores.csv has no learned scores. Run: score --gpu")
    df["tier"] = df["tier"].astype(str).str.strip()
    df["fluency"] = pd.to_numeric(df["fluency"], errors="coerce")
    df["part"] = df["part"].astype(str)
    rep, per_tag = s7.full_report(df, target_recall=args.recall)
    thr = rep["threshold"]
    share_full = float((items["cometkiwi_min"] <= thr["threshold"]).mean()) if not np.isnan(thr["threshold"]) else float("nan")
    rep["flag_share_full_set"] = share_full
    print(json.dumps(rep, indent=2, default=float))
    print("recall per tag at threshold", thr["threshold"])
    print(per_tag.to_string(index=False))
    per_tag.to_csv(out / "recall_per_tag.csv", index=False, encoding=CSV_ENC)
    thresholds = {"cometkiwi_min_threshold": thr["threshold"],
                  "rank_gap": rep["rank_gap"]["rank_gap"] if rep["rank_gap"] else float("nan"),
                  "roundtrip_chrf_cut": rep["chrf_cut"]["roundtrip_chrf_cut"],
                  "recall_in_sample": thr["recall"], "precision_in_sample": thr["precision"],
                  "recall_cv": rep["cv"]["recall_mean"], "precision_cv": rep["cv"]["precision_mean"],
                  "auroc_random": rep["auroc"]["random"]["auroc"], "auroc_all_enriched": rep["auroc"]["all_enriched"]["auroc"],
                  "spearman_random": rep["correlation_primary"]["spearman"],
                  "flag_share_in_sample": thr["flag_share"], "flag_share_full_set": share_full}
    json.dump(thresholds, open(out / "thresholds.json", "w"), indent=2, default=float)
    json.dump(rep, open(out / "stats_report.json", "w"), indent=2, default=float)
    print("wrote thresholds.json and stats_report.json:", json.dumps(thresholds, default=float))


# ---------------------------------------------------------------- CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("score")
    a.add_argument("--json", default=None); a.add_argument("--xlsx", default=None); a.add_argument("--sheet", default="Translation run sheet")
    a.add_argument("--bt", default=None); a.add_argument("--out", required=True); a.add_argument("--gpu", action="store_true")
    a.add_argument("--metricx-dir", default="/content/metricx")
    a.add_argument("--allow", default=None, help="Latin allow list, one token per line (superseded by --bengali-only on 20 Sep 2026)")
    a.add_argument("--bengali-only", action="store_true", help="D12 policy: every Latin run of 2+ letters and every Greek letter is a defect")
    a = sub.add_parser("summary"); a.add_argument("--out", required=True)
    a = sub.add_parser("sample"); a.add_argument("--out", required=True); a.add_argument("--n-random", type=int, default=150); a.add_argument("--n-targeted", type=int, default=50)
    a = sub.add_parser("queue"); a.add_argument("--out", required=True); a.add_argument("--json", required=True)
    a = sub.add_parser("stats"); a.add_argument("--out", required=True); a.add_argument("--ratings", required=True); a.add_argument("--ratings2", default=None); a.add_argument("--recall", type=float, default=0.90)
    a = sub.add_parser("decide"); a.add_argument("--out", required=True); a.add_argument("--thresholds", default=None)
    args = ap.parse_args()
    if args.cmd == "score" and not (args.json or args.xlsx):
        raise SystemExit("score needs --json FILE or --xlsx WORKBOOK")
    {"score": score, "summary": summary, "sample": sample, "queue": queue, "stats": stats, "decide": decide}[args.cmd](args)
