"""
bangla_qe_pipeline.py  --  reference-free quality scoring for Bangla translations of MedQA items.
Author: Hrishin Debnath

Shape of the data
  input     : one record per (item, translator). Either a JSON list from the Bangla Gap GUI
              (fields item_id, english, bangla, model) or a Panel workbook (one Bangla column per translator).
  long table: one row per (item_id, translator, seg_id). Segment scoring happens here.
  item table: one row per (item_id, translator): counts, flags, means and minimums.

Every metric is one function: fn(src_list, hyp_list, **kw) -> list of floats (same length, same order).
run_metric() batches, checkpoints to a JSONL file after every batch, and skips rows already scored,
so a Colab disconnect costs at most one batch.

Step numbers below follow the "Translation Quality Scoring" tab of the research document.

Verified on 19 Sep 2026 (CPU, this machine): Steps 1 to 5 on the 500 gpt-oss:120b records.
NOT run here (no GPU, no HuggingFace access): cometkiwi, metricx_qe, labse_cosine, option_swap.
They follow the model cards and the official metricx24/predict.py; the Colab notebook runs a
three-pair smoke test before the full run so a wrong call signature fails early.
"""
import functools
import json
import pathlib
import re
import unicodedata

import pandas as pd
import sacrebleu

# ---------------------------------------------------------------- 0. normalisation and segmentation
PRECOMPOSED = ("\u09DC", "\u09DD", "\u09DF")          # ড় ঢ় য় precomposed; NFC decomposes them
ASSAMESE_ONLY = ("\u09F0", "\u09F1")                  # ৰ ৱ : Assamese letters, not standard Bangla
BENGALI_LETTER = re.compile(r"[\u0980-\u09FF]")
DEVANAGARI = re.compile(r"[\u0900-\u0963\u0966-\u097F]")   # excludes danda U+0964/0965, shared with Bangla
GREEK = re.compile(r"[\u0370-\u03FF]")
GREEK_SYMBOLS = set("\u03b1\u03b2\u03b3\u03b4\u03b5\u03ba\u03bb\u03bc\u03c3\u03c9\u0394\u03a9")   # α β γ δ ε κ λ μ σ ω Δ Ω
# letters outside the Latin, Greek and Bengali blocks: Cyrillic, Gurmukhi, Tamil, Lao ... (Devanagari counted separately)
OTHER_SCRIPT = re.compile(r"[^\u0000-\u024F\u0370-\u03FF\u0900-\u09FF\u2000-\u206F\s\d\W]")
OPTION_LINE = re.compile(r"^\s*([A-L]|[\u0985-\u09B9][\u0980-\u09FF]{0,3})[.:\u0964]\s", re.M)   # A. to L., or a 1 to 4 letter Bengali label
LABEL_BN = re.compile(r"^\s*[\u0985-\u09B9][\u0980-\u09FF]{0,3}[.:\u0964]\s", re.M)
BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
OPTION_LABELS = "ABCDEFGHIJKL"

# the GUI prompt asks the model to echo the item id and a QUESTION: line. They are a wrapper, not translation.
PREFIX_ITEM_ID = re.compile(r"^\s*ITEM_ID:[^\n]*\n?")
PREFIX_QUESTION = re.compile(r"^\s*QUESTION:\s*")
# hyphen look-alikes the model emits (U+2010 hyphen, U+2011 non-breaking hyphen, U+2012, U+2013, U+2212 minus)
HYPHENS = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2212": "-"})
# sub- and superscript digits and signs (HCO₃⁻, PaCO₂, CD4⁺) mapped to ASCII so the number and script checks read them
SUBSUP = str.maketrans({
    "\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3", "\u2084": "4", "\u2085": "5", "\u2086": "6", "\u2087": "7", "\u2088": "8", "\u2089": "9",
    "\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3", "\u2074": "4", "\u2075": "5", "\u2076": "6", "\u2077": "7", "\u2078": "8", "\u2079": "9",
    "\u207a": "+", "\u207b": "-", "\u208a": "+", "\u208b": "-"})


def nfc(text):
    """Step 1. NFC-normalise. Returns (text, count of precomposed chars found before normalising)."""
    text = "" if text is None else str(text)
    n_pre = sum(text.count(c) for c in PRECOMPOSED)
    return unicodedata.normalize("NFC", text), n_pre


def normalise_symbols(text):
    """Map hyphen variants to '-' and sub/superscript digits and signs to ASCII. Used on both languages."""
    return ("" if text is None else str(text)).translate(HYPHENS).translate(SUBSUP)


def clean_bangla(text):
    """Working copy of a GUI output for scoring: drop the ITEM_ID and QUESTION wrapper lines, then
    normalise_symbols. The stored record is never changed. Apply after nfc()."""
    text = "" if text is None else str(text)
    text = PREFIX_ITEM_ID.sub("", text, count=1)
    text = PREFIX_QUESTION.sub("", text, count=1)
    return normalise_symbols(text).strip()


def segment(text):
    """Step 2. Split an item cell into [stem, option A, option B, ...]. Labels are Latin A to L (with . or :)
    or a short Bengali label such as ক. or এ. (Google Translate emits them; counted by script_checks)."""
    parts = OPTION_LINE.split(text.strip())
    stem = parts[0].strip()
    opts = [parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)]
    return [stem] + opts


def segment_labelled(text):
    """Like segment() but returns (stem, [(label, option_text), ...]) with the labels as the model wrote them.
    The back-translation notebook uses it to keep the model's own labels in the English output."""
    parts = OPTION_LINE.split(text.strip())
    stem = parts[0].strip()
    opts = [(parts[i].strip(), parts[i + 1].strip()) for i in range(1, len(parts) - 1, 2)]
    return stem, opts


# ---------------------------------------------------------------- 1. cheap structural checks (CPU), Step 3
def script_checks(hyp, src="", bengali_only=False):
    """Script leakage. Bengali fraction of letters, Assamese-only letters, Devanagari letters, double virama,
    and other-script letters. A letter is not counted when the English source contains the same character.
    Greek letters that are common symbols (α β γ δ ε κ λ μ σ ω Δ Ω) are allowed; any other Greek letter
    that is not in the source counts as other-script (this catches leaked Greek words such as ειπ)."""
    src_chars = set(src or "")
    letters = [c for c in hyp if c.isalpha()]
    other = [c for c in OTHER_SCRIPT.findall(hyp) if c not in src_chars]
    greek = ([c for c in GREEK.findall(hyp)] if bengali_only                       # D12: Greek letters become Bangla names
             else [c for c in GREEK.findall(hyp) if c not in src_chars and c not in GREEK_SYMBOLS])
    deva = [c for c in DEVANAGARI.findall(hyp) if c not in src_chars]
    return {
        "bengali_frac": (sum(bool(BENGALI_LETTER.match(c)) for c in letters) / len(letters)) if letters else 0.0,
        "assamese_n": sum(hyp.count(c) for c in ASSAMESE_ONLY),
        "devanagari_n": len(deva),
        "double_virama_n": hyp.count("\u09CD\u09CD"),
        "other_script_n": len(other) + len(greek),
        "other_script_chars": "".join(sorted(set(other + greek + deva))),
        "bn_option_labels_n": len(LABEL_BN.findall(hyp)),
    }


# units and lab abbreviations that stay in Latin script in Bangla clinical text (lower-case forms)
UNITS = {
    "mm", "hg", "mmhg", "cm", "km", "kg", "mg", "mcg", "ug", "µg", "ng", "pg", "dl", "ml", "ul", "µl", "fl",
    "meq", "mmol", "umol", "µmol", "nmol", "pmol", "mol", "mosm", "mosmol", "iu", "miu", "hpf", "lpf", "ph",
    "bpm", "min", "hr", "hrs", "sec", "ms", "msec", "kcal", "cal", "kpa", "torr", "cmh2o", "mci", "gy", "cgy",
    "ppm", "mcl", "au", "rpm", "cc", "lb", "oz", "ft", "kb", "mb",
}
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*")      # Latin run, digits and hyphens allowed inside
GLOSS = re.compile(r"\(([^()]*[A-Za-z]{2,}[^()]*)\)")


def untranslated_english(src, hyp, extra_allow=None, bengali_only=False):
    """Latin-script words in the Bangla output, split into two counts.
    extra_allow: extra lower-case tokens to allow, from a Latin allow list.
    bengali_only: the policy of 20 September 2026 (D12): the translation is Bengali script only, so every Latin
    run of two or more letters counts, whatever the source did (units, symbols and abbreviations included).
    gloss_n: Latin words inside parentheses, the '(prolonged labor)' habit.
    latin_word_n: Latin words outside parentheses, minus tokens the source keeps in Latin form
    (any source token with an upper-case letter or a digit: Apgar, Tdap, MRI, HbA1c, HCO3, D-dimer, anti-dsDNA,
    and the capitalised parts of such tokens: anti-Ro allows Ro),
    minus units (mm Hg, kg, mg, hpf, pH ...), and minus two source unit tokens the translator joined (mm Hg -> mmHg).
    A Latin fragment glued to a Bengali word (kolch+icine) is counted. Single letters are ignored."""
    src_tokens = TOKEN.findall(src)
    keep = [t for t in src_tokens if any(ch.isupper() or ch.isdigit() for ch in t)]
    allow = {t.lower() for t in keep}
    allow |= {part.lower() for t in keep for part in t.split("-") if any(ch.isupper() or ch.isdigit() for ch in part)}   # anti-Ro -> Ro
    allow |= {(a + b).lower() for a, b in zip(src_tokens, src_tokens[1:]) if a.lower() in UNITS and b.lower() in UNITS}
    allow |= UNITS
    if extra_allow:
        allow |= {a.lower() for a in extra_allow}
    if bengali_only:
        allow = set()
    glosses = GLOSS.findall(hyp)
    gloss_words = [w for g in glosses for w in TOKEN.findall(g) if len(w) > 1 and w.lower() not in allow]
    rest = GLOSS.sub(" ", hyp)
    found = [w for w in TOKEN.findall(rest) if len(w) > 1 and w.lower() not in allow]
    return {"latin_word_n": len(found), "latin_words": " ".join(found), "gloss_n": len(gloss_words)}


def _canon_number(n):
    return n.replace(",", "")


# Bangla number words and ordinals for 0 to 20, the tens and 100. A one- or two-digit source number counts as
# found when its digits appear or one of these words appears anywhere in the Bangla (substring match, so একটি
# and একজন count for 1). Three-digit numbers and decimals are never written as words in clinical text.
BN_NUMBER_WORDS = {
    "0": ["শূন্য"], "1": ["এক", "প্রথম"], "2": ["দুই", "দু", "দ্বিতীয়"], "3": ["তিন", "তৃতীয়"], "4": ["চার", "চতুর্থ"],
    "5": ["পাঁচ", "পঞ্চম"], "6": ["ছয়", "ছ'", "ষষ্ঠ"], "7": ["সাত", "সপ্তম"], "8": ["আট", "অষ্টম"], "9": ["নয়", "নবম"],
    "10": ["দশ", "দশম"], "11": ["এগারো", "এগার"], "12": ["বারো", "বার", "দ্বাদশ"], "13": ["তেরো", "তের"], "14": ["চৌদ্দ"],
    "15": ["পনেরো", "পনের"], "16": ["ষোলো", "ষোল"], "17": ["সতেরো", "সতের"], "18": ["আঠারো", "আঠার"], "19": ["উনিশ"],
    "20": ["বিশ", "কুড়ি"], "30": ["ত্রিশ"], "40": ["চল্লিশ"], "50": ["পঞ্চাশ"], "60": ["ষাট"], "70": ["সত্তর"], "80": ["আশি"],
    "90": ["নব্বই"], "100": ["একশ", "একশো", "শত"],
}


def load_allow_list(path):
    """One token per line, # starts a comment. Returns a set of lower-case tokens."""
    out = set()
    for line in open(path, encoding="utf-8"):
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line.lower())
    return out


def numbers_preserved(src, hyp):
    """Every number in the source should appear in the output (Bengali digits mapped to ASCII, thousands
    commas removed on both sides). Missing numbers are sorted by shape:
    hard: a decimal point, or three or more digits (doses, lab values, temperatures, weights).
    soft: one or two digits that are neither present as digits nor as a Bangla number word (BN_NUMBER_WORDS)."""
    src_nums = sorted({_canon_number(n) for n in NUMBER.findall(src)})
    hyp_nums = {_canon_number(n) for n in NUMBER.findall(hyp.translate(BN_DIGITS))}
    hyp_nfc = unicodedata.normalize("NFC", hyp)
    missing = [n for n in src_nums if n not in hyp_nums
               and not any(unicodedata.normalize("NFC", w) in hyp_nfc for w in BN_NUMBER_WORDS.get(n, []))]
    hard = [n for n in missing if "." in n or len(re.sub(r"\D", "", n)) >= 3]
    soft = [n for n in missing if n not in hard]
    return {"num_src": len(src_nums), "num_missing": len(missing),
            "num_hard_n": len(hard), "num_soft_n": len(soft),
            "num_missing_list": " ".join(missing),
            "num_recall": 1.0 if not src_nums else 1 - len(missing) / len(src_nums)}


def duplicate_options(hyp_segs):
    """Hard flag when two Bangla options are identical after trimming spaces (segments 1..n)."""
    opts = [re.sub(r"\s+", " ", o).strip() for o in hyp_segs[1:]]
    return {"dup_option_n": len(opts) - len(set(opts))}


def length_ratio(src, hyp):
    return len(hyp) / max(len(src), 1)          # characters; converted to a robust z per translator later


def robust_z(series):
    med = series.median()
    mad = 1.483 * (series - med).abs().median()
    return (series - med) / mad if mad else series * 0


# ---------------------------------------------------------------- 2. Steps 4 and 5 (CPU)
def consensus_chrf(hyps_by_translator):
    """Step 5. For each translator: mean chrF of its Bangla output against every other translator's output.
    Low value = odd one out. Needs 3+ translators. Agreement is not correctness."""
    out = {}
    for t, h in hyps_by_translator.items():
        others = [o for k, o in hyps_by_translator.items() if k != t and o]
        out[t] = (sum(sacrebleu.sentence_chrf(h, [o]).score for o in others) / len(others)) if others else float("nan")
    return out


def roundtrip_chrf(src_list, back_list):
    """Step 4. English side: back-translation vs original source, whole item. Blind to untranslated English."""
    return [sacrebleu.sentence_chrf(b, [s]).score for s, b in zip(src_list, back_list)]


# ---------------------------------------------------------------- 3. learned metrics (GPU), Step 6 -- NOT run here
@functools.lru_cache(maxsize=None)
def _comet_model(model_name):
    from comet import download_model, load_from_checkpoint   # pip install "unbabel-comet>=2.2.0"; HF login needed (gated)
    return load_from_checkpoint(download_model(model_name))


@functools.lru_cache(maxsize=None)
def _labse_model():
    from sentence_transformers import SentenceTransformer                   # sentence-transformers/LaBSE, 109 languages incl. bn
    return SentenceTransformer("sentence-transformers/LaBSE")


@functools.lru_cache(maxsize=None)
def _metricx_model(model_name, tokenizer_name, metricx_dir):
    """MetricX-24 loaded in-process, the way the official metricx24/predict.py does it."""
    import sys
    import torch
    import transformers
    if metricx_dir and metricx_dir not in sys.path:
        sys.path.append(metricx_dir)                     # git clone https://github.com/google-research/metricx.git
    from metricx24 import models
    tok = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.MT5ForRegression.from_pretrained(model_name).to(device).eval()
    return tok, model, device


def cometkiwi(src_list, hyp_list, model_name="Unbabel/wmt22-cometkiwi-da", batch_size=16):
    """Primary metric. Reads source and translation, returns 0..1, higher = better. Encoder cap 512 tokens per side."""
    import torch
    model = _comet_model(model_name)                          # loaded once; run_metric calls this once per batch
    data = [{"src": s, "mt": h} for s, h in zip(src_list, hyp_list)]
    out = model.predict(data, batch_size=batch_size, gpus=1 if torch.cuda.is_available() else 0)
    return [float(x) for x in out.scores]


def metricx_qe(src_list, hyp_list, model="google/metricx-24-hybrid-large-v2p6", tokenizer="google/mt5-large",
               metricx_dir="/content/metricx", max_input_length=1536, batch_size=8):
    """Secondary metric. MetricX-24 hybrid in reference-free mode. Returns an error score 0..25, LOWER = better.
    Input text, tokenisation and EOS removal mirror metricx24/predict.py (QE branch, reference = "")."""
    import torch
    tok, mdl, device = _metricx_model(model, tokenizer, metricx_dir)
    texts = [f"source: {s} candidate: {h}" for s, h in zip(src_list, hyp_list)]
    out = []
    for i in range(0, len(texts), batch_size):
        encs = [tok(t, max_length=max_input_length, truncation=True) for t in texts[i:i + batch_size]]
        feats = [{"input_ids": e["input_ids"][:-1], "attention_mask": e["attention_mask"][:-1]} for e in encs]
        batch = tok.pad(feats, return_tensors="pt")
        with torch.no_grad():
            pred = mdl(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).predictions
        out += [float(x) for x in pred.detach().cpu()]
    return out


def labse_cosine(src_list, hyp_list, batch_size=64):
    """Cheap not-a-translation check: cosine between LaBSE embeddings of the English and the Bangla segment."""
    m = _labse_model()                                        # loaded once
    a = m.encode(src_list, batch_size=batch_size, normalize_embeddings=True)
    b = m.encode(hyp_list, batch_size=batch_size, normalize_embeddings=True)
    return [float(x) for x in (a * b).sum(axis=1)]


def option_swap(long_df, batch_size=64):
    """Step 6 option alignment. For every Bangla option, LaBSE similarity against every English option of the
    same item. If the closest English option is not its own, the option is swapped. No threshold needed.
    Input: long table rows with seg_mismatch == 0 (segment 0 is the stem, 1..n the options).
    Returns one row per (item_id, translator): option_swap_n and option_swaps such as 'B->C'."""
    import numpy as np
    m = _labse_model()
    opts = long_df[(long_df["seg_mismatch"] == 0) & (long_df["seg_id"] >= 1)].copy()
    if opts.empty:
        return pd.DataFrame(columns=["item_id", "translator", "option_swap_n", "option_swaps"])
    src_emb = m.encode(opts["src_en"].tolist(), batch_size=batch_size, normalize_embeddings=True)
    hyp_emb = m.encode(opts["hyp_bn"].tolist(), batch_size=batch_size, normalize_embeddings=True)
    opts["_i"] = range(len(opts))
    rows = []
    for (item, tr), g in opts.groupby(["item_id", "translator"], sort=False):
        g = g.sort_values("seg_id")
        idx = g["_i"].to_numpy()
        sims = hyp_emb[idx] @ src_emb[idx].T                  # rows: Bangla option, columns: English option
        best = sims.argmax(axis=1)
        swaps = [f"{OPTION_LABELS[i]}->{OPTION_LABELS[j]}" for i, j in enumerate(best) if i != j]
        rows.append({"item_id": item, "translator": tr, "option_swap_n": len(swaps), "option_swaps": ";".join(swaps)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 4. checkpointed runner
def run_metric(long_df, metric_name, fn, ckpt_dir, batch=64, hyp_col="hyp_bn", src_col="src_en"):
    """Scores rows not yet in <ckpt_dir>/<metric_name>.jsonl. Appends after every batch."""
    ckpt = pathlib.Path(ckpt_dir) / f"{metric_name}.jsonl"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if ckpt.exists():
        done = {tuple(json.loads(l)["key"]) for l in open(ckpt, encoding="utf-8")}
    todo = long_df[[tuple(k) not in done for k in long_df[["item_id", "translator", "seg_id"]].itertuples(index=False)]]
    if done:
        print(f"  {metric_name}: {len(done)} segments already in the checkpoint, {len(todo)} to go")
    with open(ckpt, "a", encoding="utf-8") as f:
        for n, start in enumerate(range(0, len(todo), batch), 1):
            chunk = todo.iloc[start:start + batch]
            scores = fn(chunk[src_col].tolist(), chunk[hyp_col].tolist())
            for key, s in zip(chunk[["item_id", "translator", "seg_id"]].itertuples(index=False), scores):
                f.write(json.dumps({"key": list(key), "value": s}) + "\n")
            f.flush()
            if n % 10 == 0:
                print(f"  {metric_name}: {min(start + batch, len(todo))} of {len(todo)} segments")
    rows = [json.loads(l) for l in open(ckpt, encoding="utf-8")]
    return pd.DataFrame([{"item_id": r["key"][0], "translator": r["key"][1], "seg_id": r["key"][2], metric_name: r["value"]}
                         for r in rows])


# ---------------------------------------------------------------- 5. workbook helpers (Panel workbooks only)
def build_long(df, src_col, translator_cols, id_col="item_id"):
    """Wide sheet -> long table with one row per (item, translator, segment). seg_mismatch=1 when the
    Bangla cell does not split into the same number of segments as the English cell (scored whole)."""
    rows = []
    for _, r in df.iterrows():
        src_segs = segment(str(r[src_col]))
        for t in translator_cols:
            hyp, n_pre = nfc(r[t])
            hyp_segs = segment(hyp)
            mismatch = int(len(hyp_segs) != len(src_segs))
            pairs = zip(src_segs, hyp_segs) if not mismatch else [(str(r[src_col]), hyp)]
            for i, (s, h) in enumerate(pairs):
                rows.append({"item_id": r[id_col], "translator": t, "seg_id": i, "src_en": s, "hyp_bn": h,
                             "precomposed_n": n_pre, "seg_mismatch": mismatch})
    return pd.DataFrame(rows)


def to_wide(long_scores, metric_cols):
    """Aggregate segments to item level (mean and min) and pivot to one row per item."""
    agg = long_scores.groupby(["item_id", "translator"])[metric_cols].agg(["mean", "min"])
    agg.columns = [f"{m}__{a}" for m, a in agg.columns]
    wide = agg.unstack("translator")
    wide.columns = [f"{t}__{c}" for c, t in wide.columns]
    return wide.reset_index()
