import json
import pathlib
import sys
import time

import bangla_gap_gui as G

FAILS = []


def check(name, condition, detail=""):
    if condition:
        print("PASS", name)
    else:
        print("FAIL", name, repr(detail)[:300])
        FAILS.append(name)


MARKER = "BN_OK"
BANGLA = "\u09ac\u09be\u0982\u09b2\u09be"

TEST_ITEMS = [
    {"id": "T-0001", "en": "A 4670-g male newborn is delivered at term.\nA. One\nB. Two"},
    {"id": "T-0002", "en": "A 40-year-old man has watery diarrhea.\nA. One\nB. Two"},
    {"id": "T-0003", "en": "A mother brings her 3-year-old son.\nA. One\nB. Two"},
]
with open("_test_items.json", "w", encoding="utf-8") as handle:
    json.dump(TEST_ITEMS, handle, ensure_ascii=False)


# ---- fake Ollama -------------------------------------------------------
class FakeReply:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self.payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


CALLS = []


def fake_urlopen(request, timeout=None):
    url = request if isinstance(request, str) else request.full_url
    if url.endswith("/api/tags"):
        return FakeReply({"models": [{"name": "qwen2.5:7b-instruct"},
                                     {"name": "gpt-oss:120b"}]})
    body = json.loads(request.data.decode("utf-8"))
    CALLS.append(body)
    return FakeReply({
        "response": "ITEM_ID: X\nQUESTION: " + MARKER + " " + BANGLA + "\nA. one\nB. two",
        "done_reason": "stop",
        "load_duration": 1_500_000_000,
        "eval_count": 40,
        "eval_duration": 2_000_000_000,
    })


G.urllib.request.urlopen = fake_urlopen

# ---- fake dialogs ------------------------------------------------------
PICKS = {}
G.filedialog.askopenfilename = lambda **k: PICKS.get("open", "")
G.filedialog.asksaveasfilename = lambda **k: PICKS.get("save", "")
WARNINGS = []
G.messagebox.showwarning = lambda t, m: WARNINGS.append((t, m))
G.messagebox.showerror = lambda t, m: WARNINGS.append(("ERROR " + t, m))
G.messagebox.askyesno = lambda t, m: True

app = G.App()


def pump(seconds=2.0):
    end = time.time() + seconds
    while time.time() < end:
        app.update()
        time.sleep(0.02)


def logtext():
    return app.log_text.get("1.0", "end")


# ---- 1. startup --------------------------------------------------------
check("status bar exists", app.status.get() != "")
check("status bar shows start line", "App started" in app.status.get())
check("five tabs", len(app.tabs.tabs()) == 5, str(len(app.tabs.tabs())))

# ---- 2. load items -----------------------------------------------------
PICKS["open"] = "_test_items.json"
app.load_items()
pump(0.2)
check("3 items listed", app.items_list.size() == 3, app.items_list.size())
check("first auto selected", app.items_list.curselection() == (0,))
check("source pane filled", "4670-g" in app.src_text.get("1.0", "end"))
check("header shows id", "T-0001" in app.now_showing.get(), app.now_showing.get())

# ---- 3. models ---------------------------------------------------------
app.refresh_models()
pump(1.0)
check("model list filled", list(app.model_box["values"]) != [])
check("model preselected", app.model_box.get() == "gpt-oss:120b" or app.model_box.get() != "")

# ---- 4. single run -----------------------------------------------------
app.model_box.set("qwen2.5:7b-instruct")
app.ctx_box.set("8192")
app.run_translation()
pump(1.5)
pane = app.out_text.get("1.0", "end")
check("response returned", MARKER in pane, pane)
check("unicode preserved in pane", BANGLA in pane, pane)
check("num_ctx sent", CALLS[-1]["options"].get("num_ctx") == 8192, str(CALLS[-1]["options"]))
check("temperature 0 sent", CALLS[-1]["options"].get("temperature") == 0)
check("keep_alive sent", CALLS[-1].get("keep_alive") == "30m")
check("item id in prompt", "ITEM_ID: T-0001" in CALLS[-1]["prompt"], CALLS[-1]["prompt"])
check("prompt file used", "Prompt read from prompt_pass1.txt" in logtext())
check("timing logged", "tokens at" in logtext())
check("buttons re-enabled", not app.run_button.instate(["disabled"]))
check("last_result stored", app.last_result is not None)
check("last_result has english", bool(app.last_result.get("english")))

# ---- 5. model default context -----------------------------------------
app.ctx_box.set(G.MODEL_DEFAULT)
app.run_translation()
pump(1.5)
check("no num_ctx on model default", "num_ctx" not in CALLS[-1]["options"], str(CALLS[-1]["options"]))

# ---- 6. copy -----------------------------------------------------------
app.copy_bangla()
clip = app.clipboard_get()
check("clipboard has the answer", MARKER in clip, clip)
check("unicode preserved in clipboard", BANGLA in clip, clip)

# ---- 7. save one -------------------------------------------------------
PICKS["save"] = "one.json"
app.save_one()
one = json.loads(open("one.json", encoding="utf-8").read())
check("unicode survives the json file", BANGLA in one[0]["bangla"], one[0]["bangla"])
check("save_one is a list", isinstance(one, list) and len(one) == 1)
check("save_one has fields",
      all(k in one[0] for k in ("item_id", "model", "num_ctx", "bangla", "english")),
      str(sorted(one[0])))

# ---- 8. batch ----------------------------------------------------------
import glob
before = set(glob.glob("translations_*.json"))
app.items_list.selection_set(0, 2)
app.ctx_box.set("8192")
app.run_batch()
pump(3.0)
check("batch finished line", "Batch finished" in logtext())
check("batch header moved", "of 3" in app.now_showing.get(), app.now_showing.get())
check("english pane follows batch", "3-year-old" in app.src_text.get("1.0", "end"),
      app.src_text.get("1.0", "end"))

made = sorted(set(glob.glob("translations_*.json")) - before)
check("batch file written", len(made) == 1, made)
batch_files = made or sorted(glob.glob("translations_*.json"))
recs = json.loads(open(batch_files[-1], encoding="utf-8").read())
check("batch has 3 records", len(recs) == 3, str(len(recs)))
check("batch record has english", bool(recs[0].get("english")))
check("batch record has timing", "tokens at" in recs[0].get("timing", ""))
check("batch buttons re-enabled", not app.batch_button.instate(["disabled"]))

# ---- 9. rating ---------------------------------------------------------
WARNINGS.clear()
PICKS["open"] = batch_files[-1]
app.load_for_rating()
check("blocked without rater name", any("rater" in w[0].lower() for w in WARNINGS), str(WARNINGS))

app.rater_entry.insert(0, "hrishin")
app.load_for_rating()
check("queue loaded", len(app.queue) == 3, str(len(app.queue)))
check("queue shuffled with seed", app.seed is not None)
check("bangla shown", MARKER in app.rate_bangla.get("1.0", "end"),
      app.rate_bangla.get("1.0", "end"))
check("english hidden at start", app.rate_english.get("1.0", "end").strip() == "")
check("tier disabled at start", app.tier_buttons[0].instate(["disabled"]))
check("progress shows 1 of 3", "1 of 3" in app.progress.get(), app.progress.get())

WARNINGS.clear()
app.reveal_english()
check("reveal blocked without fluency", app.rate_english.get("1.0", "end").strip() == "")
check("warned about fluency", any("Fluency" in w[0] for w in WARNINGS), str(WARNINGS))

app.fluency_var.set("4")
app.reveal_english()
revealed = app.rate_english.get("1.0", "end")
check("english revealed", any(w in revealed for w in ("newborn", "diarrhea", "3-year-old")),
      revealed)
check("fluency locked after reveal", app.fluency_buttons[0].instate(["disabled"]))
check("tier enabled after reveal", not app.tier_buttons[0].instate(["disabled"]))

WARNINGS.clear()
app.save_rating()
check("blocked without tier", any("tier" in w[1].lower() for w in WARNINGS), str(WARNINGS))

app.tier_var.set("D2")
WARNINGS.clear()
app.save_rating()
check("blocked without acceptable", any("acceptable" in w[1].lower() for w in WARNINGS), str(WARNINGS))

app.acceptable_var.set("no")
app.tag_vars["number"].set(True)
app.save_rating()
pump(0.2)
check("advanced to item 2", "2 of 3" in app.progress.get(), app.progress.get())

import csv as csvmod
rows = list(csvmod.reader(open(app.ratings_path, encoding="utf-8")))
check("csv header + 1 row", len(rows) == 2, str(len(rows)))
check("csv tier saved", rows[1][3] == "D2", str(rows[1]))
check("csv tags saved", rows[1][6] == "number", str(rows[1]))
check("csv hides nothing needed", "shuffle_seed" in rows[0])

check("form reset for item 2", app.fluency_var.get() == "" and app.tier_var.get() == "")
check("tags reset for item 2", not any(v.get() for v in app.tag_vars.values()))
check("english hidden again", app.rate_english.get("1.0", "end").strip() == "")

# finish the queue
for _ in range(2):
    app.fluency_var.set("3")
    app.reveal_english()
    app.tier_var.set("D0")
    app.acceptable_var.set("yes")
    app.save_rating()
pump(0.2)
check("queue finished cleanly", "Finished" in app.progress.get(), app.progress.get())

WARNINGS.clear()
app.save_rating()
check("save after finish is safe", any("Nothing to rate" in w[0] for w in WARNINGS), str(WARNINGS))
app.next_item()
check("next after finish is safe", True)

# ---- 10. error path ----------------------------------------------------
def broken_urlopen(request, timeout=None):
    raise OSError("Connection refused")

G.urllib.request.urlopen = broken_urlopen
app.tabs.select(1)
app.run_translation()
pump(1.0)
check("failure logged", "Failed." in logtext())
check("buttons re-enabled after failure", not app.run_button.instate(["disabled"]))

# ---- 11. clean up our own files ---------------------------------------
app.destroy()
for junk in ["_test_items.json", "one.json"] + list(made) + (
    [str(app.ratings_path)] if app.ratings_path else []
):
    try:
        pathlib.Path(junk).unlink()
    except OSError:
        pass

print()
print("FAILURES:", FAILS if FAILS else "none")
sys.exit(1 if FAILS else 0)
