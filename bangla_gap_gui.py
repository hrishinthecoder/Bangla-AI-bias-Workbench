
import csv
import datetime
import json
import os
import queue
import random
import re
import sys
import threading
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont


def scrolled_text(parent, **kwargs):
    """A Text widget with a working vertical scrollbar."""
    frame = ttk.Frame(parent)
    text = tk.Text(frame, **kwargs)
    bar = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    text.configure(yscrollcommand=bar.set)
    text.pack(side="left", fill="both", expand=True)
    bar.pack(side="right", fill="y")
    return frame, text


OPTION_LINE = re.compile(r"^\s*([A-L])\s*[.:]\s+", re.MULTILINE)


def count_options(text):
    return len(set(OPTION_LINE.findall(text or "")))


def bengali_chars(text):
    return sum(1 for c in text if "\u0980" <= c <= "\u09ff")


def foreign_letters(text):
    """Letters that are neither Bengali nor plain Latin. Tamil, Arabic, Cyrillic."""
    out = set()
    for c in text:
        if not c.isalpha():
            continue
        if c.isascii() or "\u0980" <= c <= "\u09ff":
            continue
        out.add(c)
    return out


def option_lines(text):
    lines = []
    for line in (text or "").splitlines():
        if OPTION_LINE.match(line):
            lines.append(line)
    return lines


def answer_key_problem(answer, bangla):
    """The answer key is a letter (A, B, C ...). It is never translated and never
    sent to the model. This only checks that the Bangla output still has an
    option line for that letter, so the key still points at something."""
    letter = str(answer or "").strip().upper()
    if not letter or not (bangla or "").strip():
        return None
    for line in option_lines(bangla):
        if line.strip()[0].upper() == letter:
            return None
    return "answer_option_missing_" + letter


def find_problems(english, bangla, done_reason, eval_count=None):
    """Cheap checks run at translation time so a broken cell is never silent."""
    problems = []
    body = (bangla or "").strip()
    if not body:
        problems.append("empty_output")
        return problems

    if done_reason and done_reason != "stop":
        problems.append("stopped_" + str(done_reason))
    if eval_count == 0:
        problems.append("zero_token_count")

    if bengali_chars(body) < 20:
        problems.append("untranslated")
    else:
        english_options = [
            line for line in option_lines(body) if bengali_chars(line) == 0
        ]
        if english_options:
            problems.append("untranslated_options_" + str(len(english_options)))

    strays = foreign_letters(body) - foreign_letters(english or "")   # a letter the English source also uses (Greek mu, beta) is not leakage
    if strays:
        problems.append("foreign_script_" + "".join(sorted(strays))[:8])

    if "QUESTION:" not in body:
        problems.append("no_question_line")

    en_count = count_options(english)
    bn_count = count_options(body)
    if en_count and en_count != bn_count:
        problems.append("option_count_" + str(bn_count) + "_vs_" + str(en_count))
    return problems


def pick_bangla_font(root):
    """Bengali needs a font that has the glyphs. Fall back quietly if none is here."""
    families = set(tkfont.families(root))
    for name in ("Nirmala UI", "Shonar Bangla", "Vrinda", "Noto Sans Bengali", "Kalpurush"):
        if name in families:
            return (name, 12)
    return ("TkDefaultFont", 12)

APP_TITLE = "Bangla Gap workbench (skeleton)"
FLUENCY = ["1", "2", "3", "4", "5"]
TIERS = [
    ("D0", "D0  Wording differs, clinical content identical"),
    ("D1", "D1  Register or word order changed, no clinical fact changed"),
    ("D2", "D2  A clinical fact changed, was added, dropped or reversed"),
    ("D3", "D3  The Bangla supports a different option than the key"),
]
MEANING_TAGS = [
    "number",
    "negation",
    "drug_or_disease",
    "clause_dropped",
    "option_changed",
    "other_fact",
]
FORM_TAGS = ["untranslated", "gloss", "script", "option_missing"]
RATING_COLUMNS = [
    "rated_at", "rater", "item_id", "tier", "fluency", "acceptable", "tags",
    "note", "model", "num_ctx", "shuffle_seed", "source_file",
]

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
NUM_CTX = 8192
MODEL_DEFAULT = "Model default"
# A batch stops after this many failures in a row. A dropped tunnel or a full
# Mac would otherwise fail every remaining item in seconds and fill the file
# with junk. Press Run batch again to resume once the cause is fixed.
MAX_CONSECUTIVE_FAILURES = 3
PROMPT_FILE = "prompt_pass1.txt"
# Pass 2 and later: python bangla_gap_gui.py --prompt prompt_pass2.txt
# The prompt file's stem goes into the output file name and into every record, so a run's provenance is on disk.
if "--prompt" in sys.argv:
    PROMPT_FILE = sys.argv[sys.argv.index("--prompt") + 1]
# The Translate tab has a Prompt selector listing every prompt_*.txt next to the script.
# PROMPT_FILE is only its starting value. The chosen file's stem goes into the output file name,
# the resume lookup and every record, so a run's provenance is on disk.
FALLBACK_PROMPT = (
    "Translate the following English medical exam item into Bangla. "
    "Return only the Bangla translation and nothing else.\n\n{text}"
)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("980x640")
        self.minsize(820, 540)

        self.items = []
        self.loaded_path = None
        self.last_result = None
        self.pending = None
        self.queue = []
        self.queue_pos = 0
        self.seed = None
        self.ratings_path = None
        self.ratings = {}

        self.bn_font = pick_bangla_font(self)
        self.jobs = queue.Queue()

        self._build_status_bar()
        self._build_tabs()
        self.after(100, self._poll_jobs)
        self.log("App started. Nothing is connected yet.")

    def _poll_jobs(self):
        """Worker threads must not touch Tk. They post here, the main thread runs it."""
        while True:
            try:
                callback, payload = self.jobs.get_nowait()
            except queue.Empty:
                break
            callback(*payload)
        self.after(100, self._poll_jobs)

    # ---------- layout ----------

    def _build_tabs(self):
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        self.tab_items = ttk.Frame(self.tabs)
        self.tab_translate = ttk.Frame(self.tabs)
        self.tab_quality = ttk.Frame(self.tabs)
        self.tab_rate = ttk.Frame(self.tabs)
        self.tab_log = ttk.Frame(self.tabs)

        self.tabs.add(self.tab_items, text="Items")
        self.tabs.add(self.tab_translate, text="Translate")
        self.tabs.add(self.tab_quality, text="Quality")
        self.tabs.add(self.tab_rate, text="Rate")
        self.tabs.add(self.tab_log, text="Log")

        self._build_items_tab()
        self._build_translate_tab()
        self._build_quality_tab()
        self._build_rate_tab()
        self._build_log_tab()

    def _build_status_bar(self):
        self.status = tk.StringVar(value="Ready")
        bar = ttk.Frame(self)
        bar.pack(fill="x", side="bottom")
        ttk.Separator(bar, orient="horizontal").pack(fill="x")
        ttk.Label(bar, textvariable=self.status, anchor="w").pack(
            fill="x", padx=8, pady=4
        )

    # ---------- tabs ----------

    def _build_items_tab(self):
        top = ttk.Frame(self.tab_items)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Button(top, text="Load items JSON", command=self.load_items).pack(side="left")
        self.items_label = tk.StringVar(value="No file loaded")
        ttk.Label(top, textvariable=self.items_label).pack(side="left", padx=12)

        list_frame = ttk.Frame(self.tab_items)
        list_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.items_list = tk.Listbox(
            list_frame, exportselection=False, selectmode="extended"
        )
        list_bar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.items_list.yview
        )
        self.items_list.configure(yscrollcommand=list_bar.set)
        self.items_list.pack(side="left", fill="both", expand=True)
        list_bar.pack(side="right", fill="y")
        self.items_list.bind("<<ListboxSelect>>", self.on_item_selected)

        pick = ttk.Frame(self.tab_items)
        pick.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(pick, text="Range from").pack(side="left")
        self.range_from = ttk.Entry(pick, width=10)
        self.range_from.pack(side="left", padx=(4, 6))
        ttk.Label(pick, text="to").pack(side="left")
        self.range_to = ttk.Entry(pick, width=10)
        self.range_to.pack(side="left", padx=(4, 6))
        ttk.Button(pick, text="Select range", command=self.select_range).pack(side="left")

        ttk.Label(pick, text="   Single item").pack(side="left")
        self.single_entry = ttk.Entry(pick, width=10)
        self.single_entry.pack(side="left", padx=(4, 6))
        ttk.Button(pick, text="Select item", command=self.select_single).pack(side="left")
        ttk.Button(pick, text="Select all", command=self.select_all).pack(side="left", padx=(12, 0))
        ttk.Button(pick, text="Clear", command=self.clear_selection).pack(side="left", padx=(4, 0))
        for entry in (self.range_from, self.range_to):
            entry.bind("<Return>", lambda _e: self.select_range())
        self.single_entry.bind("<Return>", lambda _e: self.select_single())

        ttk.Label(
            self.tab_items,
            text="Type an item number (12) or an id (US-00012). Or click one item, "
            "shift click a range, ctrl click several. "
            "Then use Run on selected item or Run batch on the Translate tab.",
        ).pack(anchor="w", padx=8, pady=(0, 8))

    def _build_translate_tab(self):
        top = ttk.Frame(self.tab_translate)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Label(top, text="Model").pack(side="left")
        self.model_box = ttk.Combobox(top, width=34, state="readonly", values=[])
        self.model_box.pack(side="left", padx=(6, 12))
        ttk.Button(top, text="Refresh models", command=self.refresh_models).pack(side="left")

        ttk.Label(top, text="Context").pack(side="left", padx=(12, 0))
        self.ctx_box = ttk.Combobox(
            top, width=16, state="readonly", values=[str(NUM_CTX), MODEL_DEFAULT]
        )
        self.ctx_box.set(str(NUM_CTX))
        self.ctx_box.pack(side="left", padx=6)

        self.run_button = ttk.Button(
            top, text="Run on selected item", command=self.run_translation
        )
        self.run_button.pack(side="left", padx=8)

        self.batch_button = ttk.Button(
            top, text="Run batch on all selected", command=self.run_batch
        )
        self.batch_button.pack(side="left")

        prow = ttk.Frame(self.tab_translate)
        prow.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(prow, text="Prompt").pack(side="left")
        self.prompt_box = ttk.Combobox(prow, width=34, state="readonly", values=[])
        self.prompt_box.pack(side="left", padx=(6, 12))
        self.prompt_box.bind("<<ComboboxSelected>>", self.prompt_changed)
        ttk.Button(prow, text="Refresh prompts", command=self.refresh_prompts).pack(side="left")
        ttk.Button(prow, text="Browse...", command=self.browse_prompt).pack(side="left", padx=8)
        ttk.Button(prow, text="Show prompt", command=self.show_prompt).pack(side="left")
        self.refresh_prompts(quiet=True)

        actions = ttk.Frame(self.tab_translate)
        actions.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Button(actions, text="Copy Bangla", command=self.copy_bangla).pack(side="left")
        ttk.Button(actions, text="Save this one to JSON", command=self.save_one).pack(
            side="left", padx=8
        )

        self.now_showing = tk.StringVar(value="No item selected")
        ttk.Label(self.tab_translate, textvariable=self.now_showing).pack(
            anchor="w", padx=8, pady=(0, 4)
        )

        panes = ttk.Panedwindow(self.tab_translate, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        left = ttk.Labelframe(panes, text="English source")
        right = ttk.Labelframe(panes, text="Bangla output")
        panes.add(left, weight=1)
        panes.add(right, weight=1)

        src_frame, self.src_text = scrolled_text(left, wrap="word")
        src_frame.pack(fill="both", expand=True, padx=4, pady=4)
        out_frame, self.out_text = scrolled_text(right, wrap="word", font=self.bn_font)
        out_frame.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_quality_tab(self):
        top = ttk.Frame(self.tab_quality)
        top.pack(fill="x", padx=8, pady=8)
        ttk.Button(top, text="Run steps 1 to 6", command=self.run_quality).pack(side="left")
        ttk.Button(top, text="Apply decision", command=self.apply_decision).pack(
            side="left", padx=8
        )

        ttk.Label(
            self.tab_quality,
            text="Not built yet. This tab will run steps 1 to 6 of the quality pipeline "
            "and show one row per translated cell.\nIt is blocked until the 200 item hand "
            "sample is rated and the Step 7 threshold exists.",
            justify="left",
        ).pack(anchor="w", padx=8, pady=(0, 8))

        cols = ("item", "translator", "chrf_rt", "chrf_cons", "metricx", "decision")
        self.quality_table = ttk.Treeview(
            self.tab_quality, columns=cols, show="headings", height=16
        )
        for col, width in zip(cols, (110, 150, 90, 100, 90, 100)):
            self.quality_table.heading(col, text=col)
            self.quality_table.column(col, width=width, anchor="w")
        self.quality_table.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def _build_rate_tab(self):
        top = ttk.Frame(self.tab_rate)
        top.pack(fill="x", padx=8, pady=8)
        ttk.Button(
            top, text="Load translations JSON", command=self.load_for_rating
        ).pack(side="left")
        ttk.Label(top, text="Rater").pack(side="left", padx=(12, 0))
        self.rater_entry = ttk.Entry(top, width=18)
        self.rater_entry.pack(side="left", padx=6)
        self.progress = tk.StringVar(value="No file loaded")
        ttk.Label(top, textvariable=self.progress).pack(side="left", padx=12)

        # Buttons sit at the very bottom so they can never be pushed off screen.
        buttons = ttk.Frame(self.tab_rate)
        buttons.pack(fill="x", side="bottom", padx=8, pady=8)
        self.back_button = ttk.Button(
            buttons, text="Back to previous", command=self.previous_item
        )
        self.back_button.pack(side="left")
        self.save_button = ttk.Button(
            buttons, text="Save and next", command=self.save_rating
        )
        self.save_button.pack(side="left", padx=8)
        ttk.Button(buttons, text="Skip this one", command=self.next_item).pack(
            side="left"
        )
        self.saved_to = tk.StringVar(value="Ratings file: none yet")
        ttk.Label(buttons, textvariable=self.saved_to).pack(side="right")

        body = ttk.Panedwindow(self.tab_rate, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8)
        left = ttk.Panedwindow(body, orient="vertical")
        right = ttk.Frame(body)
        body.add(left, weight=3)
        body.add(right, weight=2)

        bangla_box = ttk.Labelframe(left, text="Bangla, read this first")
        bn_frame, self.rate_bangla = scrolled_text(
            bangla_box, wrap="word", state="disabled", font=self.bn_font, height=10
        )
        bn_frame.pack(fill="both", expand=True, padx=4, pady=4)

        english_box = ttk.Labelframe(left, text="English source")
        self.reveal_button = ttk.Button(
            english_box, text="Show English", command=self.reveal_english
        )
        self.reveal_button.pack(anchor="w", padx=4, pady=4)
        en_frame, self.rate_english = scrolled_text(
            english_box, wrap="word", state="disabled", height=10
        )
        en_frame.pack(fill="both", expand=True, padx=4, pady=4)

        left.add(bangla_box, weight=1)
        left.add(english_box, weight=1)

        form = ttk.Frame(right)
        form.pack(fill="both", expand=True)

        fluency_box = ttk.Labelframe(form, text="Fluency, before you see the English")
        fluency_box.pack(fill="x", pady=(0, 6))
        self.fluency_var = tk.StringVar(value="")
        self.fluency_buttons = []
        for value in FLUENCY:
            button = ttk.Radiobutton(
                fluency_box, text=value, value=value, variable=self.fluency_var
            )
            button.pack(side="left", padx=6, pady=3)
            self.fluency_buttons.append(button)

        tier_box = ttk.Labelframe(form, text="Drift tier, after you see the English")
        tier_box.pack(fill="x", pady=(0, 6))
        self.tier_var = tk.StringVar(value="")
        self.tier_buttons = []
        for value, description in TIERS:
            button = ttk.Radiobutton(
                tier_box, text=description, value=value, variable=self.tier_var
            )
            button.pack(anchor="w", padx=6, pady=1)
            self.tier_buttons.append(button)

        ok_box = ttk.Labelframe(form, text="Acceptable as is")
        ok_box.pack(fill="x", pady=(0, 6))
        self.acceptable_var = tk.StringVar(value="")
        for value in ["yes", "no"]:
            ttk.Radiobutton(
                ok_box, text=value, value=value, variable=self.acceptable_var
            ).pack(side="left", padx=6, pady=3)

        # Tags go in two columns so the form fits without scrolling.
        self.tag_vars = {}
        meaning_box = ttk.Labelframe(form, text="The meaning changed")
        meaning_box.pack(fill="x", pady=(0, 6))
        for index, tag in enumerate(MEANING_TAGS):
            self.tag_vars[tag] = tk.BooleanVar(value=False)
            ttk.Checkbutton(meaning_box, text=tag, variable=self.tag_vars[tag]).grid(
                row=index // 2, column=index % 2, sticky="w", padx=6, pady=1
            )

        form_box = ttk.Labelframe(form, text="Translator problems")
        form_box.pack(fill="x", pady=(0, 6))
        for index, tag in enumerate(FORM_TAGS):
            self.tag_vars[tag] = tk.BooleanVar(value=False)
            ttk.Checkbutton(form_box, text=tag, variable=self.tag_vars[tag]).grid(
                row=index // 2, column=index % 2, sticky="w", padx=6, pady=1
            )

        ttk.Label(form, text="Note").pack(anchor="w")
        self.note_text = tk.Text(form, height=3, wrap="word")
        self.note_text.pack(fill="x")

    def _build_log_tab(self):
        self.log_text = tk.Text(self.tab_log, wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=8)

    # ---------- actions ----------

    def load_items(self):
        path = filedialog.askopenfilename(
            title="Choose an items JSON file",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            messagebox.showerror("Could not read file", str(err))
            self.log("Failed to load " + path)
            return

        self.items = data if isinstance(data, list) else [data]
        self.loaded_path = path
        self.items_label.set(Path(path).name + ", " + str(len(self.items)) + " items")

        self.items_list.delete(0, "end")
        for index, item in enumerate(self.items):
            label = str(item.get("id", index)) if isinstance(item, dict) else str(index)
            self.items_list.insert("end", label)

        self.log("Loaded " + str(len(self.items)) + " items from " + Path(path).name)

        if self.items:
            self.items_list.selection_set(0)
            self.items_list.see(0)
            self.show_item(0)

    def find_index(self, text):
        """Turn what the user typed into a list index. A plain number is the
        1-based position (US-00012 is item 12). Anything else must match an id."""
        wanted = (text or "").strip()
        if not wanted:
            return None
        if wanted.isdigit():
            position = int(wanted)
            return position - 1 if 1 <= position <= len(self.items) else None
        for index in range(len(self.items)):
            if self.read_item(index)[0].lower() == wanted.lower():
                return index
        return None

    def apply_selection(self, first, last):
        self.items_list.selection_clear(0, "end")
        self.items_list.selection_set(first, last)
        self.items_list.see(first)
        self.show_item(first)
        count = last - first + 1
        self.log(
            "Selected " + str(count) + " items: "
            + self.read_item(first)[0] + " to " + self.read_item(last)[0]
            if count > 1 else "Selected " + self.read_item(first)[0]
        )

    def select_range(self):
        if not self.items:
            messagebox.showwarning("No items", "Load an items JSON file first.")
            return
        first = self.find_index(self.range_from.get())
        last = self.find_index(self.range_to.get())
        if first is None or last is None:
            messagebox.showwarning(
                "Range not found",
                "Type an item number (1 to " + str(len(self.items))
                + ") or an id such as " + self.read_item(0)[0] + " in both boxes.",
            )
            return
        if last < first:
            first, last = last, first
        self.apply_selection(first, last)

    def select_single(self):
        if not self.items:
            messagebox.showwarning("No items", "Load an items JSON file first.")
            return
        index = self.find_index(self.single_entry.get())
        if index is None:
            messagebox.showwarning(
                "Item not found",
                "Type an item number (1 to " + str(len(self.items))
                + ") or an id such as " + self.read_item(0)[0] + ".",
            )
            return
        self.apply_selection(index, index)

    def select_all(self):
        if not self.items:
            messagebox.showwarning("No items", "Load an items JSON file first.")
            return
        self.apply_selection(0, len(self.items) - 1)

    def clear_selection(self):
        self.items_list.selection_clear(0, "end")
        self.log("Selection cleared")

    def on_item_selected(self, _event):
        selection = self.items_list.curselection()
        if not selection:
            return
        if len(selection) > 1:
            self.log(str(len(selection)) + " items selected")
        self.show_item(selection[0])

    def show_item(self, index):
        item = self.items[index]
        if isinstance(item, dict):
            body = item.get("en") or item.get("question") or ""
            item_id = str(item.get("id", index))
        else:
            body = str(item)
            item_id = str(index)

        self.src_text.delete("1.0", "end")
        self.src_text.insert("1.0", body)
        self.out_text.delete("1.0", "end")
        self.now_showing.set("Item " + item_id)
        self.log("Selected " + item_id + ", " + str(len(body)) + " characters in the source pane")

    def prompt_file(self):
        """The prompt file chosen in the Translate tab. Main thread only; workers get it as an argument."""
        return self.prompt_box.get() or PROMPT_FILE

    def prompt_stem(self):
        return Path(self.prompt_file()).stem

    def refresh_prompts(self, quiet=False):
        folder = Path(__file__).parent
        names = sorted(p.name for p in folder.glob("prompt_*.txt"))
        current = self.prompt_box.get() or PROMPT_FILE
        if current not in names and (folder / current).exists():
            names.append(current)
        self.prompt_box["values"] = names
        if current in names:
            self.prompt_box.set(current)
        elif names:
            self.prompt_box.set(names[0])
        if not quiet:
            self.log("Prompt files next to the script: " + (", ".join(names) if names else "none"))

    def prompt_changed(self, event=None):
        self.log("Prompt set to " + self.prompt_file() + ". Output files will carry the stem " + self.prompt_stem())

    def browse_prompt(self):
        path = filedialog.askopenfilename(
            title="Choose a prompt file",
            initialdir=str(Path(__file__).parent),
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        chosen = Path(path)
        if chosen.parent != Path(__file__).parent:
            messagebox.showwarning(
                "Wrong folder",
                "Put the prompt file next to bangla_gap_gui.py first. Output file names and records use its name.",
            )
            return
        values = list(self.prompt_box["values"])
        if chosen.name not in values:
            values.append(chosen.name)
            self.prompt_box["values"] = values
        self.prompt_box.set(chosen.name)
        self.prompt_changed()

    def show_prompt(self):
        template, note = self.load_template()
        self.log(note)
        win = tk.Toplevel(self)
        win.title(self.prompt_file())
        box = tk.Text(win, wrap="word", width=100, height=40)
        box.pack(fill="both", expand=True)
        box.insert("1.0", template)
        box.configure(state="disabled")

    def refresh_models(self):
        self.log("Asking " + OLLAMA_HOST + " for the model list")
        self.in_background(self.fetch_models, self.models_ready)

    def fetch_models(self):
        with urllib.request.urlopen(OLLAMA_HOST + "/api/tags", timeout=15) as reply:
            data = json.loads(reply.read().decode("utf-8"))
        return sorted(entry["name"] for entry in data.get("models", []))

    def models_ready(self, names):
        self.model_box["values"] = names
        if names and not self.model_box.get():
            self.model_box.set(names[0])
        self.log("Found " + str(len(names)) + " models on the server")

    def run_translation(self):
        model = self.model_box.get()
        source = self.src_text.get("1.0", "end").strip()
        if not model:
            messagebox.showwarning("No model", "Press Refresh models and pick one.")
            return
        if not source:
            messagebox.showwarning("No item", "Load a file and select an item first.")
            return

        num_ctx = self.chosen_context()
        template, note = self.load_template()
        self.log(note)

        selection = self.items_list.curselection()
        item_id = self.read_item(selection[0])[0] if selection else "manual"
        prompt = self.build_prompt(template, item_id, source)

        self.pending = {
            "item_id": item_id,
            "model": model,
            "prompt_file": self.prompt_file(),
            "num_ctx": num_ctx if num_ctx else "model default",
            "temperature": 0,
            "started": datetime.datetime.now(),
        }
        if selection:
            self.pending.update(self.item_meta(selection[0]))

        self.out_text.delete("1.0", "end")
        self.out_text.insert("1.0", "Running. This can take minutes on a large model.")
        self.set_busy(True)
        shown_ctx = str(num_ctx) if num_ctx else MODEL_DEFAULT
        self.log("Sent to " + model + ", context " + shown_ctx + ", temperature 0")
        self.in_background(
            lambda: self.call_ollama(model, prompt, num_ctx), self.translation_ready
        )

    def run_batch(self):
        model = self.model_box.get()
        picked = list(self.items_list.curselection())
        if not model:
            messagebox.showwarning("No model", "Press Refresh models and pick one.")
            return
        if not picked:
            messagebox.showwarning(
                "Nothing selected", "Go to the Items tab and select one or more items."
            )
            return

        num_ctx = self.chosen_context()
        template, note = self.load_template()
        self.log(note)

        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = model.replace(":", "_").replace("/", "_")
        prompt_file = self.prompt_file()
        out_path = Path(__file__).with_name(
            "translations_" + safe_model + "_" + self.prompt_stem() + "_" + stamp + ".json"
        )

        # Resume: if the newest file for this model already holds good results
        # for some of the selected items, offer to skip them and keep writing
        # into that same file. Failed or empty items are always run again.
        existing = []
        previous = self.latest_batch_file(safe_model)
        if previous is not None:
            done_ids = self.finished_ids(previous)
            already = [i for i in picked if self.read_item(i)[0] in done_ids]
            if already and messagebox.askyesno(
                "Resume earlier batch?",
                str(len(already)) + " of the " + str(len(picked))
                + " selected items already have a good translation in\n"
                + previous.name + "\n\n"
                + "Yes: skip them and add the rest to that file.\n"
                + "No: start a new file and run every selected item.",
            ):
                picked = [i for i in picked if self.read_item(i)[0] not in done_ids]
                out_path = previous
                existing = self.read_batch_file(previous)
                self.log(
                    "Resuming " + previous.name + ": skipping "
                    + str(len(already)) + " finished items"
                )
                if not picked:
                    self.log("Nothing left to run. Every selected item is already done.")
                    return

        self.set_busy(True)
        shown_ctx = str(num_ctx) if num_ctx else MODEL_DEFAULT
        self.log(
            "Batch started. "
            + str(len(picked))
            + " items, "
            + model
            + ", context "
            + shown_ctx
        )
        threading.Thread(
            target=self.batch_worker,
            args=(picked, model, num_ctx, template, out_path, existing, prompt_file),
            daemon=True,
        ).start()

    def latest_batch_file(self, safe_model):
        folder = Path(__file__).parent
        files = sorted(folder.glob("translations_" + safe_model + "_" + self.prompt_stem() + "_*.json"))
        return files[-1] if files else None

    def read_batch_file(self, path):
        try:
            records = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        return records if isinstance(records, list) else []

    def finished_ids(self, path):
        ids = set()
        for record in self.read_batch_file(path):
            if not isinstance(record, dict):
                continue
            if record.get("error") or not str(record.get("bangla", "")).strip():
                continue
            ids.add(str(record.get("item_id", "")))
        return ids

    def batch_worker(self, indices, model, num_ctx, template, out_path, existing=None, prompt_file=PROMPT_FILE):
        rerun = {self.read_item(i)[0] for i in indices}
        results = [
            r for r in (existing or [])
            if isinstance(r, dict) and str(r.get("item_id", "")) not in rerun
        ]
        kept = len(results)
        total = len(indices)
        failures_in_a_row = 0
        stopped = False
        for position, index in enumerate(indices, start=1):
            item_id, source = self.read_item(index)
            meta = self.item_meta(index)
            started = datetime.datetime.now()
            try:
                answer, timing, done_reason, eval_count = self.call_ollama(
                    model, self.build_prompt(template, item_id, source), num_ctx
                )
                error = ""
            except Exception as err:
                answer = ""
                timing = ""
                done_reason = ""
                eval_count = None
                error = str(err)
            problems = (
                find_problems(source, answer, done_reason, eval_count)
                if not error
                else []
            )
            key_problem = answer_key_problem(meta.get("answer"), answer) if not error else None
            if key_problem:
                problems.append(key_problem)
            seconds = round((datetime.datetime.now() - started).total_seconds(), 1)

            results.append(
                {
                    "item_id": item_id,
                    **meta,
                    "model": model,
                    "prompt_file": prompt_file,
                    "num_ctx": num_ctx if num_ctx else "model default",
                    "temperature": 0,
                    "seconds": seconds,
                    "timing": timing,
                    "done_reason": done_reason,
                    "problems": problems,
                    "english": source,
                    "bangla": answer,
                    "error": error,
                }
            )
            try:
                out_path.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except OSError as err:
                error = error or str(err)

            self.jobs.put(
                (
                    self.batch_progress,
                    (position, total, item_id, seconds, error, answer, timing, source,
                     problems),
                )
            )

            if error or "empty_output" in problems:
                failures_in_a_row += 1
            else:
                failures_in_a_row = 0
            if failures_in_a_row >= MAX_CONSECUTIVE_FAILURES and position < total:
                stopped = True
                break

        ran = results[kept:]
        bad = sum(1 for r in ran if r["problems"] or r["error"])
        self.jobs.put((self.batch_done, (out_path, total, bad, len(ran), len(results),
                                          stopped)))

    def batch_progress(self, position, total, item_id, seconds, error, answer, timing,
                       english, problems=()):
        self.now_showing.set(
            "Item " + str(position) + " of " + str(total) + ", id " + item_id
        )
        self.src_text.delete("1.0", "end")
        self.src_text.insert("1.0", english)
        if error:
            self.out_text.delete("1.0", "end")
            self.log(
                "Item " + str(position) + " of " + str(total) + ", " + item_id + " failed. " + error
            )
            return
        self.out_text.delete("1.0", "end")
        self.out_text.insert("1.0", answer)
        self.log(
            "Item "
            + str(position)
            + " of "
            + str(total)
            + ", "
            + item_id
            + " done in "
            + str(seconds)
            + " seconds. "
            + timing
        )
        if problems:
            self.log("  PROBLEMS on " + item_id + ": " + ", ".join(problems))

    def batch_done(self, out_path, total, bad=0, ran=None, in_file=None, stopped=False):
        self.set_busy(False)
        ran = total if ran is None else ran
        in_file = ran if in_file is None else in_file
        if stopped:
            self.log(
                "Batch STOPPED after " + str(MAX_CONSECUTIVE_FAILURES)
                + " failures in a row, " + str(ran) + " of " + str(total)
                + " attempted. Check the tunnel and the Mac Ollama log, then press "
                + "Run batch again with the same selection to resume."
            )
            return
        self.log(
            "Batch finished. "
            + str(ran)
            + " items written to "
            + out_path.name
            + " (file now holds "
            + str(in_file)
            + "). "
            + str(bad)
            + " of "
            + str(ran)
            + " have problems."
        )

    def load_template(self):
        name = self.prompt_file()
        path = Path(__file__).with_name(name)
        if path.exists():
            return path.read_text(encoding="utf-8"), "Prompt read from " + name
        return (
            FALLBACK_PROMPT,
            "No " + name + " next to the script. Using the placeholder prompt.",
        )

    def build_prompt(self, template, item_id, source):
        block = "ITEM_ID: " + item_id + "\n" + source
        if "{text}" in template:
            return template.replace("{text}", block)
        return template.rstrip() + "\n\n" + block

    def read_item(self, index):
        item = self.items[index]
        if isinstance(item, dict):
            body = item.get("en") or item.get("question") or ""
            item_id = str(item.get("id", index))
        else:
            body = str(item)
            item_id = str(index)
        return item_id, body

    def item_meta(self, index):
        """Fields carried from the item file into every output record, so the
        scoring stage has the key without going back to items500.json."""
        item = self.items[index]
        if not isinstance(item, dict):
            return {}
        meta = {}
        for key in ("answer", "answer_text", "n_options", "meta_info"):
            if key in item:
                meta[key] = item[key]
        return meta

    def chosen_context(self):
        choice = self.ctx_box.get()
        return None if choice == MODEL_DEFAULT else int(choice)

    def set_busy(self, busy):
        flag = ["disabled"] if busy else ["!disabled"]
        self.run_button.state(flag)
        self.batch_button.state(flag)

    def call_ollama(self, model, prompt, num_ctx):
        options = {"temperature": 0}
        if num_ctx:
            options["num_ctx"] = num_ctx
        body = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "30m",
                "options": options,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            OLLAMA_HOST + "/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=1800) as reply:
            data = json.loads(reply.read().decode("utf-8"))
        return (
            data.get("response", ""),
            self.timing_note(data),
            data.get("done_reason", ""),
            data.get("eval_count"),
        )

    def timing_note(self, data):
        load_seconds = data.get("load_duration", 0) / 1e9
        eval_count = data.get("eval_count", 0)
        eval_seconds = data.get("eval_duration", 0) / 1e9
        speed = round(eval_count / eval_seconds, 1) if eval_seconds else 0
        return (
            "load "
            + str(round(load_seconds, 1))
            + "s, "
            + str(eval_count)
            + " tokens at "
            + str(speed)
            + " per second"
        )

    def translation_ready(self, result):
        text, timing, done_reason, eval_count = result
        self.set_busy(False)
        self.out_text.delete("1.0", "end")
        self.out_text.insert("1.0", text)

        record = dict(self.pending) if self.pending else {"item_id": "manual"}
        started = record.pop("started", None)
        record["seconds"] = (
            round((datetime.datetime.now() - started).total_seconds(), 1) if started else None
        )
        english = self.src_text.get("1.0", "end").strip()
        problems = find_problems(english, text, done_reason, eval_count)
        key_problem = answer_key_problem(record.get("answer"), text)
        if key_problem:
            problems.append(key_problem)
        record["done_reason"] = done_reason
        record["problems"] = problems
        record["bangla"] = text
        record["english"] = english
        record["error"] = ""
        self.last_result = record

        self.log("Answer received, " + str(len(text)) + " characters. " + timing)
        if problems:
            self.log("PROBLEMS: " + ", ".join(problems))

    def copy_bangla(self):
        text = self.out_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Nothing to copy", "Run a translation first.")
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()
        self.log("Copied " + str(len(text)) + " characters to the clipboard")

    def save_one(self):
        if not self.last_result:
            messagebox.showwarning("Nothing to save", "Run a translation first.")
            return
        safe_id = str(self.last_result.get("item_id", "item")).replace(":", "_")
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            title="Save this translation",
            defaultextension=".json",
            initialfile="translation_" + safe_id + "_" + stamp + ".json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            Path(path).write_text(
                json.dumps([self.last_result], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as err:
            messagebox.showerror("Could not save", str(err))
            return
        self.log("Saved to " + Path(path).name)

    def run_quality(self):
        self.stub("Quality steps 1 to 6. Will call bangla_qe_pipeline.py.")

    def apply_decision(self):
        self.stub("Step 8 decision. Will write PASS, REVIEW or FLAG per item.")

    def load_for_rating(self):
        rater = self.rater_entry.get().strip()
        if not rater:
            messagebox.showwarning("No rater", "Type your name in the Rater box first.")
            return

        path = filedialog.askopenfilename(
            title="Choose a translations JSON file",
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            records = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            messagebox.showerror("Could not read file", str(err))
            return
        if not isinstance(records, list) or not records:
            messagebox.showerror("Empty file", "That file holds no translation records.")
            return

        missing = [r.get("item_id", "?") for r in records if not r.get("english")]
        if missing:
            messagebox.showwarning(
                "No English source",
                "These records have no english field, so the drift tier cannot be judged:\n"
                + ", ".join(str(m) for m in missing[:10]),
            )

        self.seed = random.randrange(1, 1000000)
        self.queue = list(records)
        random.Random(self.seed).shuffle(self.queue)
        self.queue_pos = 0
        self.ratings = {}

        safe_rater = "".join(c for c in rater if c.isalnum() or c in "._-") or "rater"
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.ratings_path = Path(path).with_name(
            "ratings_" + safe_rater + "_" + stamp + ".csv"
        )
        with self.ratings_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(RATING_COLUMNS)

        self.log(
            "Rating queue loaded. "
            + str(len(self.queue))
            + " records, shuffle seed "
            + str(self.seed)
            + ", writing to "
            + self.ratings_path.name
        )
        self.show_current()

    def show_current(self):
        if self.queue_pos >= len(self.queue):
            self.progress.set("Finished. " + str(len(self.queue)) + " records done.")
            self.set_text(self.rate_bangla, "")
            self.set_text(self.rate_english, "")
            self.log("Rating finished. File is " + self.ratings_path.name)
            return

        record = self.queue[self.queue_pos]
        self.progress.set(
            "Item " + str(self.queue_pos + 1) + " of " + str(len(self.queue))
        )
        self.set_text(self.rate_bangla, record.get("bangla", ""))
        self.set_text(self.rate_english, "")

        self.back_button.state(["!disabled"] if self.queue_pos else ["disabled"])

        saved = self.ratings.get(self.queue_pos)
        if saved:
            self.fluency_var.set(saved["fluency"])
            self.tier_var.set(saved["tier"])
            self.acceptable_var.set(saved["acceptable"])
            for tag, var in self.tag_vars.items():
                var.set(tag in saved["tags"])
            self.note_text.delete("1.0", "end")
            self.note_text.insert("1.0", saved["note"])
            self.set_text(self.rate_english, record.get("english", ""))
            for button in self.fluency_buttons:
                button.state(["disabled"])
            for button in self.tier_buttons:
                button.state(["!disabled"])
            self.reveal_button.state(["disabled"])
            self.progress.set(
                "Item " + str(self.queue_pos + 1) + " of " + str(len(self.queue))
                + ", already rated"
            )
            return

        self.fluency_var.set("")
        self.tier_var.set("")
        self.acceptable_var.set("")
        for var in self.tag_vars.values():
            var.set(False)
        self.note_text.delete("1.0", "end")

        for button in self.fluency_buttons:
            button.state(["!disabled"])
        for button in self.tier_buttons:
            button.state(["disabled"])
        self.reveal_button.state(["!disabled"])

    def reveal_english(self):
        if not self.queue or self.queue_pos >= len(self.queue):
            return
        if not self.fluency_var.get():
            messagebox.showwarning(
                "Fluency first",
                "Score fluency before you see the English. The guide says to judge the "
                "Bangla on its own.",
            )
            return
        self.set_text(self.rate_english, self.queue[self.queue_pos].get("english", ""))
        for button in self.fluency_buttons:
            button.state(["disabled"])
        for button in self.tier_buttons:
            button.state(["!disabled"])
        self.reveal_button.state(["disabled"])

    def save_rating(self):
        if not self.queue or self.queue_pos >= len(self.queue):
            messagebox.showwarning("Nothing to rate", "Load a translations file first.")
            return
        if not self.fluency_var.get():
            messagebox.showwarning("Missing", "Score fluency.")
            return
        if not self.tier_var.get():
            messagebox.showwarning("Missing", "Show the English and pick a drift tier.")
            return
        if not self.acceptable_var.get():
            messagebox.showwarning("Missing", "Pick yes or no for acceptable as is.")
            return

        tier = self.tier_var.get()
        tags = [tag for tag, var in self.tag_vars.items() if var.get()]
        meaning_hit = [tag for tag in tags if tag in MEANING_TAGS]
        if tier in ("D2", "D3") and not meaning_hit:
            keep = messagebox.askyesno(
                "Check this",
                tier + " means the meaning changed, but no meaning tag is ticked.\n\n"
                "Save it anyway?",
            )
            if not keep:
                return

        record = self.queue[self.queue_pos]
        self.ratings[self.queue_pos] = {
            "rated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "rater": self.rater_entry.get().strip(),
            "item_id": record.get("item_id", ""),
            "tier": tier,
            "fluency": self.fluency_var.get(),
            "acceptable": self.acceptable_var.get(),
            "tags": tags,
            "note": self.note_text.get("1.0", "end").strip().replace("\n", " "),
            "model": record.get("model", ""),
            "num_ctx": record.get("num_ctx", ""),
        }
        if not self.write_ratings():
            return

        self.log(
            "Saved "
            + str(record.get("item_id", ""))
            + ", tier "
            + tier
            + ", fluency "
            + self.fluency_var.get()
            + ", tags "
            + (";".join(tags) if tags else "none")
        )
        self.next_item()

    def write_ratings(self):
        """Rewrite the whole file, so going Back and editing replaces a row."""
        try:
            with self.ratings_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(RATING_COLUMNS)
                for position in sorted(self.ratings):
                    r = self.ratings[position]
                    writer.writerow(
                        [
                            r["rated_at"], r["rater"], r["item_id"], r["tier"],
                            r["fluency"], r["acceptable"], ";".join(r["tags"]),
                            r["note"], r["model"], r["num_ctx"], self.seed,
                            self.ratings_path.name,
                        ]
                    )
        except OSError as err:
            messagebox.showerror("Could not save", str(err))
            return False
        return True

    def next_item(self):
        if not self.queue:
            return
        self.queue_pos += 1
        self.show_current()

    def previous_item(self):
        if not self.queue or self.queue_pos == 0:
            return
        self.queue_pos -= 1
        self.show_current()

    def set_text(self, widget, content):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", content)
        widget.configure(state="disabled")

    # ---------- helpers ----------

    def in_background(self, work, done):
        def runner():
            try:
                result = work()
            except Exception as err:
                self.jobs.put((self.fail, (str(err),)))
                return
            self.jobs.put((done, (result,)))

        threading.Thread(target=runner, daemon=True).start()

    def fail(self, message):
        self.set_busy(False)
        self.out_text.delete("1.0", "end")
        self.log("Failed. " + message)
        messagebox.showerror(
            "Could not reach Ollama",
            message + "\n\nIs the SSH tunnel open in another window?",
        )

    def stub(self, message):
        self.log("Not built yet. " + message)
        self.status.set("Not built yet")

    def log(self, message):
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        line = "[" + stamp + "] " + message + "\n"
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.status.set(message)


if __name__ == "__main__":
    App().mainloop()
