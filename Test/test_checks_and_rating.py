import json, time, sys, glob, csv as csvmod
import bangla_gap_gui as G
F=[]
def check(n,c,d=""):
    print(("PASS " if c else "FAIL ")+n+("  "+repr(d)[:200] if not c else "")); 
    if not c: F.append(n)

# --- structural checks, pure function ---
EN="Q text\n\nA. one\nB. two\nC. three"
CLEAN="ITEM_ID: X\nQUESTION: \u098f\u0995\u099f\u09bf \u09aa\u09cd\u09b0\u09b6\u09cd\u09a8 \u09af\u09be \u09ac\u09be\u0982\u09b2\u09be\u09df \u09b2\u09c7\u0996\u09be\nA: \u098f\u0995\nB: \u09a6\u09c1\u0987\nC: \u09a4\u09bf\u09a8"
check("clean passes", G.find_problems(EN,CLEAN,"stop")==[], G.find_problems(EN,CLEAN,"stop"))
check("empty caught", G.find_problems(EN,"","stop")==["empty_output"])
check("whitespace only caught", G.find_problems(EN,"   \n ","stop")==["empty_output"])
check("length stop caught", "stopped_length" in G.find_problems(EN,CLEAN,"length"))
check("untranslated caught", "untranslated" in G.find_problems(EN,EN,"stop"))
check("untranslated survives an ITEM_ID line", "untranslated" in G.find_problems(EN,"ITEM_ID: X\n"+EN,"stop"), G.find_problems(EN,"ITEM_ID: X\n"+EN,"stop"))
check("english options caught", "untranslated_options_3" in G.find_problems(EN,"\n".join(CLEAN.splitlines()[:2])+"\nA. one\nB. two\nC. three","stop"), G.find_problems(EN,"\n".join(CLEAN.splitlines()[:2])+"\nA. one\nB. two\nC. three","stop"))
check("foreign script caught", any(x.startswith("foreign_script") for x in G.find_problems(EN,CLEAN.replace("\u098f\u0995","\u098f\u0995 \u062c\u0631\u0627\u062d",1),"stop")))
check("zero token count caught", "zero_token_count" in G.find_problems(EN,CLEAN,"stop",0))
check("latin abbreviation is not foreign", G.find_problems(EN,CLEAN.replace("A: ","A: MRI ",1),"stop")==[], G.find_problems(EN,CLEAN.replace("A: ","A: MRI ",1),"stop"))
check("missing question line caught", "no_question_line" in G.find_problems(EN,CLEAN.replace("QUESTION:","XX:"),"stop"))
check("option count mismatch caught", "option_count_2_vs_3" in G.find_problems(EN,"\n".join(CLEAN.splitlines()[:-1]),"stop"), G.find_problems(EN,"\n".join(CLEAN.splitlines()[:-1]),"stop"))
check("counts A. and A: alike", G.count_options("A. x\nB: y")==2)
check("no false option from prose", G.count_options("QUESTION: what is 5. something")==0, G.count_options("QUESTION: what is 5. something"))
check("answer key present passes", G.answer_key_problem("B", CLEAN) is None, G.answer_key_problem("B", CLEAN))
check("answer key lowercase passes", G.answer_key_problem("b", CLEAN) is None)
check("answer key missing option caught", G.answer_key_problem("D", CLEAN)=="answer_option_missing_D", G.answer_key_problem("D", CLEAN))
check("no key means no check", G.answer_key_problem("", CLEAN) is None and G.answer_key_problem(None, CLEAN) is None)
check("empty output is not double flagged", G.answer_key_problem("A", "") is None)

# --- live app: back button + edit ---
class R:
    def __init__(s,p): s.p=json.dumps(p).encode()
    def read(s): return s.p
    def __enter__(s): return s
    def __exit__(s,*a): return False
N={"i":0}
def fake(req,timeout=None):
    u=req if isinstance(req,str) else req.full_url
    if u.endswith("/api/tags"): return R({"models":[{"name":"m1"}]})
    N["i"]+=1
    if N["i"]==2:   # one broken item on purpose
        return R({"response":"","done_reason":"length","load_duration":0,"eval_count":0,"eval_duration":1})
    return R({"response":CLEAN,"done_reason":"stop","load_duration":0,"eval_count":9,"eval_duration":1_000_000_000})
G.urllib.request.urlopen=fake
P={}; W=[]
G.filedialog.askopenfilename=lambda **k:P.get("open","")
G.messagebox.showwarning=lambda t,m:W.append((t,m))
G.messagebox.showerror=lambda t,m:W.append(("ERROR "+t,m))
G.messagebox.askyesno=lambda t,m:True
json.dump([{"id":"A-1","en":EN,"answer":"B","answer_text":"two","options":{"A":"one","B":"two","C":"three"},"n_options":3,"meta_info":"step2"},{"id":"A-2","en":EN,"answer":"C","n_options":3,"meta_info":"step2"},{"id":"A-3","en":EN,"answer":"D","n_options":3,"meta_info":"step2"}], open("it3.json","w"))
app=G.App()
def pump(s=2.0):
    e=time.time()+s
    while time.time()<e: app.update(); time.sleep(0.02)
P["open"]="it3.json"; app.load_items(); pump(.2)
app.refresh_models(); pump(1); app.model_box.set("m1"); app.ctx_box.set("8192")
before=set(glob.glob("translations_*.json"))
app.items_list.selection_set(0,2); app.run_batch(); pump(3)
f=sorted(set(glob.glob("translations_*.json"))-before)[0]
recs=json.loads(open(f,encoding="utf-8").read())
check("done_reason recorded", recs[0]["done_reason"]=="stop", recs[0]["done_reason"])
check("problems field present", "problems" in recs[0])
check("clean item has no problems", recs[0]["problems"]==[], recs[0]["problems"])
check("answer key carried into record", recs[0].get("answer")=="B" and recs[0].get("answer_text")=="two" and recs[0].get("n_options")==3 and recs[0].get("meta_info")=="step2", {k:recs[0].get(k) for k in ("answer","answer_text","n_options","meta_info")})
check("options dict not copied into record", "options" not in recs[0])
check("key pointing at a missing option is flagged", "answer_option_missing_D" in recs[2]["problems"], recs[2]["problems"])
check("answer key never sent to the model", "answer" not in json.dumps(recs[0]["english"]) and "D" not in "".join(l for l in recs[2]["english"].splitlines() if l.startswith("answer")), recs[2]["english"])
check("broken item flagged empty", "empty_output" in recs[1]["problems"], recs[1]["problems"])
check("batch summary counts bad", "2 of 3 have problems" in app.log_text.get("1.0","end"), app.log_text.get("1.0","end")[-200:])

# --- resume: same selection again skips the good items, reruns the broken one, same file ---
files_before=set(glob.glob("translations_*.json"))
app.items_list.selection_set(0,2); app.run_batch(); pump(3)
check("resume made no new file", set(glob.glob("translations_*.json"))==files_before, glob.glob("translations_*.json"))
check("resume logged the skip", "skipping 2 finished items" in app.log_text.get("1.0","end"), app.log_text.get("1.0","end")[-300:])
recs=json.loads(open(f,encoding="utf-8").read())
check("resume kept 3 records, no duplicates", sorted(r["item_id"] for r in recs)==["A-1","A-2","A-3"], [r["item_id"] for r in recs])
byid={r["item_id"]:r for r in recs}
check("resume repaired the broken item", all(r["bangla"] for r in recs) and byid["A-2"]["problems"]==[], [(r["item_id"],r["problems"]) for r in recs])
check("resume summary counts the run, not the file", "1 items written" in app.log_text.get("1.0","end") and "file now holds 3" in app.log_text.get("1.0","end"), app.log_text.get("1.0","end")[-300:])
app.items_list.selection_set(0,2); app.run_batch(); pump(.5)
check("nothing left to run is handled", "Nothing left to run" in app.log_text.get("1.0","end"), app.log_text.get("1.0","end")[-200:])
check("buttons free after nothing-to-run", not app.batch_button.instate(["disabled"]))

# rating: back and edit
app.rater_entry.insert(0,"h"); P["open"]=f; app.load_for_rating(); pump(.2)
check("back disabled on first item", app.back_button.instate(["disabled"]))
app.fluency_var.set("2"); app.reveal_english(); app.tier_var.set("D0"); app.acceptable_var.set("yes")
app.save_rating(); pump(.2)
check("moved to item 2", "2 of 3" in app.progress.get(), app.progress.get())
check("back enabled now", not app.back_button.instate(["disabled"]))
app.previous_item(); pump(.2)
check("back returns to item 1", app.progress.get().startswith("Item 1 of 3"), app.progress.get())
check("marked already rated", "already rated" in app.progress.get(), app.progress.get())
check("restores tier", app.tier_var.get()=="D0")
check("restores fluency", app.fluency_var.get()=="2")
check("english already shown on revisit", app.rate_english.get("1.0","end").strip()!="")
check("fluency stays locked on revisit", app.fluency_buttons[0].instate(["disabled"]))
check("tier editable on revisit", not app.tier_buttons[0].instate(["disabled"]))
app.tier_var.set("D2"); app.tag_vars["number"].set(True); app.save_rating(); pump(.2)
rows=list(csvmod.reader(open(app.ratings_path,encoding="utf-8")))
check("edit did not duplicate the row", len(rows)==2, len(rows))
check("edit replaced the tier", rows[1][3]=="D2", rows[1])
check("edit saved the tag", rows[1][6]=="number", rows[1])

# --- range and single selection by number or id ---
P["open"]="it3.json"; app.load_items(); pump(.2)
app.range_from.delete(0,"end"); app.range_from.insert(0,"2"); app.range_to.delete(0,"end"); app.range_to.insert(0,"3")
app.select_range(); pump(.1)
check("range by number selects 2 to 3", app.items_list.curselection()==(1,2), app.items_list.curselection())
check("range shows the first item", "A-2" in app.now_showing.get(), app.now_showing.get())
app.range_from.delete(0,"end"); app.range_from.insert(0,"a-3"); app.range_to.delete(0,"end"); app.range_to.insert(0,"A-1")
app.select_range(); pump(.1)
check("range by id, reversed, case-insensitive", app.items_list.curselection()==(0,1,2), app.items_list.curselection())
app.single_entry.delete(0,"end"); app.single_entry.insert(0,"A-2"); app.select_single(); pump(.1)
check("single by id replaces the selection", app.items_list.curselection()==(1,), app.items_list.curselection())
app.single_entry.delete(0,"end"); app.single_entry.insert(0,"3"); app.select_single(); pump(.1)
check("single by number", app.items_list.curselection()==(2,), app.items_list.curselection())
W.clear(); app.single_entry.delete(0,"end"); app.single_entry.insert(0,"9"); app.select_single(); pump(.1)
check("out of range warns and keeps selection", W and app.items_list.curselection()==(2,), (W, app.items_list.curselection()))
W.clear(); app.range_from.delete(0,"end"); app.range_from.insert(0,"1"); app.range_to.delete(0,"end"); app.select_range()
check("empty range box warns", bool(W), W)
app.select_all(); pump(.1)
check("select all", app.items_list.curselection()==(0,1,2), app.items_list.curselection())
app.clear_selection()
check("clear", app.items_list.curselection()==(), app.items_list.curselection())

# --- stop after repeated failures (dropped tunnel) ---
def dead(req,timeout=None):
    u=req if isinstance(req,str) else req.full_url
    if u.endswith("/api/tags"): return R({"models":[{"name":"m1"}]})
    raise OSError("Connection refused")
G.urllib.request.urlopen=dead
json.dump([{"id":"B-%d"%i,"en":EN} for i in range(1,6)], open("it5.json","w"))
P["open"]="it5.json"; app.load_items(); pump(.2)
files_before=set(glob.glob("translations_*.json"))
app.items_list.selection_set(0,4); app.run_batch(); pump(3)
g=sorted(set(glob.glob("translations_*.json"))-files_before)
check("failure batch wrote one file", len(g)==1, g)
recs=json.loads(open(g[0],encoding="utf-8").read()) if g else []
check("stopped after 3 failures, not 5", len(recs)==G.MAX_CONSECUTIVE_FAILURES, len(recs))
check("stop is logged with resume advice", "Batch STOPPED" in app.log_text.get("1.0","end") and "resume" in app.log_text.get("1.0","end"), app.log_text.get("1.0","end")[-300:])
check("buttons free after stop", not app.batch_button.instate(["disabled"]))

app.destroy()
import os, pathlib as _pl
for j in ["it3.json", "it5.json", f] + g + ([str(app.ratings_path)] if app.ratings_path else []):
    try: _pl.Path(j).unlink()
    except OSError: pass
print(); print("FAILURES:", F if F else "none"); sys.exit(1 if F else 0)
