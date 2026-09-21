# Bangla AI bias Workbench

Author: Hrishin Debnath. Part of The Bangla Gap study, CreateBud Summer Research 2026.

## What this is

The study asks whether language models answer medical exam questions worse in Bangla than in English. That needs the same items in both languages. This repository holds the tools that translate MedQA items into Bangla and check the translations without a human reference translation, plus every file the checks produced on the first 500 items.

Two translation runs exist, on the same 500 items, with the same model and settings. Pass 1 used a prompt that gives only the output structure. Pass 2 used the same structure plus 11 rules, mainly that everything except the option labels is written in Bengali script. The human rating of a sample has not run yet.

## Folder map

- `bangla_gap_gui.py`. The desktop program. It sends one item at a time to a model through Ollama, stores every answer with its settings, and has a Rate tab for the human rating.
- `bangla_qe_pipeline.py`. The checks: encoding fix, segment split, structural checks, round-trip chrF, learned scores, option swap.
- `run_all.py`. The driver. Commands: `score`, `decide`, `summary`, `sample`, `queue`, `stats`.
- `step7_stats.py`. The statistics for the human check: correlation, AUROC, threshold, cross-validation, kappa.
- `prompt_pass1.txt` and `prompt_pass2.txt`. The two prompts. The program reads them from this folder.
- `Bangla_Gap_QE_500.ipynb`. The Colab notebook that runs `score` with the GPU scorers.
- `Dataset/`. `items500.json`, the 500 items. `US_qbank.jsonl`, the MedQA source bank.
- `Pass_01/`. The Pass 1 prompt, the frozen translation file, the back-translation and its notebook, and `Results/` with the scores, the summary table and the rating sheets.
- `Pass_02/`. The same for Pass 2. Its results folder is `results500_pass2/`.
- `Panel_03/`. Four local models on items 1 to 50 with the Pass 2 prompt, and a second gpt-oss:120b run of the same 50 items. See `note.txt` there.
- `Test/`. Two test scripts for the program and the checks.
- `DNT_term_register_Bangla_Gap_checked.xlsx`. The term register with the decisions taken on 20 and 21 September 2026.

## How the checks run

The structural checks need no GPU:

```
pip install pandas sacrebleu scipy scikit-learn openpyxl
python run_all.py score --json Pass_02/Translation/500_translated_gptoss_pass2.json --out Pass_02/results500_pass2 --bengali-only
python run_all.py decide --out Pass_02/results500_pass2
python run_all.py summary --out Pass_02/results500_pass2
```

The learned scores need a GPU. Open `Bangla_Gap_QE_500.ipynb` in Colab on a T4, run the cells in order, and restart the session once after cell 1. The notebook adds the back-translation file with `--bt` and runs `score` with `--gpu`.

`--bengali-only` is the scoring setting for Pass 2. Every Latin run of two or more letters counts as untranslated English. Pass 1 was scored without it, so a Latin token the English source also keeps in Latin form, such as MRI, was allowed. A comparison of the two passes is fair only under one setting.

## The human rating

```
python run_all.py queue --out Pass_02/results500_pass2 --json Pass_02/Translation/500_translated_gptoss_pass2.json
```

This writes `rating_queue.json`, `rating_queue_rater2.json` and `anchor_queue.json` into the results folder. The Rate tab of the program loads them and writes `ratings_<rater>_<stamp>.csv`. Then:

```
python run_all.py stats --out Pass_02/results500_pass2 --ratings <ratings file> --ratings2 <second rater file>
python run_all.py decide --out Pass_02/results500_pass2 --thresholds Pass_02/results500_pass2/thresholds.json
```

## Tests

```
python Test/test_checks_and_rating.py
python Test/test_bangla_gap_gui.py
```

Both need a display. The second one needs `prompt_pass1.txt` next to `bangla_gap_gui.py`.
