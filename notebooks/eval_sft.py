import json, os, subprocess, sys

import pandas as pd

os.makedirs('results', exist_ok=True)

subprocess.run([
    sys.executable, "-m", "training.evaluate",
    "--checkpoints", "policy_0=BASE", "policy_1=./ckpts/sft",
    "--samples-per-layout", "4",
    "--n-val", "40",
    "--out", "results/eval_sft_vs_base.json",
], check=True)

d = json.load(open('results/eval_sft_vs_base.json'))
rows = [{'ckpt': c['name'],
         'composite': round(c['mean_composite'], 4),
         'green_sci/s': round(c['mean_green_science'], 4),
         'valid %': round(c['valid_output_pct'], 1),
         'parse_ok %': round(c['parse_ok_pct'], 1),
         'materials': round(c['mean_materials'], 1),
         'cells': round(c['mean_cells'], 1),
         'machines': round(c['mean_machines'], 2)} for c in d]
pd.DataFrame(rows).set_index('ckpt')
