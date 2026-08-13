import json, os, pandas as pd

os.makedirs('results', exist_ok=True)

os.system(
    'python -m training.evaluate '
    '--checkpoints policy_0=BASE policy_1=./ckpts/sft '
    '--samples-per-layout 4 --n-val 20 '
    '--out results/eval_sft_vs_base.json'
)

d = json.load(open('results/eval_sft_vs_base.json'))
rows = [{'ckpt': c['name'],
         'composite': round(c['mean_composite'], 4),
         'green_sci/s': round(c['mean_green_science'], 4),
         'valid %': round(c['valid_output_pct'], 1),
         'parse_ok %': round(c['parse_ok_pct'], 1),
         'materials': round(c['mean_materials'], 1),
         'cells': round(c['mean_cells'], 1),
         'machines': round(c['mean_machines'], 2)} for c in d]
print(pd.DataFrame(rows).set_index('ckpt').to_string())
