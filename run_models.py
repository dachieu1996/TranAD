import subprocess
import csv
import re
from datetime import datetime

models = ['TranAD', 'GDN', 'MAD_GAN', 'MTAD_GAT', 'MSCRED', 'USAD', 'OmniAnomaly', 'LSTM_AD']
datasets = ['SMAP', 'MSL', 'SWaT', 'WADI', 'SMD', 'MSDS', 'MBA', 'UCR', 'NAB']

output_csv = "results.csv"

with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Model', 'Dataset', 'ROC/AUC', 'F1'])

    for model in models:
        for dataset in datasets:
            print(f"Running model: {model} on dataset: {dataset}")
            try:
                result = subprocess.run(
                    ['python3', 'main.py', '--model', model, '--dataset', dataset, '--retrain'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=6000
                )

                roc_auc_match = re.search(r"'ROC/AUC':\s*([\d.]+)", result.stdout)
                f1_match = re.search(r"'f1':\s*([\d.]+)", result.stdout)

                if roc_auc_match and f1_match:
                    roc_auc = float(roc_auc_match.group(1))
                    f1 = float(f1_match.group(1))
                else:
                    roc_auc = 'N/A'
                    f1 = 'N/A'

                writer.writerow([model, dataset, roc_auc, f1])

            except subprocess.TimeoutExpired:
                print(f"Timeout: {model} on {dataset}")
                writer.writerow([model, dataset, 'Timeout', 'Timeout'])
            except Exception as e:
                print(f"Error running {model} on {dataset}: {str(e)}")
                writer.writerow([model, dataset, 'Error', 'Error'])

print(f"\nResults saved to {output_csv}")