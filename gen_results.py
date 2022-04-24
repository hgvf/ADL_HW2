import json
import csv
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--test_path', type=str, default='./QA/predict_predictions.json')
parser.add_argument('--pred_path', type=str, default='./QA/predictions.csv')
opt = parser.parse_args()

f = open(opt.test_path)
predict = json.load(f)

with open(opt.pred_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'answer'])

    for k, v in tqdm(predict.items()):
        writer.writerow([k, v])
