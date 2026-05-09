import sys
sys.path.append("/home3/alejandro.rodriguez/python_modules")

from functions import *
import json
import uproot
import numpy as np

DEC_ID = "38011800"
IS_SIGNAL = 0

STATS_FILE = f"stats/{'signal' if IS_SIGNAL else 'background'}_stats_{DEC_ID}.json"
with open(STATS_FILE, 'r') as f:
    global_stats = json.load(f)
print(f"Loaded global statistics from {STATS_FILE}")

print("Global statistics:")
for col, stats in global_stats.items():
    print(f"  {col}: mean = {stats['mean']}, std = {stats['std']}, min = {stats['min']}, max = {stats['max']}")