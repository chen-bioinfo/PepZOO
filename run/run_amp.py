from amp.inference import HydrAMPGenerator
import pandas as pd
import os
from amp.utils.seed import set_seed

set_seed(42)

os.environ['CUDA_VISIBLE_DEVICES'] = ""

data = pd.read_csv('data/case3.csv')
mask = data['sequence'].str.len() <= 25
peptides = data.loc[mask]['sequence'].tolist()

generator = HydrAMPGenerator('models/HydrAMP/', 'models/HydrAMP/pca_decomposer.joblib')
generator.amp_optimization(sequences=peptides, n_attempts=100, seed=42, lr=0.05, beta=0.5, variance=1, Q=32)

txt = pd.read_csv(f'results/HydrAMP/result.csv')
# txt2 = txt.sort_values(by=['amp'], ascending=True)
txt.to_csv(f'results/HydrAMP/case3.csv', index=False, encoding='utf-8')
# clear file
f = open(f'results/HydrAMP/result.csv', 'w')
f.close()

txt = pd.read_csv(f'results/PepZOO/amp/result.csv')
# txt2 = txt.sort_values(by=['amp'], ascending=True)
txt.to_csv(f'results/PepZOO/amp/case3.csv', index=False, encoding='utf-8')
# clear file
f = open(f'results/PepZOO/amp/result.csv', 'w')
f.close()
