from amp.inference import HydrAMPGenerator
import pandas as pd
import os
from amp.utils.seed import set_seed

set_seed(42)

os.environ['CUDA_VISIBLE_DEVICES'] = ""

data = pd.read_csv('data/case3.csv')
mask = data['sequence'].str.len() <= 25
peptides = data.loc[mask]['sequence'].tolist()

# pepzoo and hydramp
generator = HydrAMPGenerator('models/HydrAMP', 'models/HydrAMP/pca_decomposer.joblib')
generator.amp_mic_optimization(sequences=peptides,
                                 n_attempts=100,
                                 seed=42,
                                 lr=0.05,
                                 beta=0.5,
                                 variance=1,
                                 Q=32)
txt = pd.read_csv(f'results/HydrAMP/result.csv')
# txt2 = txt.sort_values(by=['amp', 'mic'], ascending=True)
txt.to_csv(f'results/HydrAMP/case3.csv', index=False, encoding='utf-8')
f = open(f'results/amp_mic/Hydramp/result.csv', 'w')
f.close()

txt = pd.read_csv(f'results/PepZOO/amp_mic/result.csv')
# txt2 = txt.sort_values(by=['amp', 'mic'], ascending=True)
txt.to_csv(f'results/PepZOO/amp_mic/case3.csv', index=False, encoding='utf-8')
f = open(f'results/PepZOO/amp_mic/result.csv', 'w')
f.close()