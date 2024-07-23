from amp.inference import HydrAMPGenerator
import pandas as pd
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ""
generator = HydrAMPGenerator('models/PepCVAE', 'models/PepCVAE/pca_decomposer.joblib')

data = pd.read_csv('data/case3.csv')
mask = data['sequence'].str.len() <= 25
sequences = data.loc[mask]['sequence'].tolist()



generator.pepcvae_generation(sequences=sequences,
                            filtering_criteria='improvement',
                            n_attempts=100,
                            seed=42,
                            lr=0.05,
                            Q=32,
                            variance=1,
                            beta=0.5
)

txt = pd.read_csv(f'results/PepCVAE/result.csv')
# txt2 = txt.sort_values(by=['amp','mic'], ascending=True)
txt.to_csv(f'results/PepCVAE/case3.csv', index=False, encoding='utf-8')
f = open(f'results/PepCVAE/result.csv', 'w')
f.close()