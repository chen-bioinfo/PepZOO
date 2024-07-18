from amp.inference import HydrAMPGenerator
import pandas as pd
from amp.utils.seed import set_seed

set_seed(42)
# 加载VAE模型
generator = HydrAMPGenerator('models/HydrAMP/41', 'models/HydrAMP/pca_decomposer.joblib')
# 加载数据
data = pd.read_csv('data/Antiviral_amps.csv')
mask = data['sequence'].str.len() <= 25
peptides = data.loc[mask]['sequence'].tolist()
# 调用模型开始优化
generator.toxicity_optimization(peptides, 42, 100, lr=0.025, Q=32, variance=1, beta=0.5)
