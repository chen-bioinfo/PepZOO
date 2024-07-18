from amp.inference import HydrAMPGenerator
import pandas as pd
import numpy as np
import prettytable 
from prettytable import PrettyTable, from_csv
import csv
import os
# 不使用GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ""
# 加载模型
generator = HydrAMPGenerator('models/Basic/41', 'models/Basic/pca_decomposer.joblib')
# 加载数据
data = pd.read_csv('data/inactivate_amp.csv')
mask = data['sequence'].str.len() <= 25
sequences = data.loc[mask]['sequence'].tolist()

model = "basic"
seq_size = len(sequences)

# 优化
generator.basic_analogue_generation(sequences=sequences,
                            filtering_criteria='improvement',
                            n_attempts=100,
                            seed=42,
                            lr=0.05,
                            Q=32,
                            variance=1,
                            beta=0.5
)

# 结果排序
txt = pd.read_csv(f'result/{model}/result.csv')
txt2 = txt.sort_values(by=['amp','mic','hydrophobicity', 'hydrophobic_moment', 'charge', 'isoelectric_point'], ascending=True)
txt2.to_csv(f'result/{model}/result_sortby_all.csv', index=False, encoding='utf-8')
f = open(f'result/{model}/result.csv', 'w')
f.close()

# 将每条优化序列的结果单独保存为一个csv文件
dic = {}
with open(f'result/{model}/result_sortby_all.csv','r',encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        dic[row[0]] = row

dic.pop('description')
result = {i:[] for i in range(seq_size)}

for key in dic.keys():
    k = int(key.split('_')[0])
    result[k].append(dic[key])

for i in range(seq_size): 
    f = open(f'result/{model}/seqs/seq_{i}.csv', 'w', encoding='utf-8')
    writer = csv.writer(f)
    writer.writerow([
        'description','sequence', 'amp', 'mic', 'length', 'hydrophobicity','hydrophobic_moment', 'charge', 'isoelectric_point'
    ])
    writer.writerows(result[i]) 
    f.close()  

# 将每条序列单独保存为一个prettytable
tb = PrettyTable([
    'description', 
    'sequence', 
    'amp', 
    'mic', 
    'length', 
    'hydrophobicity', 
    'hydrophobic_moment', 
    'charge', 
    'isoelectric_point'
])
tb.set_style(15)
for j in range(seq_size):
    csvfile = open(f'result/{model}/seqs/seq_{j}.csv', 'r')
    data = pd.read_csv(csvfile)
    csvfile.close()
    data = data.sort_values(by=['amp','mic','hydrophobicity', 'hydrophobic_moment', 'charge', 'isoelectric_point'], ascending=False)
    tb.add_rows(np.array(data).tolist())   
    with open(f'result/{model}/amp/seq_{j}.txt', 'w') as f:
        f.write(str(tb))      
    tb.clear_rows()

# 统计优化结果
dic_all = {'BasicZeroOpt': 0, 'BasicOpt': 0, 'original': 0}
dic_first = {'BasicZeroOpt': 0, 'BasicOpt': 0, 'original': 0}
for k in range(seq_size):
    with open(f'result/{model}/amp/seq_{k}.txt', 'r') as f:
        data = f.readlines()
    tb.clear_rows()
    for i in range(len(data)):
        if data[i][0] == '╔':
            continue
        if data[i][0] == '╠':
            continue
        if data[i][0] == '╚':
            tb.del_row(0)
            lists = tb.rows
            key = lists[0][0].split('_')[1]
            dic_first[key] += 1
            for i in range(len(lists)):
                key = lists[i][0].split('_')[1]
                dic_all[key] += 1
            continue
        seq = data[i].split('║')
        if len(seq) == 11:
            tb.add_row([
                    seq[1].strip(), 
                    seq[2].strip(), 
                    seq[3].strip(),
                    seq[4].strip(),
                    seq[5].strip(),
                    seq[6].strip(),
                    seq[7].strip(),
                    seq[8].strip(),
                    seq[9].strip(),
            ])
with open(f'result/{model}/result.txt', 'a') as f:
    f.write(str(dic_all)+'\n')
    f.write(str(dic_first)+'\n')

print(dic_all)
print(dic_first)
