import csv
import itertools
import math
import os
import pickle
import sys
from functools import reduce
from time import time
from typing import Any, Callable, Dict, Iterable, List, Literal, Tuple, Union  
import warnings
import re

import joblib
import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchsnooper
from Bio import SeqIO
from keras import layers, models
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, average_precision_score,
                             classification_report, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from amp.config import LATENT_DIM
from amp.data_utils import sequence as du_sequence
from amp.inference.filtering import get_filtering_mask
from amp.utils.basic_model_serializer import load_master_model_components
from amp.utils.generate_peptides import translate_peptide
from amp.utils.phys_chem_propterties import calculate_physchem_prop
from amp.utils.seed import set_seed

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')

def aac_comp(file,out):
    std = list("ACDEFGHIKLMNPQRSTVWY")
    df1 = pd.DataFrame(file, columns=["Seq"])
    dd = []
    for j in df1['Seq']:
        cc = []
        for i in std:
            count = 0
            for k in j:
                temp1 = k
                if temp1 == i:
                    count += 1
                composition = (count/len(j))*100
            cc.append(composition)
        dd.append(cc)
    df2 = pd.DataFrame(dd)
    head = []
    for mm in std:
        head.append('AAC_'+mm)
    df2.columns = head
    df2.to_csv(out, index=None, header=False)

def dpc_comp(file,out,q=1):
    std = list("ACDEFGHIKLMNPQRSTVWY")
    df1 = pd.DataFrame(file, columns=["Seq"])
    zz = df1.Seq
    dd = []
    for i in range(0,len(zz)):
        cc = []
        for j in std:
            for k in std:
                count = 0
                temp = j+k
                for m3 in range(0,len(zz[i])-q):
                    b = zz[i][m3:m3+q+1:q]
                    b.upper()
                    if b == temp:
                        count += 1
                    composition = (count/(len(zz[i])-(q)))*100
                cc.append(composition)
        dd.append(cc)
    df3 = pd.DataFrame(dd)
    head = []
    for s in std:
        for u in std:
            head.append("DPC"+str(q)+"_"+s+u)
    df3.columns = head
    df3.to_csv(out, index=None, header=False)


def prediction(inputfile1, inputfile2, model,out):
    df = pd.DataFrame()
    a=[]
    file_name = inputfile1
    file_name1 = out
    file_name2 = model
    file_name3 = inputfile2
    clf = joblib.load(file_name2)
    
    data_test1 = np.loadtxt(file_name, delimiter=',')
    data_test2 = np.loadtxt(file_name3, delimiter=',')
    data_test3 = np.concatenate([data_test1,data_test2], axis=1)
    X_test = data_test3
    y_p_score1=clf.predict_proba(X_test)
    y_p_s1=y_p_score1.tolist()
    df = pd.DataFrame(y_p_s1)
    df_1 = df.iloc[:,-1]
    df_1.to_csv(file_name1, index=None, header=False)

def class_assignment(file1,thr,out):
    df1 = pd.read_csv(file1, header=None)
    df1.columns = ['ML Score']
    cc = []
    for i in range(0,len(df1)):
        if df1['ML Score'][i]>=float(thr):
            cc.append('Toxin')
        else:
            cc.append('Non-Toxin')
    df1['Prediction'] = cc
    df1 =  df1.round(3)
    df1.to_csv(out, index=None)

def MERCI_Processor_p(merci_file,merci_processed,name):
    hh =[]
    jj = []
    kk = []
    qq = []
    filename = merci_file
    df = pd.DataFrame(name)
    zz = list(df[0])
    check = '>'
    with open(filename) as f:
        l = []
        for line in f:
            if not len(line.strip()) == 0 :
                l.append(line)
            if 'COVERAGE' in line:
                for item in l:
                    if item.lower().startswith(check.lower()):
                        hh.append(item)
                l = []
    if hh == []:
        ff = [w.replace('>', '') for w in zz]
        for a in ff:
            jj.append(a)
            qq.append(np.array(['0']))
            kk.append('Non-Toxin')
    else:
        ff = [w.replace('\n', '') for w in hh]
        ee = [w.replace('>', '') for w in ff]
        rr = [w.replace('>', '') for w in zz]
        ff = ee + rr
        oo = np.unique(ff)
        df1 = pd.DataFrame(list(map(lambda x:x.strip(),l))[1:])
        df1.columns = ['Name']
        df1['Name'] = df1['Name'].str.strip('(')
        df1[['Seq','Hits']] = df1.Name.str.split("(",expand=True)
        df2 = df1[['Seq','Hits']]
        df2.replace(to_replace=r"\)", value='', regex=True, inplace=True)
        df2.replace(to_replace=r'motifs match', value='', regex=True, inplace=True)
        df2.replace(to_replace=r' $', value='', regex=True,inplace=True)
        total_hit = int(df2.loc[len(df2)-1]['Seq'].split()[0])
        for j in oo:
            if j in df2.Seq.values:
                jj.append(j)
                qq.append(df2.loc[df2.Seq == j]['Hits'].values)
                kk.append('Toxin')
            else:
                jj.append(j)
                qq.append(np.array(['0']))
                kk.append('Non-Toxin')
    df3 = pd.concat([pd.DataFrame(jj),pd.DataFrame(qq),pd.DataFrame(kk)], axis=1)
    df3.columns = ['Name','Hits','Prediction']
    df3.to_csv(merci_processed,index=None)

def Merci_after_processing_p(merci_processed,final_merci_p):
    df5 = pd.read_csv(merci_processed)
    df5 = df5[['Name','Hits']]
    df5.columns = ['Subject','Hits']
    kk = []
    for i in range(0,len(df5)):
        if df5['Hits'][i] > 0:
            kk.append(0.5)
        else:
            kk.append(0)
    df5["MERCI Score Pos"] = kk
    df5 = df5[['Subject','MERCI Score Pos']]
    df5.to_csv(final_merci_p, index=None)

def MERCI_Processor_n(merci_file,merci_processed,name):
    hh =[]
    jj = []
    kk = []
    qq = []
    filename = merci_file
    df = pd.DataFrame(name)
    zz = list(df[0])
    check = '>'
    with open(filename) as f:
        l = []
        for line in f:
            if not len(line.strip()) == 0 :
                l.append(line)
            if 'COVERAGE' in line:
                for item in l:
                    if item.lower().startswith(check.lower()):
                        hh.append(item)
                l = []
    if hh == []:
        ff = [w.replace('>', '') for w in zz]
        for a in ff:
            jj.append(a)
            qq.append(np.array(['1']))
            kk.append('Toxin')
    else:
        ff = [w.replace('\n', '') for w in hh]
        ee = [w.replace('>', '') for w in ff]
        rr = [w.replace('>', '') for w in zz]
        ff = ee + rr
        oo = np.unique(ff)
        df1 = pd.DataFrame(list(map(lambda x:x.strip(),l))[1:])
        df1.columns = ['Name']
        df1['Name'] = df1['Name'].str.strip('(')
        df1[['Seq','Hits']] = df1.Name.str.split("(",expand=True)
        df2 = df1[['Seq','Hits']]
        df2.replace(to_replace=r"\)", value='', regex=True, inplace=True)
        df2.replace(to_replace=r'motifs match', value='', regex=True, inplace=True)
        df2.replace(to_replace=r' $', value='', regex=True,inplace=True)
        total_hit = int(df2.loc[len(df2)-1]['Seq'].split()[0])
        for j in oo:
            if j in df2.Seq.values:
                jj.append(j)
                qq.append(df2.loc[df2.Seq == j]['Hits'].values)
                kk.append('Non-Toxin')
            else:
                jj.append(j)
                qq.append(np.array(['0']))
                kk.append('Toxin')
    df3 = pd.concat([pd.DataFrame(jj),pd.DataFrame(qq),pd.DataFrame(kk)], axis=1)
    df3.columns = ['Name','Hits','Prediction']
    df3.to_csv(merci_processed,index=None)

def Merci_after_processing_n(merci_processed,final_merci_n):
    df5 = pd.read_csv(merci_processed)
    df5 = df5[['Name','Hits']]
    df5.columns = ['Subject','Hits']
    kk = []
    for i in range(0,len(df5)):
        if df5['Hits'][i] > 0:
            kk.append(-0.5)
        else:
            kk.append(0)
    df5["MERCI Score Neg"] = kk
    df5 = df5[['Subject','MERCI Score Neg']]
    df5.to_csv(final_merci_n, index=None)


def hybrid(ML_output,name1,merci_output_p, merci_output_n,threshold,final_output):
    df6_2 = pd.read_csv(ML_output,header=None)
    df6_1 = pd.DataFrame(name1)
    df5 = pd.read_csv(merci_output_p, dtype={'Subject': object, 'MERCI Score Pos': np.float64})
    df4 = pd.read_csv(merci_output_n, dtype={'Subject': object, 'MERCI Score Neg': np.float64})
    df6 = pd.concat([df6_1,df6_2],axis=1)
    df6.columns = ['Subject','ML Score']
    df6['Subject'] = df6['Subject'].str.replace('>','')
    df7 = pd.merge(df6,df5, how='outer',on='Subject')
    df8 = pd.merge(df7,df4, how='outer',on='Subject')
    df8.fillna(0, inplace=True)
    df8['Hybrid Score'] = df8[['ML Score', 'MERCI Score Pos', 'MERCI Score Neg']].sum(axis=1)
    df8 = df8.round(3)
    ee = []
    for i in range(0,len(df8)):
        if df8['Hybrid Score'][i] > float(threshold):
            ee.append('Toxin')
        else:
            ee.append('Non-Toxin')
    df8['Prediction'] = ee
    df8.to_csv(final_output, index=None)


def _dispose_into_bucket(intersection: np.ndarray,
                         prototype_sequences: List[str],
                         generated_sequences: np.ndarray,
                         generated_amp: np.ndarray,
                         generated_mic: np.ndarray,
                         attempts: int,
                         block_size: int) -> List[Dict[str, np.ndarray]]:
    """
    Takes block of generated peptides that corresponds to a single original (input) peptide and filter out based on
     intersection, uniquness
    @param intersection:
    @param generated_sequences:
    @param generated_amp:
    @param generated_mic:
    @param attempts:
    @param block_size:
    @return:
    """
    bucket_indices = np.arange(0, (attempts + 1) * block_size, attempts)
    disposed_generated_sequences = []
    for origin_seq, (left_index, right_index) in zip(prototype_sequences, zip(bucket_indices, bucket_indices[1:])):
        # in case of low temperature it might be the case that an analouge will be actually a peptide we start from
        intersection[left_index:right_index] &= (generated_sequences[left_index:right_index] != origin_seq)
        current_bucket_indices = intersection[left_index:right_index]
        current_bucket_sequences = generated_sequences[left_index:right_index][current_bucket_indices].tolist()
        current_bucket_sequences = np.delete(
            np.array(current_bucket_sequences), 
            [i for i in range(len(current_bucket_sequences)) if current_bucket_sequences[i] == '']
        ).tolist() 
        if not current_bucket_sequences:
            disposed_generated_sequences.append(None)
            continue
        current_amps = generated_amp[left_index:right_index][current_bucket_indices]
        current_mic = generated_mic[left_index:right_index][current_bucket_indices]

        current_bucket_sequences, indices = np.unique(current_bucket_sequences, return_index=True)
        current_amps = current_amps[indices]
        current_mic = current_mic[indices]

        bucket_data = {
            'sequence': current_bucket_sequences,
            'amp': current_amps.tolist(),
            'mic': current_mic.tolist()
        }
        bucket_data.update(calculate_physchem_prop(current_bucket_sequences))
        disposed_generated_sequences.append(bucket_data)
    return disposed_generated_sequences


def slice_blocks(flat_arrays: Tuple[np.ndarray, ...], block_size: int) -> Tuple[np.ndarray, ...]:
    """
    Changes ordering of sequence from [a, b, c, a, b, c, ...] to [a, a, b, b, c, c, ...]
    @param flat_arrays: arrays to process
    @param block_size: number of arrays before stacking
    @return: rearranged arrays into blocks
    """
    return tuple([x.reshape(-1, block_size).T.flatten() for x in flat_arrays])

amino_acid_set = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, 
				"F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
				"O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
				"U": 19, "T": 20, "W": 21, 
				"V": 22, "Y": 23, "X": 24, 
				"Z": 25 } # consider non-standard residues

amino_acid_num = 25

ss_set = {"H": 1, "C": 2, "E": 3} # revise order, not necessary if training your own model
ss_number = 3

physicochemical_set={'A': 1, 'C': 3, 'B': 7, 'E': 5, 'D': 5, 'G': 2, 'F': 1, 
			'I': 1, 'H': 6, 'K': 6, 'M': 1, 'L': 1, 'O': 7, 'N': 4, 
			'Q': 4, 'P': 1, 'S': 4, 'R': 6, 'U': 7, 'T': 4, 'W': 2, 
			'V': 1, 'Y': 4, 'X': 7, 'Z': 7}

residue_list = list(amino_acid_set.keys())
ss_list = list(ss_set.keys())


new_key_list = []
for i in residue_list:
    for j in ss_list:
        str_1 = str(i)+str(j)
        new_key_list.append(str_1)

new_value_list = [x+1 for x in list(range(amino_acid_num*ss_number))]

seq_ss_dict = dict(zip(new_key_list,new_value_list))
seq_ss_number = amino_acid_num*ss_number #75

def label_sequence(line, pad_prot_len, res_ind):
	X = np.zeros(pad_prot_len)

	for i, res in enumerate(line[:pad_prot_len]):
		X[i] = res_ind[res]
	return X

def label_seq_ss(line, pad_prot_len, res_ind):
	line = line.strip().split(',')
	X = np.zeros(pad_prot_len)
	for i ,res in enumerate(line[:pad_prot_len]):
		X[i] = res_ind[res]
	return X

def sigmoid(x):
	return 1 / (1 + math.exp(-x))

sigmoid_array=np.vectorize(sigmoid)

def padding_sigmoid_pssm(x,N):
	x = sigmoid_array(x)
	padding_array = np.zeros([N,x.shape[1]])
	if x.shape[0]>=N: # sequence is longer than N
		padding_array[:N,:x.shape[1]] = x[:N,:]
	else:
		padding_array[:x.shape[0],:x.shape[1]] = x
	return padding_array

def padding_intrinsic_disorder(x,N):
	padding_array = np.zeros([N,x.shape[1]])
	if x.shape[0]>=N: # sequence is longer than N
		padding_array[:N,:x.shape[1]] = x[:N,:]
	else:
		padding_array[:x.shape[0],:x.shape[1]] = x
	return padding_array


def cls_scores(label, pred):
	label = label.reshape(-1)
	pred = pred.reshape(-1)
	# r2_score, mean_squred_error are ignored
	return roc_auc_score(label, pred), average_precision_score(label, pred)


class HydrAMPGenerator:

    def __init__(self, model_path: str, decomposer_path: str, softmax=False):
        components = load_master_model_components(model_path, return_master=True, softmax=softmax)
        self.model_path = model_path
        self.decomposer_path = decomposer_path
        self._encoder, self._decoder, self._amp_classifier, self._mic_classifier, self.master = components
        self._latent_decomposer: PCA = joblib.load(decomposer_path)
        self._sigma_model = self.get_sigma_model()

    def get_sigma_model(self):
        inputs = layers.Input(shape=(25,))
        z_mean, z_sigma, z = self.master.encoder.output_tensor(inputs)
        return models.Model(inputs, [z_mean, z_sigma, z])

    def get_sigma(self, x):
        _, z_sigma, _ = self._sigma_model.predict(x)
        return np.exp(z_sigma / 2)

    @staticmethod
    def _transpose_sequential_results(res: Dict[str, np.array]):
        transposed_results = {}
        properties = list(res.keys())
        properties.remove('sequence')
        for index, sequence in enumerate(res['sequence']):
            seq_properties = {}
            for prop in properties:
                seq_properties[prop] = res[prop][index]
            transposed_results[sequence] = seq_properties
        return transposed_results

    @staticmethod
    def _encapsulate_sequential_results(res: List[Dict[str, np.array]]):
        transposed_results = []
        for item in res:
            if item is None:
                transposed_results.append(None)
                continue
            item_generated_sequences = []
            properties = list(item.keys())
            for index, sequence in enumerate(item['sequence']):
                seq_properties = {}
                for prop in properties:
                    seq_properties[prop] = item[prop][index]
                item_generated_sequences.append(seq_properties)
            transposed_results.append(item_generated_sequences)
        return transposed_results

    @staticmethod
    def select_peptides(peptides, amp, mic, n_attempts: int = 64, target_positive: bool = True):
        amp = amp.reshape(n_attempts, -1)
        mic = mic.reshape(n_attempts, -1)
        if target_positive:
            mask_amp = amp < 0.8  # ignore those below 0.8
            combined = ma.masked_where(mask_amp, amp)
            good = combined.argmax(axis=0)

        else:
            mask_amp = amp > 0.2  # ignore those above 0.2
            combined = ma.masked_where(mask_amp, amp)
            good = combined.argmin(axis=0)
        peptides = np.array(peptides).reshape(n_attempts, -1).T

        selective_index = list(range(peptides.shape[0]))
        good_peptides = peptides[selective_index, good]
        good_amp = amp.T[selective_index, good]
        good_mic = mic.T[selective_index, good]
        return good_peptides, good_amp, good_mic

    # compute the zeroth-order gradient for antimicrobial function and activity optimization
    def estimate_gradient(self, z, q, beta, variance):
        amp_conditions = mic_conditions = np.ones((len(z), 1))
        conditioned = np.hstack([
            z,
            amp_conditions,
            mic_conditions,
        ])
        decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)  
        seq = np.argmax(decoded, axis=-1)
        old_amp = self._amp_classifier.predict(seq, verbose=1, batch_size=80000) 
        old_mic = self._mic_classifier.predict(seq, verbose=1, batch_size=80000)  

        u = np.random.normal(0, variance, size=(q, len(z), LATENT_DIM)).astype('float32')
        u = beta * (u / np.linalg.norm(u, axis=-1, keepdims=True))
        amp_condition = mic_condition = np.ones((q, len(z), 1))

        conditioned = np.concatenate((z+u, amp_condition, mic_condition), axis=-1).reshape(-1, LATENT_DIM+2)
        
        decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
        seq = np.argmax(decoded, axis=-1)

        new_amp = self._amp_classifier.predict(seq, verbose=1, batch_size=80000)
        new_mic = self._mic_classifier.predict(seq, verbose=1, batch_size=80000)
        # compute pesudo gradient
        grads = np.zeros((len(z), LATENT_DIM))
        for i in range(q):
            for j in range(len(z)):
                grads[j] += beta * (float)(old_amp[j][0]-new_amp[i*len(z)+j][0] + old_mic[j][0]-new_mic[i*len(z)+j][0]) * u[i][j]        
        return grads

    # compute the zeroth-order gradient for antimicrobial function optimization only
    def estimate_gradient_amp(self, z, q, beta, variance):
        amp_conditions = mic_conditions = np.ones((len(z), 1))
        conditioned = np.hstack([
            z,
            amp_conditions,
            mic_conditions,
        ])
        decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)  
        seq = np.argmax(decoded, axis=-1)
        old_amp = self._amp_classifier.predict(seq, verbose=1, batch_size=80000)  

        u = np.random.normal(0, variance, size=(q, len(z), LATENT_DIM)).astype('float32')
        u = beta * (u / np.linalg.norm(u, axis=-1, keepdims=True))
        amp_condition = mic_condition = np.ones((q, len(z), 1))

        conditioned = np.concatenate((z+u, amp_condition, mic_condition), axis=-1).reshape(-1, LATENT_DIM+2)
        
        decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
        seq = np.argmax(decoded, axis=-1)

        new_amp = self._amp_classifier.predict(seq, verbose=1, batch_size=80000)
        # compute pesudo gradient
        grads = np.zeros((len(z), LATENT_DIM))
        for i in range(q):
            for j in range(len(z)):
                grads[j] += beta * (float)(old_amp[j][0]-new_amp[i*len(z)+j][0]) * u[i][j]        
        return grads

    # compute the zeroth-order gradient for antimicrobial activity optimization only
    def estimate_gradient_mic(self, z, q, beta, variance):
        amp_conditions = mic_conditions = np.ones((len(z), 1))
        conditioned = np.hstack([
            z,
            amp_conditions,
            mic_conditions,
        ])
        decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)  
        seq = np.argmax(decoded, axis=-1)
        old_mic = self._mic_classifier.predict(seq, verbose=1, batch_size=80000) 

        u = np.random.normal(0, variance, size=(q, len(z), LATENT_DIM)).astype('float32')
        u = beta * (u / np.linalg.norm(u, axis=-1, keepdims=True))
        amp_condition = mic_condition = np.ones((q, len(z), 1))

        conditioned = np.concatenate((z+u, amp_condition, mic_condition), axis=-1).reshape(-1, LATENT_DIM+2)
        
        decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
        seq = np.argmax(decoded, axis=-1)

        new_mic = self._mic_classifier.predict(seq, verbose=1, batch_size=80000)
        # compute pesudo gradient
        grads = np.zeros((len(z), LATENT_DIM))
        for i in range(q):
            for j in range(len(z)):
                grads[j] += beta * (float)(old_mic[j][0]-new_mic[i*len(z)+j][0]) * u[i][j]        
        return grads



    def amp_optimization(self, sequences: List[str], seed: int,
                            n_attempts: int = 100, temp: float = 5.0, **kwargs) -> Dict[Any, Dict[str, Any]]:
        """
        Generates new peptides based on input sequences
        @param sequences: peptides that form a template for further processing
        @param n_attempts: how many times a single latent vector is decoded - for normal Softmax models it should be set
        to 1 as every decoding call returns the same peptide.
        @param temp: creativity parameter. Controls latent vector sigma scaling
        @param seed: random seed
        """

        # set random seed
        set_seed(seed)

        block_size = len(sequences)
        padded_sequences = du_sequence.pad(du_sequence.to_one_hot(sequences))

        # variance
        sigmas = self.get_sigma(padded_sequences)
        # AMP probability of original sequences
        amp_org = self._amp_classifier.predict(padded_sequences, verbose=1, batch_size=80000)
        # probability of original sequences with MIC < 32ug/ml
        mic_org = self._mic_classifier.predict(padded_sequences, verbose=1, batch_size=80000)
        # copy sequences for n_attempts times
        padded_sequences = np.vstack([padded_sequences] * n_attempts).reshape(-1, 25)
        # embedding
        z = self._encoder.predict(padded_sequences, verbose=1, batch_size=80000)

        amp_stacked = np.vstack([amp_org] * n_attempts)
        mic_stacked = np.vstack([mic_org] * n_attempts)
        # physicochemical property
        props = calculate_physchem_prop(sequences)

        # save result
        hydramp_wf = open('results/HydrAMP/result.csv', 'a', encoding='utf-8')
        pepzoo_wf = open('results/PepZOO/amp/result.csv', 'a', encoding='utf-8')
        hydramp_writer = csv.writer(hydramp_wf)
        pepzoo_writer = csv.writer(pepzoo_wf)
        hydramp_writer.writerow(['description','sequence', 'amp', 'mic', 'length', 'hydrophobicity',
                            'hydrophobic_moment', 'charge', 'isoelectric_point'])
        pepzoo_writer.writerow(['description','sequence', 'amp', 'mic', 'length', 'hydrophobicity',
                            'hydrophobic_moment', 'charge', 'isoelectric_point'])
        for i in range(block_size):
            hydramp_writer.writerow([
                f'{i}_original',
                sequences[i], 
                amp_org[i][0], 
                mic_org[i][0], 
                props['length'][i],
                props['hydrophobicity'][i], 
                props['hydrophobic_moment'][i], 
                props['charge'][i], 
                props['isoelectric_point'][i]
            ])
            pepzoo_writer.writerow([
                f'{i}_original',
                sequences[i], 
                amp_org[i][0], 
                mic_org[i][0], 
                props['length'][i],
                props['hydrophobicity'][i], 
                props['hydrophobic_moment'][i], 
                props['charge'][i], 
                props['isoelectric_point'][i]
            ])
        # Perturbation Vector
        perturb = np.random.normal(loc=0, scale=temp * np.vstack([sigmas] * n_attempts), size=z.shape)
    
        encoded = z + perturb

        amp_condition = mic_condition = np.ones((len(padded_sequences), 1))
        # concat embedding and condition
        conditioned = np.hstack([
            encoded,
            amp_condition,
            mic_condition,
        ])
        # 
        decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
        new_peptides = np.argmax(decoded, axis=2)
        new_amp = self._amp_classifier.predict(new_peptides, verbose=1, batch_size=80000)
        new_mic = self._mic_classifier.predict(new_peptides, verbose=1, batch_size=80000)
        #
        better = new_amp > amp_stacked.reshape(-1, 1)
        better = better & (new_mic > mic_stacked.reshape(-1, 1))

        better = better.flatten()
        # 
        new_peptides = np.array([translate_peptide(x) for x in new_peptides])
        # 
        new_peptides, new_amp, new_mic, better = slice_blocks((new_peptides, new_amp, new_mic, better), block_size)
        # mask = get_filtering_mask(sequences=new_peptides, filtering_options=kwargs)
        # mask &= better
        filtered_peptides = _dispose_into_bucket(better, sequences, new_peptides, new_amp, new_mic, n_attempts,
                                                 block_size)
        filtered_peptides = self._encapsulate_sequential_results(filtered_peptides)
        # save result
        for i in range(block_size):
            if filtered_peptides[i] != None:
                for j in range(len(filtered_peptides[i])):
                    hydramp_writer.writerow([
                        f'{i}_HydrAMP',
                        filtered_peptides[i][j]['sequence'],
                        filtered_peptides[i][j]['amp'],
                        filtered_peptides[i][j]['mic'], 
                        filtered_peptides[i][j]['length'], 
                        filtered_peptides[i][j]['hydrophobicity'], 
                        filtered_peptides[i][j]['hydrophobic_moment'], 
                        filtered_peptides[i][j]['charge'], 
                        filtered_peptides[i][j]['isoelectric_point']
                    ])

        padded_sequences = du_sequence.pad(du_sequence.to_one_hot(sequences)).reshape(-1, 25)
        # get embedding of original sequences
        z = self._encoder.predict(padded_sequences, verbose=1, batch_size=80000)
        amp_condition = mic_condition = np.ones((len(padded_sequences),1))

        z = torch.from_numpy(z)
        base_lr = kwargs['lr']
        beta = kwargs['beta']
        Q = kwargs['Q']
        variance = kwargs['variance']
        adam = torch.optim.Adam([z], lr=base_lr)

        attempts = n_attempts
        seqs_dic = [{} for i in range(block_size)]

        for t in tqdm(range(attempts)):
            # # compute zeroth-order gradient
            z.grad = torch.from_numpy(self.estimate_gradient_amp(z, Q, beta, variance).astype('float32'))
            # update the embedding of peptides using zeroth order gradient
            adam.step()

            conditioned = np.hstack([
                z,
                amp_condition,
                mic_condition,
            ])

            decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
            seq = np.argmax(decoded, axis=-1)
            
            new_amp = self._amp_classifier.predict(seq, verbose=1, batch_size=80000)
            new_mic = self._mic_classifier.predict(seq, verbose=1, batch_size=80000)

            seq = np.array([translate_peptide(x) for x in seq])

            better = new_amp > amp_org
            better = better & (new_mic > mic_org)
            better = better.flatten()

            new_peptides, new_amp, new_mic, better = slice_blocks((seq, new_amp, new_mic, better), block_size)
            filtered_peptides = _dispose_into_bucket(better, sequences, new_peptides, new_amp, new_mic, 1, block_size)
            filtered_peptides = self._encapsulate_sequential_results(filtered_peptides)      
              
            for i in range(block_size):
                if filtered_peptides[i] != None:
                    if filtered_peptides[i][0]['sequence'] not in seqs_dic[i].keys():
                        pepzoo_writer.writerow([
                            f'{i}_PepZOO',
                            filtered_peptides[i][0]['sequence'], 
                            filtered_peptides[i][0]['amp'],
                            filtered_peptides[i][0]['mic'], 
                            filtered_peptides[i][0]['length'], 
                            filtered_peptides[i][0]['hydrophobicity'], 
                            filtered_peptides[i][0]['hydrophobic_moment'], 
                            filtered_peptides[i][0]['charge'], 
                            filtered_peptides[i][0]['isoelectric_point']
                        ])
                    seqs_dic[i][filtered_peptides[i][0]['sequence']] = i
        hydramp_wf.close()
        pepzoo_wf.close()

    def mic_optimization(self, sequences: List[str], seed: int,
                            n_attempts: int = 100, temp: float = 5.0, **kwargs) -> Dict[Any, Dict[str, Any]]:
        """
        Generates new peptides based on input sequences
        @param sequences: peptides that form a template for further processing
        @param filtering_criteria: 'improvement' if generated peptides should be strictly better than input sequences
        'discovery' if generated sequences should be good enough but not strictly better
        @param n_attempts: how many times a single latent vector is decoded - for normal Softmax models it should be set
        to 1 as every decoding call returns the same peptide.
        @param temp: creativity parameter. Controls latent vector sigma scaling
        @param seed: random seed
        """

        # random seed
        set_seed(seed)

        block_size = len(sequences)
        # padding
        padded_sequences = du_sequence.pad(du_sequence.to_one_hot(sequences))
        # variance
        sigmas = self.get_sigma(padded_sequences)
        # amp score of prototypes
        amp_org = self._amp_classifier.predict(padded_sequences, verbose=1, batch_size=80000)
        # mic score of prototypes
        mic_org = self._mic_classifier.predict(padded_sequences, verbose=1, batch_size=80000)
        # 
        padded_sequences = np.vstack([padded_sequences] * n_attempts).reshape(-1, 25)
        # embedding
        z = self._encoder.predict(padded_sequences, verbose=1, batch_size=80000)

        amp_stacked = np.vstack([amp_org] * n_attempts)
        mic_stacked = np.vstack([mic_org] * n_attempts)
        # physicochemical properties
        props = calculate_physchem_prop(sequences)

        # save result
        hydramp_wf = open('results/HydrAMP/result.csv', 'a', encoding='utf-8')
        pepzoo_wf = open('results/PepZOO/mic/result.csv', 'a', encoding='utf-8')
        hydramp_writer = csv.writer(hydramp_wf)
        pepzoo_writer = csv.writer(pepzoo_wf)
        hydramp_writer.writerow(['description','sequence', 'amp', 'mic', 'length', 'hydrophobicity',
                            'hydrophobic_moment', 'charge', 'isoelectric_point'])
        pepzoo_writer.writerow(['description','sequence', 'amp', 'mic', 'length', 'hydrophobicity',
                            'hydrophobic_moment', 'charge', 'isoelectric_point'])
        for i in range(block_size):
            hydramp_writer.writerow([
                f'{i}_original',
                sequences[i], 
                amp_org[i][0], 
                mic_org[i][0], 
                props['length'][i],
                props['hydrophobicity'][i], 
                props['hydrophobic_moment'][i], 
                props['charge'][i], 
                props['isoelectric_point'][i]
            ])

            pepzoo_writer.writerow([
                f'{i}_original',
                sequences[i], 
                amp_org[i][0], 
                mic_org[i][0], 
                props['length'][i],
                props['hydrophobicity'][i], 
                props['hydrophobic_moment'][i], 
                props['charge'][i], 
                props['isoelectric_point'][i]
            ])
        # Perturbation Vector
        noise = np.random.normal(loc=0, scale=temp * np.vstack([sigmas] * n_attempts), size=z.shape)
        # 
        encoded = z + noise
        # condition
        amp_condition = mic_condition = np.ones((len(padded_sequences), 1))
        
        conditioned = np.hstack([
            encoded,
            amp_condition,
            mic_condition,
        ])
        # decode
        decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
        new_peptides = np.argmax(decoded, axis=2)
        # amp score of generated peptides
        new_amp = self._amp_classifier.predict(new_peptides, verbose=1, batch_size=80000)
        # mic score of generated peptides
        new_mic = self._mic_classifier.predict(new_peptides, verbose=1, batch_size=80000)
        
        better = new_amp > amp_stacked.reshape(-1, 1)
        better = better & (new_mic > mic_stacked.reshape(-1, 1))

        better = better.flatten()
       
        new_peptides = np.array([translate_peptide(x) for x in new_peptides])
        
        new_peptides, new_amp, new_mic, better = slice_blocks((new_peptides, new_amp, new_mic, better), block_size)
        # mask = get_filtering_mask(sequences=new_peptides, filtering_options=kwargs)
        # mask &= better
        filtered_peptides = _dispose_into_bucket(better, sequences, new_peptides, new_amp, new_mic, n_attempts,
                                                 block_size)
        filtered_peptides = self._encapsulate_sequential_results(filtered_peptides)
        # save result
        for i in range(block_size):
            if filtered_peptides[i] != None:
                for j in range(len(filtered_peptides[i])):
                    hydramp_writer.writerow([
                        f'{i}_HydrAMP',
                        filtered_peptides[i][j]['sequence'],
                        filtered_peptides[i][j]['amp'],
                        filtered_peptides[i][j]['mic'], 
                        filtered_peptides[i][j]['length'], 
                        filtered_peptides[i][j]['hydrophobicity'], 
                        filtered_peptides[i][j]['hydrophobic_moment'], 
                        filtered_peptides[i][j]['charge'], 
                        filtered_peptides[i][j]['isoelectric_point']
                    ])

        padded_sequences = du_sequence.pad(du_sequence.to_one_hot(sequences)).reshape(-1, 25)
        z = self._encoder.predict(padded_sequences, verbose=1, batch_size=80000)
        amp_condition = mic_condition = np.ones((len(padded_sequences),1))

        z = torch.from_numpy(z)
        base_lr = kwargs['lr']
        beta = kwargs['beta']
        Q = kwargs['Q']
        variance = kwargs['variance']
        adam = torch.optim.Adam([z], lr=base_lr)

        attempts = n_attempts
        seqs_dic = [{} for i in range(block_size)]
        for t in tqdm(range(attempts)):
            # # compute pesudo gradient
            z.grad = torch.from_numpy(self.estimate_gradient_mic(z, Q, beta, variance).astype('float32'))
            # update
            adam.step()

            conditioned = np.hstack([
                z,
                amp_condition,
                mic_condition,
            ])

            decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
            seq = np.argmax(decoded, axis=-1)
            
            new_amp = self._amp_classifier.predict(seq, verbose=1, batch_size=80000)
            new_mic = self._mic_classifier.predict(seq, verbose=1, batch_size=80000)

            seq = np.array([translate_peptide(x) for x in seq])


            better = new_amp > amp_org
            better = better & (new_mic > mic_org)
            better = better.flatten()

            new_peptides, new_amp, new_mic, better = slice_blocks((seq, new_amp, new_mic, better), block_size)
            filtered_peptides = _dispose_into_bucket(better, sequences, new_peptides, new_amp, new_mic, 1, block_size)
            filtered_peptides = self._encapsulate_sequential_results(filtered_peptides)      
              
            for i in range(block_size):
                if filtered_peptides[i] != None:
                    if filtered_peptides[i][0]['sequence'] not in seqs_dic[i].keys():
                        pepzoo_writer.writerow([
                            f'{i}_PepZOO',
                            filtered_peptides[i][0]['sequence'], 
                            filtered_peptides[i][0]['amp'],
                            filtered_peptides[i][0]['mic'], 
                            filtered_peptides[i][0]['length'], 
                            filtered_peptides[i][0]['hydrophobicity'], 
                            filtered_peptides[i][0]['hydrophobic_moment'], 
                            filtered_peptides[i][0]['charge'], 
                            filtered_peptides[i][0]['isoelectric_point']
                        ])
                    seqs_dic[i][filtered_peptides[i][0]['sequence']] = i
        hydramp_wf.close()
        pepzoo_wf.close()

    def amp_mic_optimization(self, sequences: List[str], seed: int,
                            n_attempts: int = 100, temp: float = 5.0, **kwargs) -> Dict[Any, Dict[str, Any]]:
        """
        Generates new peptides based on input sequences
        @param sequences: peptides that form a template for further processing
        @param n_attempts: how many times a single latent vector is decoded - for normal Softmax models it should be set
        to 1 as every decoding call returns the same peptide.
        @param temp: creativity parameter. Controls latent vector sigma scaling
        @param seed: random seed
        """

        set_seed(seed)

        block_size = len(sequences)
        # padding
        padded_sequences = du_sequence.pad(du_sequence.to_one_hot(sequences))
        # variance
        sigmas = self.get_sigma(padded_sequences)
        # amp score of prototypes
        amp_org = self._amp_classifier.predict(padded_sequences, verbose=1, batch_size=80000)
        # mic score of prototypes
        mic_org = self._mic_classifier.predict(padded_sequences, verbose=1, batch_size=80000)

        padded_sequences = np.vstack([padded_sequences] * n_attempts).reshape(-1, 25)
        # embedding
        z = self._encoder.predict(padded_sequences, verbose=1, batch_size=80000)

        amp_stacked = np.vstack([amp_org] * n_attempts)
        mic_stacked = np.vstack([mic_org] * n_attempts)
        # physicochemical properties
        props = calculate_physchem_prop(sequences)

        # save result
        hydramp_wf = open('results/HydrAMP/result.csv', 'a', encoding='utf-8')
        pepzoo_wf = open('results/PepZOO/amp_mic/result.csv', 'a', encoding='utf-8')
        hydramp_writer = csv.writer(hydramp_wf)
        pepzoo_writer = csv.writer(pepzoo_wf)

        f2 = open('results/PepZOO/amp_mic/result_4_novelty.csv', 'a', encoding='utf-8')
        writer2 = csv.writer(f2)

        hydramp_writer.writerow(['description','sequence', 'amp', 'mic', 'length', 'hydrophobicity',
                            'hydrophobic_moment', 'charge', 'isoelectric_point'])
        pepzoo_writer.writerow(['description','sequence', 'amp', 'mic', 'length', 'hydrophobicity',
                            'hydrophobic_moment', 'charge', 'isoelectric_point'])
        writer2.writerow(['type','sequence', 'amp', 'mic'])
        for i in range(block_size):
            hydramp_writer.writerow([
                f'{i+1}_original',
                sequences[i], 
                amp_org[i][0], 
                mic_org[i][0], 
                props['length'][i],
                props['hydrophobicity'][i], 
                props['hydrophobic_moment'][i], 
                props['charge'][i], 
                props['isoelectric_point'][i]
            ])

            pepzoo_writer.writerow([
                f'{i+1}_original',
                sequences[i], 
                amp_org[i][0], 
                mic_org[i][0], 
                props['length'][i],
                props['hydrophobicity'][i], 
                props['hydrophobic_moment'][i], 
                props['charge'][i], 
                props['isoelectric_point'][i]
            ])

            writer2.writerow([
                f'{i+1}_original',
                sequences[i], 
                amp_org[i][0], 
                mic_org[i][0], 
            ])

        # Perturbation Vector
        noise = np.random.normal(loc=0, scale=temp * np.vstack([sigmas] * n_attempts), size=z.shape)
        
        encoded = z + noise
        # condition
        amp_condition = mic_condition = np.ones((len(padded_sequences), 1))
        
        conditioned = np.hstack([
            encoded,
            amp_condition,
            mic_condition,
        ])

        decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
        new_peptides = np.argmax(decoded, axis=2)

        new_amp = self._amp_classifier.predict(new_peptides, verbose=1, batch_size=80000)
        new_mic = self._mic_classifier.predict(new_peptides, verbose=1, batch_size=80000)
        
        better = new_amp > amp_stacked.reshape(-1, 1)
        better = better & (new_mic > mic_stacked.reshape(-1, 1))

        better = better.flatten()
        
        new_peptides = np.array([translate_peptide(x) for x in new_peptides])
        
        new_peptides, new_amp, new_mic, better = slice_blocks((new_peptides, new_amp, new_mic, better), block_size)
        # mask = get_filtering_mask(sequences=new_peptides, filtering_options=kwargs)
        # mask &= better
        filtered_peptides = _dispose_into_bucket(better, sequences, new_peptides, new_amp, new_mic, n_attempts,
                                                 block_size)
        filtered_peptides = self._encapsulate_sequential_results(filtered_peptides)
        # save result
        for i in range(block_size):
            if filtered_peptides[i] != None:
                for j in range(len(filtered_peptides[i])):
                    hydramp_writer.writerow([
                        f'{i+1}_HydrAMP',
                        filtered_peptides[i][j]['sequence'],
                        filtered_peptides[i][j]['amp'],
                        filtered_peptides[i][j]['mic'], 
                        filtered_peptides[i][j]['length'], 
                        filtered_peptides[i][j]['hydrophobicity'], 
                        filtered_peptides[i][j]['hydrophobic_moment'], 
                        filtered_peptides[i][j]['charge'], 
                        filtered_peptides[i][j]['isoelectric_point']
                    ])

        padded_sequences = du_sequence.pad(du_sequence.to_one_hot(sequences)).reshape(-1, 25)
        z = self._encoder.predict(padded_sequences, verbose=1, batch_size=80000)
        amp_condition = mic_condition = np.ones((len(padded_sequences),1))

        z = torch.from_numpy(z)
        base_lr = kwargs['lr']
        beta = kwargs['beta']
        Q = kwargs['Q']
        variance = kwargs['variance']
        adam = torch.optim.Adam([z], lr=base_lr)

        attempts = n_attempts
        seqs_dic = [{} for i in range(block_size)]
        for t in tqdm(range(attempts)):
            # # compute pesudo gradient
            z.grad = torch.from_numpy(self.estimate_gradient(z, Q, beta, variance).astype('float32'))
            # update embedding using zeroth order gradient
            adam.step()

            conditioned = np.hstack([
                z,
                amp_condition,
                mic_condition,
            ])

            decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
            seq = np.argmax(decoded, axis=-1)
            
            new_amp = self._amp_classifier.predict(seq, verbose=1, batch_size=80000)
            new_mic = self._mic_classifier.predict(seq, verbose=1, batch_size=80000)

            seq = np.array([translate_peptide(x) for x in seq])

            better = new_amp > amp_org
            better = better & (new_mic > mic_org)

            better = better.flatten()

            new_peptides, new_amp, new_mic, better = slice_blocks((seq, new_amp, new_mic, better), block_size)
            filtered_peptides = _dispose_into_bucket(better, sequences, new_peptides, new_amp, new_mic, 1, block_size)
            filtered_peptides = self._encapsulate_sequential_results(filtered_peptides)      
              
            for i in range(block_size):
                if filtered_peptides[i] != None:
                    if filtered_peptides[i][0]['sequence'] not in seqs_dic[i].keys():
                        pepzoo_writer.writerow([
                            f'{i+1}_PepZOO',
                            filtered_peptides[i][0]['sequence'], 
                            filtered_peptides[i][0]['amp'],
                            filtered_peptides[i][0]['mic'], 
                            filtered_peptides[i][0]['length'], 
                            filtered_peptides[i][0]['hydrophobicity'], 
                            filtered_peptides[i][0]['hydrophobic_moment'], 
                            filtered_peptides[i][0]['charge'], 
                            filtered_peptides[i][0]['isoelectric_point']
                        ])
                        writer2.writerow([
                            f'{i+1}_{t+1}',
                            filtered_peptides[i][0]['sequence'], 
                            filtered_peptides[i][0]['amp'],
                            filtered_peptides[i][0]['mic'], 
                        ])
                    seqs_dic[i][filtered_peptides[i][0]['sequence']] = i
        hydramp_wf.close()
        pepzoo_wf.close()
        f2.close()

    # predict the binding score of peptides to the target protein
    def binding_predict(self, peptides: List[str], protein: str, model, task: str):
        
        peps_len = len(peptides)
        device = torch.device('cuda:0')
    ################################################### step1 secondary structure #####################################################
        with open(f'data_prepare/{task}/amps.fasta', 'w') as wf:
            for i in range(peps_len):
                f = open(f'data_prepare/{task}/{i+1}.fasta','w')
                f.write('>peptide'+'\n')
                f.write(peptides[i]+'\n')
                f.close()

                wf.write(f'>{i+1}'+'\n')
                wf.write(peptides[i]+'\n')
        f = open(f'data_prepare/{task}/target_protein.fasta','w')
        f.write('>target_protein'+'\n')
        f.write(protein+'\n')
        f.close()
        # get secondary structure of peptides by SCRATCH-1D
        os.system(f"tools/SCRATCH-1D_1.2/bin/run_SCRATCH-1D_predictors.sh\
             data_prepare/{task}/amps.fasta data_prepare/{task}/amps.out 256")
        # get secondary structure of the target protein by SCRATCH-1D
        if not os.path.exists(f'data_prepare/{task}/protein.out.ss'):
            os.system(f"tools/SCRATCH-1D_1.2/bin/run_SCRATCH-1D_predictors.sh\
                data_prepare/{task}/target_protein.fasta data_prepare/{task}/protein.out 256")    
        # save 
        wf = open(f'data_prepare/{task}/features.tsv','w')
        tsv_w = csv.writer(wf, delimiter='\t')
        tsv_w.writerow(['prot_seq', 'pep_seq', 'pep_concat_seq', 'prot_concat_seq'])
        
        with open(f'data_prepare/{task}/amps.out.ss', 'r') as f:
            peps_data = f.readlines()
        for i in range(len(peps_data)):
            peps_data[i] = peps_data[i].strip('\n')
        
        with open(f'data_prepare/{task}/protein.out.ss', 'r') as f:
            prot_data = f.readlines()
        for i in range(len(prot_data)):
            prot_data[i] = prot_data[i].strip('\n')
        
        prot_seq = protein
        
        prot_concat_seq = ""
        for j in range(len(prot_seq)):
            if j < len(prot_seq) - 1:
                prot_concat_seq = prot_concat_seq + prot_seq[j] + prot_data[1][j] + ','
            else:
                prot_concat_seq = prot_concat_seq + prot_seq[j] + prot_data[1][j]
        
        filename = f'data_prepare/{task}/amps.fasta'
        iterator = SeqIO.parse(filename,'fasta')
        seqs = []
        for record in iter(iterator):
            seqs.append(record)
        
        pep_concat_seq = ""
        for i in range(len(seqs)):
            for j in range(len(seqs[i].seq)):
                if j < len(seqs[i].seq) - 1:
                    pep_concat_seq = pep_concat_seq + seqs[i].seq[j] + peps_data[2*i+1][j] + ','
                else:
                    pep_concat_seq = pep_concat_seq + seqs[i].seq[j] + peps_data[2*i+1][j]
            tsv_w.writerow([prot_seq, seqs[i].seq, pep_concat_seq, prot_concat_seq])
            pep_concat_seq = ""
        wf.close()
#############################################step2: get pssm matrix of the target protein #######################################################
        prot_pssm_dict = {}   
        
        if not os.path.exists(f'data_prepare/{task}/target_protein.pssm'):
            os.system(f"psiblast -query data_prepare/{task}/target_protein.fasta -db /geniusland/dataset/uniprot/uniref90/uniref90.fasta -num_iterations 3 -out_ascii_pssm data_prepare/{task}/target_protein.pssm")

        with open(f'data_prepare/{task}/target_protein.pssm','r') as rf:
            data = rf.readlines()
        tmp = np.zeros((len(protein),20))
        for i in range(3, 3+len(protein)):
            char = data[i].split(' ')
            count = 0
            for j in range(6, len(char)):
                if char[j] != char[0]:
                    tmp[i-3][count] = float(char[j])
                    count += 1
                if count >= 20:
                    break
        prot_pssm_dict[protein] = tmp
            
############################################# step3: get the intrinsic disorder tendencies ###################
        prot_intrinsic_dict = {}
        pep_intrinsic_dict = {}
        # peptides
        for i in range(peps_len):
            result = os.popen(f"python3 tools/iupred2a/iupred2a.py -a data_prepare/{task}/{i+1}.fasta long")
            res = result.read()
            long_val = res.splitlines()[0].split(',')
            long_list = np.zeros((len(long_val), 1))
            long_list[0][0] = float(long_val[0].split('[')[1])
            for j in range(1, len(long_val)-1):
                long_list[j][0] = float(long_val[j])
            long_list[j][0] = float(long_val[j].split(']')[0])

            result = os.popen(f"python3 tools/iupred2a/iupred2a.py -a data_prepare/{task}/{i+1}.fasta short")
            res = result.read()
            short_val = res.splitlines()[0].split(',')
            short_list = np.zeros((len(short_val), 1))
            short_list[0][0] = float(short_val[0].split('[')[1])
            for j in range(1, len(short_val)-1):
                short_list[j][0] = float(short_val[j])
            short_list[j][0] = float(short_val[j].split(']')[0])

            anchor_val = res.splitlines()[1].split(',')
            anchor_list = np.zeros((len(anchor_val), 1))
            anchor_list[0][0] = float(anchor_val[0].split('[')[1])
            for j in range(1, len(anchor_val)-1):
                anchor_list[j][0] = float(anchor_val[j])
            anchor_list[j][0] = float(anchor_val[j].split(']')[0])

            results = np.concatenate((long_list, short_list, anchor_list), axis=1)
            pep_intrinsic_dict[peptides[i]] = results
        # the target protein
        result = os.popen(f"python3 tools/iupred2a/iupred2a.py -a data_prepare/{task}/target_protein.fasta long")
        res = result.read()
        long_val = res.splitlines()[0].split(',')
        long_list = np.zeros((len(long_val), 1))
        long_list[0][0] = float(long_val[0].split('[')[1])
        for j in range(1, len(long_val)-1):
            long_list[j][0] = float(long_val[j])
        long_list[j][0] = float(long_val[j].split(']')[0])

        result = os.popen(f"python3 tools/iupred2a/iupred2a.py -a data_prepare/{task}/target_protein.fasta short")
        res = result.read()
        short_val = res.splitlines()[0].split(',')
        short_list = np.zeros((len(short_val), 1))
        short_list[0][0] = float(short_val[0].split('[')[1])
        for j in range(1, len(short_val)-1):
            short_list[j][0] = float(short_val[j])
        short_list[j][0] = float(short_val[j].split(']')[0])

        anchor_val = res.splitlines()[1].split(',')
        anchor_list = np.zeros((len(anchor_val), 1))
        anchor_list[0][0] = float(anchor_val[0].split('[')[1])
        for j in range(1, len(anchor_val)-1):
            anchor_list[j][0] = float(anchor_val[j])
        anchor_list[j][0] = float(anchor_val[j].split(']')[0])
        results = np.concatenate((long_list, short_list, anchor_list), axis=1)
        prot_intrinsic_dict[protein] = results
############################################## feature concatenation #######################################################

        protein_dense_feature_dict = np.concatenate((prot_pssm_dict[protein], prot_intrinsic_dict[protein]),axis=1)
        with open(f'data_prepare/{task}/protein_dense_feature_dict','wb') as f:
            pickle.dump(protein_dense_feature_dict,f)
        
        f = open(f'data_prepare/{task}/features.tsv')
        pep_set = set()
        seq_set = set()
        pep_ss_set = set()
        seq_ss_set = set()
        for line in f.readlines()[1:]: # if the file has headers and pay attention to the columns (whether have peptide binding site labels)
            # seq, pep, label, pep_ss, seq_ss  = line.strip().split('\t')
            seq, pep, pep_ss, seq_ss  = line.strip().split('\t')
            pep_set.add(pep)
            seq_set.add(seq)
            pep_ss_set.add(pep_ss)
            seq_ss_set.add(seq_ss)

        f.close()
        pep_len = [len(pep) for pep in pep_set]
        seq_len = [len(seq) for seq in seq_set]
        pep_ss_len = [len(pep_ss) for pep_ss in pep_ss_set]
        seq_ss_len = [len(seq_ss) for seq_ss in seq_ss_set]

        pep_len.sort()
        seq_len.sort()
        pep_ss_len.sort()
        seq_ss_len.sort()
        pad_pep_len = 50 
        pad_prot_len = seq_len[int(0.8*len(seq_len))-1]

        peptide_feature_dict = {}
        protein_feature_dict = {}

        peptide_ss_feature_dict = {}
        protein_ss_feature_dict = {}

        peptide_2_feature_dict = {}
        protein_2_feature_dict = {}

        peptide_dense_feature_dict = {}
        protein_dense_feature_dict = {}

        f = open(f'data_prepare/{task}/features.tsv')
        for line in f.readlines()[1:]:
            seq, pep, pep_ss, seq_ss  = line.strip().split('\t')
            if pep not in peptide_feature_dict:
                feature = label_sequence(pep, pad_pep_len, amino_acid_set)
                peptide_feature_dict[pep] = feature
            if seq not in protein_feature_dict:
                feature = label_sequence(seq, pad_prot_len, amino_acid_set)
                protein_feature_dict[seq] = feature
            if pep_ss not in peptide_ss_feature_dict:
                feature = label_seq_ss(pep_ss, pad_pep_len, seq_ss_dict)
                peptide_ss_feature_dict[pep_ss] = feature
            if seq_ss not in protein_ss_feature_dict:
                feature = label_seq_ss(seq_ss, pad_prot_len, seq_ss_dict)
                protein_ss_feature_dict[seq_ss] = feature
            if pep not in peptide_2_feature_dict:
                feature = label_sequence(pep, pad_pep_len, physicochemical_set)
                peptide_2_feature_dict[pep] = feature
            if seq not in protein_2_feature_dict:
                feature = label_sequence(seq, pad_prot_len, physicochemical_set)
                protein_2_feature_dict[seq] = feature
            if pep not in peptide_dense_feature_dict:
                feature = padding_intrinsic_disorder(pep_intrinsic_dict[pep], pad_pep_len)
                peptide_dense_feature_dict[pep] = feature
            if seq not in protein_dense_feature_dict:
                feature_pssm = padding_sigmoid_pssm(prot_pssm_dict[seq], pad_prot_len)
                feature_intrinsic = padding_intrinsic_disorder(prot_intrinsic_dict[seq], pad_prot_len)
                feature_dense = np.concatenate((feature_pssm, feature_intrinsic), axis=1)
                protein_dense_feature_dict[seq] = feature_dense

        f.close()

        print('load feature dict')
        X_pep, X_prot, X_pep_SS, X_prot_SS, X_pep_2, X_prot_2 = [], [], [], [], [], []
        X_dense_pep,X_dense_prot = [],[]
        pep_sequence, prot_sequence, Y = [], [], []
        with open(f'data_prepare/{task}/features.tsv') as f:  # change your own data here
            for line in f.readlines()[1:]:
                protein, peptide, pep_ss, prot_ss  = line.strip().split('\t')
                # protein, peptide,label, pep_ss, prot_ss  = line.strip().split('\t')
                pep_sequence.append(peptide)
                prot_sequence.append(protein)

                X_pep.append(peptide_feature_dict[peptide])
                X_prot.append(protein_feature_dict[protein])
                X_pep_SS.append(peptide_ss_feature_dict[pep_ss])
                X_prot_SS.append(protein_ss_feature_dict[prot_ss])
                X_pep_2.append(peptide_2_feature_dict[peptide])
                X_prot_2.append(protein_2_feature_dict[protein])
                X_dense_pep.append(peptide_dense_feature_dict[peptide])
                X_dense_prot.append(protein_dense_feature_dict[protein])
                
        X_pep = torch.from_numpy(np.array(X_pep)).to(device)
        X_prot = torch.from_numpy(np.array(X_prot)).to(device)
        X_pep_ss = torch.from_numpy(np.array(X_pep_SS)).to(device)
        X_prot_ss = torch.from_numpy(np.array(X_prot_SS)).to(device)
        X_pep_2 = torch.from_numpy(np.array(X_pep_2)).to(device)
        X_prot_2 = torch.from_numpy(np.array(X_prot_2)).to(device)
        X_pep_dense = torch.from_numpy(np.array(X_dense_pep)).to(device)
        X_prot_dense = torch.from_numpy(np.array(X_dense_prot)).to(device)

        model.eval()
        preds = []
        pred=model(X_pep,X_prot,X_pep_ss,X_prot_ss,X_pep_2,X_prot_2,X_pep_dense,X_prot_dense)
        preds.extend(pred.detach().cpu().numpy().tolist())
        preds = np.array(preds)                
        return preds

    def binding_optimization(self, peptides: List[str], protein: str, binding_model, seed: int, n_attempts: int =100, **kwargs):
        """
        Optimized new peptides toward better binding score with the target protein based on input peptides
        @param peptides: peptides that form a template for further processing
        @param protein: target protein
        @param binding_model: predict the binding score of peptides to the target protein
        @param seed: random seed
        """
        set_seed(seed)

        block_size = len(peptides)
        padded_sequences = du_sequence.pad(du_sequence.to_one_hot(peptides)).reshape(-1, 25)
        # save peptides
        f = open('peptides_binding.fasta', 'w')
        for i in range(len(peptides)):
            f.write('>' + str(i)+'\n')
            f.write(peptides[i]+'\n')
        f.close()
        #  
        binding_score = self.binding_predict(peptides, protein, binding_model, 'binding')
        # compute the toxicity of peptides
        # Parameter initialization or assigning variable for command level arguments
        Sequence= 'peptides_binding.fasta'        # Input variable      
        # Threshold 
        Threshold= 0.38
        #------------------ Read input file ---------------------
        with open(Sequence) as f:
                records = f.read()
        records = records.split('>')[1:]
        seqid = []
        seq = []
        for fasta in records:
            array = fasta.split('\n')
            name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '', ''.join(array[1:]).upper())
            seqid.append(name)
            seq.append(sequence)
        if len(seqid) == 0:
            f=open(Sequence,"r")
            data1 = f.readlines()
            for each in data1:
                seq.append(each.replace('\n',''))
            for i in range (1,len(seq)+1):
                seqid.append("Seq_"+str(i))

        seqid_1 = list(map(">{}".format, seqid))
        CM = pd.concat([pd.DataFrame(seqid_1),pd.DataFrame(seq)],axis=1)
        CM.to_csv("Sequence_1",header=False,index=None,sep="\n")
        f.close()
        #======================= Prediction Module start from here =====================
        merci = 'tools/toxinpred3/merci/MERCI_motif_locator.pl'
        motifs_p = 'tools/toxinpred3/motifs/pos_motif.txt'
        motifs_n = 'tools/toxinpred3/motifs/neg_motif.txt'
        aac_comp(seq,'seq.aac')
        os.system("perl -pi -e 's/,$//g' seq.aac")
        dpc_comp(seq,'seq.dpc')
        os.system("perl -pi -e 's/,$//g' seq.dpc")
        prediction('seq.aac', 'seq.dpc', 'tools/toxinpred3/model/toxinpred3.0_model.pkl','seq.pred')
        os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_p + " -o merci_p.txt")
        os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_n + " -o merci_n.txt")
        MERCI_Processor_p('merci_p.txt','merci_output_p.csv',seqid)
        Merci_after_processing_p('merci_output_p.csv','merci_hybrid_p.csv')
        MERCI_Processor_n('merci_n.txt','merci_output_n.csv',seqid)
        Merci_after_processing_n('merci_output_n.csv','merci_hybrid_n.csv')
        hybrid('seq.pred',seqid,'merci_hybrid_p.csv','merci_hybrid_n.csv',Threshold,'final_output')
        df44 = pd.read_csv('final_output')
        df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
        df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
        toxic_score = df44['Hybrid Score'].tolist()
        os.remove('seq.aac')
        os.remove('seq.dpc')
        os.remove('seq.pred')
        os.remove('final_output')
        os.remove('merci_hybrid_p.csv')
        os.remove('merci_hybrid_n.csv')
        os.remove('merci_output_p.csv')
        os.remove('merci_output_n.csv')
        os.remove('merci_p.txt')
        os.remove('merci_n.txt')
        os.remove('Sequence_1')
        # save result
        f = open('results/PepZOO/affinity/result_binding.txt', 'a')
        f.write('original\n')
        f.write('ID'+'\t'+'sequence'+'\t'+'binding score'+'\t'+'toxic score'+'\n')
        for j in range(block_size):
            f.write(str(j+1)+'\t'+peptides[j]+'\t'+str(binding_score[j])+'\t'+str(toxic_score[j])+'\n')
        f.write('optimized\n')
        f.close()
        # embedding
        z = self._encoder.predict(padded_sequences, verbose=1, batch_size=80000)
        z = torch.from_numpy(z)
        # condition
        amp_condition = mic_condition = np.ones((len(z), 1))
        
        base_lr = kwargs['lr']
    
        beta = kwargs['beta']
        q = kwargs['Q']
        variance = kwargs['variance']
        
        adam = torch.optim.Adam([z], lr=base_lr)
        
        for i in range(n_attempts):
            
            u = np.random.normal(0, variance, size=(q, len(z), LATENT_DIM)).astype('float32')
            u = beta * (u / np.linalg.norm(u, axis=-1, keepdims=True))
            
            amp_conditions = mic_conditions = np.ones((q, len(z), 1))
            conditioned = np.concatenate((z+u, amp_conditions, mic_conditions), axis=-1).reshape(-1, LATENT_DIM+2)
            
            decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
            peps = np.argmax(decoded, axis=-1)
            peps = np.array([translate_peptide(x) for x in peps])
            
            f = open('peptides_binding.fasta', 'w')
            for j in range(len(peps)):
                f.write('>' + str(j)+'\n')
                f.write(peps[j]+'\n')
            f.close()
            
            binding_score_neighbors = self.binding_predict(peps, protein, binding_model, 'binding')
            # compute the zeroth order gradient
            grads = np.zeros((len(z), LATENT_DIM))
            for k in range(q):
                for j in range(len(z)):
                    grads[j] += beta * (float)(binding_score[j] - binding_score_neighbors[k*len(z)+j]) * u[k][j]
            z.grad = torch.from_numpy(grads.astype('float32'))
            # update embedding
            adam.step()

           
            conditioned = np.hstack([
                z,
                amp_condition,
                mic_condition,
            ])
            
            decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
            peps= np.argmax(decoded, axis=-1)        
            peps = np.array([translate_peptide(x) for x in peps])
            f = open('peptides_binding.fasta', 'w')
            for j in range(len(peps)):
                f.write('>' + str(j)+'\n')
                f.write(peps[j]+'\n')
            f.close()
            
            with open(Sequence) as f:
                    records = f.read()
            records = records.split('>')[1:]
            seqid = []
            seq = []
            for fasta in records:
                array = fasta.split('\n')
                name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '', ''.join(array[1:]).upper())
                seqid.append(name)
                seq.append(sequence)
            if len(seqid) == 0:
                f=open(Sequence,"r")
                data1 = f.readlines()
                for each in data1:
                    seq.append(each.replace('\n',''))
                for i in range (1,len(seq)+1):
                    seqid.append("Seq_"+str(i))

            seqid_1 = list(map(">{}".format, seqid))
            CM = pd.concat([pd.DataFrame(seqid_1),pd.DataFrame(seq)],axis=1)
            CM.to_csv("Sequence_1",header=False,index=None,sep="\n")
            f.close()
            #======================= Prediction Module start from here =====================
            merci = 'tools/toxinpred3/merci/MERCI_motif_locator.pl'
            motifs_p = 'tools/toxinpred3/motifs/pos_motif.txt'
            motifs_n = 'tools/toxinpred3/motifs/neg_motif.txt'
            aac_comp(seq,'seq.aac')
            os.system("perl -pi -e 's/,$//g' seq.aac")
            dpc_comp(seq,'seq.dpc')
            os.system("perl -pi -e 's/,$//g' seq.dpc")
            prediction('seq.aac', 'seq.dpc', 'tools/toxinpred3/model/toxinpred3.0_model.pkl','seq.pred')
            os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_p + " -o merci_p.txt")
            os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_n + " -o merci_n.txt")
            MERCI_Processor_p('merci_p.txt','merci_output_p.csv',seqid)
            Merci_after_processing_p('merci_output_p.csv','merci_hybrid_p.csv')
            MERCI_Processor_n('merci_n.txt','merci_output_n.csv',seqid)
            Merci_after_processing_n('merci_output_n.csv','merci_hybrid_n.csv')
            hybrid('seq.pred',seqid,'merci_hybrid_p.csv','merci_hybrid_n.csv',Threshold,'final_output')
            df44 = pd.read_csv('final_output')
            df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
            df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
            toxic_score = df44['Hybrid Score'].tolist()
            os.remove('seq.aac')
            os.remove('seq.dpc')
            os.remove('seq.pred')
            os.remove('final_output')
            os.remove('merci_hybrid_p.csv')
            os.remove('merci_hybrid_n.csv')
            os.remove('merci_output_p.csv')
            os.remove('merci_output_n.csv')
            os.remove('merci_p.txt')
            os.remove('merci_n.txt')
            os.remove('Sequence_1')
            
            binding_score = self.binding_predict(peps, protein, binding_model, 'binding')
            # save result
            f = open('results/PepZOO/affinity/result_binding.txt', 'a')
            f.write(f'iter{i+1}'+'\n')
            f.write('ID'+'\t' + 'sequence'+'\t' + 'binding score'+'\t' + 'toxic score'+'\n')
            for j in range(block_size):
                f.write(str(j+1) + '\t' + peps[j] + '\t' + str(binding_score[j]) + '\t' + str(toxic_score[j]) + '\n')
            f.close()

    def toxicity_optimization(self, peptides: List[str], protein: str, binding_model, seed: int, n_attempts: int =100, **kwargs):
        """
        Optimized new peptides toward lower toxicity based on input peptides
        @param peptides: peptides that form a template for further processing
        @param seed:
        @param kwargs:additional boolean arguments for zeroth order optimization. This include
        -
        """
        set_seed(seed)
        
        f = open('peptides_4_toxic.fasta', 'w')
        for i in range(len(peptides)):
            f.write('>' + str(i)+'\n')
            f.write(peptides[i]+'\n')
        f.close()
       
        block_size = len(peptides)
        padded_sequences = du_sequence.pad(du_sequence.to_one_hot(peptides)).reshape(-1, 25)

        # Parameter initialization or assigning variable for command level arguments
        Sequence= 'peptides_4_toxic.fasta'        # Input variable      
        # Threshold 
        Threshold= 0.38
        #------------------ Read input file ---------------------
        # f=open(Sequence,"r")
        # len1 = f.read().count('>')
        # f.close()
        with open(Sequence) as f:
            records = f.read()
        records = records.split('>')[1:]
        seqid = []
        seq = []
        for fasta in records:
            array = fasta.split('\n')
            name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '', ''.join(array[1:]).upper())
            seqid.append(name)
            seq.append(sequence)
        if len(seqid) == 0:
            f=open(Sequence,"r")
            data1 = f.readlines()
            for each in data1:
                seq.append(each.replace('\n',''))
            for i in range (1,len(seq)+1):
                seqid.append("Seq_"+str(i))

        seqid_1 = list(map(">{}".format, seqid))
        CM = pd.concat([pd.DataFrame(seqid_1),pd.DataFrame(seq)],axis=1)
        CM.to_csv("Sequence_1",header=False,index=None,sep="\n")
        f.close()
        #======================= Prediction Module start from here =====================
        merci = 'tools/toxinpred3/merci/MERCI_motif_locator.pl'
        motifs_p = 'tools/toxinpred3/motifs/pos_motif.txt'
        motifs_n = 'tools/toxinpred3/motifs/neg_motif.txt'
        aac_comp(seq,'seq.aac')
        os.system("perl -pi -e 's/,$//g' seq.aac")
        dpc_comp(seq,'seq.dpc')
        os.system("perl -pi -e 's/,$//g' seq.dpc")
        prediction('seq.aac', 'seq.dpc', 'tools/toxinpred3/model/toxinpred3.0_model.pkl','seq.pred')
        os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_p + " -o merci_p.txt")
        os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_n + " -o merci_n.txt")
        MERCI_Processor_p('merci_p.txt','merci_output_p.csv',seqid)
        Merci_after_processing_p('merci_output_p.csv','merci_hybrid_p.csv')
        MERCI_Processor_n('merci_n.txt','merci_output_n.csv',seqid)
        Merci_after_processing_n('merci_output_n.csv','merci_hybrid_n.csv')
        hybrid('seq.pred',seqid,'merci_hybrid_p.csv','merci_hybrid_n.csv',Threshold,'final_output')
        df44 = pd.read_csv('final_output')
        df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
        df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
        toxic_pre = df44['Hybrid Score'].tolist()
        os.remove('seq.aac')
        os.remove('seq.dpc')
        os.remove('seq.pred')
        os.remove('final_output')
        os.remove('merci_hybrid_p.csv')
        os.remove('merci_hybrid_n.csv')
        os.remove('merci_output_p.csv')
        os.remove('merci_output_n.csv')
        os.remove('merci_p.txt')
        os.remove('merci_n.txt')
        os.remove('Sequence_1')
        
        binding_score = self.binding_predict(peptides, protein, binding_model, 'toxicity')
        
        f = open('result_toxic.txt', 'a')
        f.write('original\n')
        f.write('ID'+'\t'+'Peptide'+'\t'+'toxic score'+'\t'+'binding score'+'\n')
        for j in range(block_size):
            f.write(str(j+1)+'\t'+peptides[j]+'\t'+str(toxic_pre[j])+'\t'+str(binding_score[j])+'\n')
        f.write('optimized\n')
        f.close()
        
        z = self._encoder.predict(padded_sequences, verbose=1, batch_size=80000)
        z = torch.from_numpy(z)
        
        amp_condition = mic_condition = np.ones((len(z), 1))
        
        base_lr = kwargs['lr']
        
        beta = kwargs['beta']
        q = kwargs['Q']
        variance = kwargs['variance']
        
        adam = torch.optim.Adam([z], lr=base_lr)
        
        for i in range(n_attempts):
            
            u = np.random.normal(0, variance, size=(q, len(z), LATENT_DIM)).astype('float32')
            
            u = beta * (u / np.linalg.norm(u, axis=-1, keepdims=True))
            
            amp_conditions = mic_conditions = np.ones((q, len(z), 1))
            conditioned = np.concatenate((z+u, amp_conditions, mic_conditions), axis=-1).reshape(-1, LATENT_DIM+2)
            
            decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
            peps = np.argmax(decoded, axis=-1)
            peps = np.array([translate_peptide(x) for x in peps])
            
            f = open('peptides_4_toxic.fasta', 'w')
            for j in range(len(peps)):
                f.write('>' + str(j)+'\n')
                f.write(peps[j]+'\n')
            f.close()
            
            #------------------ Read input file ---------------------
            with open(Sequence) as f:
                    records = f.read()
            records = records.split('>')[1:]
            seqid = []
            seq = []
            for fasta in records:
                array = fasta.split('\n')
                name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '', ''.join(array[1:]).upper())
                seqid.append(name)
                seq.append(sequence)
            if len(seqid) == 0:
                f=open(Sequence,"r")
                data1 = f.readlines()
                for each in data1:
                    seq.append(each.replace('\n',''))
                for i in range (1,len(seq)+1):
                    seqid.append("Seq_"+str(i))

            seqid_1 = list(map(">{}".format, seqid))
            CM = pd.concat([pd.DataFrame(seqid_1),pd.DataFrame(seq)],axis=1)
            CM.to_csv("Sequence_1",header=False,index=None,sep="\n")
            f.close()
            #======================= Prediction Module start from here =====================
            merci = 'tools/toxinpred3/merci/MERCI_motif_locator.pl'
            motifs_p = 'tools/toxinpred3/motifs/pos_motif.txt'
            motifs_n = 'tools/toxinpred3/motifs/neg_motif.txt'
            aac_comp(seq,'seq.aac')
            os.system("perl -pi -e 's/,$//g' seq.aac")
            dpc_comp(seq,'seq.dpc')
            os.system("perl -pi -e 's/,$//g' seq.dpc")
            prediction('seq.aac', 'seq.dpc', 'tools/toxinpred3/model/toxinpred3.0_model.pkl','seq.pred')
            os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_p + " -o merci_p.txt")
            os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_n + " -o merci_n.txt")
            MERCI_Processor_p('merci_p.txt','merci_output_p.csv',seqid)
            Merci_after_processing_p('merci_output_p.csv','merci_hybrid_p.csv')
            MERCI_Processor_n('merci_n.txt','merci_output_n.csv',seqid)
            Merci_after_processing_n('merci_output_n.csv','merci_hybrid_n.csv')
            hybrid('seq.pred',seqid,'merci_hybrid_p.csv','merci_hybrid_n.csv',Threshold,'final_output')
            df44 = pd.read_csv('final_output')
            df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
            df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
            toxic_neighbors = df44['Hybrid Score'].tolist()
            os.remove('seq.aac')
            os.remove('seq.dpc')
            os.remove('seq.pred')
            os.remove('final_output')
            os.remove('merci_hybrid_p.csv')
            os.remove('merci_hybrid_n.csv')
            os.remove('merci_output_p.csv')
            os.remove('merci_output_n.csv')
            os.remove('merci_p.txt')
            os.remove('merci_n.txt')
            os.remove('Sequence_1')
            
            grads = np.zeros((len(z), LATENT_DIM))
            for k in range(q):
                for j in range(len(z)):
                    grads[j] += beta * (float)(toxic_neighbors[k*len(z)+j] - toxic_pre[j]) * u[k][j]
            z.grad = torch.from_numpy(grads.astype('float32'))
            
            adam.step()

            
            conditioned = np.hstack([
                z,
                amp_condition,
                mic_condition,
            ])
            
            decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
            peps= np.argmax(decoded, axis=-1)        
            peps = np.array([translate_peptide(x) for x in peps])
            f = open('peptides_4_toxic.fasta', 'w')
            for j in range(len(peps)):
                f.write('>' + str(j)+'\n')
                f.write(peps[j]+'\n')
            f.close()

            
            with open(Sequence) as f:
                    records = f.read()
            records = records.split('>')[1:]
            seqid = []
            seq = []
            for fasta in records:
                array = fasta.split('\n')
                name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '', ''.join(array[1:]).upper())
                seqid.append(name)
                seq.append(sequence)
            if len(seqid) == 0:
                f=open(Sequence,"r")
                data1 = f.readlines()
                for each in data1:
                    seq.append(each.replace('\n',''))
                for i in range (1,len(seq)+1):
                    seqid.append("Seq_"+str(i))

            seqid_1 = list(map(">{}".format, seqid))
            CM = pd.concat([pd.DataFrame(seqid_1),pd.DataFrame(seq)],axis=1)
            CM.to_csv("Sequence_1",header=False,index=None,sep="\n")
            f.close()
            #======================= Prediction Module start from here =====================
            merci = 'tools/toxinpred3/merci/MERCI_motif_locator.pl'
            motifs_p = 'tools/toxinpred3/motifs/pos_motif.txt'
            motifs_n = 'tools/toxinpred3/motifs/neg_motif.txt'
            aac_comp(seq,'seq.aac')
            os.system("perl -pi -e 's/,$//g' seq.aac")
            dpc_comp(seq,'seq.dpc')
            os.system("perl -pi -e 's/,$//g' seq.dpc")
            prediction('seq.aac', 'seq.dpc', 'tools/toxinpred3/model/toxinpred3.0_model.pkl','seq.pred')
            os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_p + " -o merci_p.txt")
            os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_n + " -o merci_n.txt")
            MERCI_Processor_p('merci_p.txt','merci_output_p.csv',seqid)
            Merci_after_processing_p('merci_output_p.csv','merci_hybrid_p.csv')
            MERCI_Processor_n('merci_n.txt','merci_output_n.csv',seqid)
            Merci_after_processing_n('merci_output_n.csv','merci_hybrid_n.csv')
            hybrid('seq.pred',seqid,'merci_hybrid_p.csv','merci_hybrid_n.csv',Threshold,'final_output')
            df44 = pd.read_csv('final_output')
            df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
            df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
            toxic_pre = df44['Hybrid Score'].tolist()
            os.remove('seq.aac')
            os.remove('seq.dpc')
            os.remove('seq.pred')
            os.remove('final_output')
            os.remove('merci_hybrid_p.csv')
            os.remove('merci_hybrid_n.csv')
            os.remove('merci_output_p.csv')
            os.remove('merci_output_n.csv')
            os.remove('merci_p.txt')
            os.remove('merci_n.txt')
            os.remove('Sequence_1')

            
            binding_score = self.binding_predict(peps, protein, binding_model, 'toxicity')
            
            f = open('result_toxic.txt', 'a')
            f.write(f'iter{i+1}'+'\n')
            f.write('ID'+'\t'+'Peptide'+'\t'+'toxic score'+'\t'+'binding score'+'\n')
            for j in range(block_size):
                f.write(str(j+1) + '\t' + peps[j] +'\t' + str(toxic_pre[j]) + '\t' + str(binding_score[j]) + '\n')
            f.close()

    def binding_toxicity_optimization(self, peptides: List[str], protein: str, binding_model, seed: int, n_attempts: int =100, **kwargs):
        """
        Optimized new peptides toward better binding score with the target protein based on input peptides
        @param peptides: peptides that form a template for further processing
        @param seed:
        @param kwargs:additional boolean arguments for filtering. This include
        """
        
        set_seed(seed)
        block_size = len(peptides)
        padded_sequences = du_sequence.pad(du_sequence.to_one_hot(peptides)).reshape(-1, 25)

        f = open('peptides_binding_toxic.fasta', 'w')
        for i in range(len(peptides)):
            f.write('>' + str(i)+'\n')
            f.write(peptides[i]+'\n')
        f.close()

        binding_score = self.binding_predict(peptides, protein, binding_model, 'binding_toxicity')
        # Parameter initialization or assigning variable for command level arguments
        Sequence= 'peptides_binding_toxic.fasta'        # Input variable      
        # Threshold 
        Threshold= 0.38
        #------------------ Read input file ---------------------
        with open(Sequence) as f:
                records = f.read()
        records = records.split('>')[1:]
        seqid = []
        seq = []
        for fasta in records:
            array = fasta.split('\n')
            name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '', ''.join(array[1:]).upper())
            seqid.append(name)
            seq.append(sequence)
        if len(seqid) == 0:
            f=open(Sequence,"r")
            data1 = f.readlines()
            for each in data1:
                seq.append(each.replace('\n',''))
            for i in range (1,len(seq)+1):
                seqid.append("Seq_"+str(i))

        seqid_1 = list(map(">{}".format, seqid))
        CM = pd.concat([pd.DataFrame(seqid_1),pd.DataFrame(seq)],axis=1)
        CM.to_csv("Sequence_1",header=False,index=None,sep="\n")
        f.close()
        #======================= Prediction Module start from here =====================
        merci = 'tools/toxinpred3/merci/MERCI_motif_locator.pl'
        motifs_p = 'tools/toxinpred3/motifs/pos_motif.txt'
        motifs_n = 'tools/toxinpred3/motifs/neg_motif.txt'
        aac_comp(seq,'seq.aac')
        os.system("perl -pi -e 's/,$//g' seq.aac")
        dpc_comp(seq,'seq.dpc')
        os.system("perl -pi -e 's/,$//g' seq.dpc")
        prediction('seq.aac', 'seq.dpc', 'tools/toxinpred3/model/toxinpred3.0_model.pkl','seq.pred')
        os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_p + " -o merci_p.txt")
        os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_n + " -o merci_n.txt")
        MERCI_Processor_p('merci_p.txt','merci_output_p.csv',seqid)
        Merci_after_processing_p('merci_output_p.csv','merci_hybrid_p.csv')
        MERCI_Processor_n('merci_n.txt','merci_output_n.csv',seqid)
        Merci_after_processing_n('merci_output_n.csv','merci_hybrid_n.csv')
        hybrid('seq.pred',seqid,'merci_hybrid_p.csv','merci_hybrid_n.csv',Threshold,'final_output')
        df44 = pd.read_csv('final_output')
        df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
        df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
        toxic_pre = df44['Hybrid Score'].tolist()
        os.remove('seq.aac')
        os.remove('seq.dpc')
        os.remove('seq.pred')
        os.remove('final_output')
        os.remove('merci_hybrid_p.csv')
        os.remove('merci_hybrid_n.csv')
        os.remove('merci_output_p.csv')
        os.remove('merci_output_n.csv')
        os.remove('merci_p.txt')
        os.remove('merci_n.txt')
        os.remove('Sequence_1')
        
        f = open('result_binding_toxic.txt', 'a')
        f.write('original\n')
        f.write('ID'+'\t'+'sequence'+'\t'+'binding score'+'\t'+'toxicity'+'\n')
        for j in range(block_size):
            f.write(str(j+1)+'\t'+peptides[j]+'\t'+str(binding_score[j])+'\t'+str(toxic_pre[j])+'\n')
        f.write('optimized\n')
        f.close()
        
        z = self._encoder.predict(padded_sequences, verbose=1, batch_size=80000)
        z = torch.from_numpy(z)
        
        amp_condition = mic_condition = np.ones((len(z), 1))
        
        base_lr = kwargs['lr']
        
        beta = kwargs['beta']
        q = kwargs['Q']
        variance = kwargs['variance']
        
        adam = torch.optim.Adam([z], lr=base_lr)
        
        for i in range(n_attempts):
            
            u = np.random.normal(0, variance, size=(q, len(z), LATENT_DIM)).astype('float32')
            
            u = beta * (u / np.linalg.norm(u, axis=-1, keepdims=True))
            
            amp_conditions = mic_conditions = np.ones((q, len(z), 1))
            conditioned = np.concatenate((z+u, amp_conditions, mic_conditions), axis=-1).reshape(-1, LATENT_DIM+2)
            
            decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
            peps = np.argmax(decoded, axis=-1)
            peps = np.array([translate_peptide(x) for x in peps])
            
            f = open('peptides_binding_toxic.fasta', 'w')
            for j in range(len(peps)):
                f.write('>' + str(j)+'\n')
                f.write(peps[j]+'\n')
            f.close()
            
            #------------------ Read input file ---------------------
            with open(Sequence) as f:
                    records = f.read()
            records = records.split('>')[1:]
            seqid = []
            seq = []
            for fasta in records:
                array = fasta.split('\n')
                name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '', ''.join(array[1:]).upper())
                seqid.append(name)
                seq.append(sequence)
            if len(seqid) == 0:
                f=open(Sequence,"r")
                data1 = f.readlines()
                for each in data1:
                    seq.append(each.replace('\n',''))
                for i in range (1,len(seq)+1):
                    seqid.append("Seq_"+str(i))

            seqid_1 = list(map(">{}".format, seqid))
            CM = pd.concat([pd.DataFrame(seqid_1),pd.DataFrame(seq)],axis=1)
            CM.to_csv("Sequence_1",header=False,index=None,sep="\n")
            f.close()
            #======================= Prediction Module start from here =====================
            merci = 'tools/toxinpred3/merci/MERCI_motif_locator.pl'
            motifs_p = 'tools/toxinpred3/motifs/pos_motif.txt'
            motifs_n = 'tools/toxinpred3/motifs/neg_motif.txt'
            aac_comp(seq,'seq.aac')
            os.system("perl -pi -e 's/,$//g' seq.aac")
            dpc_comp(seq,'seq.dpc')
            os.system("perl -pi -e 's/,$//g' seq.dpc")
            prediction('seq.aac', 'seq.dpc', 'tools/toxinpred3/model/toxinpred3.0_model.pkl','seq.pred')
            os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_p + " -o merci_p.txt")
            os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_n + " -o merci_n.txt")
            MERCI_Processor_p('merci_p.txt','merci_output_p.csv',seqid)
            Merci_after_processing_p('merci_output_p.csv','merci_hybrid_p.csv')
            MERCI_Processor_n('merci_n.txt','merci_output_n.csv',seqid)
            Merci_after_processing_n('merci_output_n.csv','merci_hybrid_n.csv')
            hybrid('seq.pred',seqid,'merci_hybrid_p.csv','merci_hybrid_n.csv',Threshold,'final_output')
            df44 = pd.read_csv('final_output')
            df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
            df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
            toxic_neighbors = df44['Hybrid Score'].tolist()
            os.remove('seq.aac')
            os.remove('seq.dpc')
            os.remove('seq.pred')
            os.remove('final_output')
            os.remove('merci_hybrid_p.csv')
            os.remove('merci_hybrid_n.csv')
            os.remove('merci_output_p.csv')
            os.remove('merci_output_n.csv')
            os.remove('merci_p.txt')
            os.remove('merci_n.txt')
            os.remove('Sequence_1')
            
            binding_score_neighbor = self.binding_predict(peps, protein, binding_model, 'binding_toxicity')
            
            grads = np.zeros((len(z), LATENT_DIM))
            for k in range(q):
                for j in range(len(z)):
                    grads[j] += beta * (float)(binding_score[j] - binding_score_neighbor[k*len(z)+j] + toxic_neighbors[k*len(z)+j] - toxic_pre[j]) * u[k][j]
            z.grad = torch.from_numpy(grads.astype('float32'))
            
            adam.step()

        
            conditioned = np.hstack([
                z,
                amp_condition,
                mic_condition,
            ])
            
            decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
            peps= np.argmax(decoded, axis=-1)        
            peps = np.array([translate_peptide(x) for x in peps])
            f = open('peptides_binding_toxic.fasta', 'w')
            for j in range(len(peps)):
                f.write('>' + str(j)+'\n')
                f.write(peps[j]+'\n')
            f.close()
            
            with open(Sequence) as f:
                    records = f.read()
            records = records.split('>')[1:]
            seqid = []
            seq = []
            for fasta in records:
                array = fasta.split('\n')
                name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '', ''.join(array[1:]).upper())
                seqid.append(name)
                seq.append(sequence)
            if len(seqid) == 0:
                f=open(Sequence,"r")
                data1 = f.readlines()
                for each in data1:
                    seq.append(each.replace('\n',''))
                for i in range (1,len(seq)+1):
                    seqid.append("Seq_"+str(i))

            seqid_1 = list(map(">{}".format, seqid))
            CM = pd.concat([pd.DataFrame(seqid_1),pd.DataFrame(seq)],axis=1)
            CM.to_csv("Sequence_1",header=False,index=None,sep="\n")
            f.close()
            #======================= Prediction Module start from here =====================
            merci = 'tools/toxinpred3/merci/MERCI_motif_locator.pl'
            motifs_p = 'tools/toxinpred3/motifs/pos_motif.txt'
            motifs_n = 'tools/toxinpred3/motifs/neg_motif.txt'
            aac_comp(seq,'seq.aac')
            os.system("perl -pi -e 's/,$//g' seq.aac")
            dpc_comp(seq,'seq.dpc')
            os.system("perl -pi -e 's/,$//g' seq.dpc")
            prediction('seq.aac', 'seq.dpc', 'tools/toxinpred3/model/toxinpred3.0_model.pkl','seq.pred')
            os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_p + " -o merci_p.txt")
            os.system("perl " + merci + " -p " + "Sequence_1" +  " -i " + motifs_n + " -o merci_n.txt")
            MERCI_Processor_p('merci_p.txt','merci_output_p.csv',seqid)
            Merci_after_processing_p('merci_output_p.csv','merci_hybrid_p.csv')
            MERCI_Processor_n('merci_n.txt','merci_output_n.csv',seqid)
            Merci_after_processing_n('merci_output_n.csv','merci_hybrid_n.csv')
            hybrid('seq.pred',seqid,'merci_hybrid_p.csv','merci_hybrid_n.csv',Threshold,'final_output')
            df44 = pd.read_csv('final_output')
            df44.loc[df44['Hybrid Score'] > 1, 'Hybrid Score'] = 1
            df44.loc[df44['Hybrid Score'] < 0, 'Hybrid Score'] = 0
            toxic_pre = df44['Hybrid Score'].tolist()
            os.remove('seq.aac')
            os.remove('seq.dpc')
            os.remove('seq.pred')
            os.remove('final_output')
            os.remove('merci_hybrid_p.csv')
            os.remove('merci_hybrid_n.csv')
            os.remove('merci_output_p.csv')
            os.remove('merci_output_n.csv')
            os.remove('merci_p.txt')
            os.remove('merci_n.txt')
            os.remove('Sequence_1')
            
            binding_score = self.binding_predict(peps, protein, binding_model, 'binding_toxicity')
            
            f = open('results/PepZOO/affinity_toxicity/result_binding_toxic.txt', 'a')
            f.write(f'iter{i+1}'+'\n')
            f.write('ID'+'\t'+'sequence'+'\t'+'binding score'+'\t'+'toxicity'+'\n')
            for j in range(block_size):
                f.write(str(j+1) + '\t' + peps[j] + '\t' + str(binding_score[j]) + '\t' + str(toxic_pre[j])+'\n')
            f.close()

    def cvae_generation(self, sequences: List[str], seed: int,
                            n_attempts: int = 100, temp: float = 5.0, **kwargs) -> Dict[Any, Dict[str, Any]]:
        """
        Generates new peptides based on input sequences
        @param sequences: peptides that form a template for further processing
        @param n_attempts: how many times a single latent vector is decoded - for normal Softmax models it should be set
        to 1 as every decoding call returns the same peptide.
        @param temp: creativity parameter. Controls latent vector sigma scaling
        @param seed: random seed
        """
        set_seed(seed)

        block_size = len(sequences)
        padded_sequences = du_sequence.pad(du_sequence.to_one_hot(sequences))
        sigmas = self.get_sigma(padded_sequences)
        amp_org = self._amp_classifier.predict(padded_sequences, verbose=1, batch_size=80000)
        mic_org = self._mic_classifier.predict(padded_sequences, verbose=1, batch_size=80000)
        padded_sequences = np.vstack([padded_sequences] * n_attempts).reshape(-1, 25)
        z = self._encoder.predict(padded_sequences, verbose=1, batch_size=80000)
        amp_stacked = np.vstack([amp_org] * n_attempts)
        mic_stacked = np.vstack([mic_org] * n_attempts)
        props = calculate_physchem_prop(sequences)
        # save result
        basic_wf = open('results/CVAE/result.csv', 'a', encoding='utf-8')
        basic_writer = csv.writer(basic_wf)
        basic_writer.writerow(['description','sequence', 'amp', 'mic', 'length', 'hydrophobicity',
                            'hydrophobic_moment', 'charge', 'isoelectric_point'])
        for i in range(block_size):
            basic_writer.writerow([
                f'{i}_original',
                sequences[i], 
                amp_org[i][0], 
                mic_org[i][0], 
                props['length'][i],
                props['hydrophobicity'][i], 
                props['hydrophobic_moment'][i], 
                props['charge'][i], 
                props['isoelectric_point'][i]
            ])

        noise = np.random.normal(loc=0, scale=temp * np.vstack([sigmas] * n_attempts), size=z.shape)
        encoded = z + noise

        amp_condition = mic_condition = np.ones((len(padded_sequences), 1))

        conditioned = np.hstack([
            encoded,
            amp_condition,
            mic_condition,
        ])
        decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
        new_peptides = np.argmax(decoded, axis=2)

        new_amp = self._amp_classifier.predict(new_peptides, verbose=1, batch_size=80000)
        new_mic = self._mic_classifier.predict(new_peptides, verbose=1, batch_size=80000)

        better = new_amp > amp_stacked.reshape(-1, 1)
        better = better & (new_mic > mic_stacked.reshape(-1, 1))

        better = better.flatten()

        new_peptides = np.array([translate_peptide(x) for x in new_peptides])
        new_peptides, new_amp, new_mic, better = slice_blocks((new_peptides, new_amp, new_mic, better), block_size)
        # mask = get_filtering_mask(sequences=new_peptides, filtering_options=kwargs)
        # mask &= better
                
        filtered_peptides = _dispose_into_bucket(better, sequences, new_peptides, new_amp, new_mic, n_attempts,
                                                 block_size)
        filtered_peptides = self._encapsulate_sequential_results(filtered_peptides)

        for i in range(block_size):
            if filtered_peptides[i] != None:
                for j in range(len(filtered_peptides[i])):
                    basic_writer.writerow([
                        f'{i}_CVAE',
                        filtered_peptides[i][j]['sequence'],
                        filtered_peptides[i][j]['amp'],
                        filtered_peptides[i][j]['mic'], 
                        filtered_peptides[i][j]['length'], 
                        filtered_peptides[i][j]['hydrophobicity'], 
                        filtered_peptides[i][j]['hydrophobic_moment'], 
                        filtered_peptides[i][j]['charge'], 
                        filtered_peptides[i][j]['isoelectric_point']
                    ])
        basic_wf.close()

    def pepcvae_generation(self, sequences: List[str], seed: int,
                            n_attempts: int = 100, temp: float = 5.0, **kwargs) -> Dict[Any, Dict[str, Any]]:
        """
        Generates new peptides based on input sequences
        @param sequences: peptides that form a template for further processing
        @param n_attempts: how many times a single latent vector is decoded - for normal Softmax models it should be set
        to 1 as every decoding call returns the same peptide.
        @param temp: creativity parameter. Controls latent vector sigma scaling
        @param seed: random seed
        """
        set_seed(seed)

        block_size = len(sequences)
        padded_sequences = du_sequence.pad(du_sequence.to_one_hot(sequences))
        sigmas = self.get_sigma(padded_sequences)
        amp_org = self._amp_classifier.predict(padded_sequences, verbose=1, batch_size=80000)
        mic_org = self._mic_classifier.predict(padded_sequences, verbose=1, batch_size=80000)
        padded_sequences = np.vstack([padded_sequences] * n_attempts).reshape(-1, 25)
        z = self._encoder.predict(padded_sequences, verbose=1, batch_size=80000)
        amp_stacked = np.vstack([amp_org] * n_attempts)
        mic_stacked = np.vstack([mic_org] * n_attempts)
        props = calculate_physchem_prop(sequences)
        # save result
        pepcvae_wf = open('results/PepCVAE/result.csv', 'a', encoding='utf-8')
        pepcvae_writer = csv.writer(pepcvae_wf)
        pepcvae_writer.writerow(['description','sequence', 'amp', 'mic', 'length', 'hydrophobicity',
                            'hydrophobic_moment', 'charge', 'isoelectric_point'])
        for i in range(block_size):
            pepcvae_writer.writerow([
                f'{i}_original',
                sequences[i], 
                amp_org[i][0], 
                mic_org[i][0], 
                props['length'][i],
                props['hydrophobicity'][i], 
                props['hydrophobic_moment'][i], 
                props['charge'][i], 
                props['isoelectric_point'][i]
            ])

        noise = np.random.normal(loc=0, scale=temp * np.vstack([sigmas] * n_attempts), size=z.shape)
        encoded = z + noise

        amp_condition = mic_condition = np.ones((len(padded_sequences), 1))

        conditioned = np.hstack([
            encoded,
            amp_condition,
            mic_condition,
        ])
        decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
        new_peptides = np.argmax(decoded, axis=2)

        new_amp = self._amp_classifier.predict(new_peptides, verbose=1, batch_size=80000)
        new_mic = self._mic_classifier.predict(new_peptides, verbose=1, batch_size=80000)

        better = new_amp > amp_stacked.reshape(-1, 1)
        better = better & (new_mic > mic_stacked.reshape(-1, 1))

        better = better.flatten()

        new_peptides = np.array([translate_peptide(x) for x in new_peptides])
        new_peptides, new_amp, new_mic, better = slice_blocks((new_peptides, new_amp, new_mic, better), block_size)
        # mask = get_filtering_mask(sequences=new_peptides, filtering_options=kwargs)
        # mask &= better
        filtered_peptides = _dispose_into_bucket(better, sequences, new_peptides, new_amp, new_mic, n_attempts,
                                                 block_size)
        filtered_peptides = self._encapsulate_sequential_results(filtered_peptides)

        for i in range(block_size):
            if filtered_peptides[i] != None:
                for j in range(len(filtered_peptides[i])):
                    pepcvae_writer.writerow([
                        f'{i}_PepCVAE',
                        filtered_peptides[i][j]['sequence'],
                        filtered_peptides[i][j]['amp'],
                        filtered_peptides[i][j]['mic'], 
                        filtered_peptides[i][j]['length'], 
                        filtered_peptides[i][j]['hydrophobicity'], 
                        filtered_peptides[i][j]['hydrophobic_moment'], 
                        filtered_peptides[i][j]['charge'], 
                        filtered_peptides[i][j]['isoelectric_point']
                    ])

        pepcvae_wf.close()
        # generation_result = {
        #     'sequence': sequences,
        #     'amp': amp_org.flatten().tolist(),
        #     'mic': mic_org.flatten().tolist(),
        #     'generated_sequences': filtered_peptides
        # }
        # generation_result.update(calculate_physchem_prop(sequences))
        # return self._transpose_sequential_results(generation_result)