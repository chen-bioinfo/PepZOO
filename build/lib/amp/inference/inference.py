import itertools
from functools import reduce
from typing import Any, Callable, Dict, Iterable, List, Literal, Tuple, Union

import joblib
import numpy as np
import numpy.ma as ma
import pandas as pd
from keras import layers, models
from sklearn.decomposition import PCA
from tqdm import tqdm
import csv
import os
import torch
from time import time

from amp.config import LATENT_DIM
from amp.data_utils import sequence as du_sequence
from amp.inference.filtering import get_filtering_mask
from amp.utils.basic_model_serializer import load_master_model_components
from amp.utils.generate_peptides import translate_peptide
from amp.utils.phys_chem_propterties import calculate_physchem_prop
from amp.utils.seed import set_seed


def _get_comb_iterator(means: List[float], stds: List[float]) -> Iterable:
    iterator = itertools.product(means, stds)
    next(iterator)
    return iterator


def _unroll_to_batch(a: np.array, batch_size: int, combinations: int, attempts: int) -> np.array:
    return a.reshape(batch_size, combinations, attempts, -1)


def _apply_along_axis_improved(func: Callable[[np.ndarray, int], np.ndarray], axis: int, arr: np.array, ):
    return np.array([func(v, i) for i, v in enumerate(np.rollaxis(arr, axis))])


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

    def amp_mic_optimization(self, sequences: List[str], seed: int,
                            filtering_criteria: Literal['improvement', 'discovery'] = 'improvement',
                            n_attempts: int = 100, temp: float = 5.0, **kwargs) -> Dict[Any, Dict[str, Any]]:
        """
        Generates new peptides based on input sequences
        @param sequences: peptides that form a template for further processing
        @param filtering_criteria: 'improvement' if generated peptides should be strictly better than input sequences
        'discovery' if generated sequences should be good enough but not strictly better
        @param n_attempts: how many times a single latent vector is decoded - for normal Softmax models it should be set
        to 1 as every decoding call returns the same peptide.
        @param temp: creativity parameter. Controls latent vector sigma scaling
        @param seed:
        @param kwargs:additional boolean arguments for filtering. This include
        - filter_positive_clusters
        - filter_repetitive_clusters or filter_hydrophobic_clusters
        - filter_cysteins
        - filter_known_amps

        @return: dict, each key corresponds to a single input sequence.
        """
        # 设置随机种子
        set_seed(seed)
        
        filtering_criteria = filtering_criteria.strip().lower()
        assert filtering_criteria == 'improvement' or filtering_criteria == 'discovery', \
            "Unrecognised filtering constraint"

        # 序列数目
        block_size = len(sequences)

        # 补全序列
        padded_sequences = du_sequence.pad(du_sequence.to_one_hot(sequences))
        # 方差
        sigmas = self.get_sigma(padded_sequences)
        # 原始序列的amp值
        amp_org = self._amp_classifier.predict(padded_sequences, verbose=1, batch_size=80000)
        # 原始序列的mic值
        mic_org = self._mic_classifier.predict(padded_sequences, verbose=1, batch_size=80000)
        # 将每条序列复制n_attempts条
        padded_sequences = np.vstack([padded_sequences] * n_attempts).reshape(-1, 25)
        # 中间编码
        z = self._encoder.predict(padded_sequences, verbose=1, batch_size=80000)

        amp_stacked = np.vstack([amp_org] * n_attempts)
        mic_stacked = np.vstack([mic_org] * n_attempts)
        # 理化性质
        props = calculate_physchem_prop(sequences)

        # save result
        f = open('result/hydramp/result.csv', 'a', encoding='utf-8')
        writer = csv.writer(f)
        writer.writerow(['description','sequence', 'amp', 'mic', 'length', 'hydrophobicity',
                            'hydrophobic_moment', 'charge', 'isoelectric_point'])
        for i in range(block_size):
            writer.writerow([
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
        # 生成扰动向量
        noise = np.random.normal(loc=0, scale=temp * np.vstack([sigmas] * n_attempts), size=z.shape)
        # 编码向量=原始中间向量+扰动
        encoded = z + noise
        # 控制条件（amp值和mic值）
        amp_condition = mic_condition = np.ones((len(padded_sequences), 1))
        # 将编码和控制条件拼接
        conditioned = np.hstack([
            encoded,
            amp_condition,
            mic_condition,
        ])
        # 解码
        decoded = self._decoder.predict(conditioned, verbose=1, batch_size=80000)
        new_peptides = np.argmax(decoded, axis=2)
        # 计算amp值
        new_amp = self._amp_classifier.predict(new_peptides, verbose=1, batch_size=80000)
        # 计算mic值
        new_mic = self._mic_classifier.predict(new_peptides, verbose=1, batch_size=80000)
        # 筛选规则为improvement：生成序列的amp值和mic值高于原始序列的amp值和mic值
        if filtering_criteria == 'improvement':
            better = new_amp > amp_stacked.reshape(-1, 1)
            better = better & (new_mic > mic_stacked.reshape(-1, 1))
        else:   # 筛选规则为discovery：生成序列的amp值 >= 0.8，mic值 > 0.5
            better = new_amp >= 0.8
            better = better & (new_mic > 0.5)

        better = better.flatten()
        # 将数字表示的序列还原为字符表示的序列
        new_peptides = np.array([translate_peptide(x) for x in new_peptides])
        # 过滤
        new_peptides, new_amp, new_mic, better = slice_blocks((new_peptides, new_amp, new_mic, better), block_size)
        # mask = get_filtering_mask(sequences=new_peptides, filtering_options=kwargs)
        # mask &= better
        filtered_peptides = _dispose_into_bucket(better, sequences, new_peptides, new_amp, new_mic, n_attempts,
                                                 block_size)
        filtered_peptides = self._encapsulate_sequential_results(filtered_peptides)
        # 保存结果
        for i in range(block_size):
            if filtered_peptides[i] != None:
                for j in range(len(filtered_peptides[i])):
                    writer.writerow([
                        f'{i}_HydrampOpt',
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

            if filtering_criteria == 'improvement':
                better = new_amp > amp_org
                better = better & (new_mic > mic_org)
            else:
                better = new >= 0.8
                better = better & (new_mic > 0.5)
            better = better.flatten()

            new_peptides, new_amp, new_mic, better = slice_blocks((seq, new_amp, new_mic, better), block_size)
            filtered_peptides = _dispose_into_bucket(better, sequences, new_peptides, new_amp, new_mic, 1, block_size)
            filtered_peptides = self._encapsulate_sequential_results(filtered_peptides)      
              
            for i in range(block_size):
                if filtered_peptides[i] != None:
                    if filtered_peptides[i][0]['sequence'] not in seqs_dic[i].keys():
                        writer.writerow([
                            f'{i}_HydrampZeroOpt',
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
        f.close()

    def binding_optimization(self, peptides: List[str], seed: int, **kwargs) -> Dict[Any, Dict[str, Any]]:
        """
        Optimized new peptides toward better binding score with the target protein based on input peptides
        @param peptides: peptides that form a template for further processing
        @param seed:
        @param kwargs:additional boolean arguments for filtering. This include
        - filter_positive_clusters
        - filter_repetitive_clusters or filter_hydrophobic_clusters
        - filter_cysteins
        - filter_known_amps

        @return: dict, each key corresponds to a single input peptide.
        """
        # 设置随机种子
        set_seed(seed)
        # 序列数目
        block_size = len(peptides)
        with open('amps.fasta', 'w') as wf:
            for i in range(block_size):
                wf.write(f'>{i}')
                wf.write(peptides[i])

        os.system("/geniusland/home/liuxianliang1/SCRATCH-1D_1.2/bin/run_SCRATCH-1D_predictors.sh\
             amps.fasta test.out 4")


    def basic_analogue_generation(self, sequences: List[str], seed: int,
                            filtering_criteria: Literal['improvement', 'discovery'] = 'improvement',
                            n_attempts: int = 100, temp: float = 5.0, **kwargs) -> Dict[Any, Dict[str, Any]]:
        """
        Generates new peptides based on input sequences
        @param sequences: peptides that form a template for further processing
        @param filtering_criteria: 'improvement' if generated peptides should be strictly better than input sequences
        'discovery' if generated sequences should be good enough but not strictly better
        @param n_attempts: how many times a single latent vector is decoded - for normal Softmax models it should be set
        to 1 as every decoding call returns the same peptide.
        @param temp: creativity parameter. Controls latent vector sigma scaling
        @param seed:
        @param kwargs:additional boolean arguments for filtering. This include
        - filter_positive_clusters
        - filter_repetitive_clusters or filter_hydrophobic_clusters
        - filter_cysteins
        - filter_known_amps

        @return: dict, each key corresponds to a single input sequence.
        """
        set_seed(seed)
        filtering_criteria = filtering_criteria.strip().lower()
        assert filtering_criteria == 'improvement' or filtering_criteria == 'discovery', \
            "Unrecognised filtering constraint"

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
        f = open('result/basic/result.csv', 'a', encoding='utf-8')
        writer = csv.writer(f)
        writer.writerow(['description','sequence', 'amp', 'mic', 'length', 'hydrophobicity',
                            'hydrophobic_moment', 'charge', 'isoelectric_point'])
        for i in range(block_size):
            writer.writerow([
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

        if filtering_criteria == 'improvement':
            better = new_amp > amp_stacked.reshape(-1, 1)
            better = better & (new_mic > mic_stacked.reshape(-1, 1))
        else:
            better = new_amp >= 0.8
            better = better & (new_mic > 0.5)

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
                    writer.writerow([
                        f'{i}_BasicOpt',
                        filtered_peptides[i][j]['sequence'],
                        filtered_peptides[i][j]['amp'],
                        filtered_peptides[i][j]['mic'], 
                        filtered_peptides[i][j]['length'], 
                        filtered_peptides[i][j]['hydrophobicity'], 
                        filtered_peptides[i][j]['hydrophobic_moment'], 
                        filtered_peptides[i][j]['charge'], 
                        filtered_peptides[i][j]['isoelectric_point']
                    ])

        # generation_result = {
        #     'sequence': sequences,
        #     'amp': amp_org.flatten().tolist(),
        #     'mic': mic_org.flatten().tolist(),
        #     'generated_sequences': filtered_peptides
        # }
        # generation_result.update(calculate_physchem_prop(sequences))
        # return self._transpose_sequential_results(generation_result)

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

            if filtering_criteria == 'improvement':
                better = new_amp > amp_org
                better = better & (new_mic > mic_org)
            else:
                better = new >= 0.8
                better = better & (new_mic > 0.5)
            better = better.flatten()

            new_peptides, new_amp, new_mic, better = slice_blocks((seq, new_amp, new_mic, better), block_size)
            filtered_peptides = _dispose_into_bucket(better, sequences, new_peptides, new_amp, new_mic, 1, block_size)
            filtered_peptides = self._encapsulate_sequential_results(filtered_peptides)      
              
            for i in range(block_size):
                if filtered_peptides[i] != None:
                    if filtered_peptides[i][0]['sequence'] not in seqs_dic[i].keys():
                        writer.writerow([
                            f'{i}_BasicZeroOpt',
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
        f.close()

    def pepcvae_analogue_generation(self, sequences: List[str], seed: int,
                            filtering_criteria: Literal['improvement', 'discovery'] = 'improvement',
                            n_attempts: int = 100, temp: float = 5.0, **kwargs) -> Dict[Any, Dict[str, Any]]:
        """
        Generates new peptides based on input sequences
        @param sequences: peptides that form a template for further processing
        @param filtering_criteria: 'improvement' if generated peptides should be strictly better than input sequences
        'discovery' if generated sequences should be good enough but not strictly better
        @param n_attempts: how many times a single latent vector is decoded - for normal Softmax models it should be set
        to 1 as every decoding call returns the same peptide.
        @param temp: creativity parameter. Controls latent vector sigma scaling
        @param seed:
        @param kwargs:additional boolean arguments for filtering. This include
        - filter_positive_clusters
        - filter_repetitive_clusters or filter_hydrophobic_clusters
        - filter_cysteins
        - filter_known_amps

        @return: dict, each key corresponds to a single input sequence.
        """
        set_seed(seed)
        filtering_criteria = filtering_criteria.strip().lower()
        assert filtering_criteria == 'improvement' or filtering_criteria == 'discovery', \
            "Unrecognised filtering constraint"

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
        f = open('result/pepcvae/result.csv', 'a', encoding='utf-8')
        writer = csv.writer(f)
        writer.writerow(['description','sequence', 'amp', 'mic', 'length', 'hydrophobicity',
                            'hydrophobic_moment', 'charge', 'isoelectric_point'])
        for i in range(block_size):
            writer.writerow([
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

        if filtering_criteria == 'improvement':
            better = new_amp > amp_stacked.reshape(-1, 1)
            better = better & (new_mic > mic_stacked.reshape(-1, 1))
        else:
            better = new_amp >= 0.8
            better = better & (new_mic > 0.5)

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
                    writer.writerow([
                        f'{i}_PepCVAEOpt',
                        filtered_peptides[i][j]['sequence'],
                        filtered_peptides[i][j]['amp'],
                        filtered_peptides[i][j]['mic'], 
                        filtered_peptides[i][j]['length'], 
                        filtered_peptides[i][j]['hydrophobicity'], 
                        filtered_peptides[i][j]['hydrophobic_moment'], 
                        filtered_peptides[i][j]['charge'], 
                        filtered_peptides[i][j]['isoelectric_point']
                    ])

        # generation_result = {
        #     'sequence': sequences,
        #     'amp': amp_org.flatten().tolist(),
        #     'mic': mic_org.flatten().tolist(),
        #     'generated_sequences': filtered_peptides
        # }
        # generation_result.update(calculate_physchem_prop(sequences))
        # return self._transpose_sequential_results(generation_result)

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

            if filtering_criteria == 'improvement':
                better = new_amp > amp_org
                better = better & (new_mic > mic_org)
            else:
                better = new >= 0.8
                better = better & (new_mic > 0.5)
            better = better.flatten()

            new_peptides, new_amp, new_mic, better = slice_blocks((seq, new_amp, new_mic, better), block_size)
            filtered_peptides = _dispose_into_bucket(better, sequences, new_peptides, new_amp, new_mic, 1, block_size)
            filtered_peptides = self._encapsulate_sequential_results(filtered_peptides)      
              
            for i in range(block_size):
                if filtered_peptides[i] != None:
                    if filtered_peptides[i][0]['sequence'] not in seqs_dic[i].keys():
                        writer.writerow([
                            f'{i}_PepCVAEZeroOpt',
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
        f.close()