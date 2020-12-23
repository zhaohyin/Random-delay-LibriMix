import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm

# Global parameters
# eps secures log and division
EPS = 1e-10
# max amplitude in sources and mixtures
MAX_AMP = 0.9
# In LibriSpeech all the sources are at 16K Hz
RATE = 16000
# We will randomize loudness between this range
MIN_LOUDNESS = -33
MAX_LOUDNESS = -25

# A random seed is used for reproducibility
random.seed(72)

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--librispeech_dir', type=str, required=True,
                    help='Path to librispeech root directory')
parser.add_argument('--librispeech_md_dir', type=str, required=True,
                    help='Path to librispeech metadata directory')
parser.add_argument('--metadata_outdir', type=str, default=None,
                    help='Where librimix metadata files will be stored.')
parser.add_argument('--n_src', type=int, required=True,
                    help='Number of sources desired to create the mixture')

# 获取不同语音组合的方式
def main(args):
    librispeech_dir = args.librispeech_dir    # librispeech_dir = ./LibriSpeech
    librispeech_md_dir = args.librispeech_md_dir  # librispeech_md_dir = ./LibriSpeech/metadata
    #
    n_src = args.n_src  # n_src = 2
    # Create Librimix metadata directory
    md_dir = args.metadata_outdir   # metadata_outdir = ./mixDataInfo
    if md_dir is None:
        root = os.path.dirname(librispeech_dir)
        md_dir = os.path.join(root, f'Libri{n_src}Mix_metadata')   # 生成文件,存放混合数据的混合方式
    os.makedirs(md_dir, exist_ok=True)
    create_librimix_metadata(librispeech_dir, librispeech_md_dir, md_dir, n_src)

def create_librimix_metadata(librispeech_dir, librispeech_md_dir, md_dir, n_src):
    """ Generate LibriMix metadata according to LibriSpeech metadata """
    dataset = f'libri{n_src}mix'      # 确定数据集的名字
    # Dataset name
    # List metadata files in LibriSpeech
    librispeech_md_files = os.listdir(librispeech_md_dir)
    # Go through each metadata file and create metadata accordingly
    for librispeech_md_file in librispeech_md_files:
        if not librispeech_md_file.endswith('.csv'):
            print(f"{librispeech_md_file} is not a csv file, continue.")
            continue
        # Open .csv files from LibriSpeech
        librispeech_md = pd.read_csv(os.path.join(
            librispeech_md_dir, librispeech_md_file), engine='python')   # ./LibriSpeech/metadata/test-clean.csv
        #
        # Filenames
        save_path = os.path.join(md_dir,
                                 '_'.join([dataset, librispeech_md_file]))
        info_name = '_'.join([dataset, librispeech_md_file.strip('.csv'),
                              'info']) + '.csv'
        info_save_path = os.path.join(md_dir, info_name)
        print(f"Creating {os.path.basename(save_path)} file in {md_dir}")
        #
        # Create dataframe
        mixtures_md, mixtures_info = create_librimix_df(librispeech_md, librispeech_dir, n_src)
        #
        # Round number of files
        mixtures_md = mixtures_md[:len(mixtures_md) // 100 * 100]
        mixtures_info = mixtures_info[:len(mixtures_info) // 100 * 100]

        # Save csv files
        mixtures_md.to_csv(save_path, index=False)
        mixtures_info.to_csv(info_save_path, index=False)


def create_librimix_df(librispeech_md_file, librispeech_dir, n_src):
    """ Generate librimix dataframe from a LibriSpeech and wha md file"""
    # Create a dataframe that will be used to generate sources and mixtures
    mixtures_md = pd.DataFrame(columns=['mixture_ID'])
    # Create a dataframe with additional infos.
    mixtures_info = pd.DataFrame(columns=['mixture_ID'])
    # Add columns (depends on the number of sources)
    for i in range(n_src):
        mixtures_md[f"source_{i + 1}_path"] = {}      # mixtures_md : ID_source1Path_source1Gain_source2Path_source2Gain 1292*5
        mixtures_md[f"source_{i + 1}_gain"] = {}
        mixtures_info[f"speaker_{i + 1}_ID"] = {}
        mixtures_info[f"speaker_{i + 1}_sex"] = {}
    #
    # Generate pairs of sources to mix
    pairs= set_pairs(librispeech_md_file, n_src)                         # 1292 dimention
    clip_counter = 0
    # For each combination create a new line in the dataframe
    for pair in tqdm(pairs, total=len(pairs)):                          # zip表示打包成元组
        # return infos about the sources, generate sources
        sources_info, sources_list_max = read_sources(
            librispeech_md_file, pair, n_src, librispeech_dir)
        # compute initial loudness, randomize loudness and normalize sources
        loudness, _, sources_list_norm = set_loudness(sources_list_max)
        # Do the mixture
        mixture_max = mix(sources_list_norm)
        # Check the mixture for clipping and renormalize if necessary
        renormalize_loudness, did_clip = check_for_cliping(mixture_max,
                                                           sources_list_norm)
        clip_counter += int(did_clip)
        # Compute gain
        gain_list = compute_gain(loudness, renormalize_loudness)

        # Add information to the dataframe
        row_mixture, row_info = get_row(sources_info, gain_list, n_src)
        mixtures_md.loc[len(mixtures_md)] = row_mixture      # 多添加了一个gain
        mixtures_info.loc[len(mixtures_info)] = row_info
    print(f"Among {len(mixtures_md)} mixtures, {clip_counter} clipped.")
    return mixtures_md, mixtures_info

def set_pairs(librispeech_md_file, n_src):
    """ set pairs of sources to make the mixture """
    # Initialize list for pairs sources
    utt_pairs = []
    # In train sets utterance are only used once
    if 'train' in librispeech_md_file.iloc[0]['subset']:
        utt_pairs = set_utt_pairs(librispeech_md_file, utt_pairs, n_src)
    else:
        while len(utt_pairs) < 3000:
            utt_pairs = set_utt_pairs(librispeech_md_file, utt_pairs, n_src)
            utt_pairs= remove_duplicates(utt_pairs)
        utt_pairs = utt_pairs[:3000]
    print(len(utt_pairs))
    return utt_pairs


def set_utt_pairs(librispeech_md_file, pair_list, n_src):
    # A counter
    c = 0
    # Index of the rows in the metadata file
    index = list(range(len(librispeech_md_file)))

    # Try to create pairs with different speakers end after 200 fails
    while len(index) >= n_src and c < 200:
        couple = random.sample(index, n_src)
        # Check that speakers are different
        speaker_list = set([librispeech_md_file.iloc[couple[i]]['speaker_ID']
                            for i in range(n_src)])
        # If there are duplicates then increment the counter
        if len(speaker_list) != n_src:
            c += 1
        # Else append the combination to pair_list and erase the combination
        # from the available indexes
        else:
            for i in range(n_src):
                index.remove(couple[i])      # 删除已有的语音
            pair_list.append(couple)
            c = 0
    return pair_list    # [[1,2],[4,5]....]

def remove_duplicates(utt_pairs):
    print('Removing duplicates')
    # look for identical mixtures O(n²)
    for i, (pair) in enumerate(zip(utt_pairs)):
        for j, (du_pair) in enumerate(
                zip(utt_pairs)):
            # sort because [s1,s2] = [s2,s1]
            if sorted(pair) == sorted(du_pair) and i != j:
                utt_pairs.remove(du_pair)
    return utt_pairs


def read_sources(metadata_file, pair, n_src, librispeech_dir):
    # Read lines corresponding to pair
    sources = [metadata_file.iloc[pair[i]] for i in range(n_src)]   # Speakers Info
    # Get sources info
    speaker_id_list = [source['speaker_ID'] for source in sources]
    sex_list = [source['sex'] for source in sources]
    length_list = [source['length'] for source in sources]
    path_list = [source['origin_path'] for source in sources]
    id_l = [os.path.split(source['origin_path'])[1].strip('.flac')
            for source in sources]
    mixtures_id = "_".join(id_l)

    # Get the longest and shortest source len
    max_length = max(length_list)
    sources_list = []

    # Read the source and compute some info
    for i in range(n_src):
        source = metadata_file.iloc[pair[i]]
        absolute_path = os.path.join(librispeech_dir,
                                     source['origin_path'])
        s, _ = sf.read(absolute_path, dtype='float32')
        sources_list.append(
            np.pad(s, (0, max_length - len(s)), mode='constant'))

    sources_info = {'mixtures_id': mixtures_id,
                    'speaker_id_list': speaker_id_list, 'sex_list': sex_list,
                    'path_list': path_list}
    return sources_info, sources_list

def set_loudness(sources_list):
    """ Compute original loudness and normalise them randomly """
    # Initialize loudness
    loudness_list = []
    # In LibriSpeech all sources are at 16KHz hence the meter
    meter = pyln.Meter(RATE)
    # Randomize sources loudness
    target_loudness_list = []
    sources_list_norm = []

    # Normalize loudness
    for i in range(len(sources_list)):
        # Compute initial loudness
        loudness_list.append(meter.integrated_loudness(sources_list[i]))
        # Pick a random loudness
        target_loudness = random.uniform(MIN_LOUDNESS, MAX_LOUDNESS)
        # Noise has a different loudness
        if i == len(sources_list) - 1:
            target_loudness = random.uniform(MIN_LOUDNESS - 5,
                                             MAX_LOUDNESS - 5)
        # Normalize source to target loudness

        with warnings.catch_warnings():
            # We don't want to pollute stdout, but we don't want to ignore
            # other warnings.
            warnings.simplefilter("ignore")
            src = pyln.normalize.loudness(sources_list[i], loudness_list[i],
                                          target_loudness)
        # If source clips, renormalize
        if np.max(np.abs(src)) >= 1:
            src = sources_list[i] * MAX_AMP / np.max(np.abs(sources_list[i]))
            target_loudness = meter.integrated_loudness(src)
        # Save scaled source and loudness.
        sources_list_norm.append(src)
        target_loudness_list.append(target_loudness)
    return loudness_list, target_loudness_list, sources_list_norm


def mix(sources_list_norm):
    """ Do the mixture for min mode and max mode """
    # Initialize mixture
    mixture_max = np.zeros_like(sources_list_norm[0])
    for i in range(len(sources_list_norm)):
        mixture_max += sources_list_norm[i]
    return mixture_max


def check_for_cliping(mixture_max, sources_list_norm):
    """Check the mixture (mode max) for clipping and re normalize if needed."""
    # Initialize renormalized sources and loudness
    renormalize_loudness = []
    clip = False
    # Recreate the meter
    meter = pyln.Meter(RATE)
    # Check for clipping in mixtures
    if np.max(np.abs(mixture_max)) > MAX_AMP:
        clip = True
        weight = MAX_AMP / np.max(np.abs(mixture_max))
    else:
        weight = 1
    # Renormalize
    for i in range(len(sources_list_norm)):
        new_loudness = meter.integrated_loudness(sources_list_norm[i] * weight)
        renormalize_loudness.append(new_loudness)
    return renormalize_loudness, clip


def compute_gain(loudness, renormalize_loudness):
    """ Compute the gain between the original and target loudness"""
    gain = []
    for i in range(len(loudness)):
        delta_loudness = renormalize_loudness[i] - loudness[i]
        gain.append(np.power(10.0, delta_loudness / 20.0))
    return gain


def get_row(sources_info, gain_list, n_src):
    """ Get new row for each mixture/info dataframe """
    row_mixture = [sources_info['mixtures_id']]
    row_info = [sources_info['mixtures_id']]
    for i in range(n_src):
        row_mixture.append(sources_info['path_list'][i])
        row_mixture.append(gain_list[i])
        row_info.append(sources_info['speaker_id_list'][i])
        row_info.append(sources_info['sex_list'][i])
    return row_mixture, row_info


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
