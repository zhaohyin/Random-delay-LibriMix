import os
import argparse
import soundfile as sf
import pandas as pd
import numpy as np
import functools
from scipy.signal import resample_poly
import tqdm.contrib.concurrent
import random

# eps secures log and division
EPS = 1e-10
# Rate of the sources in LibriSpeech
RATE = 16000
#
parser = argparse.ArgumentParser()
parser.add_argument('--librispeech_dir', type=str, required=True,
                    help='Path to librispeech root directory')
parser.add_argument('--metadata_dir', type=str, required=True,
                    help='Path to the LibriMix metadata directory')
parser.add_argument('--librimix_outdir', type=str, default=None,
                    help='Path to the desired dataset root directory')
parser.add_argument('--n_src', type=int, required=True,
                    help='Number of sources in mixtures')
parser.add_argument('--freqs', nargs='+', default=['16k'],
                    help='--freqs 16k 8k will create 2 directories wav8k '
                         'and wav16k')
parser.add_argument('--modes', nargs='+', default=['max'],
                    help='--modes min max will create 2 directories in '
                         'each freq directory')
parser.add_argument('--types', nargs='+', default=['mix_clean'],
                    help='--types mix_clean')


def main(args):
    # Get librispeech root path
    librispeech_dir = args.librispeech_dir
    # Get Metadata directory
    metadata_dir = args.metadata_dir
    # Get LibriMix root path
    librimix_outdir = args.librimix_outdir
    n_src = args.n_src
    if librimix_outdir is None:
        librimix_outdir = os.path.dirname(metadata_dir)
    librimix_outdir = os.path.join(librimix_outdir, f'Libri{n_src}Mix')
    # Get the desired frequencies
    freqs = args.freqs
    freqs = [freq.lower() for freq in freqs]
    # Get the desired modes
    modes = args.modes
    modes = [mode.lower() for mode in modes]
    types = args.types
    types = [t.lower() for t in types]
    print(modes, types)
    # Get the number of sources
    create_librimix(librispeech_dir, librimix_outdir, metadata_dir,
                    freqs, n_src, modes, types)


def create_librimix(librispeech_dir, out_dir, metadata_dir,
                    freqs, n_src, modes, types):
    """ Generate sources mixtures and saves them in out_dir"""
    # Get metadata files
    md_filename_list = [file for file in os.listdir(metadata_dir)
                        if 'info' not in file]
    # Create all parts of librimix
    for md_filename in md_filename_list:
        csv_path = os.path.join(metadata_dir, md_filename)
        process_metadata_file(csv_path, freqs, n_src, librispeech_dir,
                              out_dir, modes, types)


def process_metadata_file(csv_path, freqs, n_src, librispeech_dir,
                          out_dir, modes, types):
    """ Process a metadata generation file to create sources and mixtures"""
    md_file = pd.read_csv(csv_path, engine='python')
    for freq in freqs:
        # Get the frequency directory path
        freq_path = os.path.join(out_dir, 'wav' + freq)
        # Transform freq = "16k" into 16000
        freq = int(freq.strip('k')) * 1000

        for mode in modes:
            # Path to the mode directory
            mode_path = os.path.join(freq_path, mode)
            # Subset metadata path
            subset_metadata_path = os.path.join(mode_path, 'metadata')
            os.makedirs(subset_metadata_path, exist_ok=True)
            # Directory where the mixtures and sources will be stored
            dir_name = os.path.basename(csv_path).replace(
                f'libri{n_src}mix_', '').replace('-clean', '').replace(
                '.csv', '')
            dir_path = os.path.join(mode_path, dir_name)
            # If the files already exist then continue the loop
            if os.path.isdir(dir_path):
                print(f"Directory {dir_path} already exist. "
                      f"Files won't be overwritten")
                continue

            print(f"Creating mixtures and sources from {csv_path} "
                  f"in {dir_path}")
            # Create subdir
            subdirs = [f's{i + 1}' for i in range(n_src)] + ['mix_clean']  # subdirs is s1,S2,mix_clean
            # Create directories accordingly
            for subdir in subdirs:
                os.makedirs(os.path.join(dir_path, subdir))
            # Go through the metadata file
            print(subdirs, dir_name)
            process_utterances(md_file, librispeech_dir, freq, mode,
                               subdirs, dir_path, subset_metadata_path, n_src)


def process_utterances(md_file, librispeech_dir, freq, mode, subdirs,
                       dir_path, subset_metadata_path, n_src):
    # Dictionary that will contain all metadata
    md_dic = {}
    # Get dir name
    dir_name = os.path.basename(dir_path)
    # Create Dataframes
    for subdir in subdirs:
        if subdir.startswith('mix'):
            md_dic[f'metrics_{dir_name}_{subdir}'] = create_empty_metrics_md(
                n_src, subdir)
            md_dic[f'mixture_{dir_name}_{subdir}'] = create_empty_mixture_md(
                n_src, subdir)

    # Go through the metadata file and generate mixtures ????
    for results in tqdm.contrib.concurrent.process_map(
            functools.partial(
                process_utterance,
                n_src, librispeech_dir, freq, mode, subdirs, dir_path),
            [row for _, row in md_file.iterrows()],
            chunksize=10,
    ):
        if mode == 'move':
            for mix_id, snr_list, abs_mix_path, abs_source_path_list, length, subdir,delay in results:
                # Add line to the dataframes
                add_to_metrics_metadata(md_dic[f"metrics_{dir_name}_{subdir}"],
                                        mix_id, snr_list)
                add_to_mixture_metadata(md_dic[f'mixture_{dir_name}_{subdir}'],
                                        mix_id, abs_mix_path, abs_source_path_list,
                                        length,delay)
        else:
            for mix_id, snr_list, abs_mix_path, abs_source_path_list, length, subdir in results:
                # Add line to the dataframes
                add_to_metrics_metadata(md_dic[f"metrics_{dir_name}_{subdir}"],
                                        mix_id, snr_list)
                add_to_mixture_metadata(md_dic[f'mixture_{dir_name}_{subdir}'],
                                        mix_id, abs_mix_path, abs_source_path_list,
                                        length)

    # Save the metadata files
    for md_df in md_dic:
        # Save the metadata in out_dir ./data/wavxk/mode/subset
        save_path_mixture = os.path.join(subset_metadata_path, md_df + '.csv')
        md_dic[md_df].to_csv(save_path_mixture, index=False)


def process_utterance(n_src, librispeech_dir, freq, mode, subdirs, dir_path, row):
    res = []
    # Get sources and mixture infos
    mix_id, gain_list, sources = read_sources(row, n_src, librispeech_dir)
    # Transform sources
    if mode == 'move':
        transformed_sources,delay = transform_sources(sources, freq, mode, gain_list, n_src)
    else:
        transformed_sources = transform_sources(sources, freq, mode, gain_list, n_src)
    # Write the sources and get their paths
    abs_source_path_list = write_sources(mix_id,
                                         transformed_sources,
                                         subdirs, dir_path, freq,
                                         n_src)
    # Mixtures are different depending on the subdir
    for subdir in subdirs:
        if subdir == 'mix_clean':
            sources_to_mix = transformed_sources[:n_src]
        else:
            continue

        # Mix sources
        mixture = mix(sources_to_mix)
        # Write mixture and get its path
        abs_mix_path = write_mix(mix_id, mixture, dir_path, subdir, freq)
        length = len(mixture)
        # Compute SNR
        snr_list = compute_snr_list(mixture, sources_to_mix)
        #
        if mode == 'move':
            res.append((mix_id, snr_list, abs_mix_path, abs_source_path_list, length, subdir,delay))
        else:
            res.append((mix_id, snr_list, abs_mix_path, abs_source_path_list, length, subdir))

    return res


def create_empty_metrics_md(n_src, subdir):
    """ Create the metrics dataframe"""
    metrics_dataframe = pd.DataFrame()
    metrics_dataframe['mixture_ID'] = {}
    if subdir == 'mix_clean':
        for i in range(n_src):
            metrics_dataframe[f"source_{i + 1}_SNR"] = {}
    return metrics_dataframe


def create_empty_mixture_md(n_src, subdir):
    """ Create the mixture dataframe"""
    mixture_dataframe = pd.DataFrame()
    mixture_dataframe['mixture_ID'] = {}
    mixture_dataframe['mixture_path'] = {}
    if subdir == 'mix_clean':
        for i in range(n_src):
            mixture_dataframe[f"source_{i + 1}_path"] = {}
    mixture_dataframe['length'] = {}
    if subdir == 'mix_clean':
        for i in range(n_src):
            mixture_dataframe[f"source_{i + 1}_delay"] = {}
    return mixture_dataframe


def read_sources(row, n_src, librispeech_dir):
    """ Get sources and info to mix the sources """
    # Get info about the mixture
    mixture_id = row['mixture_ID']
    sources_path_list = get_list_from_csv(row, 'source_path', n_src)  #
    gain_list = get_list_from_csv(row, 'source_gain', n_src)  #
    sources_list = []
    # Read the files to make the mixture
    for sources_path in sources_path_list:
        sources_path = os.path.join(librispeech_dir,
                                    sources_path)
        source, _ = sf.read(sources_path, dtype='float32')  # read sound file
        sources_list.append(source)
    return mixture_id, gain_list, sources_list


def get_list_from_csv(row, column, n_src):
    """ Transform a list in the .csv in an actual python list """
    python_list = []
    for i in range(n_src):
        current_column = column.split('_')  # source,path
        current_column.insert(1, str(i + 1))  # source,1,path
        current_column = '_'.join(current_column)  # source_1_path
        python_list.append(row[current_column])
    return python_list


def transform_sources(sources_list, freq, mode, gain_list, n_src):
    """ Transform libriSpeech sources to librimix """
    # Normalize sources
    sources_list_norm = loudness_normalize(sources_list, gain_list)
    # Resample the sources
    sources_list_resampled = resample_list(sources_list_norm, freq)
    # Reshape sources
    if mode == 'move':
        reshaped_sources,delay = fit_lengths(sources_list_resampled, mode, n_src, freq)
        return reshaped_sources,delay
    else:
        reshaped_sources = fit_lengths(sources_list_resampled, mode, n_src, freq)
        return reshaped_sources


def loudness_normalize(sources_list, gain_list):
    """ Normalize sources loudness"""
    # Create the list of normalized sources
    normalized_list = []
    for i, source in enumerate(sources_list):
        normalized_list.append(source * gain_list[i])
    return normalized_list


def resample_list(sources_list, freq):
    """ Resample the source list to the desired frequency"""
    # Create the resampled list
    resampled_list = []
    # Resample each source
    for source in sources_list:
        resampled_list.append(resample_poly(source, freq, RATE))
    return resampled_list


def fit_lengths(source_list, mode, n_src, freq):
    """ Make the sources to match the target length """
    sources_list_reshaped = []
    # Check the mode
    if mode == 'min':
        target_length = min([len(source) for source in source_list])
        for source in source_list:
            sources_list_reshaped.append(source[:target_length])
        return sources_list_reshaped
    elif mode == 'max':
        # The mixture of voice start with the same time point
        target_length = max([len(source) for source in source_list])  # source_list is a two-demension matrix
        for source in source_list:
            sources_list_reshaped.append(
                np.pad(source, (0, target_length - len(source)),
                       mode='constant'))
        return sources_list_reshaped
    elif mode == 'move':
        # The mixture of voice start with different time point
        random_delay = determine_delay(source_list, n_src, freq)
        target_length = determine_lengths(source_list, n_src, random_delay)
        # print(target_length)
        # add 0 in difference source
        #  00000000000000000  source  000000000000000000000000000000000000000000000
        #  |sum(delay[:n])|  source  |target_length - sum(delay[:n])-len(source)|
        n = 1
        for source in source_list:
            sources_list_reshaped.append(
                np.pad(source,
                       (sum(random_delay[:n]),
                        target_length - sum(random_delay[:n]) - len(source)),
                       mode='constant', constant_values=0))
            n = n+1
        return sources_list_reshaped,random_delay


def determine_delay(source_list, n_src, freq):
    '''
    set the delay between voice,min_delay is 0.5sec,max_delay is length of the last voice
    ---------------------
    |<-0.5sec->|-------------
    |<-- max_delay -->|----------------
    '''
    delay = np.zeros((n_src)).astype(int)  # delay is a n demension matrix
    for i in range(1,n_src):
        # set the delay of each voice in mixture data (In this situation,i use sample point as the minimum unit)
        min_delay = 0.5 * freq
        max_delay = len(source_list[i-1]) - 0.5 * freq
        delay[i] = int(min_delay + random.random() * (max_delay - min_delay))
    return delay
    pass


def determine_lengths(source_list, n_src, delay):
    '''determine the length of combined voice through the min_delay and max_delay
    source_list:the component of mixture voice
    n_src:the number of speaker
    delay:n_src demension matrix
                                        for example:
                                        ------------
                                           ----------
                                             -----
    '''
    if n_src == 1:
        target_length = len(source_list[0])  # when mixture process is not needed
        return target_length
    else:
        length_end = sum(delay[:n_src]) + len(source_list[n_src - 1])  # delay=[0,xx,xx,xx,xx]
        # print(length_end)
        return max([determine_lengths(source_list, n_src - 1, delay), length_end])  # return the maximum data
    pass


def write_sources(mix_id, transformed_sources, subdirs, dir_path, freq, n_src):
    # Write sources and mixtures and save their path
    abs_source_path_list = []
    ex_filename = mix_id + '.wav'
    for src, src_dir in zip(transformed_sources[:n_src], subdirs[:n_src]):  # only insert s1 and s2
        save_path = os.path.join(dir_path, src_dir, ex_filename)
        abs_save_path = os.path.abspath(save_path)
        sf.write(abs_save_path, src, freq)
        abs_source_path_list.append(abs_save_path)
    return abs_source_path_list


def mix(sources_list):
    """ Do the mixing """
    # Initialize mixture
    mixture = np.zeros_like(sources_list[0])
    for source in sources_list:
        mixture += source
    return mixture


def write_mix(mix_id, mixture, dir_path, subdir, freq):
    # Write noise save it's path
    ex_filename = mix_id + '.wav'
    save_path = os.path.join(dir_path, subdir, ex_filename)
    abs_save_path = os.path.abspath(save_path)
    sf.write(abs_save_path, mixture, freq)
    return abs_save_path


def compute_snr_list(mixture, sources_list):
    """Compute the SNR on the mixture mode min"""
    snr_list = []
    # Compute SNR for min mode
    for i in range(len(sources_list)):
        noise_min = mixture - sources_list[i]
        snr_list.append(snr_xy(sources_list[i], noise_min))
    return snr_list


def snr_xy(x, y):
    return 10 * np.log10(np.mean(x ** 2) / (np.mean(y ** 2) + EPS) + EPS)


def add_to_metrics_metadata(metrics_df, mixture_id, snr_list):
    """ Add a new line to metrics_df"""
    row_metrics = [mixture_id] + snr_list
    metrics_df.loc[len(metrics_df)] = row_metrics


def add_to_mixture_metadata(mix_df, mix_id, abs_mix_path, abs_sources_path, length,delays=None):
    """ Add a new line to mixture_df """
    sources_path = abs_sources_path
    noise_path = []
    list_delay = []
    for delay in delays:
        list_delay.append(str(delay))
        pass
    row_mixture = [mix_id, abs_mix_path] + sources_path + noise_path + [length] + list_delay
    mix_df.loc[len(mix_df)] = row_mixture  # the next row


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
