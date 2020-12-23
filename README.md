# Random-delay-LibriMix
Edit by zhyin in Dec23,2020
* Description : Create a new overlapped Speech dataset based on LibriSpeech 
* Principle: 
1. Set the loudness of each voice. A linear transformation was used in this progress.
2. Set the delay between voice randomly.The min_delay is 0.5sec, and the max_delay is length of the last voice. The mixture progress of three voice can be expressed as:  
  
                   ---------------------               [voice1]    
                   |<-0.5sec->|-------------           [voice2]    
                   |<-- max_delay -->|---------------- [voice3]     
                   
  
    
* Use:
   * <b>[1] Description</b> \
    Create a metadate for the subset of LibriSpeech.\
    *[speaker_ID]+[sex]+[subset]+[length]+[origin_path]*

                python create_librispeech_metadata.py  --librispeech_dir 
  
    * <b>Example</b> 
    
          python create_librispeech_metadata.py  --librispeech_dir D:OSR\2020-12\LibriMix\scripts\LibriSpeech
     
   * <b>Result</b> 
   
          D:\OSR\2020-12\LibriMix\scripts\LibriSpeech\metadata\test-clean.csv
   
   * <b>[2] Description</b> \
    Create a metadata for the  overlapped speech \
    *miture-[mixture_ID]+[source_x_path]+[source_x_gain] for x in range(n_src)*
    *info-[mixture_ID]+[speaker_x_ID]+[speaker_x_sex] for x in range(n_src)*
    
          python create_librimix_metadata.py --librispeech_dir --metadata_dir --n_src
          
    * <b>Example</b>: 
    
          python create_librimix_metadata.py --librispeech_dir D:\OSR\2020-12\LibriMix\scripts\LibriSpeech --librispeech_md_dir D:\OSR\2020-12\LibriMix\scripts\LibriSpeech\metadata --n_src 3

    * <b>Result</b>: 

          D:\OSR\2020-12\LibriMix\scripts\Libri3Mix_metadata\libri3mix_test-clean.csv
          D:\OSR\2020-12\LibriMix\scripts\Libri3Mix_metadata\libri3mix_test-clean_info.csv
   
   * <b>[3] Description</b> \
    Create a 3-Speakers overlapped speech \
    Create the metadata for this overlapped speech \
    *[mixture_ID]+[mixture_path]+[source_x_path]+[length]+[source_x_delay] for x in range(n_src)* 
    
          python create_libri2mix_random_delay.py --librispeech_dir --metadata_dir --n_src --modes 
          
    * <b>Example1</b>:
    
          D:\OSR\2020-12\LibriMix\scripts>python create_libri2mix_random_delay.py --librispeech_dir D:\OSR\2020-12\LibriMix\scripts\LibriSpeech --metadata_dir D:\OSR\2020-12\LibriMix\scripts\Libri3Mix_metadata --n_src 2 --modes move
          
    * <b>Example2</b>: 
    
          D:\OSR\2020-12\LibriMix\scripts>python create_libri2mix_random_delay.py --librispeech_dir D:\OSR\2020-12\LibriMix\scripts\LibriSpeech --metadata_dir D:\OSR\2020-12\LibriMix\scripts\Libri3Mix_metadata --n_src 3 --modes move
          
    * <b>Result</b>:
    
          D:\OSR\2020-12\LibriMix\scripts\Libri2Mix-randomdelay\wav16k\move\test\mix_clean\mixture_ID.wav
          D:\OSR\2020-12\LibriMix\scripts\Libri2Mix-randomdelay\wav16k\move\metadata\mixture_test_mix_clean.csv
