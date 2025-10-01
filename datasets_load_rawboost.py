import numpy as np
import os
import torch
from torch.utils.data import Dataset
import librosa
from Rawboost import ISD_additive_noise,LnL_convolutive_noise,SSI_additive_noise,normWav
from random import randrange
import random

def pad_random(x: np.ndarray, max_len: int = 64000):
    x_len = x.shape[0]
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))
    return pad_random(padded_x, max_len)

class SVDD2024(Dataset):
    """
    Dataset class for the SVDD 2024 dataset.
    """
    def __init__(self, base_dir, partition="train", max_len=64000, args=None, algo=5):
        assert partition in ["train", "dev", "test"], "Invalid partition. Must be one of ['train', 'dev', 'test']"
        self.base_dir = base_dir
        self.partition = partition
        self.base_dir = os.path.join(base_dir, partition + "_set")
        self.max_len = max_len
        self.args=args
        self.algo=algo
        
        try:
            with open(os.path.join(base_dir, f"{partition}.txt"), "r") as f:
                self.file_list = f.readlines()
        except FileNotFoundError:
            if partition == "test":
                self.file_list = []
                # get all *.flac files in the test_set directory
                for root, _, files in os.walk(self.base_dir):
                    for file in files:
                        if file.endswith(".flac"):
                            self.file_list.append(file)
            else:
                raise FileNotFoundError(f"File {partition}.txt not found in {base_dir}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):            
        try:
            if self.partition == "test":
                file_name = self.file_list[index].strip()
                label = 0  # dummy label. Not used for test set.
            else:
                file = self.file_list[index]
                file_name = file.split(" ")[2].strip()
                bonafide_or_spoof = file.split(" ")[-1].strip()
                label = 1 if bonafide_or_spoof == "bonafide" else 0

            # Append the .flac extension to the file name
            file_path = os.path.join(self.base_dir, file_name + ".flac")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Load and process audio
            x, sr = librosa.load(file_path, sr=16000, mono=True)
            
            # Apply processing based on partition
            if self.partition == "train":
                # Apply data augmentation for training
                x = process_Rawboost_feature(x, sr, self.args, self.algo)
                x = pad_random(x, self.max_len)
            else:  # dev or test
                # For dev/test, just pad or trim to max_len
                if len(x) > self.max_len:
                    # Randomly select a segment of max_len
                    start = np.random.randint(0, len(x) - self.max_len)
                    x = x[start:start + self.max_len]
                else:
                    # Pad with zeros if shorter than max_len
                    x = np.pad(x, (0, self.max_len - len(x)), mode='constant')

            # Convert to PyTorch tensor with proper dtype
            x = torch.tensor(x, dtype=torch.float32)
            label = torch.tensor(label, dtype=torch.long)
            
            return x, label, file_name

        except Exception as e:
            print(f"Error processing {file_name if 'file_name' in locals() else 'unknown file'}: {str(e)}")
            # Return a zero tensor with proper shape in case of error
            return torch.zeros(self.max_len), torch.tensor(0, dtype=torch.long), file_name
            # Convert to PyTorch tensor
            x_inp = torch.tensor(x, dtype=torch.float32)
            
            # file_name is used for generating the score file for submission
            return x_inp, label, file_name
        except Exception as e:
            print(f"Error loading {file_name}: {str(e)}")
            raise RuntimeError(f"Error loading {file_name}: {str(e)}")



#--------------RawBoost data augmentation algorithms---------------------------##

def process_Rawboost_feature(feature, sr, args, algo):
    
    # Data process by Convolutive noise (1st algo)
    if algo == 1:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF,
                                      args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
                                      args.minG, args.maxG, args.minBiasLinNonLin,
                                      args.maxBiasLinNonLin, sr)
                            
    # Data process by Impulsive noise (2nd algo)
    elif algo == 2:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        
    # Data process by Stationary noise (3rd algo)
    elif algo == 3:
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands,
                                    args.minF, args.maxF, args.minBW, args.maxBW,
                                    args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    
    # Data process by all three algo in series (4th algo)
    elif algo == 4:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF,
                                       args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
                                       args.minG, args.maxG, args.minBiasLinNonLin,
                                       args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands,
                                    args.minF, args.maxF, args.minBW, args.maxBW,
                                    args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    
    # Data process by 1st and 2nd algo in series (5th algo)
    elif algo == 5:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF,
                                       args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
                                       args.minG, args.maxG, args.minBiasLinNonLin,
                                       args.maxBiasLinNonLin, sr)
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
    
    # Data process by 1st and 3rd algo in series (6th algo)
    elif algo == 6:
        feature = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF,
                                       args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
                                       args.minG, args.maxG, args.minBiasLinNonLin,
                                       args.maxBiasLinNonLin, sr)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands,
                                    args.minF, args.maxF, args.minBW, args.maxBW,
                                    args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    
    # Data process by 2nd and 3rd algo in series (7th algo)
    elif algo == 7:
        feature = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = SSI_additive_noise(feature, args.SNRmin, args.SNRmax, args.nBands,
                                    args.minF, args.maxF, args.minBW, args.maxBW,
                                    args.minCoeff, args.maxCoeff, args.minG, args.maxG, sr)
    
    # Data process by 1st and 2nd algo in parallel (8th algo)
    elif algo == 8:
        feature1 = LnL_convolutive_noise(feature, args.N_f, args.nBands, args.minF, args.maxF,
                                        args.minBW, args.maxBW, args.minCoeff, args.maxCoeff,
                                        args.minG, args.maxG, args.minBiasLinNonLin,
                                        args.maxBiasLinNonLin, sr)
        feature2 = ISD_additive_noise(feature, args.P, args.g_sd)
        feature = (feature1 + feature2) / 2
    
    return feature
                            
    # Data process by coloured additive noise (3rd algo)
    elif algo==3:
        
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)
    
    # Data process by all 3 algo. together in series (1+2+3)
    elif algo==4:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)  
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr)                 

    # Data process by 1st two algo. together in series (1+2)
    elif algo==5:
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=ISD_additive_noise(feature, args.P, args.g_sd)                
                            

    # Data process by 1st and 3rd algo. together in series (1+3)
    elif algo==6:  
        
        feature =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 

    # Data process by 2nd and 3rd algo. together in series (2+3)
    elif algo==7: 
        
        feature=ISD_additive_noise(feature, args.P, args.g_sd)
        feature=SSI_additive_noise(feature,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,sr) 
   
    # Data process by 1st two algo. together in Parallel (1||2)
    elif algo==8:
        
        feature1 =LnL_convolutive_noise(feature,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                 args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,sr)                         
        feature2=ISD_additive_noise(feature, args.P, args.g_sd)

        feature_para=feature1+feature2
        feature=normWav(feature_para,0)  #normalized resultant waveform
 
    # original data without Rawboost processing           
    else:
        
        feature=feature
    
    return feature
    # return torch.tensor(feature, dtype=torch.float32)

