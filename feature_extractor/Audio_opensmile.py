import os
import pickle
import numpy as np
import argparse
import opensmile
import audiofile
import scipy.signal as spsig

# Argument parser
parser = argparse.ArgumentParser(description="Process WAV files for OpenSMILE feature extraction.")
parser.add_argument('--raw_dir', type=str, required=True, help='Path to the directory containing WAV files.')
parser.add_argument('--pkl_dir', type=str, required=True, help='Path to the directory where extracted features will be saved.')
args = parser.parse_args()

raw_dir = args.raw_dir
pkl_dir = args.pkl_dir

# OpenSMILE init  => feature level = Functional로 하면 sequence 정보 없음
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

smile2 = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
)

# 고정 시퀀스 길이
# sequence_length = 375  # MOSI
# resample_threshold = 700

sequence_length = 500  # MOSEI
resample_threshold = 1000

for root, dirs, files in os.walk(raw_dir):
    for file in files:
        if file.endswith('.wav'):
            wav_file = os.path.join(root, file)
            print(wav_file)

            rel_path = os.path.relpath(root, raw_dir)
            save_dir = os.path.join(pkl_dir, rel_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # WAV 파일 로드
            x, sr = audiofile.read(wav_file)

            # OpenSMILE 특징 추출
            features1 = smile.process_signal(x, sr).to_numpy()
            features2 = smile2.process_signal(x, sr).to_numpy()
            
            # Concat
            features = np.concatenate((features1, features2), axis=1)  # feature_shape = (seq, 90)
            original_length = features.shape[0]

            # 3. Threshold 리샘플링
            if original_length <= resample_threshold:
                # Resample to sequence_length
                sampled_features = spsig.resample_poly(features, up=sequence_length, down=original_length, axis=0)
                print(f"Resampled: {original_length} -> {sequence_length}")
            else:
                # Center crop
                start = (original_length - sequence_length) // 2
                sampled_features = features[start:start + sequence_length, :]
                print(f"Cropped: {original_length} -> {sequence_length} (center)")

            save_file_path = os.path.join(save_dir, file.replace('.wav', '.pkl'))
            with open(save_file_path, 'wb') as f:
                pickle.dump(sampled_features, f)

            print(f"Saved features for {wav_file} to {save_file_path}")
