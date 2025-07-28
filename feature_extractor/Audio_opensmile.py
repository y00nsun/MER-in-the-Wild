import os
import pickle
import numpy as np
import argparse
import opensmile
import audiofile
import scipy.signal as spsig

def extract_features(wav_path, sequence_length=375, resample_threshold=700):
    """
    Extract OpenSMILE features from a wav file and fix sequence length.

    Args:
        wav_path (str): Path to the input .wav file.
        sequence_length (int): Target number of frames in output. (375 for MOSI, 500 for MOSEI)
        resample_threshold (int): If input length ≤ threshold, resample; else, crop.

    Returns:
        np.ndarray: Feature array of shape (sequence_length, 90)
    """
    x, sr = audiofile.read(wav_path)

    # OpenSMILE features
    smile1 = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    smile2 = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    feat1 = smile1.process_signal(x, sr).to_numpy()
    feat2 = smile2.process_signal(x, sr).to_numpy()
    features = np.concatenate((feat1, feat2), axis=1)  # shape: (T, 90)

    original_length = features.shape[0]

    # Resample
    if original_length <= resample_threshold:
        resampled = spsig.resample_poly(features, up=sequence_length, down=original_length, axis=0)
        print(f"Resampled: {original_length} -> {sequence_length}")
    else:
        start = (original_length - sequence_length) // 2
        resampled = features[start:start + sequence_length, :]
        print(f"Cropped: {original_length} -> {sequence_length} (center)")

    return resampled

def process_all_files(raw_dir, pkl_dir):
    for root, _, files in os.walk(raw_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_path = os.path.join(root, file)
                print(wav_path)

                relative_path = os.path.relpath(root, raw_dir)
                save_dir = os.path.join(pkl_dir, relative_path)
                os.makedirs(save_dir, exist_ok=True)

                features = extract_features(wav_path)

                save_path = os.path.join(save_dir, file.replace('.wav', '.pkl'))
                with open(save_path, 'wb') as f:
                    pickle.dump(features, f)

                print(f"Saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract OpenSMILE features from WAV files.")
    parser.add_argument('--raw_dir', type=str, required=True, help='Directory containing .wav files')
    parser.add_argument('--pkl_dir', type=str, required=True, help='Directory to save .pkl feature files')
    args = parser.parse_args()

    process_all_files(args.raw_dir, args.pkl_dir)