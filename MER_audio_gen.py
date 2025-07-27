from Audio_perturb import *
import librosa
import soundfile as sf
import os
import argparse
import random
import numpy as np

random.seed(1)

def audio_gen(args):
    """Generate corrupted audio using additive noise at various SNR levels."""
    snrs = [-5, 0, 5, 10, 15]

    # Collect target audio files
    file_list = []
    for root, _, files in os.walk(args.main_dir):
        for file in files:
            if file.endswith('.wav'):
                relative_path = os.path.relpath(os.path.join(root, file), args.main_dir)
                file_list.append(os.path.splitext(relative_path)[0])

    # Collect available noise samples
    noise_list = []
    for root, _, files in os.walk(args.noise_dir):
        for file in files:
            if file.endswith('.wav'):
                noise_list.append(os.path.join(root, file))

    for snr in snrs:
        for kk, test_file in enumerate(file_list):
            input_path = os.path.join(args.main_dir, test_file + '.wav')
            aud, _ = librosa.load(input_path, sr=16000)

            audio = noise_injection(aud, noise_list, snr=snr, part=True)

            # Save corrupted audio
            save_dir = os.path.join(f'{args.save_loc}_{snr}')
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, test_file + '.wav')
            sf.write(save_path, audio, 16000)

            print(f'{snr}: {kk + 1}/{len(file_list)} | Saved: {save_path}', end='\r')

            if np.isnan(audio).any() or np.isinf(audio).any():
                print(f"\nInvalid audio data detected in: {test_file}")
                continue

    print('\nAll audio perturbations complete.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--main_dir', type=str, default='./sample_data/wav',
                        help='Directory of the original audio dataset')
    parser.add_argument('-o', '--save_loc', type=str, default='./results/audio_perturb/snr',
                        help='Directory to save corrupted audio')
    parser.add_argument('--noise_dir', type=str, default='./noise/sample_wav',
                        help='Directory containing noise .wav files')
    args = parser.parse_args()

    audio_gen(args)
