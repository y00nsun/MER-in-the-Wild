### Audio Noise Generation Utilities ###
import numpy as np
import random
import os
import soundfile as sf
import librosa
import scipy.signal as spsig  # currently unused

def cal_adjusted_rms(clean_rms, snr):
    """
    Calculate the adjusted RMS of noise given clean RMS and target SNR (in dB).
    """
    return clean_rms / (10 ** (float(snr) / 20))

def cal_rms(amp):
    """
    Calculate the RMS (Root Mean Square) value of an audio signal.
    """
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def noise_injection(clean_amp, ns_list, snr=None, part=False):
    """
    Injects noise into clean audio with a specific SNR.
    
    Args:
        clean_amp (np.ndarray): Clean audio waveform.
        ns_list (list): List of noise audio file paths.
        snr (int, optional): Target Signal-to-Noise Ratio (in dB). Randomly chosen if None.
        part (bool): If True, randomly zero out small parts of the noise (simulates packet loss).
        
    Returns:
        np.ndarray: Noised audio signal.
    """
    ns_file = random.choice(ns_list)
    noise_amp, _ = librosa.load(ns_file, sr=16000)

    if snr is None:
        snr = random.choice([-5, 0, 5, 10, 15])

    # Extend noise if it's shorter than the clean signal
    if len(noise_amp) < len(clean_amp):
        repeat_factor = (len(clean_amp) // len(noise_amp)) + 1
        noise_amp = np.tile(noise_amp, repeat_factor)

    start = random.randint(0, len(noise_amp) - len(clean_amp))
    split_noise_amp = noise_amp[start: start + len(clean_amp)]

    clean_rms = cal_rms(clean_amp)
    noise_rms = cal_rms(split_noise_amp)
    adjusted_noise_rms = cal_adjusted_rms(clean_rms, snr)
    adjusted_noise_amp = split_noise_amp * (adjusted_noise_rms / noise_rms)

    # Optional partial masking (simulating packet drop or severe degradation)
    if part:
        repeat = random.randint(1, 5)
        min_len = int(0.05 * 16000)  # minimum 50ms
        max_len = int(len(adjusted_noise_amp) / repeat / 2)
        max_len = max(min_len + 1, max_len)
        st_inds = random.sample(range(len(adjusted_noise_amp) - max_len + 1), repeat)
        for st_ind in st_inds:
            segment_len = random.randint(min_len, max_len)
            adjusted_noise_amp[st_ind:st_ind + segment_len] = 0.0

    mixed_amp = clean_amp + adjusted_noise_amp

    # Normalize to avoid clipping
    max_val = np.max(np.abs(mixed_amp))
    if max_val > 1:
        mixed_amp = mixed_amp / max_val

    return mixed_amp
