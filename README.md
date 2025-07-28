# Multimodal Emotion Recognition in the Wild: Corruption Modeling and Relevance-Guided Scoring

This repository provides the official implementation for the paper  
**"Multimodal Emotion Recognition in the Wild: Corruption Modeling and Relevance-Guided Scoring."**  
We present a corruption modeling framework and relevance-guided scoring mechanism for robust multimodal emotion recognition (MER) under real-world degraded conditions.


## Repository Structure

```yaml
├── MER_gen.py
├── Visual_perturb.py
├── MER_audio_gen.py
├── Audio_perturb.py
├── feature_extractor/
│  ├── Audio_opensmile.py
│  └── face_detection_CLIP.py
```

## Dataset Preparation

This project uses ```CMU-MOSI``` and ```CMU-MOSEI``` datasets with external audio/visual resources for corruption modeling. Please follow the instructions below to prepare the required files.

### 1. CMU-MOSI / CMU-MOSEI
- Download from [CMU Multimodal SDK](https://github.com/A2Zadeh/CMU-MultimodalSDK)
- Required:
  - Raw `.mp4` video files
  - Corresponding `.wav` audio files


### 2. Audio Noise: MUSAN
- We use the [MUSAN dataset](https://www.openslr.org/17/) for audio corruption.
- Recommended categories:
  - `speech`, `music`, `noise` subfolders
- Use `--noise_dir ./noise/sample_wav` in the audio corruption module.


### 3. Visual Occlusion Sources

- We utilize various occlusion patches from publicly available datasets:

  | Source         | Usage                          | Download Link |
  |----------------|---------------------------------|----------------|
  | **11k Hands**  | Hand occlusion patches          | [11k Hands](https://sites.google.com/view/11khands) |
  | **COCO Object**| Object-shaped masks and regions | [COCO](https://cocodataset.org/#download) *(2017 images)* |
  | **DTD**        | Texture occlusion (random noise-like) | [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/) |
  

**Directory structure example:**
  ```yaml
  occlusion_patch/
  ├── image/
  │  ├── hand_img/
  │  ├── object_img/
  │  └── dtd/
  ├── mask/
  │  ├── hand_masks/
  │  ├── object_masks/
  ```


### 4. Landmark Extraction (Optional)
- Facial landmarks for occlusion positioning can be extracted using Mediapipe:
  ```bash
  python MER_gen.py \
      --MOSI_main_dir ./sample_data/video \
      --MOSI_landmark_dir ./sample_data/mediapipe_landmark \
      --MOSI_save_loc ./tmp \
      --occlusion_img ./occlusion_patch/image \
      --occlusion_mask ./occlusion_patch/mask \
      --mode occlusion \
      --noise_percent 0.0
  ```


## Corruption Modeling
We simulate real-world corruptions in both the audio and visual modalities to evaluate model robustness.

### 1. Audio Corruption
- ```MER_audio_gen.py```

  Injects noise into .wav files using additive SNR-based corruption.
  Noise types include speech, music, environmental, and white noise.

- ```Audio_perturb.py```

  Noise injection utility with optional segment masking and configurable SNR levels.


### 2. Visual Corruption
- ```MER_gen.py```

  Corrupts video frames via occlusion (hand/object/texture) or additive noise (blur, Gaussian).
  Uses Mediapipe to extract facial landmarks for occlusion positioning.

- ```Visual_perturb.py```

  Contains core functions for visual occlusion, noise injection, and corruption scheduling.


## Feature Extraction
We extract audio and visual embeddings for model input using the following scripts:

### 1. Audio
- ```Audio_opensmile.py```

  Uses OpenSMILE to extract low-level descriptors (eGeMAPS, ComParE-2016).
  Outputs a fixed-length feature matrix using resampling or cropping.
  ```bash
  python feature_extractor/Audio_opensmile.py \
      --raw_dir ./results/audio_perturb/snr_10 \
      --pkl_dir ./features/audio
  ```

### 2. Visual
- ```Visual_face_detection_CLIP.py```

  Detects faces using Mediapipe, extracts regions, and encodes them using CLIP (ViT-B/32).
  Embeddings are resampled to a fixed number of frames (500 by default).
  ```bash
  python feature_extractor/Visual_face_detection_CLIP.py \
      --input_folder ./results/visual_perturb \
      --output_folder ./features/visual
  ```  


## How to Run
### 1. Audio Corruption
  ```bash
  python MER_audio_gen.py \
      --main_dir ./sample_data/wav \
      --noise_dir ./noise/sample_wav \
      --save_loc ./results/audio_perturb/snr
  ```
###  2. Visual Corruption
  ```bash
  python MER_gen.py \
      --MOSI_main_dir ./sample_data/video \
      --MOSI_landmark_dir ./sample_data/mediapipe_landmark \
      --MOSI_save_loc ./results/visual_perturb \
      --occlusion_img ./occlusion_patch/image \
      --occlusion_mask ./occlusion_patch/mask \
      --mode occlusion \
      --noise_percent 0.3
  ```

## Installation

Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
  Note: For audio feature extraction, please install OpenSMILE separately:
  - [OpenSMILE GitHub](https://github.com/audeering/opensmile)

## Citation
If you use this code or find it helpful for your research, please cite our [paper]():

```yaml
@article{lee2025cars,
  title     = {Multimodal Emotion Recognition in the Wild: Corruption Modeling and Relevance-Guided Scoring},
  author    = {Lee, Yoonsun and Cho, Sunyoung},
  journal   = {Under Review},
  year      = {2025}
}

@article{hong2023watch,
  title={Watch or Listen: Robust Audio-Visual Speech Recognition with Visual Corruption Modeling and Reliability Scoring},
  author={Hong, Joanna and Kim, Minsu and Choi, Jeongsoo and Ro, Yong Man},
  journal={arXiv preprint arXiv:2303.08536},
  year={2023}
}
```
This repository extends and adapts [AV-RelScore](https://github.com/joannahong/AV-RelScore) for emotion recognition.
