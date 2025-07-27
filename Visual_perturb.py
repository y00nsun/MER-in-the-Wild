import cv2
import os
import albumentations as A
import numpy as np
import random
from skimage.util import random_noise
import torchvision
import torch
import pickle
import math
import matplotlib.pyplot as plt

random.seed(10)
np.random.seed(10)
_window_margin = 12

### ---------- Occluder and Noise Augmentation ---------- ###

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Overlay an image with alpha mask at position (x, y)."""
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return img

    img_crop = img[y1:y2, x1:x2]
    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha
    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop
    return img

def get_occluder_augmentor():
    """Return Albumentations augmenter for occluders."""
    aug = A.Compose([
        A.AdvancedBlur(),
        A.OneOf([A.ImageCompression(quality_lower=70)], p=0.5),
        A.Affine(scale=(0.8, 1.2), rotate=(-15, 15), shear=(-8, 8), fit_output=True, p=0.7),
        A.RandomBrightnessContrast(p=0.5, brightness_limit=0.1, contrast_limit=0.1)
    ])
    return aug

def extract_mask_bbox_crop(occluder_img, occluder_mask):    
    """Crop region around the mask contour."""
    contours, _ = cv2.findContours(occluder_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    occluder_crop = occluder_img[y:y+h, x:x+w]
    mask_crop = occluder_mask[y:y+h, x:x+w]
    return occluder_crop, mask_crop


def get_occluders(d, d_mask, data='MOSI'):
    """Select and apply an occluder to the frame."""
    aug = get_occluder_augmentor()
    occlude_type = random.choice(os.listdir(d))

    if occlude_type == 'dtd':  # texture
        d = os.path.join(d, occlude_type)
        image_files = []
        for root, _, files in os.walk(d):
            for file in files:
                if file.endswith(('.jpg')):
                    image_files.append(os.path.join(root, file))
        occlude_img = random.choice(image_files)
        occluder_img = cv2.imread(occlude_img)
        occluder_img = cv2.cvtColor(occluder_img, cv2.COLOR_BGR2RGB)
        occluder_mask = np.ones((occluder_img.shape[0], occluder_img.shape[1]), dtype=np.uint8) * 255     
    else: # object, hand
        d_mask = os.path.join(d_mask, occlude_type.split('_')[0]+'_masks')
        occlude_masks = os.listdir(d_mask)
        occlude_mask = random.choice(occlude_masks)
        print("occlude mask: ", os.path.join(d_mask, occlude_mask))
        occluder_mask = cv2.imread(os.path.join(d_mask, occlude_mask))
        occluder_mask = cv2.cvtColor(occluder_mask, cv2.COLOR_BGR2GRAY)
        occlude_img = occlude_mask.replace('png', 'jpeg') if (occlude_type == 'coco-object_img') else occlude_mask.replace('png', 'jpg')
        d = os.path.join(d, occlude_type)
        ori_occluder_img = cv2.imread(os.path.join(d, occlude_img), -1)
        ori_occluder_img = cv2.cvtColor(ori_occluder_img, cv2.COLOR_BGR2RGB)
        occluder_mask = cv2.resize(occluder_mask, (ori_occluder_img.shape[1], ori_occluder_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        occluder_img = cv2.bitwise_and(ori_occluder_img, ori_occluder_img, mask=occluder_mask)
        occluder_img, occluder_mask = extract_mask_bbox_crop(occluder_img, occluder_mask)
    
    # Apply augmentation
    transformed = aug(image=occluder_img, mask=occluder_mask)
    occluder_img, occluder_mask = transformed["image"], transformed["mask"]
    return occlude_img, occluder_img, occluder_mask


def calculate_landmark_bounds(landmark, frame):
    """Calculate bounding box dimensions for the landmark."""
    xmin = landmark[0, 0] * frame.shape[1]
    ymin = landmark[0, 1] * frame.shape[0]
    width = landmark[0, 2] * frame.shape[1]
    height = landmark[0, 3] * frame.shape[0]
    return xmin, ymin, width, height


def get_random_occluder_position(xmin, ymin, width, height, occ_size):
    """Get random occluder position within the landmark bounds."""
    x = random.randint(int(xmin), int(xmin + width) - occ_size)
    y = random.randint(int(ymin), int(ymin + height) - occ_size)
    return x, y


def apply_occlusion(frame, occluder_img, occluder_mask, occ_size, x, y):
    """Apply occluder to the frame."""
    # Resize the occluder image and mask
    occluder_img = cv2.resize(occluder_img, (occ_size, occ_size), interpolation=cv2.INTER_LANCZOS4)
    occluder_mask = cv2.resize(occluder_mask, (occ_size, occ_size), interpolation=cv2.INTER_LANCZOS4)

    # Create alpha mask and apply occlusion
    alpha_mask = np.expand_dims(occluder_mask, axis=2)
    alpha_mask = np.repeat(alpha_mask, 3, axis=2) / 255.0
    frame = overlay_image_alpha(frame, occluder_img, x, y, alpha_mask)
    return frame

def occlude_sequence(d, d_mask, img_seq, landmarks, freq, noise_percent, bgr=False, data='MOSI'):
    """
    Occlude sequence with fixed occluder for each video.
    Apply the same occluder and fixed position for all frames in a video.
    Randomize occluder and position for the next video.
    """
    total_frames = img_seq.shape[0]
    overlay_percent = random.choice([0.1, 0.2, 0.3, 0.4, 0.5])  # Face overlay percent

    # Keep the same occluder for the video sequence
    occlude_img, occluder_img, occluder_mask = get_occluders(d, d_mask, data=data)

    # Apply occlusion if no landmarks are available
    if landmarks is None:
        print("No landmarks available. Applying occlusion to random positions.")
        for i in range(total_frames):
            fr = img_seq[i]
            height, width, _ = fr.shape
            max_occ_size = int(min(width, height) * 0.95)
            occ_size = min(occ_size, max_occ_size)
            x = random.randint(0, width - occ_size)
            y = random.randint(0, height - occ_size)
            fr = apply_occlusion(fr, occluder_img, occluder_mask, occ_size, x, y)
            img_seq[i] = fr
        return np.array(img_seq), occlude_img
    
     # Apply occlusion based on first valid landmark frame
    first_landmark_idx = next((i for i, lm in enumerate(landmarks) if lm is not None), None)
    if first_landmark_idx is None:
        print("No valid landmarks found. Skipping occlusion.")
        return img_seq, occlude_img

    fr = img_seq[first_landmark_idx]
    xmin, ymin, width, height = calculate_landmark_bounds(landmarks[first_landmark_idx], fr)
    occ_size = int(math.sqrt(width * height * overlay_percent))

    if occ_size <= 0:
        print("Occluder size is too small. Skipping occlusion.")
        return img_seq, occlude_img

    x, y = get_random_occluder_position(xmin, ymin, width, height, occ_size)

    # Apply occlusion in specified frequency
    if freq == 0:  # Apply occlusion at regular intervals
        interval = int(1 / noise_percent)
        for i in range(0, total_frames, interval):
            fr = img_seq[i]
            fr = apply_occlusion(fr, occluder_img, occluder_mask, occ_size, x, y)
            img_seq[i] = fr

    elif freq == 1:  # Apply once at a random start position
        occ_len = int(total_frames * noise_percent)
        start_fr = random.randint(0, total_frames - occ_len)

        for i in range(occ_len):
            fr = img_seq[i + start_fr]
            fr = apply_occlusion(fr, occluder_img, occluder_mask, occ_size, x, y)
            img_seq[i + start_fr] = fr

    else:  # Apply at multiple random segments
        segment_length = total_frames // freq

        for j in range(freq):
            occ_len = int(segment_length * noise_percent)
            start_fr = random.randint(j * segment_length, (j + 1) * segment_length - occ_len)

            for i in range(occ_len):
                fr = img_seq[i + start_fr]
                fr = apply_occlusion(fr, occluder_img, occluder_mask, occ_size, x, y)
                img_seq[i + start_fr] = fr

    # Convert back to BGR if necessary
    if bgr:
        img_seq = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in img_seq]

    return np.array(img_seq), occlude_img

def occlude_sequence_noise(img_seq, freq, noise_percent, save_path=None):
    """Apply noise corruption to frames."""
    total_frames = img_seq.shape[0]
    occ_len = int(total_frames * noise_percent)

    if total_frames == 0 or occ_len == 0:
        print(f"Video too short or no valid frames for noise. Saving original video to {save_path}")
        if save_path:
            write_video(img_seq, save_path)
        return img_seq

    if freq == 0:   # Apply noise at regular intervals
        interval = total_frames // occ_len
        for i in range(occ_len):
            frame_idx = i * interval
            if frame_idx >= total_frames:
                break
            raw_frame = img_seq[frame_idx:frame_idx + 1]
            prob = random.random()
            if prob < 0.5:
                var = random.random() * 0.2
                raw_sequence = random_noise(raw_sequence, mode='gaussian', mean=0, var=var, clip=True) * 255
            elif prob < 1.0:
                blur = torchvision.transforms.GaussianBlur(kernel_size=(11, 11), sigma=(2.0, 2.0))
                raw_frame = blur(torch.tensor(raw_frame).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()
            img_seq[frame_idx:frame_idx + 1] = raw_frame

    elif freq == 1:   # Apply noise once at a random start position
        start_fr = random.randint(0, max(0, total_frames - occ_len))
        if start_fr + occ_len > total_frames:
            print(f"Invalid frame range for noise. Saving original video to {save_path}")
            if save_path:
                write_video(img_seq, save_path)
            return img_seq
        raw_sequence = img_seq[start_fr:start_fr + occ_len]
        prob = random.random()
        if prob < 0.5:
            var = random.random() * 0.2
            raw_sequence = random_noise(raw_sequence, mode='gaussian', mean=0, var=var, clip=True) * 255
        elif prob < 1.0:
            blur = torchvision.transforms.GaussianBlur(kernel_size=(11, 11), sigma=(2.0, 2.0))
            raw_sequence = blur(torch.tensor(raw_sequence).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()

        img_seq[start_fr:start_fr + occ_len] = raw_sequence

    else:   # Apply at multiple random segments
        len_segment = total_frames // freq
        for j in range(freq):
            start_range = max(0, len_segment * j)
            end_range = max(0, len_segment * (j + 1) - occ_len)
            if start_range >= end_range:
                print(f"Skipping range for segment {j}. Saving original video to {save_path}")
                if save_path:
                    write_video(img_seq, save_path)
                return img_seq
            start_fr = random.randint(start_range, end_range)
            raw_sequence = img_seq[start_fr:start_fr + occ_len]
            prob = random.random()
            if prob < 0.5:
                var = random.random() * 0.2
                raw_sequence = random_noise(raw_sequence, mode='gaussian', mean=0, var=var, clip=True) * 255
            elif prob < 1.0:
                blur = torchvision.transforms.GaussianBlur(kernel_size=(11, 11), sigma=(2.0, 2.0))
                raw_sequence = blur(torch.tensor(raw_sequence).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).numpy()
            img_seq[start_fr:start_fr + occ_len] = raw_sequence
    return img_seq


def landmarks_interpolate(landmarks):
    """landmarks_interpolate.

    :param landmarks: List, the raw landmark (in-place)

    """
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None

    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark" 

    return landmarks


def crop_patch(video_pathname, landmarks):
    """crop_patch.

    :param video_pathname: str, the filename for the processed video.
    :param landmarks: List, the interpolated landmarks.
    """
    frame_idx = 0
    frame_gen = read_video(video_pathname)
    while True:
        try:
            frame = frame_gen.__next__()  ## -- BGR
        except StopIteration:
            break
        if frame_idx == 0:
            sequence = []
        sequence.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1
    return np.array(sequence)

def normalize_landmarks(preprocessed_landmarks):
    normalized_landmarks = []
    for frame in preprocessed_landmarks:
        if isinstance(frame, (list, np.ndarray)) and np.array(frame).size == 4:
            normalized_landmarks.append(np.array(frame).reshape(1, 4))
        else:
            normalized_landmarks.append(np.array([[0.0, 0.0, 0.0, 0.0]]))
    return np.array(normalized_landmarks, dtype=float).reshape(-1, 1, 4)

def preprocess(video_pathname, landmarks_pathname):
    """
    Preprocess the video and landmarks (interpolation, normalization).
    """
    print("landmark path: ", landmarks_pathname)
    if isinstance(landmarks_pathname, str):
        with open(landmarks_pathname, "rb") as pkl_file:
            landmarks = pickle.load(pkl_file)
    else:
        landmarks = landmarks_pathname

    # Interpolate missing landmarks
    preprocessed_landmarks = landmarks_interpolate(landmarks)
    if preprocessed_landmarks is not None:
        preprocessed_landmarks = normalize_landmarks(preprocessed_landmarks)

    # Crop video frames based on the landmarks
    sequence = crop_patch(video_pathname, preprocessed_landmarks if preprocessed_landmarks is not None else [])

    return sequence, np.array(preprocessed_landmarks)


def load_video(data_filename, landmarks_filename=None):
    """load_video.

    :param data_filename: str, the filename of input sequence.
    :param landmarks_filename: str, the filename of landmarks.
    """
    assert landmarks_filename is not None

    sequence, landmark = preprocess(
        video_pathname=data_filename,
        landmarks_pathname=landmarks_filename,
    )
    return sequence, landmark


def read_video(filename):
    """load_video.
f
    :param filename: str, the fileanme for a video sequence.
    """
    cap = cv2.VideoCapture(filename)
    while (cap.isOpened()):
        ret, frame = cap.read()  # BGR
        if ret:
            yield frame
        else:
            break
    cap.release()


def write_video(video, filename):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = video[0].shape[:2]   # Set dimensions from first frame
    output = cv2.VideoWriter(filename, fourcc, 30, (width, height))

    for i, frame in enumerate(video):
        # Resize frame if dimensions don't match
        if frame.shape[:2] != (height, width):
            print(f"Frame {i} has unexpected size {frame.shape[:2]} (expected: {(height, width)})")
            frame = cv2.resize(frame, (width, height))

        # RGB to BGR
        if frame.shape[-1] == 3:  
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = np.clip(frame, 0, 255).astype(np.uint8)
        output.write(frame)

    output.release()


def linear_interpolate(landmarks, start_idx, stop_idx):
    """linear_interpolate.

    :param landmarks: ndarray, input landmarks to be interpolated.
    :param start_idx: int, the start index for linear interpolation.
    :param stop_idx: int, the stop for linear interpolation.
    """
    start_landmarks = np.array(landmarks[start_idx])
    stop_landmarks = np.array(landmarks[stop_idx])
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = start_landmarks + idx / float(stop_idx - start_idx) * delta
    return landmarks