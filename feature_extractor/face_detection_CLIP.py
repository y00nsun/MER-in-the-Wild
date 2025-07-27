import os
import cv2
import mediapipe as mp
import torch
import clip
from PIL import Image
import pickle
import numpy as np
import argparse
import time
import scipy.signal as spsig

# Mediapipe face detection
mp_face_detection = mp.solutions.face_detection

# CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

TARGET_FRAME = 500          # Target number of frames for resampling
EMBEDDING_DIM = 512         # CLIP model output embedding dimension
RESAMPLE_THRESHOLD = 30     # Minimum number of frames required before resampling

def process_video(video_path):
    """
    Process video to extract face features using CLIP.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_faces = []

    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        for _ in range(frame_count):
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            if results.detections:
                detection = results.detections[0]
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = (bboxC.xmin * iw, bboxC.ymin * ih,
                              bboxC.width * iw, bboxC.height * ih)

                if w > 0 and h > 0:
                    face = frame_rgb[int(y):int(y+h), int(x):int(x+w)]
                    face_pil = Image.fromarray(face)

                    if face_pil.size[0] > 0 and face_pil.size[1] > 0:
                        face_preprocessed = preprocess(face_pil).unsqueeze(0).to(device)
                        with torch.no_grad():
                            face_features = model.encode_image(face_preprocessed).squeeze(0).cpu().numpy()
                        video_faces.append(face_features)

    cap.release()

    # Sampling
    num_faces = len(video_faces)
    print(f"[INFO] Detected {num_faces} faces in {os.path.basename(video_path)}")
    if num_faces == 0:
        print(f"[WARN] No face detected in {video_path}")
        return np.zeros((TARGET_FRAME, EMBEDDING_DIM), dtype=np.float32)
    video_faces = np.array(video_faces)  # (T, 512)
    if video_faces.ndim != 2 or video_faces.shape[1] != EMBEDDING_DIM:
        print(f"[ERROR] Unexpected feature shape in {video_path}: {video_faces.shape}")
        return np.zeros((TARGET_FRAME, EMBEDDING_DIM), dtype=np.float32)

    # Resampling or padding
    if num_faces < RESAMPLE_THRESHOLD:  
        sampled_faces = np.zeros((TARGET_FRAME, EMBEDDING_DIM), dtype=np.float32)
        sampled_faces[:num_faces] = video_faces
    elif num_faces == TARGET_FRAME:
        sampled_faces = video_faces
    else: 
        sampled_faces = spsig.resample_poly(video_faces, up=TARGET_FRAME, down=num_faces, axis=0)

    return np.squeeze(sampled_faces)

def process_folder(input_folder, output_folder):
    """Process all video files in a folder."""
    for parent_folder, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".mp4"):
                video_id = os.path.splitext(file)[0]
                input_file_path = os.path.join(parent_folder, file)
                print(f"Processing: {input_file_path}")
                start_time = time.time()

                video_faces = process_video(input_file_path)

                relative_path = os.path.relpath(parent_folder, input_folder)
                output_folder_path = os.path.join(output_folder, relative_path)
                os.makedirs(output_folder_path, exist_ok=True)

                output_file_path = os.path.join(output_folder_path, f"{video_id}.pkl")
                with open(output_file_path, 'wb') as f:
                    pickle.dump(video_faces, f)

                elapsed = time.time() - start_time
                print(f"Saved to: {output_file_path} | Time: {elapsed:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video files and extract face features using CLIP.")
    parser.add_argument("--input_folder", required=True, help="Path to the input folder containing video files.")
    parser.add_argument("--output_folder", required=True, help="Path to the output folder to save processed features.")
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder)
