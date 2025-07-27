from Visual_perturb import *
import random
import numpy as np
import os
import argparse
import mediapipe as mp


def video_gen(args):
    test_list = []
    # Collect all video file paths
    for root, _, files in os.walk(args.MOSI_main_dir):
        for file in files:
            if file.endswith('.mp4'):
                relative_path = os.path.relpath(os.path.join(root, file), args.MOSI_main_dir)
                test_list.append(os.path.splitext(relative_path)[0])

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
    
    random.seed(10)
    np.random.seed(10)
    
    for kk, test_file in enumerate(test_list):
        f_name = os.path.join(args.MOSI_main_dir, test_file + '.mp4')
        l_name = os.path.join(args.MOSI_landmark_dir, test_file + '.pkl')

        if not os.path.exists(l_name):
            os.makedirs(os.path.dirname(l_name), exist_ok=True)
            cap = cv2.VideoCapture(f_name)
            face_boxes = []
        
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(frame_rgb)
                
                if results.detections:
                    boxes = []
                    for detection in results.detections:
                        bboxC = detection.location_data.relative_bounding_box
                        boxes.append([bboxC.xmin, bboxC.ymin, bboxC.width, bboxC.height])
                    face_boxes.append(boxes)
                else:
                    face_boxes.append([[0.0, 0.0, 0.0, 0.0]])  # No face detected
            cap.release()
            
            with open(l_name, 'wb') as f:
                pickle.dump(face_boxes, f)
            print(f"Saved peakle to {l_name}")

        sequence, landmarks = load_video(f_name, l_name)

        # Skip fiels with missing data
        if sequence is None or landmarks is None:
            print(f"Skipping file {test_file} due to missing data.")
            continue 
        
        mode = args.mode if args.mode != 'random' else random.choice(['occlusion', 'noise'])
        print(f"Selected mode: {mode} for {test_file}")

        # Apply corruption
        noise_percent = args.noise_percent 

        if mode != 'noise':
            # freq = random.choice([0, 1, 2, 3])
            freq = 1
            bgr = True if mode == 'occlusion' else False
            sequence, _ = occlude_sequence(args.occlusion_img, args.occlusion_mask, sequence, landmarks, freq=freq, noise_percent=noise_percent, bgr=bgr, data='MOSI')
        if mode != 'occlusion':
            # freq = random.choice([0, 1, 2, 3])
            freq = 1
            sequence = occlude_sequence_noise(sequence, freq=freq, noise_percent=noise_percent)

        save_path = os.path.join(args.MOSI_save_loc, test_file)
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        write_video(sequence, save_path + '.mp4')
        print(f'{mode}:{kk+1}/{len(test_list)}', end='\r')
        print(f"Video Saved {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--MOSI_main_dir', type=str, default='./sample_data/video',
                        required=False, help='directory of the original MOSI video dataset')
    parser.add_argument('-l', '--MOSI_landmark_dir', type=str, default='./sample_data/mediapipe_landmark',
                        required=False, help='directory of the MOSI dataset face landmarks')
    parser.add_argument('-o', '--MOSI_save_loc', type=str, default='./results/visual_perturb',
                        help='directory to save corrupted videos')
    parser.add_argument('--occlusion_img', type=str, default='./occlusion_patch/image',
                        help='location of occlusion patch')
    parser.add_argument('--occlusion_mask', type=str, default='./occlusion_patch/mask',
                        help='location of occlusion patch mask')
    parser.add_argument('--noise_percent', type=float, default=0.1,
                        help='percentage of frames to apply noise (e.g., 0.1 for 10%)')
    parser.add_argument('--mode', type=str, default='random',
                        help='corruption mode ("occlusion", "noise", "random")')

    args = parser.parse_args()
    video_gen(args)
