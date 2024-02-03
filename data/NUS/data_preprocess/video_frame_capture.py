import cv2
import os

import pandas as pd
from tqdm import tqdm
import numpy as np


def extract_frames(input_video, input_video_id, k):
    cap = cv2.VideoCapture(input_video)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    samples = (np.linspace(0, total_frames - 1, k)).tolist()

    samples = [int(i) for i in samples]

    output_folder = r'D:\MultiModalPopularityPrediction\data\vine\video_frames'

    os.makedirs(output_folder, exist_ok=True)

    frame_count = 0

    index = 0

    while True:
        try:
            ret, frame = cap.read()

            if not ret:
                if index < k:
                    output_path = os.path.join(output_folder, f"{input_video_id}_{index}.jpg")
                    cv2.imwrite(output_path, old_frame)
                break

            if frame_count in samples:
                output_path = os.path.join(output_folder, f"{input_video_id}_{index}.jpg")
                cv2.imwrite(output_path, frame)
                index += 1

            frame_count += 1
            old_frame = frame

        except Exception as e:

            pass

    cap.release()


if __name__ == '__main__':

    df = pd.read_pickle(r'D:\MultiModalPopularityPrediction\wrong.pkl')

    video_id = df['video_id'].tolist()

    video_id = [f"{i}.mp4" for i in video_id]

    path = r'D:\MultiModalPopularityPrediction\data\vine\vine_video\video'

    files = os.listdir(path)

    k = 4

    for i in tqdm(range(3375 + 2 + 2 + 1 + 2 + 1 + 1 + 1 + 1 + 506 + 3122, len(video_id))):
        input_video_path = os.path.join(path, video_id[i])

        input_video_id = (video_id[i])[:-4]

        extract_frames(input_video_path, input_video_id, k)
