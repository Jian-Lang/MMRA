import pandas as pd
import torch
from tqdm import tqdm
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import os


def load_vit_model():
    processor = ViTImageProcessor.from_pretrained(
        r'D:\MultiModalPopularityPrediction\LLM\model\visual_feature_extraction\vit-base-patch16-224-in21k')

    model = ViTModel.from_pretrained(
        r'D:\MultiModalPopularityPrediction\LLM\model\visual_feature_extraction\vit-base-patch16-224-in21k')

    return processor, model


def vit_visual_feature_extraction(processor, model, image_path):
    image = Image.open(image_path)

    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)

    cls_output = outputs.last_hidden_state[:, 0, :]

    mean_pooling_output = torch.mean(outputs.last_hidden_state, dim=1)

    return (cls_output[0]).tolist(), (mean_pooling_output[0]).tolist()


if __name__ == '__main__':

    processor, model = load_vit_model()

    visual_feature_embedding_cls = []

    visual_feature_embedding_mean = []

    frames_path = r'D:\MultiModalPopularityPredictionData\vine\video_frames'

    files = os.listdir(frames_path)

    for i in tqdm(range(len(files))):
        image_path = os.path.join(frames_path, files[i])

        cls_output, mean_pooling_output = vit_visual_feature_extraction(processor, model, image_path)

        visual_feature_embedding_cls.append(cls_output)

        visual_feature_embedding_mean.append(mean_pooling_output)

    df = pd.DataFrame()

    df['image_id'] = files

    df['visual_feature_embedding_cls'] = visual_feature_embedding_cls

    df['visual_feature_embedding_mean'] = visual_feature_embedding_mean

    df.to_pickle(r'vine_image.pkl')
