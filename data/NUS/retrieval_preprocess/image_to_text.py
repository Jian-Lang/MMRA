
import os

import pandas as pd
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


def loading_model():

    local_path = r"D:\MultiModalPopularityPrediction\LLM\model\image_to_text\BLIP"

    processor = BlipProcessor.from_pretrained(local_path)

    model = BlipForConditionalGeneration.from_pretrained(local_path).to("cuda")

    return processor, model


def convert_image_to_text(processor, model, image_path):

    raw_image = Image.open(image_path).convert('RGB')

    inputs = processor(raw_image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)

    text = processor.decode(out[0], skip_special_tokens=True)

    return text


if __name__ == "__main__":

    processor, model = loading_model()

    image_to_text_list = []

    df = pd.read_pickle(r'D:\MultiModalPopularityPrediction\data\vine\data.pkl')

    path = r'D:\MultiModalPopularityPrediction\data\vine\video_frames'

    files = os.listdir(path)

    for i in tqdm(range(len(df))):

        current_id = df['item_id'][i]

        current_text_list = []

        for j in range(0,4):

            image_path = os.path.join(path, f"{current_id}_{j}.jpg")

            text = convert_image_to_text(processor, model, image_path)

            current_text_list.append(text)

        image_to_text_list.append(current_text_list)

    df['image_to_text_list'] = image_to_text_list

    df.to_pickle(r'D:\MultiModalPopularityPrediction\data\vine\data.pkl')
