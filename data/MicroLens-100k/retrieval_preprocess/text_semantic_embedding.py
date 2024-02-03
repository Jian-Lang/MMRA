

import pandas as pd
from angle_emb import AnglE, Prompts
from tqdm import tqdm


def loading_model():

    local_model_path = r"D:\MultiModalPopularityPrediction\LLM\model\text_semantic_embedding\UAE-Large-V1"

    angle = AnglE.from_pretrained(local_model_path, pooling_strategy='cls').cuda()

    angle.set_prompt(prompt=Prompts.C)

    return angle


def convert_text_to_embedding(angle, text):

    vec = angle.encode({'text': text}, to_numpy=True)

    return vec


if __name__ == "__main__":

    df = pd.read_pickle(r'D:\MultiModalPopularityPrediction\data\MicroLens-100k\data.pkl')

    angle = loading_model()

    text_semantic_embedding = []

    for i in tqdm(range(len(df))):

        text = df['image_to_text_list'][i]

        text = '[SEP]'.join(text)

        text = df['text'][i] + '[SEP]' + text

        vec = ((convert_text_to_embedding(angle, text)).tolist())[0]

        text_semantic_embedding.append(vec)

    df['retrieval_feature'] = text_semantic_embedding

    df.to_pickle(r'D:\MultiModalPopularityPrediction\data\MicroLens-100k\data.pkl')
