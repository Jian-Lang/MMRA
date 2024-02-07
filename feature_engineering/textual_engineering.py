import pandas as pd
from angle_emb import AnglE
from tqdm import tqdm


def load_angle_bert_model():
    angle = AnglE.from_pretrained(
        r'D:\MultiModalPopularityPrediction\LLM\model\textual_feature_extraction\angle-bert-base-uncased-nli-en-v1',
        pooling_strategy='cls_avg').cuda()

    return angle


def angle_bert_textual_feature_extraction(angle, text):
    text_embedding = angle.encode(text, to_numpy=True)

    return (text_embedding[0]).tolist()


if __name__ == "__main__":

    angle = load_angle_bert_model()

    data_path = r'D:\MultiModalPopularityPrediction\data\NUS\data.pkl'

    df = pd.read_pickle(data_path)

    textual_feature_embedding = []

    for i in tqdm(range(len(df))):
        text = df['text'][i]

        output = angle_bert_textual_feature_extraction(angle, text)

        textual_feature_embedding.append(output)

    df['textual_feature_embedding'] = textual_feature_embedding

    df.to_pickle(data_path)
