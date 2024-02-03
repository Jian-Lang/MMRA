import pandas as pd
from angle_emb import AnglE, Prompts
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

    df = pd.read_pickle(r'D:\MultiModalPopularityPrediction\data\vine\train.pkl')

    angle = load_angle_bert_model()

    text_semantic_embedding = []

    for i in tqdm(range(len(df))):
        text = df['image_to_text_list'][i]

        text = '[SEP]'.join(text)

        text = df['text'][i] + '[SEP]' + text

        vec = angle_bert_textual_feature_extraction(angle, text)

        text_semantic_embedding.append(vec)

    df['retrieval_feature_2'] = text_semantic_embedding

    df.to_pickle(r'D:\MultiModalPopularityPrediction\data\vine\train.pkl')

    text_semantic_embedding = []

    df = pd.read_pickle(r'D:\MultiModalPopularityPrediction\data\vine\valid.pkl')

    for i in tqdm(range(len(df))):
        text = df['image_to_text_list'][i]

        text = '[SEP]'.join(text)

        text = df['text'][i] + '[SEP]' + text

        vec = angle_bert_textual_feature_extraction(angle, text)

        text_semantic_embedding.append(vec)

    df['retrieval_feature_2'] = text_semantic_embedding

    df.to_pickle(r'D:\MultiModalPopularityPrediction\data\vine\valid.pkl')

    text_semantic_embedding = []

    df = pd.read_pickle(r'D:\MultiModalPopularityPrediction\data\vine\test.pkl')

    for i in tqdm(range(len(df))):
        text = df['image_to_text_list'][i]

        text = '[SEP]'.join(text)

        text = df['text'][i] + '[SEP]' + text

        vec = angle_bert_textual_feature_extraction(angle, text)

        text_semantic_embedding.append(vec)

    df['retrieval_feature_2'] = text_semantic_embedding

    df.to_pickle(r'D:\MultiModalPopularityPrediction\data\vine\test.pkl')
