import torch.utils.data
import pandas as pd
from model import TransductiveModel


def custom_collate_fn(batch):

    visual_feature_embedding, textual_feature_embedding, label = zip(*batch)

    return torch.tensor(visual_feature_embedding,dtype=torch.float), torch.tensor(textual_feature_embedding,dtype=torch.float), \
    torch.tensor(label,dtype=torch.float).unsqueeze(1)


class MyData(torch.utils.data.Dataset):

    def __init__(self, path):
        super().__init__()

        self.path = path

        self.dataframe = pd.read_pickle(path)

        self.visual_feature_embedding_list = self.dataframe['visual_feature_embedding_cls'].tolist()

        self.textual_feature_embedding_list = self.dataframe['textual_feature_embedding'].tolist()

        self.label_list = self.dataframe['label'].tolist()

    def __getitem__(self, index):

        visual_feature_embedding = self.visual_feature_embedding_list[index]

        textual_feature_embedding = self.textual_feature_embedding_list[index]

        label = self.label_list[index]

        return visual_feature_embedding, textual_feature_embedding, label

    def __len__(self):
        return len(self.dataframe)


if __name__ == "__main__":

    model = TransductiveModel(num_modalities=2, feature_dims=[768, 768], hidden_dim=256, output_dim=1)

    dataset = MyData(r'D:\MultiModalPopularityPrediction\data\tmall_microlens\train.pkl')

    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=64, collate_fn=custom_collate_fn)

    for batch in data_loader:

        batch = [item.to('cuda') if isinstance(item, torch.Tensor) else item for item in batch]

        visual_feature_embedding, textual_feature_embedding,visual_feature_embedding_test,textual_feature_embedding_test,label = batch

        visual_feature_embedding = torch.cat([visual_feature_embedding,visual_feature_embedding_test],dim=0)

        textual_feature_embedding = torch.cat([textual_feature_embedding,textual_feature_embedding_test],dim=0)

        output = model.forward([visual_feature_embedding, textual_feature_embedding])

        print(output)


