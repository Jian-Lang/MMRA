import pandas as pd
from tqdm import tqdm

df_train = pd.read_pickle(r'D:\MultiModalPopularityPrediction\data\vine\train.pkl')

df_test = pd.read_pickle(r'D:\MultiModalPopularityPrediction\data\vine\test.pkl')

df_valid = pd.read_pickle(r'D:\MultiModalPopularityPrediction\data\vine\valid.pkl')

df_img_feature = pd.read_pickle(r'D:\MultiModalPopularityPrediction\data\vine\vine_image.pkl')

visual_feature_embedding_cls_list = []

visual_feature_embedding_mean_list = []

for i in tqdm(range(len(df_train))):

    item_id = df_train['item_id'][i]

    current_visual_feature_embedding_cls_list = []

    current_visual_feature_embedding_mean_list = []

    for j in range(4):

        image_id = f"{item_id}_{j}.jpg"

        index = df_img_feature[df_img_feature['image_id'] == image_id].index[0]

        visual_feature_embedding_cls = df_img_feature['visual_feature_embedding_cls'][index]

        visual_feature_embedding_mean = df_img_feature['visual_feature_embedding_mean'][index]

        current_visual_feature_embedding_cls_list.append(visual_feature_embedding_cls)

        current_visual_feature_embedding_mean_list.append(visual_feature_embedding_mean)

    visual_feature_embedding_cls_list.append(current_visual_feature_embedding_cls_list)

    visual_feature_embedding_mean_list.append(current_visual_feature_embedding_mean_list)

df_train['visual_feature_embedding_cls'] = visual_feature_embedding_cls_list

df_train['visual_feature_embedding_mean'] = visual_feature_embedding_mean_list

df_train.to_pickle(r'D:\MultiModalPopularityPrediction\data\vine\train.pkl')

visual_feature_embedding_cls_list = []

visual_feature_embedding_mean_list = []

for i in tqdm(range(len(df_test))):

    item_id = df_test['item_id'][i]

    current_visual_feature_embedding_cls_list = []

    current_visual_feature_embedding_mean_list = []

    for j in range(4):

        image_id = f"{item_id}_{j}.jpg"

        index = df_img_feature[df_img_feature['image_id'] == image_id].index[0]

        visual_feature_embedding_cls = df_img_feature['visual_feature_embedding_cls'][index]

        visual_feature_embedding_mean = df_img_feature['visual_feature_embedding_mean'][index]

        current_visual_feature_embedding_cls_list.append(visual_feature_embedding_cls)

        current_visual_feature_embedding_mean_list.append(visual_feature_embedding_mean)

    visual_feature_embedding_cls_list.append(current_visual_feature_embedding_cls_list)

    visual_feature_embedding_mean_list.append(current_visual_feature_embedding_mean_list)

df_test['visual_feature_embedding_cls'] = visual_feature_embedding_cls_list

df_test['visual_feature_embedding_mean'] = visual_feature_embedding_mean_list

df_test.to_pickle(r'D:\MultiModalPopularityPrediction\data\vine\test.pkl')

visual_feature_embedding_cls_list = []

visual_feature_embedding_mean_list = []

for i in tqdm(range(len(df_valid))):

    item_id = df_valid['item_id'][i]

    current_visual_feature_embedding_cls_list = []

    current_visual_feature_embedding_mean_list = []

    for j in range(4):

        image_id = f"{item_id}_{j}.jpg"

        index = df_img_feature[df_img_feature['image_id'] == image_id].index[0]

        visual_feature_embedding_cls = df_img_feature['visual_feature_embedding_cls'][index]

        visual_feature_embedding_mean = df_img_feature['visual_feature_embedding_mean'][index]

        current_visual_feature_embedding_cls_list.append(visual_feature_embedding_cls)

        current_visual_feature_embedding_mean_list.append(visual_feature_embedding_mean)

    visual_feature_embedding_cls_list.append(current_visual_feature_embedding_cls_list)

    visual_feature_embedding_mean_list.append(current_visual_feature_embedding_mean_list)

df_valid['visual_feature_embedding_cls'] = visual_feature_embedding_cls_list

df_valid['visual_feature_embedding_mean'] = visual_feature_embedding_mean_list

df_valid.to_pickle(r'D:\MultiModalPopularityPrediction\data\vine\valid.pkl')