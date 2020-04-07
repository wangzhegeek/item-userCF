import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time
from CF import Collaborative_Filtering
import gc


def data_process(data_path):
    rating_data = pd.read_csv(data_path, sep=',', usecols=['user_id', 'movie_id', 'rating']).drop_duplicates()
    train, test = train_test_split(rating_data, test_size=0.2)
    return rating_data, test


if __name__ == "__main__":
    data_path = './movielens_sample.txt'
    train_data, test_data = data_process(data_path)
    userId = train_data['user_id'].unique()
    itemId = train_data['movie_id'].unique()
    test_data = test_data[test_data['user_id'].isin(userId)]
    test_data = test_data[test_data['movie_id'].isin(itemId)]

    sparse_features = ['user_id', 'movie_id']

    # 1.Label Encoding for sparse features,and process sequence features
    for feat in sparse_features:
        lbe = LabelEncoder()
        train_data[feat] = lbe.fit_transform(train_data[feat])
        test_data[feat] = lbe.transform(test_data[feat])
    user_col, item_col, label_col = 'user_id', 'movie_id', 'rating'

    start_time = time.time()
    cf = Collaborative_Filtering(train_data, user_col, item_col, label_col, method_type='user', use_sparse=True,
                                 user_norm=False, item_norm=True)

    # train
    score_mat = cf.get_score_mat().astype(np.int)
    sim_mat = cf.get_sim_matrix(score_mat).astype(np.float16)

    gc.collect()
    # test
    print(score_mat.shape, sim_mat.shape)
    preds = cf.predict(score_mat, sim_mat, filter=True)
    cf.evaluate(test_data, preds)
    print("time cost: ", time.time() - start_time)