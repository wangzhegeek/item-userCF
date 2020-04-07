# -*- coding:utf-8 -*-

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from utils import get_precision_recall, get_ndcg
from sklearn.preprocessing import MaxAbsScaler


class Collaborative_Filtering(object):

    def __init__(self, train_data, user_col, item_col, label_col, method_type='item', use_sparse=False, cut_num=20, user_norm=True,
                 item_norm=True):
        if method_type not in ['user', 'item']:
            raise ValueError("method_type must be one of 'item' or 'user' ")
        self.data = train_data
        self.user_col = user_col
        self.item_col = item_col
        self.label_col = label_col
        self.n_item = train_data[item_col].nunique()
        self.n_user = train_data[user_col].nunique()
        self.type = method_type
        self.use_sparse = use_sparse
        self.cut_num = cut_num
        self.user_norm = user_norm
        self.item_norm = item_norm
        self.tfidf = TfidfTransformer()

    def get_score_mat(self):
        if self.use_sparse == True:
            score_mat = lil_matrix((self.n_user, self.n_item))
        else:
            score_mat = np.zeros((self.n_user, self.n_item))
        for i in self.data.itertuples():
            score_mat[getattr(i, self.user_col), getattr(i, self.item_col)] = getattr(i, self.label_col)
        if self.use_sparse == True:
            score_mat = csr_matrix(score_mat)
        if self.user_norm:
            score_mat = self.tfidf.fit_transform(score_mat)
        return score_mat

    def get_sim_matrix(self, score_mat):
        if self.type == 'item':
            score_mat = score_mat.T
        sim_mat = cosine_similarity(score_mat)
        if self.item_norm:
            scaler = MaxAbsScaler()
            sim_mat = scaler.fit_transform(sim_mat)
        return sim_mat

    def predict(self, score_mat, sim_mat, filter=True):
        if type(score_mat) != np.ndarray:
            score_mat = score_mat.toarray()
        if self.type =='item':
            pred = score_mat.dot(sim_mat)
        else:
            pred = sim_mat.dot(score_mat)
        if filter == True:
            mask = np.where(score_mat == 0, 1, 0)
            pred = pred*mask
        result = np.argsort(-pred, axis=1)[:, :self.cut_num]
        return result

    def evaluate(self, test_data, preds):
        test_data['labels'] = test_data[self.item_col].map(str) + ':' + test_data[self.label_col].map(str)
        data_group = test_data[[self.user_col, 'labels']].groupby(self.user_col).agg(list).reset_index()
        def rank(row):
            new_row = []
            for item in row:
                pair = (int(item.split(':')[0]), float(item.split(':')[1]))
                new_row.append(pair)
            sort_row = sorted(new_row, key=lambda x: x[1], reverse=True)
            sort_row = list(map(lambda x: x[0], sort_row))
            res = np.array(sort_row)
            return res
        data_group['labels'] = data_group['labels'].apply(rank)
        labels = data_group.set_index(self.user_col).to_dict()['labels']
        precision, recall, ndcg = 0, 0, 0
        for i, label in labels.items():
            tmp_pre, tmp_recall = get_precision_recall(label, preds[i])
            tmp_ndcg = get_ndcg(label, preds[i], top_k=self.cut_num)
            precision += tmp_pre
            recall += tmp_recall
            ndcg += tmp_ndcg
        print("precision: {:.3f}, recall: {:.3f}, ndcg: {:.3f}".format(precision/len(labels), recall/len(labels), ndcg/len(labels)))








