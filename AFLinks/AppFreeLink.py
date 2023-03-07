import os
import glob
import torch
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment

from os.path import join
from random import randint, normalvariate
from torch.utils.data import Dataset, DataLoader
from AFLinks.dataset import LinkData
from AFLinks.model import PostLinker


INFINITY = 1e5
thrT=(-10, 30) # (-10, 30) for CenterTrack, FairMOT, TransTrack.
thrS=35
thrP=0.05

dataset = LinkData('', '')

class AFlink:
    def __init__(self, folder, inpath, outpath, model_path="MOT20/AFLink_epoch20.pth"):
        self.track = np.loadtxt(os.path.join(folder, inpath), delimiter=',')
        self.model = self.aflink_model(model_path)
        self.save_name = os.path.join(folder, outpath)
        print("track", self.track.shape)
        
    def aflink_model(self, path_AFLink):
        model = PostLinker()
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(path_AFLink))
        else:
            model.load_state_dict(torch.load(path_AFLink, map_location=torch.device('cpu')))
        return model

    def gather_info(self, track):
        id2info = defaultdict(list)
        track = track[np.argsort(track[:, 0])]  # 按帧排序
        for row in track:
            f, i, x, y, w, h = row[:6]
            id2info[i].append([f, x, y, w, h])
        track = np.array(track)
        id2info = {k: np.array(v) for k, v in id2info.items()}
        return id2info

    def predict(self, track1, track2):
        #print(track1.shape, track2.shape)
        track1, track2 = dataset.transform(track1, track2)
        # torch.Size([1, 30, 5]) torch.Size([1, 30, 5])
        track1, track2 = track1.unsqueeze(0), track2.unsqueeze(0)
        # torch.Size([1, 1, 30, 5]) torch.Size([1, 1, 30, 5])
        score = self.model(track1, track2)[0, 1].detach().cpu().numpy()
        return 1 - score

    def compression(self, cost_matrix, ids):
        # 行压缩
        mask_row = cost_matrix.min(axis=1) < thrP
        matrix = cost_matrix[mask_row, :]
        ids_row = ids[mask_row]
        # 列压缩
        mask_col = cost_matrix.min(axis=0) < thrP
        matrix = matrix[:, mask_col]
        ids_col = ids[mask_col]
        # 矩阵压缩
        return matrix, ids_row, ids_col

    def deduplicate(self, tracks):
        _, index = np.unique(tracks[:, :2], return_index=True, axis=0)  # 保证帧号和ID号的唯一性
        return tracks[index]
    
    def link(self, save=None):
        id2info = self.gather_info(self.track)
        num = len(id2info)
        ids = np.array(list(id2info))  # target ID
        fn_l2 = lambda x, y: np.sqrt(x ** 2 + y ** 2)  # L2 distance
        cost_matrix = np.ones((num, num)) * INFINITY  # cont matrix
        for i, id_i in enumerate(ids):
            for j, id_j in enumerate(ids):
                if id_i == id_j: continue
                info_i, info_j = id2info[id_i], id2info[id_j]
                fi, bi = info_i[-1][0], info_i[-1][1:3]
                fj, bj = info_j[0][0], info_j[0][1:3]
                if not thrT[0] <= fj - fi < thrT[1]: continue
                if thrS < fn_l2(bi[0] - bj[0], bi[1] - bj[1]): continue
                cost = self.predict(info_i, info_j)
                if cost <= thrP: cost_matrix[i, j] = cost

        id2id = dict()
        ID2ID = dict()
        cost_matrix, ids_row, ids_col = self.compression(cost_matrix, ids)
        indices = linear_sum_assignment(cost_matrix)
        for i, j in zip(indices[0], indices[1]):
            if cost_matrix[i, j] < thrP:
                id2id[ids_row[i]] = ids_col[j]
        for k, v in id2id.items():
            if k in ID2ID:
                ID2ID[v] = ID2ID[k]
            else:
                ID2ID[v] = k

        '''strage of results'''
        res = self.track.copy()
        for k, v in ID2ID.items():
            print(k, v)
            res[res[:, 1] == k, 1] = v
        res = self.deduplicate(res)
        if save:
            np.savetxt(self.save_name, res, fmt='%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%d,%d,%d')
        return res
