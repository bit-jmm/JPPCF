# -*- coding: utf-8 -*-
# gaussian-nmf
import math
import numpy as np
import scipy.sparse as sp
from sys import float_info
import logging


class NMF(object):

    # ランダム疎行列
    def convert_sparse_matrix(self, data):
        s1, s2 = data.shape
        if s1 == 3:
            data = data.T
        vals = data[:, 2]
        rows = data[:, 0]
        cols = data[:, 1]
        n = rows.max()
        m = cols.max()
        A = sp.csr_matrix((vals, (rows, cols)), shape=(n + 1, m + 1))
        return A

    # ランダム疎行列
    def rand_sparse_matrix(self, n=10, m=10, alpha=0.1):
        num = int(n * m * alpha)
        vals = np.random.rand(num)
        rows = np.random.randint(0, n, size=num)
        cols = np.random.randint(0, m, size=num)
        A = sp.csr_matrix((vals, (rows, cols)), shape=(n, m))
        return A

    # セットアップ
    def setup(self, A, k=5, iter_num=100, lambd=0.0005,
              epsilon=0.01, verbose = True):
        self.iter_num = iter_num
        self.k = k
        self.lambd = lambd
        self.epsilon = epsilon
        self.verbose = verbose
        n, m = A.shape
        # sigma = ((float)(A.size)) / n / m
        W0 = np.random.rand(n, k)
        H0 = np.random.rand(k, m)

        self.errors = []
        self.A = A
        self.H = H0
        self.W = W0

        self.clusterH = None
        self.clusterW = None

    def __tr(self, A, B):
        return (A * B).sum()

    def __computeLoss(self, A, W, H, lambd, trAA):
        WH = W.dot(H)
        tr1 = trAA - 2*self.__tr(A, WH) + self.__tr(WH, WH)
        tr2 = lambd * (W.sum() + H.sum())
        obj = tr1 + tr2
        return obj

    # nmf実行本体
    # iter_num: イタレーション回数
    # calc_error: エラー計算するかどうか．ここが一番重い
    # calc_error_num: 何回に1回エラー計算するか
    def run(self):
        # 初期化
        H = self.H
        W = self.W
        eps = float_info.min
        A = self.A
        iter_num = self.iter_num
        epsilon = self.epsilon
        lambd = self.lambd
        verbose = self.verbose

        trAA = self.__tr(A, A)

        obj = 1000000
        pre_obj = 2*obj
        delta = obj

        # イテレーション
        for i in range(1, iter_num):
            if delta < epsilon:
                break
            # update W
            # W=W.*(A*H')./(W*(H*H')+eps);
            W = W * ((A.dot(H.T)) / np.maximum(W.dot(H.dot(H.T)) + lambd, eps))
            WtW = W.T.dot(W)
            WtA = W.T.dot(A)
            # update H
            # H=H.*(W'*A)./(W'*W*H+eps)
            H = H * ((WtA) / np.maximum(WtW.dot(H)+lambd, eps))

            pre_obj = obj
            obj = self.__computeLoss(A, W, H, lambd, trAA)
            delta = abs(pre_obj-obj)

            if verbose:
                logging.info('Iter: ' + str(i) + '\t Loss: ' + str(obj) + '\t Delta: ' + str(delta) + '\n')
        self.H = H
        self.W = W

        logging.info('end')

    # W,Hから最大になった因数を取る
    def clusters(self):
        self.clusterH = np.argmax(self.H, 0)
        self.clusterW = np.argmax(self.W, 1)
