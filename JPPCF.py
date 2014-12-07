# -*- coding: utf-8 -*-
import math
import numpy as np
# import scipy.sparse as sp
from sys import float_info
import logging

def tr(A, B):
    return (A * B).sum()


def computeLoss(R, P, Q, S, Po, C, alpha, lambd, trRR, I):
    SPo = np.dot(S, Po)
    PQC = np.dot(P, Q) + C
    SPoQC = np.dot(SPo, Q) + C
    tr1 = trRR - 2*tr(R,PQC) + tr(PQC,PQC)
    tr2 = trRR - 2*tr(R,SPoQC) + tr(SPoQC,SPoQC)
    tr3 = lambd*(tr(S,S) - 2*(S.sum())+ I.sum())
    tr4 = alpha*(P.sum() + Q.sum() + S.sum())
    obj = tr1+ tr2 + tr3+ tr4
    return obj

def JPPCF(R, Po, k, lambd, alpha, epsilon, maxiter, verbose):

    # fix seed for reproducable experiments

    # initilasation
    n,v1 = R.shape

    # randomly initialize W, Hu, Hs.
    P  = np.random.rand(n,k)
    Q = np.random.rand(k, v1)
    S = np.random.rand(n, n)
    C = np.zeros(R.shape)
    I = np.eye(n)

    # constants
    trRR = tr(R, R)

    obj = 1000000
    eps = 0.001
    prev_obj = 2 * obj
    delta = obj

    for i in range(1, maxiter):

        if delta < epsilon and i > 10:
            break
        QT = Q.T
        QQT = np.dot(Q, QT)
        
        P =  P * ( (np.dot((2*R -2*C), QT)) / \
            np.maximum(2*np.dot(P, QQT + alpha),eps) )

        PT = P.T
        PoTST = np.dot(Po.T, S.T)
        PQ = np.dot(P, Q)
        SPo = np.dot(S, Po)
        SPoQ = np.dot(SPo, Q)
        
        Q = Q * ((np.dot((PT + PoTST), R)) / \
            np.maximum(np.dot(PT, C + PQ) + np.dot(PoTST, \
            C+SPoQ) + alpha*Q,eps))

        QTPoT = np.dot(Q.T, Po.T)
        SPoQ = np.dot(SPo, Q)

        S = S * ( ((np.dot(R, QTPoT)) + (lambd*I)) / \
            np.maximum( (np.dot((SPoQ+C), QTPoT)) \
            + ((lambd + alpha)*S),eps) )

        prev_obj = obj
        obj = computeLoss(R,P,Q,S,Po,C,alpha,lambd, trRR, I)
        delta = abs(prev_obj-obj)
        if verbose:
            logging.info('Iter: ' + str(i) + '\t Loss: ' + str(obj) + '\t Delta: ' \
                + str(delta) + '\n')

    logging.info('JPPCF OK\n')

    return (P, Q, S)
