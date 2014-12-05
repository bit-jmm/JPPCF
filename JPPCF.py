# -*- coding: utf-8 -*-
import math
import numpy as np
import scipy.sparse as sp
from sys import float_info


def tr(A, B):
    return (A * B).sum()


def computeLoss(R, P, Q, S, Po, C, alpha, lambd, trRR, I):
    SPo = S.dot(Po)
    PQC = P.dot(Q) + C
    SPoQC = SPo.dot(Q) + C
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

        P =  P * ( ((2*R -2*C).dot(Q.T)) / np.maximum(2*P.dot(Q.dot(Q.T) + alpha),eps) )
        Q = Q * (((P.T + Po.T.dot(S.T)).dot(R)) / np.maximum(P.T.dot(C+P.dot(Q)) + Po.T.dot(S.T).dot(C+S.dot(Po).dot(Q)) + alpha*Q,eps))
        S = S * ( ((R.dot(Q.T).dot(Po.T)) + (lambd*I)) / np.maximum( ((S.dot(Po).dot(Q)+C).dot(Q.T).dot(Po.T)) + ((lambd + alpha)*S),eps) )

        prev_obj = obj
        obj = computeLoss(R,P,Q,S,Po,C,alpha,lambd, trRR, I)
        delta = abs(prev_obj-obj)
        if verbose:
            print 'Iter: ', i, '\t Loss: ', obj, '\t Delta: ', delta, '\n'


    print 'end\n'

    return (P, Q, S)
