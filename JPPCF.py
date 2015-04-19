# -*- coding: utf-8 -*-
import numpy as np
import logging


def tr(A, B):
    return (A * B).sum()


def computeTopicLoss(X, W, H, M, R, alpha, lambd, trXX, I):
    MR = M.dot(R)
    WH = W.dot(H)
    WMR = W.dot(MR)
    tr1 = trXX - 2*tr(X, WH) + tr(WH, WH)
    tr2 = trXX - 2*tr(X, WMR) + tr(WMR, WMR)
    tr3 = lambd*(tr(M,M) - 2*(M.sum())+ I.sum())
    tr4 = alpha*( H.sum() + W.sum() + M.sum() )
    obj = tr1+ tr2 + tr3+ tr4
    return obj

def computeLoss(R, P, Q, S, Po, alpha, lambd, trRR, I):
    SPo = S.dot(Po)
    PQ = P.dot(Q)
    SPoQ = SPo.dot(Q)
    tr1 = trRR - 2*tr(R, PQ) + tr(PQ, PQ)
    tr2 = trRR - 2*tr(R, SPoQ) + tr(SPoQ, SPoQ)
    tr3 = lambd*(tr(S, S) - 2*(S.sum())+ I.sum())
    tr4 = alpha*(P.sum() + Q.sum() + S.sum())
    obj = tr1+ tr2 + tr3+ tr4
    return obj


def computeLoss_with_topic(R, P, Q, S, Po, C, eta, alpha, lambd, trRR, I):
    reta = 1 - eta
    SPo = S.dot(Po)
    PQC = (reta*(P.dot(Q))) + (eta*C)
    SPoQC = (reta*(SPo.dot(Q))) + (eta*C)
    tr1 = trRR - 2*tr(R, PQC) + tr(PQC, PQC)
    tr2 = trRR - 2*tr(R, SPoQC) + tr(SPoQC, SPoQC)
    tr3 = lambd*(tr(S, S) - 2*(S.sum())+ I.sum())
    tr4 = alpha*(P.sum() + Q.sum() + S.sum())
    obj = tr1 + tr2 + tr3 + tr4
    return obj


def JPPTopic(X, R, k, lambd, alpha, epsilon, maxiter, verbose):

    # initilasation
    n,m = X.shape

    # randomly initialize W, Hu, Hs.
    W  = np.random.rand(n, k)
    H = np.random.rand(k, m)
    M = np.random.rand(k, k)
    kI = np.eye(k, k)
    # constants
    trXX = tr(X, X)

    obj = 1000000
    eps = 0.001
    prev_obj = 2 * obj
    delta = obj

    for i in range(1, maxiter):

        if delta < epsilon and i > 10:
            break
        J = M.dot(R)
        JT = J.T
        HT = H.T
        RT = R.T

        W = W * ((X.dot(HT+JT)) / (np.maximum(W.dot(J.dot(JT) + \
                H.dot(HT) + alpha), eps)) )

        WT = W.T
        WTW = WT.dot(W)
        WTX = WT.dot(X)

        M = M * ((WTX.dot(RT) + (lambd*kI)) / (np.maximum( (WTW.dot(J).dot(RT)) + \
                (lambd*M) + alpha, eps)))
        H = H * ( (WTX) / (np.maximum( WTW.dot(H) + alpha, eps)) )

        prev_obj = obj
        obj = computeTopicLoss(X, W, H, M, R, alpha, lambd, trXX, kI)
        delta = abs(prev_obj-obj)
        if verbose:
            logging.info('Iter: ' + str(i) + '\t Loss: ' + str(obj) + '\t Delta: ' \
                + str(delta) + '\n')

    logging.info('JPPTopic OK\n')

    return (W, H, M)

def JPPCF(R, Po, k, lambd, alpha, epsilon, maxiter, verbose):

    # fix seed for reproducable experiments

    # initilasation
    n,m = R.shape

    # randomly initialize W, Hu, Hs.
    P  = np.random.rand(n, k)
    Q = np.random.rand(k, m)
    S = np.random.rand(n, n)
    nI = np.eye(n)
    kI = np.eye(k, k)
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
        QQT = Q.dot(QT)

        P =  P * ( (R.dot(QT)) / \
            np.maximum(P.dot(QQT + (alpha*kI)),eps) )

        PT = P.T
        PoTST = Po.T.dot(S.T)
        PQ = P.dot(Q)
        SPo = S.dot(Po)

        Q = Q * (((PT + PoTST).dot(R)) / \
            np.maximum((PT.dot(P) + PoTST.dot(\
            SPo) + (alpha*kI)).dot(Q),eps))

        QTPoT = Q.T.dot(Po.T)
        SPoQ = SPo.dot(Q)

        S = S * ( ((R.dot(QTPoT)) + (lambd*nI)) / \
            np.maximum( (SPoQ.dot(QTPoT)) \
            + ((lambd + alpha)*S),eps) )

        prev_obj = obj
        obj = computeLoss(R, P, Q, S, Po, alpha, lambd, trRR, nI)
        delta = abs(prev_obj-obj)
        if verbose:
            logging.info('Iter: ' + str(i) + '\t Loss: ' + str(obj) + '\t Delta: ' \
                + str(delta) + '\n')

    logging.info('JPPCF OK\n')

    return (P, Q, S)

def matrix_sub(A, B):
    m, n = A.shape()
    R = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            r = A[i,j] - B[i,j]
            if r > 0:
                R[i,j] = r
    return R


def JPPCF_with_topic(R, Po, C, k, eta, lambd, alpha, epsilon, maxiter, verbose):

    # fix seed for reproducable experiments

    # initilasation
    n,m = R.shape

    # randomly initialize W, Hu, Hs.
    P  = np.random.rand(n, k)
    Q = np.random.rand(k, m)
    S = np.random.rand(n, n)
    nI = np.eye(n)
    kI = np.eye(k)

    # constants
    trRR = tr(R, R)

    obj = 1000000
    eps = 0.001
    prev_obj = 2 * obj
    delta = obj
    reta = 1 - eta
    etareta = eta * reta
    retareta = reta * reta

    for i in range(1, maxiter):

        if delta < epsilon and i > 10:
            break
        QT = Q.T
        QQT = np.dot(Q, QT)

        P =  P * ( ((np.maximum((reta*R - etareta*C), 0)).dot(QT)) / \
            np.maximum(P.dot(retareta*QQT + (alpha*kI)),eps) )

        PT = P.T
        PoTST = np.dot(Po.T, S.T)
        PQ = np.dot(P, Q)
        SPo = np.dot(S, Po)
        SPoQ = np.dot(SPo, Q)

        Q = Q * (((reta*(PT + PoTST)).dot(R)) / \
            np.maximum((reta*(PT.dot((eta*C) + (reta*PQ)) + PoTST.dot( \
            (eta*C) + (reta*SPoQ)))) + (alpha*Q),eps))

        QTPoT = np.dot(Q.T, Po.T)
        SPoQ = np.dot(SPo, Q)

        S = S * ( ((reta*(R.dot(QTPoT))) + (lambd*nI)) / \
            np.maximum( (reta*(((reta*SPoQ)+(eta*C)).dot(QTPoT))) \
            + ((lambd + alpha)*S),eps) )

        prev_obj = obj
        obj = computeLoss_with_topic(R, P, Q, S, Po, C, eta, alpha, lambd, trRR, nI)

        delta = abs(prev_obj-obj)
        if verbose:
            logging.info('Iter: ' + str(i) + '\t Loss: ' + str(obj) + '\t Delta: ' \
                + str(delta) + '\n')

    logging.info('JPPCF_with_topic OK\n')

    return (P, Q, S)
