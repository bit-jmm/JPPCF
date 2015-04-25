# -*- coding: utf-8 -*-
import numpy as np
import logging
import multiprocessing as mul

def tr(A, B):
    return (A * B).sum()

def mult(matices):
    return np.dot(matices[0], matices[1])

def matrix_dots_map_async(matrices):
    pool_size = len(matrices)
    cpu_num = mul.cpu_count()
    if pool_size > cpu_num:
        pool_size = cpu_num
    p = mul.Pool(pool_size)
    result = p.map_async(mult, matrices)
    p.close()
    p.join()
    return result.get()

def tran(matrix):
    return matrix.T

def matrix_trans_map_async(matrices):
    pool_size = len(matrices)
    cpu_num = mul.cpu_count()
    if pool_size > cpu_num:
        pool_size = cpu_num
    p = mul.Pool(pool_size)
    result = p.map_async(tran, matrices)
    p.close()
    p.join()
    return result.get()


def computeTopicLoss(X, W, H, M, R, alpha, lambd, trXX, I):
    MR = np.dot(M, R)
    WH = np.dot(W, H)
    WMR = np.dot(W, MR)
    tr1 = trXX - 2*tr(X, WH) + tr(WH, WH)
    tr2 = trXX - 2*tr(X, WMR) + tr(WMR, WMR)
    tr3 = lambd*(tr(M,M) - 2*(M.sum())+ I.sum())
    tr4 = alpha*( H.sum() + W.sum() + M.sum() )
    obj = tr1+ tr2 + tr3+ tr4
    return obj

def computeLoss(R, P, Q, S, Po, alpha, lambd, trRR, I):
    SPo = np.dot(S, Po)
    PQ = np.dot(P, Q)
    SPoQ = np.dot(SPo, Q)
    tr1 = trRR - 2*tr(R, PQ) + tr(PQ, PQ)
    tr2 = trRR - 2*tr(R, SPoQ) + tr(SPoQ, SPoQ)
    tr3 = lambd*(tr(S, S) - 2*(S.sum())+ I.sum())
    tr4 = alpha*(P.sum() + Q.sum() + S.sum())
    obj = tr1+ tr2 + tr3+ tr4
    return obj


def computeLoss_with_topic(R, P, Q, S, Po, C, eta, alpha, lambd, trRR, I):
    reta = 1 - eta
    SPo = np.dot(S, Po)
    PQC = (reta*(np.dot(P, Q))) + (eta*C)
    SPoQC = (reta*(np.dot(SPo, Q))) + (eta*C)
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
        J = np.dot(M, R)
        result = matrix_trans_map_async([J, H, R])
        JT = result[0]
        HT = result[1]
        RT = result[2]

        result = matrix_dots_map_async([(X, HT+JT), (J, JT), (H, HT)])

        W = W * ((result[0]) / (np.maximum(np.dot(W, result[1] + \
                result[2] + alpha), eps)) )

        WT = W.T
        result = matrix_dots_map_async([(WT, W), (WT, X)])
        WTW = result[0]
        WTX = result[1]

        result = matrix_dots_map_async([(WTX, RT), (WTW, J), (WTW, H)])

        M = M * ((result[0] + (lambd*kI)) / (np.maximum( (np.dot(result[1], RT)) + \
                (lambd*M) + alpha, eps)))

        H = H * ( (WTX) / (np.maximum( result[2] + alpha, eps)) )

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
        result = matrix_dots_map_async([(Q, QT), (R, QT)])
        QQT = result[0]
        RQT = result[1]

        P =  P * ( (RQT) / \
            np.maximum(np.dot(P, QQT + (alpha*kI)), eps) )

        result = matrix_trans_map_async([P, Po, S])
        PT = result[0]
        PoTST = np.dot(result[1], result[2])

        result = matrix_dots_map_async([(P, Q), (S, Po), (PT+PoTST, R),
                                        (PT, P), (PoTST, SPo)])
        PQ = result[0]
        SPo = result[1]

        Q = Q * ((result[2]) / np.maximum(np.dot((result[3] + result[4] +\
                (alpha*kI)), Q), eps))

        result = matrix_trans_map_async([Q, Po])

        new_result = matrix_dots_map_async([(result[0], result[1]), (SPo, Q)])
        QTPoT = new_result[0]
        SPoQ = new_result[1]

        result = matrix_dots_map_async([(R, QTPoT), (SPoQ, QTPoT)])

        S = S * ( ((result[0]) + (lambd*nI)) / \
            np.maximum( (result[1]) + ((lambd + alpha)*S), eps ) )

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
        result = matrix_trans_map_async([Q, Po, S])
        QT = result[0]

        new_result = matrix_dots_map_async([(Q, QT), (result[1], result[2]),
                                            (S, Po)])
        QQT = new_result[0]
        PoTST = new_result[1]
        SPo = new_result[2]

        P =  P * ( (np.dot((np.maximum((reta*R - etareta*C), 0)), QT)) / \
            np.maximum(np.dot(P, retareta*QQT + (alpha*kI)), eps) )

        PT = P.T
        result = matrix_dots_map_async([(SPo, Q), (P, Q),
                                        ((reta*(PT + PoTST)), R)])
        SPoQ = result[0]
        PQ = result[1]

        result = matrix_dots_map_async([(reta*(PT + PoTST), R),
                                        (PT, (eta*C) + (reta*PQ)),
                                        (PoTST, (eta*C) + (reta*SPoQ))])

        Q = Q * ((result[0]) / \
            np.maximum((reta*(result[1] + result[2])) + (alpha*Q), eps))

        result = matrix_trans_map_async([Q, Po])
        new_result = matrix_dots_map_async([(result[0], result[1]), (SPo, Q)])

        QTPoT = new_result[0]
        SPoQ = new_result[1]

        result = matrix_dots_map_async([(R, QTPoT), (((reta*SPoQ)+(eta*C)), QTPoT)])

        S = S * ( ((reta*(result[0])) + (lambd*nI)) / \
            np.maximum( (reta*(result[1])) + ((lambd + alpha)*S), eps ) )

        prev_obj = obj
        obj = computeLoss_with_topic(R, P, Q, S, Po, C, eta, alpha, lambd, trRR, nI)

        delta = abs(prev_obj-obj)
        if verbose:
            logging.info('Iter: ' + str(i) + '\t Loss: ' + str(obj) + '\t Delta: ' \
                + str(delta) + '\n')

    logging.info('JPPCF_with_topic OK\n')

    return (P, Q, S)
