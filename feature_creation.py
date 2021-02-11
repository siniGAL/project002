import numpy as np
import statistics
import scipy
import math


def CalculateHurst(times, a, N_steps):
    hurst = np.zeros(N_steps)

    for i in range(0, N_steps):
        mx = max(times[0:a * (i + 1)])
        mn = min(times[0:a * (i + 1)])
        devi = np.std(times[0:a * (i + 1)])
        hurst[i] = (mx - mn) / (devi + 1e-18)
        if math.isnan(hurst[i]):
            print("Error!")
    return hurst


def integral(y):
    L = len(y)
    integ = np.zeros(L)
    integ[0] = y[0]
    for i in range(1, L):
        for j in range(1, i):
            integ[i] += (y[j] + y[j - 1]) / 2
    return integ


def value(array, mean):
    val = []
    range_array = max(array) - min(array)
    for index in range(len(array)):
        area = (array[index] - mean) / range_array
        val.append(area)
    return val


def trap_sum(array):
    list_trap_sum = []
    for index in range(1, len(array) + 1):
        area = scipy.integrate.trapz(array[0:index])
        list_trap_sum.append(area)
    return list_trap_sum

def feature_creation(times, features_num): # Old features
    if features_num == 0:
        if len(times) > 2:
            iat_min = np.amin(times)
            iat_max = np.amax(times)
            iat_mean = statistics.mean(times)
            iat_std = statistics.stdev(times)
            features = np.array([iat_mean, iat_std, iat_max, iat_min])
        else:
            features = np.zeros((4,), dtype=int)
    elif features_num == 1:
        if len(times) > 2:
                iat_p1 = statistics.mean(times)
                pp = np.array(times) - iat_p1
                yup = np.array(pp[pp > 0.0])
                ydn = np.array(pp[pp < 0.0])
                iat_p2 = np.amax(yup) - np.amin(ydn)
                iat_p3 = np.amax(yup) - np.absolute(np.amin(ydn))
                iat_p4 = scipy.integrate.trapz(yup) - scipy.integrate.trapz(ydn)
                iat_p5 = np.amax(times) - statistics.mean(times) / (statistics.mean(times) - np.amin(times))
                iat_p6 = len(yup) - len(ydn)
                pppp = value(times, iat_p1)
                pppp = trap_sum(pppp)
                iat_p8 = max(pppp) - min(pppp)
                features = np.array([iat_p1, iat_p2, iat_p3, iat_p4, iat_p5, iat_p6, iat_p8])
        else:
            features = np.zeros((7,), dtype=int)
    elif features_num == 2:
        if len(times) > 2:
            a = 10
            N_steps = int(120/a)
            hurst = CalculateHurst(times, a, N_steps)

            L = len(hurst)
            Y = np.multiply(range(0, L), hurst)
            Y = Y - np.mean(Y)
            X0 = np.expand_dims(integral(hurst) - np.mean(integral(hurst)), axis=1)
            X1 = np.expand_dims(np.linspace(0, L - 1, L) - np.mean(np.linspace(0, L - 1, L)), axis=1)
            # plt.plot(X0)
            tmp = np.concatenate((X0, X1), axis=1)

            # plt.figure(figsize=(20,10))
            # plt.plot(Y)

            C = np.linalg.lstsq(tmp, Y, rcond=-1)
            C0 = C[0][0]
            C1 = C[0][1]
            # plt.plot(C0*X0+C1*X1)
            H = C0 - 1
            A = C1 / (1 - C0)
            # now we need to estimate slope B

            B_cands = np.linspace(-10, 10, 100)
            metrics = np.zeros(len(B_cands))
            for i in range(0, len(B_cands)):
                Y_est = A + B_cands[i] * np.power(np.linspace(1, len(hurst), len(hurst)), H)
                metrics[i] = np.mean(np.power(np.subtract(hurst, Y_est), 2))
            idx = np.argmin(metrics)
            # if (idx == 0 or idx == (len(B_cands) - 1)):
            #     print("Expand B candidates array")
            B = B_cands[idx]

            Y_est = A + B * np.power(np.linspace(1, len(hurst), len(hurst)), H)
            # err = np.std(np.subtract(Y_est, hurst)) / np.mean(np.abs(hurst))
            # plt.plot(metrics)
            features = np.array([H, A, B])
        else:
            features = np.zeros((3,), dtype=int)

    return features
