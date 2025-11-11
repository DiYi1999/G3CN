from statsmodels.tsa.seasonal import STL
import ptwt, pywt
from ptwt import WaveletPacket
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch


def Decompose_fuc(X, args):
    if args.Decompose == "STL":
        X_array = X.cpu().numpy().astype(np.float32)
        for i in np.arange(X_array.shape[0]):
            for j in np.arange(X_array.shape[1]):
                signal = X_array[i, j, :]
                stl = STL(signal, period=args.STL_seasonal, robust=True)
                res = stl.fit()
                if j == 0:
                    X_decompose = np.stack((res.trend, res.seasonal, res.resid), axis=0)
                else:
                    X_decompose = np.concatenate((X_decompose, np.stack((res.trend, res.seasonal, res.resid), axis=0)), axis=0)
            if i == 0:
                X_decompose_all = X_decompose[np.newaxis, :, :]
            else:
                X_decompose_all = np.concatenate((X_decompose_all, X_decompose[np.newaxis, :, :]), axis=0)
        X_decompose_batch = torch.from_numpy(X_decompose_all).to(X.device)

    elif args.Decompose == "Wavelet":
        coefficients = ptwt.wavedec(X, args.Wavelet_wave, 'reflect', level=args.Wavelet_level)
        X_decompose_batch = Wavelet_coef_to_signal_tensor(coefficients, args.Wavelet_wave)

    elif args.Decompose == "WaveletPacket":
        for i in range(2**args.Wavelet_level):
            tree = ptwt.WaveletPacket(data=X, wavelet=args.Wavelet_wave, mode="boundary", maxlevel=args.Wavelet_level)
            tree[tree.get_level(args.Wavelet_level)[i]].data *= 0
            tree.reconstruct()
            signals = X - tree[""][..., :X.shape[-1]]
            if i == 0:
                signals_level_tensor = signals
            else:
                signals_chunk = torch.chunk(signals, signals.size(1), dim=1)
                signals_level_tensor_chunk = torch.chunk(signals_level_tensor, signals.size(1), dim=1)
                signals_level_tensor_chunk = [torch.cat((signals_level_tensor_chunk[j], signals_chunk[j]), dim=1)
                                              for j in np.arange(len(signals_chunk))]
                signals_level_tensor = torch.cat(signals_level_tensor_chunk, dim=1)
        X_decompose_batch = signals_level_tensor
    else:
        X_decompose_batch = X

    return X_decompose_batch


def Wavelet_coef_to_signal_tensor(coefficients_level, Wavelet_wave):
    for i in np.arange(len(coefficients_level)):
        zero_coefficients_level = [torch.zeros_like(tensor) for tensor in coefficients_level]
        if i == 0:
            zero_coefficients_level[i] = coefficients_level[i]
            signals = ptwt.waverec(zero_coefficients_level, Wavelet_wave)
            signals_level_tensor = signals
        else:
            zero_coefficients_level[i] = coefficients_level[i]
            signals = ptwt.waverec(zero_coefficients_level, Wavelet_wave)
            signals_chunk = torch.chunk(signals, signals.size(1), dim=1)
            signals_level_tensor_chunk = torch.chunk(signals_level_tensor, signals.size(1), dim=1)
            signals_level_tensor_chunk = [torch.cat((signals_level_tensor_chunk[j], signals_chunk[j]), dim=1)
                                          for j in np.arange(len(signals_chunk))]
            signals_level_tensor = torch.cat(signals_level_tensor_chunk, dim=1)

    return signals_level_tensor


def Reconstruct_fuc(args, H):
    if args.Decompose == "STL":
        chunks = torch.chunk(H, H.size(1) // 3, dim=1)
        H_stacked = torch.stack(chunks, dim=2)
        H_sum = torch.sum(H_stacked, dim=1)
        X_reconstruct_batch = H_sum

    elif args.Decompose == "Wavelet":
        chunks = torch.chunk(H, H.size(1) // (args.Wavelet_level + 1), dim=1)
        H_stacked = torch.stack(chunks, dim=2)
        H_sum = torch.sum(H_stacked, dim=1)
        X_reconstruct_batch = H_sum

    elif args.Decompose == "WaveletPacket":
        chunks = torch.chunk(H, H.size(1) // (2 ** args.Wavelet_level), dim=1)
        H_stacked = torch.stack(chunks, dim=2)
        H_sum = torch.sum(H_stacked, dim=1)
        X_reconstruct_batch = H_sum

    else:
        raise ValueError("Decompose should be STL or Wavelet")

    return X_reconstruct_batch
