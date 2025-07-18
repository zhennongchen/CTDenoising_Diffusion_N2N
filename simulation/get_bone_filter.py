'''
Get the bone filter
'''

# %%
import numpy as np
import os
import matplotlib.pyplot as plt

from locations import data_dir


# %%
def main():
    input_dir = os.path.join(data_dir, '20220720_1x1/processed')
    custom_filter = np.fromfile(os.path.join(input_dir, '6x5_Bone'), np.float32)

    ninterp = 20
    nu = 864
    du = 0.3515700101852417
    nview = 1440

    # compose rl filter
    # 这里做RL filter的原因在于有可能直接用custom filter会造成数据的bias（但具体要不要用，可以先用custom filter试试看，如果没有bias就不用）
    # 这里最后存的filter是ratio_fix
    rl_filter = np.zeros([2 * nu - 1], np.float32)
    k = np.arange(len(rl_filter)) - (nu - 1)
    for i in range(len(rl_filter)):
        if k[i] == 0:
            rl_filter[i] = 1 / (4 * du * du)
        elif k[i] % 2 != 0: 
            rl_filter[i] = -1 / (np.pi * np.pi * k[i] * k[i] * du * du)
    frl_filter = np.fft.fft(rl_filter, len(custom_filter))
    frl_filter = np.abs(frl_filter)

    frl_filter = frl_filter * len(frl_filter) / nview * du * 2

    ratio = custom_filter / frl_filter

    ratio_fix = ratio.copy()
    ratio_fix[:ninterp] = 1 + (ratio[ninterp] - 1) * np.arange(ninterp) / ninterp
    ratio_fix = ratio_fix.astype(np.float32)
    ratio_fix[len(ratio_fix) // 2:] = ratio_fix[len(ratio_fix) // 2:0:-1]

    plt.figure()
    plt.plot(frl_filter)
    plt.plot(custom_filter)
    plt.plot(ratio)
    plt.plot(ratio_fix) 

    ratio_fix.tofile(os.path.join(input_dir, 'additional_bone_kernel'))

    return frl_filter, custom_filter, ratio, ratio_fix


# %%
if __name__ == '__main__':
    res = main()
