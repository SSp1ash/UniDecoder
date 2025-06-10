import numpy as np
import numpy as np

def make_delayed(stim, delays, circpad=False):
    """Creates non-interpolated concatenated delayed versions of [stim] with the given [delays]
    (in samples).

    If [circpad], instead of being padded with zeros, [stim] will be circularly shifted.
    """
    nt, ndim = stim.shape
    dstims = []
    for di, d in enumerate(delays):
        dstim = np.zeros((nt, ndim))
        if d < 0:  ## negative delay
            dstim[:d, :] = stim[-d:, :]
            if circpad:
                dstim[d:, :] = stim[:-d, :]
        elif d > 0:
            dstim[d:, :] = stim[:-d, :]
            if circpad:
                dstim[:d, :] = stim[-d:, :]
        else:  ## d==0
            dstim = stim.copy()
        dstims.append(dstim)
    return np.hstack(dstims)

if __name__ == '__main__':
    x = np.random.randn(5,6)
    y = make_delayed(x,[-4,-3,-1, -4])
    print(y)