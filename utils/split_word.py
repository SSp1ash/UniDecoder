import numpy as np


def split_words_and_times(A, B):
    """
    将中文词语数组A拆分为字符，同时对时间数组B进行相应的拆分。
    对应的时间也要进行拆分，具体做法为在两个时间间隔里面取平均值。

    参数：
    A (ndarray): 中文词语数组
    B (ndarray): 时间数组

    返回：
    tuple: 拆分后的字符数组和时间数组
    """
    split_chars = []
    split_times = []

    for i, word in enumerate(A):
        chars = list(word)
        n = len(chars)
        split_chars.extend(chars)

        if i < len(B) - 1:
            interval_start = B[i]
            interval_end = B[i + 1]
            interval_step = (interval_end - interval_start) / n
            times = [interval_start + j * interval_step for j in range(n)]
        else:
            times = [B[i]] * n  # 最后一个词语的时间

        split_times.extend(times)

    return np.array(split_chars), np.array(split_times)

if __name__ == '__main__':


    # 示例数据
    A = np.array(["你好", "世界", '很'])
    B = np.array([1.0, 3.0, 4.0])

    split_A, split_B = split_words_and_times(A, B)
    print("拆分后的字符数组:", split_A)
    print("拆分后的时间数组:", split_B)
