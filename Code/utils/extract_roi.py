import nibabel as nib
import numpy as np
from collections import Counter
import config


def using_voxno_find_par(indices, para):
    # indices:Vox的序号 B:大脑分区
    # return 体素对应的脑区
    indices_para = []
    for i in indices:
        indices_para.append(para[i])
    return indices_para

def _find_indices(A, B):
    # 将B转换为一个NumPy数组以便与A进行比较
    B = np.array(B)
    # 创建一个空列表来存储结果
    indices = []

    # 遍历B中的每个值
    for value in B:
        # 使用np.where找到A中等于当前值的下标，并将结果添加到indices列表中
        indices.extend(np.where(A == value)[0])
    indices = sorted(indices)

    # indices_para = using_voxno_find_par(indices, A)
    # indices_para = np.array(indices_para)
    # np.save('./roivox_ref_para.npy',indices_para)

    return indices

def extract_speech(label_path):
    para = np.array(nib.load(label_path).get_fdata())
    # roi_L = [30,25,37,38,33,34,35,36,16,15,12,13,14]
    roi_L = [12,13,14,15,16,25,30,33,34,35,36,37,38]
    roi_R = [i+75 for i in roi_L]
    roi = roi_L + roi_R
    return _find_indices(para[0],roi)

def roivox2para(data):
    indices_para = np.load(config.project_lfs_path + '/roivox_ref_para.npy')
    roi_L = [12,13,14,15,16,25,30,33,34,35,36,37,38]
    roi_R = [i+75 for i in roi_L]
    roi = roi_L + roi_R


    Roi_para_data = []
    # 将ROI的值合并在一起
    for i in roi:
        index = np.where(indices_para == i)
        sum_val = data[...,index].sum(axis=-1)
        Roi_para_data.append(sum_val)
    return np.concatenate(Roi_para_data,axis=-1)


def _find_indices_change(A, B):
    mask = np.isin(A, B)
    A[~mask] = 0
    A_cat = A[A!=0]
    return A_cat


def group_by_partition_dic(A, B):
    # 创建一个空字典，用于存储分区与对应的元素
    partition_dict = {}

    # 遍历 A 和 B 数组，B 是分区 ID，A 是数据值
    for a, b in zip(A, B):
        if b not in partition_dict:
            # 如果分区 ID 不在字典中，初始化一个空列表
            partition_dict[b] = []
        # 将数据值添加到对应分区的列表中
        partition_dict[b].append(a)

    return partition_dict

def group_by_partition(A, B):
    # 创建一个空字典，用于存储分区与对应的元素（每个脑区的体素累加结果）
    partition_dict = {}

    # 遍历 A 和 B 数组，B 是分区 ID，A 是数据值
    for i in range(A.shape[0]):  # 对每个 batch 遍历
        for a, b in zip(A[i], B):
            if b not in partition_dict:
                # 如果分区 ID 不在字典中，初始化为一个 0 值
                partition_dict[b] = 0
            # 累加数据值
            partition_dict[b] += a

    # 对分区 ID 进行排序
    sorted_partition_dict = dict(sorted(partition_dict.items()))

    return sorted_partition_dict


def group_by_partition_mean(A, B):
    # 获取所有唯一的分区 ID
    unique_partitions = np.unique(B)

    # 创建一个数组用于存储每个 batch 每个分区的加和结果
    sum_per_partition = np.zeros((A.shape[0], len(unique_partitions)))  # (batch_size, num_partitions)

    # 遍历每个 batch
    for i in range(A.shape[0]):
        # 对每个分区累加体素值
        for j, partition in enumerate(unique_partitions):
            # 找到属于当前分区的所有体素的索引
            partition_indices = np.where(B == partition)[0]
            # 对应分区的体素值加和
            # sum_per_partition[i, j] = np.sum(A[i, partition_indices])
            sum_per_partition[i, j] = np.mean(A[i, partition_indices])

    # 按照分区 ID 排序，确保分区顺序从小到大
    sorted_indices = np.argsort(unique_partitions)

    # 返回排序后的结果，按照排序后的分区 ID 索引
    return sum_per_partition[:, sorted_indices]


def roivox2para_correct(data):
    para = np.array(nib.load(config.project_lfs_path + r'/SMN4Lang/CN001.aparc.a2009s.32k_fs_LR.dlabel.nii').get_fdata())
    roi_L = [12,13,14,15,16,25,30,33,34,35,36,37,38]
    roi_R = [i+75 for i in roi_L]
    roi = roi_L + roi_R
    A_cat = _find_indices_change(para[0],roi)

    result = group_by_partition_mean(data[:,0],A_cat)

    return result

def group_by_partition_new(A, B):
    # 获取所有唯一的分区 ID
    unique_partitions = np.unique(B)

    # 创建一个数组用于存储每个 batch 每个分区的加和结果
    sum_per_partition = np.zeros((A.shape[0], len(unique_partitions)))  # (batch_size, num_partitions)

    # 遍历每个 batch
    for i in range(A.shape[0]):
        # 对每个分区累加体素值
        for j, partition in enumerate(unique_partitions):
            # 找到属于当前分区的所有体素的索引
            partition_indices = np.where(B == partition)[0]
            # 对应分区的体素值加和
            # sum_per_partition[i, j] = np.sum(A[i, partition_indices])
            # sum_per_partition[i, j] = A[i,partition_indices][A[i,partition_indices] != 0].mean()
            sum_per_partition[i, j] = A[i,partition_indices][A[i,partition_indices] != 0].mean() if np.any(A[i,partition_indices] != 0) else 0
            # sum_per_partition[i, j] = A[i,partition_indices][A[i,partition_indices] != 0].sum() if np.any(A[i,partition_indices] != 0) else 0
            # sum_per_partition[i, j] = np.mean(A[i, partition_indices])

    # 按照分区 ID 排序，确保分区顺序从小到大
    sorted_indices = np.argsort(unique_partitions)

    # 返回排序后的结果，按照排序后的分区 ID 索引
    return sum_per_partition[:, sorted_indices]


def roivox2para_correct_new(data):
    para = np.array(nib.load(config.project_lfs_path + r'/SMN4Lang/CN001.aparc.a2009s.32k_fs_LR.dlabel.nii').get_fdata())
    roi_L = [12,13,14,15,16,25,30,33,34,35,36,37,38]
    roi_R = [i+75 for i in roi_L]
    roi = roi_L + roi_R
    A_cat = _find_indices_change(para[0],roi)

    result = group_by_partition_new(data,A_cat)

    return result



