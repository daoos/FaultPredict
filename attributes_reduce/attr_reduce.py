#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


# 构建差别矩阵 S信息系统
def build_discern_matrix(info_system, cond_attrs, reduce_attr):
    matrix = np.zeros((info_system.shape[0], info_system.shape[0]), dtype=set)
    for i in np.arange(info_system.shape[0]):
        for j in np.arange(i + 1, info_system.shape[0]):
            # 判断决策属性是否相同，如果决策属性，则查找条件属性中不同的属性，将其作为差别矩阵的一部分
            if info_system[i, -1] != info_system[j, -1]:
                # 判断两个不同的行为序列，在决策属性不同的情况下，有多少条件属性不同
                flag = info_system[i, :-1] != info_system[j, :-1]
                attrs = dict(zip(cond_attrs, flag))
                if reduce_attr:
                    del_attr = [i for i in cond_attrs if i not in reduce_attr]
                else:
                    del_attr = cond_attrs
                for k in del_attr:
                    attrs.pop(k)
                prop_dict = dict(filter(attr_different, attrs.items()))
                if len(prop_dict) <= 0:
                    matrix[j, i] = 0
                else:
                    matrix[j, i] = set(prop_dict.keys())

    return matrix


# 选择添加属性不相等的属性
def attr_different(attr):
    k, v = attr
    return v


def select_single_attr(item):
    if item is not 0 and len(item) == 1:
        return True
    return False


# 获得每个属性的重要性Sig^+,即Sig^+(a,R,D) =W(R并{a}|D)-W(R|D)
def get_sig_plus(info_system, condition_attrs, reduce_attrs):
    attr_dict = dict()
    # 未添加属性前的约简属性集reduce_attr的布尔差别矩阵
    bdm_old = build_discern_matrix(info_system, condition_attr, reduce_attrs)
    w_old = np.sum(np.where(bdm_old == 0, bdm_old, 1))
    # 找到一个属性在条件属性condition_attrs中，但是不在约简属性reduce_attr,即a属于C-R
    if reduce_attrs:
        extra_attrs = [attr for attr in condition_attrs if attr not in reduce_attrs]
    else:
        extra_attrs = condition_attrs
    for a in extra_attrs:
        new_attr = reduce_attrs.copy()
        new_attr.append(a)
        # 添加属性后的属性集new_attr的布尔差别矩阵
        bdm_new = build_discern_matrix(info_system, condition_attrs, new_attr)
        w_new = np.sum(np.where(bdm_new == 0, bdm_new, 1))
        # 将属性a 的重要性Sig^+以a:Sig^+这样的键值对形式添加到字典对象中，w_new-w_old的值为Sig^+
        attr_dict[a] = w_new - w_old
    return attr_dict


# C 条件属性
condition_attr = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
# s表示信息系统,灰色关联分析后的
S = np.loadtxt('data_efb_process.txt')
# 差别矩阵
discern_matrix = build_discern_matrix(S, condition_attr, condition_attr)
np.savetxt("prop_dm.txt", discern_matrix, fmt='%s')
# 1.设约简属性属性为空集
R = set()
# 2 得到差别矩阵的核属性，将核属性赋值给R
core_attr = []
for item in filter(select_single_attr, discern_matrix.flatten()):
    core_attr.extend(item)
R = core_attr
# 将Hu差别矩阵转换M_B^R成布尔差别矩阵
b_discern_matrix = build_discern_matrix(S, condition_attr, R)
b_discern_matrix = np.where(b_discern_matrix == 0, b_discern_matrix, 1)
# 计算W(R|D)
w_r = np.sum(b_discern_matrix)
# 计算W(C|D)
b_discern_matrix_c = np.where(discern_matrix == 0, discern_matrix, 1)
w_c = np.sum(b_discern_matrix_c)
# 将属性集C的布尔差别矩阵保存到bool_discern_matrix_c.txt文件中
np.savetxt("bool_discern_matrix_c.txt", b_discern_matrix_c, fmt='%s')
# 当属性集R的相对分辨相对分辨能力W(R|D)不等于条件属性集C的相对分辨能力W(C|D)循环
while w_r is not w_c:
    # 对于所有的属于C-R的a_i,得到包含a_i和Sig^+(a_i,R,D)为键值对的dict对象
    attr_dic = get_sig_plus(S, condition_attr, R)
    # 挑选Sig^+(a_i,R,D)的值最大的a_i,并把该属性添加到R中
    R.append(list(attr_dic.keys())[list(attr_dic.values()).index(max(attr_dic.values()))])
    b_discern_matrix_r = build_discern_matrix(S, condition_attr, R)
    w_r = np.sum(np.where(b_discern_matrix_r == 0, b_discern_matrix_r, 1))
print(R)
supple_attr = set(condition_attr).difference(set(R))
print(supple_attr)
dataset = np.delete(np.loadtxt("data_process.txt"), [condition_attr.index(attr) for attr in supple_attr], axis=1)
np.savetxt("data2.txt", dataset, fmt='%.2f')
