import numpy as np
from scipy.sparse import coo_matrix
from scipy.io import mmwrite

# 矩阵大小
n = 20

# 创建随机种子以确保结果可重现
np.random.seed(42)

# 生成对角线上的值（确保非零）
diag_values = np.random.randint(1, 10, size=n)

# 生成非对角线元素的数量（控制稀疏度）
num_non_diag = n * 5  # 约5%的稀疏度

# 生成非对角线元素的行和列索引
row_indices = np.random.randint(0, n, size=num_non_diag)
col_indices = np.random.randint(0, n, size=num_non_diag)

# 过滤掉对角线元素
mask = row_indices != col_indices
row_indices = row_indices[mask]
col_indices = col_indices[mask]

# 确保非对称：随机选择一些元素并确保其对称位置没有值
# 这里简单地取前一半元素作为非对称部分
num_asymmetric = len(row_indices) // 2
row = np.concatenate([row_indices[:num_asymmetric], np.arange(n)])
col = np.concatenate([col_indices[:num_asymmetric], np.arange(n)])

# 生成对应的值
values = np.concatenate([
    np.random.randint(1, 10, size=num_asymmetric),
    diag_values
])

# 构建COO格式的稀疏矩阵
sparse_matrix = coo_matrix((values, (row, col)), shape=(n, n))

# # 转换为CSR格式以按行排序
# csr_matrix = sparse_matrix.tocsr()

# # 获取排序后的行、列和值
# sorted_indices = csr_matrix.indices
# sorted_indptr = csr_matrix.indptr
# sorted_data = csr_matrix.data

# # 提取排序后的行和列索引
# sorted_rows = []
# sorted_cols = []
# sorted_vals = []

# for i in range(n):
#     for j in range(sorted_indptr[i], sorted_indptr[i+1]):
#         sorted_rows.append(i)
#         sorted_cols.append(sorted_indices[j])
#         sorted_vals.append(sorted_data[j])

# # 打印COO格式的矩阵信息（MATLAB格式，索引从1开始）
# print(f"{n} {n} {len(values)}")
# for i in range(len(sorted_rows)):
#     print(f"{sorted_rows[i]+1} {sorted_cols[i]+1} {sorted_vals[i]}")

# 将矩阵写入.mtx文件
# 转换为CSR格式以确保按行排序
sorted_matrix = sparse_matrix.tocsr()

# 转换回COO格式（此时元素已按行排序）
sorted_coo = sorted_matrix.tocoo()

# 将排序后的矩阵写入.mtx文件
mmwrite('testm2.mtx', sorted_coo, field='real')

print(f"矩阵已成功按行升序写入到testm2.mtx文件中")

# 验证矩阵是否非对称
dense_matrix = sparse_matrix.toarray()
is_symmetric = np.allclose(dense_matrix, dense_matrix.T)
print("\n矩阵是否对称:", is_symmetric)