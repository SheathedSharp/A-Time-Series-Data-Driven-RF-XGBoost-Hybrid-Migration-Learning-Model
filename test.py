import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances

def align_distributions(source_data, target_data, n_components=10, chunk_size=1000):
    print("Aligning source and target distributions with PCA...")
    
    # 标准化数据
    scaler = StandardScaler()
    source_scaled = scaler.fit_transform(source_data)
    target_scaled = scaler.fit_transform(target_data)
    
    # PCA降维
    pca = PCA(n_components=n_components)
    source_pca = pca.fit_transform(source_scaled)
    target_pca = pca.fit_transform(target_scaled)
    
    aligned_source_data = np.zeros_like(source_pca)
    
    num_chunks = len(source_pca) // chunk_size + 1
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, len(source_pca))
        
        source_chunk = source_pca[start:end]
        
        if len(source_chunk) == 0:
            continue
        
        # 计算成本矩阵
        cost_matrix = euclidean_distances(source_chunk, target_pca)
        
        # 对于每个源数据点，找到最近的目标数据点
        indices = np.argmin(cost_matrix, axis=1)
        
        # 将源数据对齐到目标数据
        aligned_source_data[start:end] = target_pca[indices]
    
    return aligned_source_data, target_pca

# 示例调用
source_data = np.random.rand(491758, 35)
target_data = np.random.rand(6024273, 35)

aligned_source_data, aligned_target_data = align_distributions(source_data, target_data)

print("Aligned source data shape:", aligned_source_data.shape)
print("Aligned target data shape:", aligned_target_data.shape)