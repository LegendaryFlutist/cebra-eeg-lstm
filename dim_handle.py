import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.preprocessing import StandardScaler

def get_tsne_loft_embedding(cebra_loft_embedding):
    # 数据标准化
    scaler = StandardScaler()
    cebra_loft_embedding_sc = scaler.fit_transform(cebra_loft_embedding)
    tsne = TSNE(n_components=3, random_state=42)
    cebra_tsne_loft_embedding = tsne.fit_transform(cebra_loft_embedding_sc)
    return cebra_tsne_loft_embedding


def get_umap_loft_embedding(cebra_loft_embedding):
    # 数据标准化
    scaler = StandardScaler()
    cebra_loft_embedding_sc = scaler.fit_transform(cebra_loft_embedding)
    umap_model = UMAP(n_components=3, random_state=42)
    cebra_umap_loft_embedding = umap_model.fit_transform(cebra_loft_embedding_sc)
    return cebra_umap_loft_embedding

def get_umap_loft_embeddingV2(cebra_loft_embedding):
    # 数据标准化
    scaler = StandardScaler()
    cebra_loft_embedding_sc = scaler.fit_transform(cebra_loft_embedding)
    umap_model = UMAP(n_components=3, random_state=42, n_jobs=1)
    cebra_umap_loft_embedding = umap_model.fit_transform(cebra_loft_embedding_sc)
    return np.hstack((cebra_umap_loft_embedding, cebra_loft_embedding))



