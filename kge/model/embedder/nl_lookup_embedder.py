import torch
import torch.nn as nn
from torch import Tensor

from kge import Config, Dataset
from kge.model.embedder.lookup_embedder import LookupEmbedder


class NLLookupEmbedder(LookupEmbedder):
    """
    用预计算的 LLaMA3 嵌入 + 可训练投影矩阵替代随机初始化的 nn.Embedding。
    对外接口与 LookupEmbedder 完全一致。
    """

    def __init__(
        self,
        config: Config,
        dataset: Dataset,
        configuration_key: str,
        vocab_size: int,
    ):
        # 1. 先调用父类初始化（会创建随机初始化的 self._embeddings）
        super().__init__(config, dataset, configuration_key, vocab_size)

        # 2. 加载预计算的 LLaMA3 嵌入
        pretrained_embeddings_path = dataset.folder
        if 'entity' in configuration_key:
            pretrained_embeddings_path += '/entity_llama_embeddings.pt'
        elif 'relation' in configuration_key:
            pretrained_embeddings_path += '/relation_llama_embeddings.pt'
        else:
            raise ValueError(f"Unknown configuration_key for nl_lookup_embedder: {configuration_key}")
        
        llama_matrix = torch.load(pretrained_embeddings_path, map_location="cpu")

        assert llama_matrix.shape[0] == vocab_size, (
            f"pretrained embeddings number ({llama_matrix.shape[0]}) does not match vocab_size ({vocab_size})"
        )

        D_llama = llama_matrix.shape[1]
        D_target = self.dim  # 你模型的目标嵌入维度（继承自父类）

        # 3. 冻结预计算嵌入（作为 buffer，不参与梯度计算）
        self.register_buffer("llama_embeddings", llama_matrix)

        # 4. 创建可训练的投影矩阵: D_llama → D_target
        #    用 nn.Linear(bias=False) 表示，其 weight 就是转换矩阵
        self.projection = nn.Linear(D_llama, D_target, bias=False)
        nn.init.xavier_uniform_(self.projection.weight)

        # 5. 删除父类创建的随机初始化 nn.Embedding，释放内存
        del self._embeddings

    def embed(self, indexes: Tensor) -> Tensor:
        """
        与 LookupEmbedder.embed 接口完全一致。
        indexes: (batch_size,) 或 (batch_size, seq_len) 的整数 id
        """
        # 查找预计算嵌入（无梯度）
        llama_vec = self.llama_embeddings[indexes.long()]  # (..., D_llama)
        # 通过可训练投影矩阵映射到目标维度

        # print(f"[GPU {llama_vec.device}] llama_vec: {llama_vec.dtype}, projection.weight: {self.projection.weight.dtype}")

        projected = self.projection(llama_vec)              # (..., D_target)
        return self._postprocess(projected)

    def embed_all(self) -> Tensor:
        """返回所有实体的嵌入，与 LookupEmbedder.embed_all 接口一致。"""
        # llama_embeddings: (vocab_size, D_llama)
        # projection.weight: (D_target, D_llama) → 转置后为 (D_llama, D_target)
        all_projected = self.llama_embeddings @ self.projection.weight.T  # (vocab_size, D_target)
        return self._postprocess(all_projected)

    def _embeddings_all(self) -> Tensor:
        """供 penalty() 等内部方法使用，与 LookupEmbedder 保持一致。"""
        return self.llama_embeddings @ self.projection.weight.T
    
    # normalize.p = -1