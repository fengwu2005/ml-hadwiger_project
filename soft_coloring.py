from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class SoftColoringResult:
    logits: torch.Tensor  # (n, k)
    probs: torch.Tensor   # (n, k)
    loss: float


class SoftColoringModel(nn.Module):
    """Differentiable k-coloring: each node has learnable logits over colors.

    Loss = sum_{(i,j) in E} <p_i, p_j>,
    where p_i is softmax of logits. 边两端的分布越相似，惩罚越大。
    若图可 k- 染，则期望通过梯度下降把每条边的颜色分开，使 loss 接近 0。
    """

    def __init__(self, n_nodes: int, n_colors: int):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(n_nodes, n_colors))

    def forward(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=-1)


def train_soft_coloring(
    n_nodes: int,
    edges: List[Tuple[int, int]],
    n_colors: int = 4,
    n_steps: int = 2000,
    lr: float = 0.05,
    n_restarts: int = 3,
    device: str | None = None,
) -> SoftColoringResult:
    """Try多次随机初始化，训练一个软 k-染色模型。

    返回最好的一次 (loss 最低)。
    若多次训练后 loss 仍然很高，说明这张图对 k-染色非常“困难”，
    可以作为进一步精确验证 (回溯法) 的候选。
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 若图中没有任何边，则不存在冲突约束，soft 4- 染色问题是“平凡可行”的，
    # 这里直接返回一个 loss=0 的结果，避免在空 edge_index 上做二维索引出错。
    if len(edges) == 0:
        model = SoftColoringModel(n_nodes, n_colors).to(device)
        with torch.no_grad():
            probs = model()
        return SoftColoringResult(
            logits=model.logits.detach().cpu(),
            probs=probs.detach().cpu(),
            loss=0.0,
        )

    edge_index = torch.tensor(edges, dtype=torch.long, device=device)

    best_result: SoftColoringResult | None = None

    for restart in range(n_restarts):
        model = SoftColoringModel(n_nodes, n_colors).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        for step in range(n_steps):
            probs = model()  # (n, k)
            # edge相似度: sum_c p_i[c] * p_j[c]
            p_i = probs[edge_index[:, 0]]
            p_j = probs[edge_index[:, 1]]
            edge_loss = (p_i * p_j).sum(dim=-1)
            loss = edge_loss.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            probs = model()
            final_loss = float(loss.item())

        if best_result is None or final_loss < best_result.loss:
            best_result = SoftColoringResult(
                logits=model.logits.detach().cpu(),
                probs=probs.detach().cpu(),
                loss=final_loss,
            )

    assert best_result is not None
    return best_result
