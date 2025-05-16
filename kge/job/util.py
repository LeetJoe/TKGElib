import torch
from torch import Tensor


def get_sp_po_coords_from_spo_batch(
    batch: Tensor, num_entities: int, sp_index: dict, po_index: dict
) -> torch.Tensor:
    """Given a set of triples , lookup matches for (s,p,?) and (?,p,o).

    Each row in batch holds an (s,p,o) triple. Returns the non-zero coordinates
    of a 2-way binary tensor with one row per triple and 2*num_entites columns.
    The first half of the columns correspond to hits for (s,p,?); the second
    half for (?,p,o).
    从说明来看，返回的结果应该是一个 3 * (2*num_entities) 的矩阵，每一列是一个三元组，前半
    num_entities 是 (s, p, ?), 后半是 (?, p, o)。
    因为 batch 是一组 (s, p, o)，返回的 shape 应该是 len(batch), 3, 2*num_entities。

    """
    num_ones = 0

    # todo 这里面怎么只有 s, p, o，没有 t？
    NOTHING = torch.zeros([0], dtype=torch.long)
    for i, triple in enumerate(batch):  # batch 就是一组切段数据，这里是应该是四元组
        s, p, o = triple[0].item(), triple[1].item(), triple[2].item()
        num_ones += len(sp_index.get((s, p), NOTHING))  # dict().get() 的第二个参数是当 key 不存在时返回的默认值
        num_ones += len(po_index.get((p, o), NOTHING))
        # sp_index 和 po_index 是从指定 split 中取得的，可以是 train/valid/test，而 batch 是从训练或者评估目标数据集中得到的。
        # 以一次在 valid 数据上的 evaluation 为例，sp_index, ps_index 取自 train，batch 是 valid 数据的切片；这样 num_ones
        # 累计的是：train 数据中，与 batch 里某条 (s, p, o, t)) 的 (s, p) 相等的数据的数量和其中的 (p, o) 相等的数据的数量之和。

    coords = torch.zeros([num_ones, 2], dtype=torch.long)
    current_index = 0
    for i, triple in enumerate(batch):
        s, p, o = triple[0].item(), triple[1].item(), triple[2].item()

        objects = sp_index.get((s, p), NOTHING)
        coords[current_index : (current_index + len(objects)), 0] = i
        coords[current_index : (current_index + len(objects)), 1] = objects
        current_index += len(objects)

        # 这里加上了 num_entities，应该是用来跟前面的 objects 进行区分。在 item[0] == i 的时候，item[1] < num_entities 就是 object，反之就是 subject
        subjects = po_index.get((p, o), NOTHING) + num_entities
        coords[current_index : (current_index + len(subjects)), 0] = i
        coords[current_index : (current_index + len(subjects)), 1] = subjects
        current_index += len(subjects)

    # coords.shape = (num_ones, 2)，它的内容为：
    # 对每一条 batch 中的第 i 条数据(s, p, o, t)，如果 sp_index 中存在 key=(s, p)，且对应的值为 [o1, o2, ... ok]，则
    # 在 coords 中放入 k 列，其中第 j 列的第 1 行是 i，第 2 行是 oj；然后再在 po_index 中找到 key=(p, o) 对应的值 [s1, s2, ... sl]，
    # 再在 coords 中放入 l 列，其中第 j 列的第 1 行还是 i, 第 2 行是 sj ———— 如此执行下去直到遍历完所有 batch 里的数据。
    return coords


def coord_to_sparse_tensor(
    nrows: int, ncols: int, coords: Tensor, device: str, value=1.0
):
    if device == "cpu":
        # torch.sparse.FloatTensor 第一个参数表示坐标，第二个参数表示数据
        labels = torch.sparse.FloatTensor(
            coords.long().t(),  # 这个参数的长度（列数）可变，重要的是每一行的两个值的取值范围要在 nrows 和 ncols 的范围内
            torch.ones([len(coords)], dtype=torch.float, device=device) * value,
            torch.Size([nrows, ncols]),
        )
    else:
        labels = torch.cuda.sparse.FloatTensor(
            coords.long().t(),
            torch.ones([len(coords)], dtype=torch.float, device=device) * value,
            torch.Size([nrows, ncols]),
            device=device,
        )
    return labels


def get_spt_pot_coords_from_spot_batch(
    batch: Tensor, num_entities: int, spt_index: dict, pot_index: dict
) -> torch.Tensor:
    """Given a set of quadruples, lookup matches for (s,p,?,t) and (?,p,o,t).

    Each row in batch holds an (s,p,o,t) quadruple. Returns the non-zero coordinates
    of a 2-way binary tensor with one row per quadruple and 2*num_entities columns.
    The first half of the columns correspond to hits for (s,p,?,t); the second
    half for (?,p,o,t).

    """
    num_ones = 0

    NOTHING = torch.zeros([0], dtype=torch.long)
    for i, triple in enumerate(batch):
        s, p, o, t = triple[0].item(), triple[1].item(), triple[2].item(), triple[3].item()
        num_ones += len(spt_index.get((s, p, t), NOTHING))
        num_ones += len(pot_index.get((p, o, t), NOTHING))

    coords = torch.zeros([num_ones, 2], dtype=torch.long)
    current_index = 0
    for i, triple in enumerate(batch):
        s, p, o, t = triple[0].item(), triple[1].item(), triple[2].item(), triple[3].item()

        objects = spt_index.get((s, p, t), NOTHING)
        coords[current_index : (current_index + len(objects)), 0] = i
        coords[current_index : (current_index + len(objects)), 1] = objects
        current_index += len(objects)

        subjects = pot_index.get((p, o), NOTHING) + num_entities
        coords[current_index : (current_index + len(subjects)), 0] = i
        coords[current_index : (current_index + len(subjects)), 1] = subjects
        current_index += len(subjects)

    return coords