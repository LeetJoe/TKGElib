import torch
from collections import defaultdict, OrderedDict
import numba
import numpy as np
import networkx as nx


rng_seed = 1234  # None
# rng_seed = None

def _group_by(keys, values) -> dict:
    """Group values by keys.

    :param keys: list of keys
    :param values: list of values
    A key value pair i is defined by (key_list[i], value_list[i]).
    :return: OrderedDict where key value pairs have been grouped by key.

     """
    result = defaultdict(list)
    for key, value in zip(keys.tolist(), values.tolist()):
        result[tuple(key)].append(value)
    for key, value in result.items():
        result[key] = torch.IntTensor(sorted(value))
    return OrderedDict(result)


# 把 split 里的数据按 key 指定的模式包装成 {(s, p): [o1, o2, ...]} 这样的形式，根据 key in {sp, po, so} 不同有不同的组织结果
def index_KvsAll(dataset: "Dataset", split: str, key: str):
    """Return an index for the triples in split (''train'', ''valid'', ''test'')
    from the specified key (''sp'' or ''po'' or ''so'') to the indexes of the
    remaining constituent (''o'' or ''s'' or ''p'' , respectively.)

    The index maps from `tuple' to `torch.LongTensor`.

    The index is cached in the provided dataset under name `{split}_sp_to_o` or
    `{split}_po_to_s`, or `{split}_so_to_p`. If this index is already present, does not
    recompute it.

    """
    value = None
    if key == "sp":
        key_cols = [0, 1]
        value_column = 2
        value = "o"
    elif key == "po":
        key_cols = [1, 2]
        value_column = 0
        value = "s"
    elif key == "so":
        key_cols = [0, 2]
        value_column = 1
        value = "p"
    elif key == "spt":
        key_cols = [0, 1, 3]
        value_column = 2
        value = "o"
    elif key == "pot":
        key_cols = [1, 2, 3]
        value_column = 0
        value = "s"
    elif key == "sot":
        key_cols = [0, 2, 3]
        value_column = 1
        value = "p"
    else:
        raise ValueError()

    name = split + "_" + key + "_to_" + value
    if not dataset._indexes.get(name):
        triples = dataset.split(split)
        dataset._indexes[name] = _group_by(
            triples[:, key_cols], triples[:, value_column]
        )

    dataset.config.log(
        "{} distinct {} pairs in {}".format(len(dataset._indexes[name]), key, split),
        prefix="  ",
    )

    return dataset._indexes.get(name)


def index_KvsAll_to_torch(index):
    """Convert `index_KvsAll` indexes to pytorch tensors.

    Returns an nx2 keys tensor (rows = keys), an offset vector
    (row = starting offset in values for corresponding key),
    a values vector (entries correspond to values of original
    index)

    Afterwards, it holds:
        index[keys[i]] = values[offsets[i]:offsets[i+1]]
    """
    keys = torch.tensor(list(index.keys()), dtype=torch.int)
    values = torch.cat(list(index.values()))
    offsets = torch.cumsum(
        torch.tensor([0] + list(map(len, index.values())), dtype=torch.int), 0
    )
    return keys, values, offsets


def index_frequency_percent(dataset):
    name = "fre"
    if not dataset._indexes.get(name):
        fre = []
        entities_fre = {}
        train_triples = dataset.split('train')
        for i in range(dataset.num_entities()):
            entities_fre[i] = 0
        for tri in train_triples:
            s, p, t, o = tri.tolist()
            if s in entities_fre:
                entities_fre[s] += 1
        for i in range(dataset.num_entities()):
            fre.append(entities_fre[i] / dataset.num_entities())
        dataset._indexes[name] = fre
    dataset.config.log("Entities_fre index finished", prefix="  ")
    return dataset._indexes[name]


def index_neighbor_multidig(dataset):
    name = "neighbor"
    if not dataset._indexes.get(name):
        train_triples = dataset.split('train')
        G = nx.MultiDiGraph()
        for tri in train_triples:
            s, p, o, t= tri.tolist()
            G.add_node(s)
            G.add_node(o)
            G.add_edge(s, o, type=p, time=t)

        # import powerlaw # Power laws are probability distributions with the form:p(x)∝x−α
        max_neighbor_num = 1000
        all_neighbor = torch.zeros((dataset.num_entities(), 3, max_neighbor_num), dtype=torch.long)
        all_neighbor_num = torch.zeros(dataset.num_entities(), dtype=torch.long)

        # get the information about all edges
        edges_attributes = G.edges(data=True)
        neighbor_all = [[] for _ in range(dataset.num_entities())]
        neighbor_edge_types_all = [[] for _ in range(dataset.num_entities())]
        neighbor_edge_times_all = [[] for _ in range(dataset.num_entities())]

        rng = np.random.default_rng(rng_seed)

        for s, o, data in edges_attributes:
            neighbor_all[s].append(o)
            neighbor_all[o].append(s)
            neighbor_edge_types_all[s].append(data["type"] + dataset.num_relations())
            neighbor_edge_times_all[s].append(data["time"])
            neighbor_edge_types_all[o].append(data["type"])
            neighbor_edge_times_all[o].append(data["time"])

        for s in range(dataset.num_entities()):
            if s not in G: continue
            rand_permut = rng.permutation(len(neighbor_all[s]))
            neighbor = np.asarray(neighbor_all[s])[rand_permut]
            neighbor_edge_types = np.asarray(neighbor_edge_types_all[s])[rand_permut]
            neighbor_edge_times = np.asarray(neighbor_edge_times_all[s])[rand_permut]
            neighbor = neighbor[:max_neighbor_num]
            neighbor_edge_types = neighbor_edge_types[:max_neighbor_num]
            neighbor_edge_times = neighbor_edge_times[:max_neighbor_num]
            all_neighbor[s, 0, 0:len(neighbor)] = torch.tensor(neighbor, dtype=torch.long)
            all_neighbor[s, 1, 0:len(neighbor)] = torch.tensor(neighbor_edge_types, dtype=torch.long)
            all_neighbor[s, 2, 0:len(neighbor)] = torch.tensor(neighbor_edge_times, dtype=torch.long)
            all_neighbor_num[s] = len(neighbor)
        dataset._indexes[name] = (all_neighbor, all_neighbor_num)

    dataset.config.log("Neighbors index finished", prefix="  ")

    return dataset._indexes.get(name)


def index_neighbor_dig(dataset):
    name = "neighbor"
    if not dataset._indexes.get(name):
        train_triples = dataset.split('train')
        G = nx.DiGraph()
        for tri in train_triples:
            s, p, o, t= tri.tolist()
            G.add_node(s)
            G.add_node(o)
            G.add_edge(s, o, type=p, time=t)
        max_neighbor_num = 300
        all_neighbor = torch.zeros((dataset.num_entities(), 3, max_neighbor_num), dtype=torch.long)
        all_neighbor_num = torch.zeros(dataset.num_entities(), dtype=torch.long)
        rng = np.random.default_rng(rng_seed)
        for s in range(dataset.num_entities()):
            if s not in G:
                continue
            suc = list(G.successors(s))
            pre = list(G.predecessors(s))
            suc_edge_types = [G.get_edge_data(s, v)['type'] + dataset.num_relations() for v in suc]
            pre_edge_types = [G.get_edge_data(v, s)['type'] for v in pre]
            suc_edge_times = [G.get_edge_data(s, v)['time'] for v in suc]
            pre_edge_times = [G.get_edge_data(v, s)['time'] for v in pre]
            rand_permut = rng.permutation(len(suc) + len(pre))
            neighbor = np.asarray(suc + pre)[rand_permut]
            neighbor_edge_types = np.asarray(suc_edge_types + pre_edge_types)[rand_permut]
            neighbor_edge_times = np.asarray(suc_edge_times + pre_edge_times)[rand_permut]
            neighbor = neighbor[:max_neighbor_num]
            neighbor_edge_types = neighbor_edge_types[:max_neighbor_num]
            neighbor_edge_times = neighbor_edge_times[:max_neighbor_num]
            all_neighbor[s, 0, 0:len(neighbor)] = torch.tensor(neighbor, dtype=torch.long)
            all_neighbor[s, 1, 0:len(neighbor)] = torch.tensor(neighbor_edge_types, dtype=torch.long)
            all_neighbor[s, 2, 0:len(neighbor)] = torch.tensor(neighbor_edge_times, dtype=torch.long)
            all_neighbor_num[s] = len(neighbor)
        dataset._indexes[name] = (all_neighbor, all_neighbor_num)

    dataset.config.log("Neighbors index finished", prefix="  ")

    return dataset._indexes.get(name)


# 这个得到的是 rel id 作为下标的，所有的 1/M-1/N 的标记，记为 relation_types
def index_relation_types(dataset):
    """Classify relations into 1-N, M-1, 1-1, M-N.

    According to Bordes et al. "Translating embeddings for modeling multi-relational
    data.", NIPS13.

    Adds index `relation_types` with list that maps relation index to ("1-N", "M-1",
    "1-1", "M-N").

    """
    if "relation_types" not in dataset._indexes:
        # 2nd dim: num_s, num_distinct_po, num_o, num_distinct_so, is_M, is_N
        relation_stats = torch.zeros((dataset.num_relations(), 6))
        for index, p in [
            (dataset.index("train_sp_to_o"), 1),
            (dataset.index("train_po_to_s"), 0),
        ]:
            # index train_xx_to_x 对应的 index，p 是 1 或者 0: 这样 prefix[p] 取到的就是其中的 predicates 了
            for prefix, labels in index.items():
                # prefix 是 (s, p) 或 (p, o)，labels 是对应的 [o] 或 [s]
                # p 为 0 时，第二个坐标是 0；p 为 1 时，第二个坐标是 2: 刚好对应于 index 的 target/direction 的位置
                relation_stats[prefix[p], 0 + p * 2] = relation_stats[
                    prefix[p], 0 + p * 2
                ] + len(labels)
                relation_stats[prefix[p], 1 + p * 2] = (
                    relation_stats[prefix[p], 1 + p * 2] + 1.0
                )
        # relation_stats 的第一个下标表示的是 relation id，第二个下标是一个 6 元组，其中第一个位置是 rel 作为 po_s 时，s 的总计数；
        # 第二个位置是 rel 作为 po_s 时，rel 出现的次数；第三个位置是 rel 作为 sp_o 时，o 的总计数；第四个位置是 rel 作为 sp_o 时，rel
        # 出现的次数。
        relation_stats[:, 4] = (relation_stats[:, 0] / relation_stats[:, 1]) > 1.5  # po_s 时，num_s/p 超过 1.5
        relation_stats[:, 5] = (relation_stats[:, 2] / relation_stats[:, 3]) > 1.5  # sp_o 时，num_o/p 超过 1.5
        relation_types = []
        # 如果 rel 相关的 head 平均数量超过 1.5 就记为 M，否则就记为 1; 相关的 tail 平均数量超过 1.5 就记为 N，否则就记为 1；
        # 最终根据上面的标记，得到 rel 的分类为：M-N, 1-N, M-1, 1-1。
        for i in range(dataset.num_relations()):
            relation_types.append(
                "{}-{}".format(
                    "1" if relation_stats[i, 4].item() == 0 else "M",
                    "1" if relation_stats[i, 5].item() == 0 else "N",
                )
            )

        dataset._indexes["relation_types"] = relation_types

    return dataset._indexes["relation_types"]


# 这里按 rel type 1/M-1/N 组织 rel id set()
def index_relations_per_type(dataset):
    if "relations_per_type" not in dataset._indexes:
        relations_per_type = {}
        for i, k in enumerate(dataset.index("relation_types")):
            relations_per_type.setdefault(k, set()).add(i)
        dataset._indexes["relations_per_type"] = relations_per_type
    else:
        relations_per_type = dataset._indexes["relations_per_type"]

    dataset.config.log("Loaded relation index")
    for k, relations in relations_per_type.items():
        dataset.config.log(
            "{} relations of type {}".format(len(relations), k), prefix="  "
        )

    return relations_per_type


def index_frequency_percentiles(dataset, recompute=False):
    """
    :return: dictionary mapping from
    {
        'subject':
        {25%, 50%, 75%, top} -> set of entities
        'relations':
        {25%, 50%, 75%, top} -> set of relations
        'object':
        {25%, 50%, 75%, top} -> set of entities
    }
    """
    if "frequency_percentiles" in dataset._indexes and not recompute:
        return
    subject_stats = torch.zeros((dataset.num_entities(), 1))
    relation_stats = torch.zeros((dataset.num_relations(), 1))
    object_stats = torch.zeros((dataset.num_entities(), 1))
    for (s, p, o) in dataset.split("train"):
        subject_stats[s] += 1
        relation_stats[p] += 1
        object_stats[o] += 1
    result = dict()
    for arg, stats, num in [
        (
            "subject",
            [
                i
                for i, j in list(
                    sorted(enumerate(subject_stats.tolist()), key=lambda x: x[1])
                )
            ],
            dataset.num_entities(),
        ),
        (
            "relation",
            [
                i
                for i, j in list(
                    sorted(enumerate(relation_stats.tolist()), key=lambda x: x[1])
                )
            ],
            dataset.num_relations(),
        ),
        (
            "object",
            [
                i
                for i, j in list(
                    sorted(enumerate(object_stats.tolist()), key=lambda x: x[1])
                )
            ],
            dataset.num_entities(),
        ),
    ]:
        for percentile, (begin, end) in [
            ("25%", (0.0, 0.25)),
            ("50%", (0.25, 0.5)),
            ("75%", (0.5, 0.75)),
            ("top", (0.75, 1.0)),
        ]:
            if arg not in result:
                result[arg] = dict()
            result[arg][percentile] = set(stats[int(begin * num) : int(end * num)])
    dataset._indexes["frequency_percentiles"] = result


class IndexWrapper:
    """Wraps a call to an index function so that it can be pickled"""

    def __init__(self, fun, **kwargs):
        self.fun = fun
        self.kwargs = kwargs

    def __call__(self, dataset: "Dataset", **kwargs):
        self.fun(dataset, **self.kwargs)


def _invert_ids(dataset, obj: str):
    if not f"{obj}_id_to_index" in dataset._indexes:
        ids = dataset.load_map(f"{obj}_ids")
        inv = {v: k for k, v in enumerate(ids)}
        dataset._indexes[f"{obj}_id_to_index"] = inv
    else:
        inv = dataset._indexes[f"{obj}_id_to_index"]
    dataset.config.log(f"Indexed {len(inv)} {obj} ids", prefix="  ")


def create_default_index_functions(dataset: "Dataset"):
    for split in dataset.files_of_type("triples"):
        for key, value in [("sp", "o"), ("po", "s"), ("so", "p"), ("spt", "o"), ("pot", "s"), ("sot", "p")]:
            # self assignment needed to capture the loop var
            # IndexWrapper 这个方法类似 map() 函数，把后面的数据应用到第一个参数指定的函数里处理
            dataset.index_functions[f"{split}_{key}_to_{value}"] = IndexWrapper(
                index_KvsAll, split=split, key=key
            )
    dataset.index_functions["neighbor"] = index_neighbor_multidig
    dataset.index_functions["relation_types"] = index_relation_types
    dataset.index_functions["relations_per_type"] = index_relations_per_type
    dataset.index_functions["frequency_percentiles"] = index_frequency_percentiles
    dataset.index_functions["fre"] = index_frequency_percent

    for obj in ["entity", "relation"]:
        dataset.index_functions[f"{obj}_id_to_index"] = IndexWrapper(
            _invert_ids, obj=obj
        )


@numba.njit
def where_in(x, y, not_in=False):
    """Retrieve the indices of the elements in x which are also in y.

    x and y are assumed to be 1 dimensional arrays.

    :params: not_in: if True, returns the indices of the elements in x
    which are not in y.

    """
    # np.isin is not supported in numba. Also: "i in y" raises an error in numba
    # setting njit(parallel=True) slows down the function
    list_y = set(y)
    return np.where(np.array([i in list_y for i in x]) != not_in)[0]
