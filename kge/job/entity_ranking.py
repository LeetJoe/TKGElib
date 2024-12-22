import math
import time

import os
import torch
import kge.job
from kge.job import EvaluationJob, Job
from kge import Config, Dataset
from collections import defaultdict


def _get_test_prediction_from_trace(job: Job, trace_entry):
    exp_dir = job.config.folder
    trace_file = os.path.join(exp_dir, 'trace.yaml')
    save_file = os.path.join(exp_dir, 'pred_kge.tsv')

    save_dict = {}
    with open(trace_file, 'r') as fr:
        with open(save_file, 'w') as fw:
            for line in fr:
                line_data = Job.trace_line_to_json(line)
                if ('event' in line_data) and (line_data['event'] == 'example_rank'):
                    s = int(line_data['s'])
                    p = int(line_data['p'])
                    o = int(line_data['o'])
                    t = int(line_data['t'])
                    task = line_data['task']
                    rank = line_data['rank_filtered']
                    candidates = line_data['candidates']

                    fw.write('{}\t{}\t{}\t{}\t{}\t{}\n{}\n'.format(
                        s, p, o, t, task, rank, candidates)
                    )
            fw.close()
        fr.close()


class EntityRankingJob(EvaluationJob):
    """ Entity ranking evaluation protocol """

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)
        self.config.check(
            "eval.tie_handling",  # tie 是指跟 target 具有相同的分数的 candidates，影响最终的 rank 取值。配置里使用的是 round mean
            ["rounded_mean_rank", "best_rank", "worst_rank"],
        )
        self.tie_handling = self.config.get("eval.tie_handling")

        if self.config.get("eval.split") == 'test':
            self.post_valid_hooks.append(_get_test_prediction_from_trace)

        self.is_prepared = False

        if self.__class__ == EntityRankingJob:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct all indexes needed to run."""

        if self.is_prepared:
            return

        # create data and precompute indexes
        # 配置文件里的 eval.split 使用的是 valid
        # 使用 np.load() 得到 shape = (n, 4) 的数据再使用 torch.from_numpy() 转化成 tensor
        self.triples = self.dataset.split(self.config.get("eval.split"))
        # filter_splits 一般包含 train 和 valid，这里按 sp_to_o 和 po_to_s 分跟 TLogic 里设置逆关系的思路应该是一样的
        for split in self.filter_splits:
            # dataset.index(key) 这个方法关键看 key 的值，比如 valid_sp_to_o 表示把 valid 数据，以 (s, p) 为键，而 [o1,o2...] 为值
            # 的形式构建成一个 OrderList。初次调用会把组织结果缓存起来，方便后面使用。
            # self.dataset.index(f"{split}_sp_to_o")
            # self.dataset.index(f"{split}_po_to_s")
            self.dataset.index(f"{split}_spt_to_o")  # todo indexing modified
            self.dataset.index(f"{split}_pot_to_s")  # todo indexing modified

        # 这个条件是为了防止 test 在 filter_splits 且 filter_with_test 为 True 重复加载 test 数据；而在 test 不在 filter_splits 里
        # 但 filter_with_test 为 True 保证要加入 test 数据。
        if "test" not in self.filter_splits and self.filter_with_test:
            # self.dataset.index("test_sp_to_o")
            # self.dataset.index("test_po_to_s")
            self.dataset.index("test_spt_to_o")  # todo indexing modified
            self.dataset.index("test_pot_to_s")  # todo indexing modified

        # and data loader
        # DataLoader 输入的第一个参数是 Dataset 类型的数据，但是这里使用的 self.triples 其实就是一个 tensor 类型的变量；
        # 而且 collate_fn 里使用的 batch 也并非是这个 tensor 的一段，需要进一步研究下。
        self.loader = torch.utils.data.DataLoader(
            self.triples,  # 看起来只包含 eval.split 数据，默认是 valid 数据，这个是四元组
            collate_fn=self._collate,
            shuffle=False,
            batch_size=self.batch_size,
            # 这个 num_workers 表示用来拆分 batch 的进程数量，配置文件里使用的是 0，表示不使用子进程。
            num_workers=self.config.get("eval.num_workers"),
            pin_memory=self.config.get("eval.pin_memory"),  # 配置文件里使用的是 False，不是 True 性能会更好吗？会有什么问题？
        )
        # let the model add some hooks, if it wants to do so
        self.model.prepare_job(self)
        self.is_prepared = True

    def _collate(self, batch):

        # todo !!!!! 这里注意 batch 格式，DataLoader 里输入的 dataset 是 shape=[n, 4] 的一个 tensor，但是 batch 并不是一个 shape
        # todo =[b, 4] 的 tensor, 而是一个长度为 b 的 tuple，其中每一项是一个 shape=[4] 的 tensor

        """Looks up true triples for each triple in the batch"""
        # 从注释来看，这个 label_coords 的目的是为了找到在那些跟 batch 的 sp 或 po 相同的 facts，寻找范围依 filter 参数来确定。
        label_coords = []
        for split in self.filter_splits:
            split_label_coords = kge.job.util.get_spt_pot_coords_from_spot_batch(  # todo indexing modified
            # split_label_coords = kge.job.util.get_sp_po_coords_from_spo_batch(
                batch,
                self.dataset.num_entities(),
                self.dataset.index(f"{split}_spt_to_o"),  # todo indexing modified
                self.dataset.index(f"{split}_pot_to_s"),  # todo indexing modified
                # self.dataset.index(f"{split}_sp_to_o"),
                # self.dataset.index(f"{split}_po_to_s"),
            )
            label_coords.append(split_label_coords)
        label_coords = torch.cat(label_coords)

        if "test" not in self.filter_splits and self.filter_with_test:
            test_label_coords = kge.job.util.get_spt_pot_coords_from_spot_batch(  # todo indexing modified
            # test_label_coords = kge.job.util.get_spt_pot_coords_from_spot_batch(
                batch,
                self.dataset.num_entities(),
                self.dataset.index("test_spt_to_o"),  # todo indexing modified
                self.dataset.index("test_pot_to_s"),  # todo indexing modified
                # self.dataset.index("test_sp_to_o"),
                # self.dataset.index("test_po_to_s"),
            )
        else:
            test_label_coords = torch.zeros([0, 3], dtype=torch.long)

        # batch 原来是由 batch_size 个 shape=[4] 的 tensor 组成，使用 cat() 后变成一个 shape=[batch_size * 4] 的 tensor；
        # 再使用 reshape((-1, 4)) 的作用是将转换成一个 shape=[batch_size, 4] 的 tensor，注意新旧 batch 的差别。
        batch = torch.cat(batch).reshape((-1, 4))
        return batch, label_coords, test_label_coords

    @torch.no_grad()
    def run(self) -> dict:
        self._prepare()

        was_training = self.model.training  # 这个是状态参数？
        self.model.eval()  # 这个是模块里提供的方法
        self.config.log(
            "Evaluating on "
            + self.eval_split
            + " data (epoch {})...".format(self.epoch)
        )
        num_entities = self.dataset.num_entities()

        # we also filter with test data if requested，这里即使是对 valid 数据进行评估，也可以指定对 test 也进行 filter
        filter_with_test = "test" not in self.filter_splits and self.filter_with_test

        # which rankings to compute (DO NOT REORDER; code assumes the order given here)
        rankings = (
            ["_raw", "_filt", "_filt_test"] if filter_with_test else ["_raw", "_filt"]
        )

        # dictionary that maps entry of rankings to a sparse tensor containing the
        # true labels for this option
        # 初始化，以面按前面的 rankings 为键值组织对应的数据
        labels_for_ranking = defaultdict(lambda: None)

        # Initiliaze dictionaries that hold the overall histogram of ranks of true
        # answers. These histograms are used to compute relevant metrics. The dictionary
        # entry with key 'all' collects the overall statistics and is the default.
        hists = dict()
        hists_filt = dict()
        hists_filt_test = dict()

        # let's go
        epoch_time = -time.time()
        for batch_number, batch_coords in enumerate(self.loader):
            # construct a sparse label tensor of shape batch_size x 2*num_entities
            # entries are either 0 (false) or infinity (true)
            # TODO add timing information
            # todo 从 _collate() 函数的实现来看，这里的 batch_coords 包括三部分：batch, label_coords 和 test_label_coords
            batch = batch_coords[0].to(self.device)  # [0] 就是 batch 本身
            s, p, o, t = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3]  # 这里终于有 t 了
            label_coords = batch_coords[1].to(self.device)
            if filter_with_test:
                test_label_coords = batch_coords[2].to(self.device)
                # create sparse labels tensor
                # 这个 coord_to_sparse_tensor 就是用来构建一个稀疏矩阵，里面用的是 torch.sparse.Tensor() 方法
                # test_label_coords 的 shape=[2, num_ones]，其中 num_ones 大小不确定，具体见 get_sp_po_coords_from_spo_batch；
                # test_label_coords 的第一行是 batch 的下标，范围 [0, batch_size - 1], 第二行是 entity id, 由于 _po 模式对
                # subject id 作了 +num_entities 操作，使其范围变成 [0, 2*num_entities-1]。
                # coord_to_sparse_tensor 第一个参数是稀疏矩阵的行数，大小是 len(batch)，第二个参数是稀疏矩阵的列数，大小是 2*num_entities，
                # 第三个参数 test_label_coords 传入后会被执行转置操作，变成了 num_ones 个二元组，结合其取值范围，刚好在稀疏矩阵的尺寸
                # 范围内，这样的二元组指定坐标处的值置为 Inf，其它位置为 0，从而构建了一个稀疏矩阵。
                test_labels = kge.job.util.coord_to_sparse_tensor(
                    len(batch),  # nrows
                    2 * num_entities,  # ncols
                    test_label_coords,
                    self.device,
                    float("Inf"),  # 非 0 默认值，默认是 1，乘以这个 value 参数
                )
                labels_for_ranking["_filt_test"] = test_labels

            # create sparse labels tensor
            labels = kge.job.util.coord_to_sparse_tensor(
                len(batch), 2 * num_entities, label_coords, self.device, float("Inf")
            )
            labels_for_ranking["_filt"] = labels

            # compute true scores beforehand, since we can't get them from a chunked
            # score table, 最后一个 "o"/"s" 表示 direction
            o_true_scores = self.model("score_spo", s, p, o, t, "o").view(-1)
            s_true_scores = self.model("score_spo", s, p, o, t, "s").view(-1)

            # default dictionary storing rank and num_ties for each key in rankings
            # as list of len 2: [rank, num_ties]
            # todo 这个 defaultdict() 的第一个参数影响的是出现重复的 key 或者取 key 的时候不存在时的行为，这里的使用方法有点看不太懂
            ranks_and_ties_for_ranking = defaultdict(
                lambda: [
                    torch.zeros(s.size(0), dtype=torch.long).to(self.device),
                    torch.zeros(s.size(0), dtype=torch.long).to(self.device),
                ]
            )

            # calculate scores in chunks to not have the complete score matrix in memory
            # a chunk here represents a range of entity_values to score against
            if self.config.get("eval.chunk_size") > -1:  # 配置文件里默认就是 -1
                chunk_size = self.config.get("eval.chunk_size")
            else:
                chunk_size = self.dataset.num_entities()

            # for trace candidates' scores
            batch_scores_sp = torch.zeros((len(batch),num_entities ), dtype=torch.float).to(self.device)
            batch_scores_po = torch.zeros((len(batch),num_entities ), dtype=torch.float).to(self.device)

            # process chunk by chunk
            # 这里 chunk 并不是对 batch 进行进一步的分批，而是在进行 _po/sp_ 预测的时候，对目标所属的 entities 集合进行
            # chunk，避免 entities 过多单批推理负载过重，限定候选范围多次推理，用时间换空间。
            for chunk_number in range(math.ceil(num_entities / chunk_size)):
                chunk_start = chunk_size * chunk_number
                chunk_end = min(chunk_size * (chunk_number + 1), num_entities)

                # compute scores of chunk
                if chunk_size == self.dataset.num_entities():
                    scores = self.model("score_sp_po", s, p, o, t, None)
                else:
                    scores = self.model("score_sp_po",
                        s, p, o, t, torch.arange(chunk_start, chunk_end).to(self.device)
                    )

                # 结果平均分两段，前半段是 sp 的结果，后半段是 po 的结果。
                scores_sp = scores[:, : chunk_end - chunk_start]
                scores_po = scores[:, chunk_end - chunk_start :]

                # replace the precomputed true_scores with the ones occurring in the
                # scores matrix to avoid floating point issues
                # 前面这两句用来选择出 s, o 中满足 chunk 范围的 mask
                s_in_chunk_mask = (chunk_start <= s) & (s < chunk_end)
                o_in_chunk_mask = (chunk_start <= o) & (o < chunk_end)
                # 这两句应该是要实现 scores_sp/po 里的下标与 id 的转换：scores_sp/po 里的 target 是 entities(子集) 的下标，
                # 但存储的时候仍然是从 0 开始计，与 entities 的实际下标有 chunk_start 的偏移，用 id - chunk_start 得到的就是
                # 对应实体 id 在 scores_sp/po 的实际下标位置。
                o_in_chunk = (o[o_in_chunk_mask] - chunk_start).long()
                s_in_chunk = (s[s_in_chunk_mask] - chunk_start).long()
                # 这里的操作是把经过 score_sp() 预测的结果 (s, p, ?, t) 里 ? 存在于 facts 中的 (s, p, o, t) 的预测结果，用
                # score_spo() 方法预测的结果进行替换，理论上两者应该是相等的，这里替换的目的说是“避免 floating issues，猜测是避
                # 免因为小数点极小的差异导致的分数浮动，统一使用 score_spo() 的结果保持分数全局来源一致。
                scores_sp[o_in_chunk_mask, o_in_chunk] = o_true_scores[o_in_chunk_mask]
                scores_po[s_in_chunk_mask, s_in_chunk] = s_true_scores[s_in_chunk_mask]

                # now compute the rankings (assumes order: None, _filt, _filt_test)
                for ranking in rankings:
                    # ranking 是 _raw 的时候，对应的 labels 是 None
                    if labels_for_ranking[ranking] is None:
                        labels_chunk = None
                    else:
                        # densify the needed part of the sparse labels tensor
                        labels_chunk = self._densify_chunk_of_labels(
                            labels_for_ranking[ranking], chunk_start, chunk_end
                        )

                        # remove current example from labels
                        # 将 query 的 target 从 labels 里删除，target 的 score 从 x_true_scores 里取
                        labels_chunk[o_in_chunk_mask, o_in_chunk] = 0
                        labels_chunk[
                            s_in_chunk_mask, s_in_chunk + (chunk_end - chunk_start)
                        ] = 0

                    # compute partial ranking and filter the scores (sets scores of true
                    # labels to infinity)
                    (
                        s_rank_chunk,
                        s_num_ties_chunk,
                        o_rank_chunk,
                        o_num_ties_chunk,
                        scores_sp_filt,
                        scores_po_filt,
                    ) = self._filter_and_rank(
                        scores_sp, scores_po, labels_chunk, o_true_scores, s_true_scores
                    )

                    # from now on, use filtered scores
                    scores_sp = scores_sp_filt
                    scores_po = scores_po_filt

                    if ranking == '_filt':
                        batch_scores_sp[:, chunk_start : chunk_end] += scores_sp
                        batch_scores_po[:, chunk_start : chunk_end] += scores_po

                    # update rankings，这里的 += 就是 cat()
                    ranks_and_ties_for_ranking["s" + ranking][0] += s_rank_chunk
                    ranks_and_ties_for_ranking["s" + ranking][1] += s_num_ties_chunk
                    ranks_and_ties_for_ranking["o" + ranking][0] += o_rank_chunk
                    ranks_and_ties_for_ranking["o" + ranking][1] += o_num_ties_chunk

                # we are done with the chunk

            # We are done with all chunks; calculate final ranks from counts
            # _get_ranks() 就是对 rank 和 ties 应用 best/worst/mean 策略
            # 这些 x_ranks 中的 rank 都是从 0 开始计数的。
            s_ranks = self._get_ranks(
                ranks_and_ties_for_ranking["s_raw"][0],
                ranks_and_ties_for_ranking["s_raw"][1],
            )
            o_ranks = self._get_ranks(
                ranks_and_ties_for_ranking["o_raw"][0],
                ranks_and_ties_for_ranking["o_raw"][1],
            )
            s_ranks_filt = self._get_ranks(
                ranks_and_ties_for_ranking["s_filt"][0],
                ranks_and_ties_for_ranking["s_filt"][1],
            )
            o_ranks_filt = self._get_ranks(
                ranks_and_ties_for_ranking["o_filt"][0],
                ranks_and_ties_for_ranking["o_filt"][1],
            )

            # Update the histograms of raw ranks and filtered ranks
            batch_hists = dict()
            batch_hists_filt = dict()
            for f in self.hist_hooks:  # 结果直接应用在 batch_hists 和 batch_hists_filt 里，没有返回
                f(batch_hists, s, p, o, s_ranks, o_ranks, job=self)
                f(batch_hists_filt, s, p, o, s_ranks_filt, o_ranks_filt, job=self)

            # and the same for filtered_with_test ranks
            if filter_with_test:
                batch_hists_filt_test = dict()
                s_ranks_filt_test = self._get_ranks(
                    ranks_and_ties_for_ranking["s_filt_test"][0],
                    ranks_and_ties_for_ranking["s_filt_test"][1],
                )
                o_ranks_filt_test = self._get_ranks(
                    ranks_and_ties_for_ranking["o_filt_test"][0],
                    ranks_and_ties_for_ranking["o_filt_test"][1],
                )
                for f in self.hist_hooks:
                    f(
                        batch_hists_filt_test,
                        s,
                        p,
                        o,
                        s_ranks_filt_test,
                        o_ranks_filt_test,
                        job=self,
                    )

            # optionally: trace ranks of each example
            # 这个设置为 true 之后，所有的 query 的相关信息都会 trace 出来，在 eval job 里会执行，输出在 trace.yaml 文件里
            # 输出有 s, p, o, t, task(sp, po), split(test), filter([train,valid,test]),rank, rank_filtered
            if self.trace_examples:
                entry = {
                    "timestamp": 0,
                    "entry_id": 0,
                    # "type": "entity_ranking",
                    # "scope": "example",
                    # "split": self.eval_split,
                    # "filter_splits": self.filter_splits,
                    # "size": len(batch),
                    # "batches": len(self.loader),
                    # "epoch": self.epoch,
                }
                for i in range(len(batch)):
                    entry["batch"] = i
                    entry["s"], entry["p"], entry["o"], entry["t"] = (
                        s[i].item(),
                        p[i].item(),
                        o[i].item(),
                        t[i].item(),
                    )
                    if filter_with_test:
                        entry["rank_filtered_with_test"] = (
                            o_ranks_filt_test[i].item() + 1
                        )
                    self.trace(
                        event="example_rank",
                        task="sp",
                        rank=o_ranks[i].item() + 1,
                        rank_filtered=o_ranks_filt[i].item() + 1,
                        candidates=self._candidates_to_str(batch_scores_sp[i]),
                        **entry,
                    )
                    if filter_with_test:
                        entry["rank_filtered_with_test"] = (
                            s_ranks_filt_test[i].item() + 1
                        )
                    self.trace(
                        event="example_rank",
                        task="po",
                        rank=s_ranks[i].item() + 1,
                        rank_filtered=s_ranks_filt[i].item() + 1,
                        candidates=self._candidates_to_str(batch_scores_po[i]),
                        **entry,
                    )

            # Compute the batch metrics for the full histogram (key "all")
            # batch_hists["all"].shape = [num_entities, ]，其中存放的是相应 rank 的计数，
            # 如 [1, 2, 3] 表示 rank=1 的有 1 个，rank=2 的有 2 个，rank=3 的有 3 个。rank = index + 1
            # 这里的 metrics 仅仅是 batch metrics，用于输出的，并不会用来进行 global 合并，global 合并的是 ranks 信息
            metrics = self._compute_metrics(batch_hists["all"])
            metrics.update(
                self._compute_metrics(batch_hists_filt["all"], suffix="_filtered")
            )
            if filter_with_test:
                metrics.update(
                    self._compute_metrics(
                        batch_hists_filt_test["all"], suffix="_filtered_with_test"
                    )
                )

            # optionally: trace batch metrics
            if self.trace_batch:  # 这里在配置文件里是 true —— 如果 trace level 是 example，这个就是 true。
                self.trace(
                    event="batch_completed",
                    type="entity_ranking",
                    scope="batch",
                    split=self.eval_split,
                    filter_splits=self.filter_splits,
                    epoch=self.epoch,
                    batch=batch_number,
                    size=len(batch),
                    batches=len(self.loader),
                    **metrics,
                )

            # output batch information to console
            # batch metrics 输出到日志， 只输出 hits_at_1 和 hits_at_10 ？ 算了，不重要
            self.config.print(
                (
                    "\r"  # go back
                    + "{}  batch:{: "
                    + str(1 + int(math.ceil(math.log10(len(self.loader)))))
                    + "d}/{}, mrr (filt.): {:4.3f} ({:4.3f}), "
                    + "hits@1: {:4.3f} ({:4.3f}), "
                    + "hits@{}: {:4.3f} ({:4.3f})"
                    + "\033[K"  # clear to right
                ).format(
                    self.config.log_prefix,
                    batch_number,
                    len(self.loader) - 1,
                    metrics["mean_reciprocal_rank"],
                    metrics["mean_reciprocal_rank_filtered"],
                    metrics["hits_at_1"],
                    metrics["hits_at_1_filtered"],
                    self.hits_at_k_s[-1],
                    metrics["hits_at_{}".format(self.hits_at_k_s[-1])],
                    metrics["hits_at_{}_filtered".format(self.hits_at_k_s[-1])],
                ),
                end="",
                flush=True,
            )

            # merge batch histograms into global histograms
            def merge_hist(target_hists, source_hists):
                for key, hist in source_hists.items():
                    if key in target_hists:
                        target_hists[key] = target_hists[key] + hist
                    else:
                        target_hists[key] = hist

            # 合并 batch hist 到 global hist
            merge_hist(hists, batch_hists)
            merge_hist(hists_filt, batch_hists_filt)
            if filter_with_test:
                merge_hist(hists_filt_test, batch_hists_filt_test)

        # we are done; compute final metrics
        self.config.print("\033[2K\r", end="", flush=True)  # clear line and go back
        for key, hist in hists.items():
            name = "_" + key if key != "all" else ""
            # todo: 这里竟然使用 update()? metric 在前面的循环里使用过而且没有清空，直接这么用没问题吗？
            #  还是说因为 global 必然会覆盖 metrics 里的所有内容所以不需要清理？
            metrics.update(self._compute_metrics(hists[key], suffix=name))
            metrics.update(
                self._compute_metrics(hists_filt[key], suffix="_filtered" + name)
            )
            if filter_with_test:
                metrics.update(
                    self._compute_metrics(
                        hists_filt_test[key], suffix="_filtered_with_test" + name
                    )
                )
        epoch_time += time.time()

        # compute trace
        trace_entry = dict(
            type="entity_ranking",
            scope="epoch",
            split=self.eval_split,
            filter_splits=self.filter_splits,
            epoch=self.epoch,
            batches=len(self.loader),
            size=len(self.triples),
            epoch_time=epoch_time,
            event="eval_completed",
            **metrics,
        )
        for f in self.post_epoch_trace_hooks:  # todo: 这个好像是空的
            f(self, trace_entry)

        # if validation metric is not present, try to compute it
        metric_name = self.config.get("valid.metric")
        if metric_name not in trace_entry:
            # 这里的 metric_expr 用于指定在 valid metric(就是用于在 epoch 间比较优劣的标准) 未找到时进行计算，
            # 这里的配置文件没给这个表达式，也就是说要保证这个 metric 一定在 trace_entry 里。
            trace_entry[metric_name] = eval(
                self.config.get("valid.metric_expr"),
                None,
                dict(config=self.config, **trace_entry),
            )

        # write out trace
        trace_entry = self.trace(**trace_entry, echo=True, echo_prefix="  ", log=True)

        # reset model and return metrics
        if was_training:
            self.model.train()  # todo: 这个操作是 reset model？？
        self.config.log("Finished evaluating on " + self.eval_split + " split.")

        for f in self.post_valid_hooks:
            f(self, trace_entry)

        return trace_entry

    # 从稀疏矩阵 labels 里截取一块子矩阵，重新组织成稀疏矩阵形式返回
    def _densify_chunk_of_labels(
        self, labels: torch.Tensor, chunk_start: int, chunk_end: int
    ) -> torch.Tensor:
        """Creates a dense chunk of a sparse label tensor.

        A chunk here is a range of entity values with 'chunk_start' being the lower
        bound and 'chunk_end' the upper bound.

        The resulting tensor contains the labels for the sp chunk and the po chunk.

        :param labels: sparse tensor containing the labels corresponding to the batch
        for sp and po

        :param chunk_start: int start index of the chunk

        :param chunk_end: int end index of the chunk

        :return: batch_size x chunk_size*2 dense tensor with labels for the sp chunk and
        the po chunk.

        """
        num_entities = self.dataset.num_entities()
        # _indices() 用于 sparse coo 矩阵，返回稀疏矩阵所有非零元素的坐标对 (x, y), x 实际上是 batch index, y 是 entities index。
        indices = labels._indices()
        mask_sp = (chunk_start <= indices[1, :]) & (indices[1, :] < chunk_end)
        mask_po = ((chunk_start + num_entities) <= indices[1, :]) & (
            indices[1, :] < (chunk_end + num_entities)
        )
        indices_sp_chunk = indices[:, mask_sp]
        indices_sp_chunk[1, :] = indices_sp_chunk[1, :] - chunk_start
        indices_po_chunk = indices[:, mask_po]
        indices_po_chunk[1, :] = (
            indices_po_chunk[1, :] - num_entities - chunk_start * 2 + chunk_end
        )
        indices_chunk = torch.cat((indices_sp_chunk, indices_po_chunk), dim=1)
        dense_labels = torch.sparse.LongTensor(
            indices_chunk,
            labels._values()[mask_sp | mask_po],
            torch.Size([labels.size()[0], (chunk_end - chunk_start) * 2]),
        ).to_dense()
        return dense_labels

    def _filter_and_rank(
        self,
        scores_sp: torch.Tensor,
        scores_po: torch.Tensor,
        labels: torch.Tensor,
        o_true_scores: torch.Tensor,
        s_true_scores: torch.Tensor,
    ):
        """Filters the current examples with the given labels and returns counts rank and
num_ties for each true score.

        :param scores_sp: batch_size x chunk_size tensor of scores

        :param scores_po: batch_size x chunk_size tensor of scores

        :param labels: batch_size x 2*chunk_size tensor of scores

        :param o_true_scores: batch_size x 1 tensor containing the scores of the actual
        objects in batch

        :param s_true_scores: batch_size x 1 tensor containing the scores of the actual
        subjects in batch

        :return: batch_size x 1 tensors rank and num_ties for s and o and filtered
        scores_sp and scores_po

        """
        chunk_size = scores_sp.shape[1]
        if labels is not None:
            # remove current example from labels
            labels_sp = labels[:, :chunk_size]
            labels_po = labels[:, chunk_size:]
            scores_sp = scores_sp - labels_sp
            scores_po = scores_po - labels_po
        o_rank, o_num_ties = self._get_ranks_and_num_ties(scores_sp, o_true_scores)
        s_rank, s_num_ties = self._get_ranks_and_num_ties(scores_po, s_true_scores)
        return s_rank, s_num_ties, o_rank, o_num_ties, scores_sp, scores_po

    @staticmethod
    def _get_ranks_and_num_ties(
        scores: torch.Tensor, true_scores: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """Returns rank and number of ties of each true score in scores.

        :param scores: batch_size x entities tensor of scores

        :param true_scores: batch_size x 1 tensor containing the actual scores of the batch

        :return: batch_size x 1 tensors rank and num_ties
        """
        # process NaN values
        scores = scores.clone()
        scores[torch.isnan(scores)] = float("-Inf")
        true_scores = true_scores.clone()
        true_scores[torch.isnan(true_scores)] = float("-Inf")

        # Determine how many scores are greater than / equal to each true answer (in its
        # corresponding row of scores)
        # rank 是 batch_size x best rank 的 矩阵。
        # 经测试如果确定 true_score 是 nx1 的 tensor，不执行 view(-1, 1) 结果也是一样的，这里可能是为了防止一些意外。
        rank = torch.sum(scores > true_scores.view(-1, 1), dim=1, dtype=torch.long)
        num_ties = torch.sum(scores == true_scores.view(-1, 1), dim=1, dtype=torch.long)
        # rank 是从 0 开始计数的
        return rank, num_ties

    def _get_ranks(self, rank: torch.Tensor, num_ties: torch.Tensor) -> torch.Tensor:
        """Calculates the final rank from (minimum) rank and number of ties.

        :param rank: batch_size x 1 tensor with number of scores greater than the one of
        the true score

        :param num_ties: batch_size x tensor with number of scores equal as the one of
        the true score

        :return: batch_size x 1 tensor of ranks

        """

        if self.tie_handling == "rounded_mean_rank":
            return rank + num_ties // 2
        elif self.tie_handling == "best_rank":
            return rank
        elif self.tie_handling == "worst_rank":
            return rank + num_ties - 1
        else:
            raise NotImplementedError

    def _compute_metrics(self, rank_hist, suffix=""):
        """Computes desired matrix from rank histogram"""
        metrics = {}
        n = torch.sum(rank_hist).item()  # n 是 rank_dist 中所有元素和，todo 应该等于 query_num?

        # torch.arange(1, 11) 生成一个 shape=[10,] 从 1 到 10 的 tensor
        ranks = torch.arange(1, self.dataset.num_entities() + 1).float().to(self.device)
        metrics["mean_rank" + suffix] = (
            # rank_hist 是所有 rank 的计数，rank_hist[0] 的值是 rank=1 的计数；rank_hist * ranks 就是所有 sum(ranks（此处的 ranks 从 1 计）)，
            # sum() 后再 /n 刚好就是 MR。
            (torch.sum(rank_hist * ranks).item() / n) if n > 0.0 else 0.0
        )

        reciprocal_ranks = 1.0 / ranks
        metrics["mean_reciprocal_rank" + suffix] = (
            (torch.sum(rank_hist * reciprocal_ranks).item() / n) if n > 0.0 else 0.0
        )

        hits_at_k = (
            # a = torch.cumsum(b, dim=0) 是对 list b 进行累加，a[0] 是 sum(b[:1]), a[1] 是 sum(b[:2]), a[n] = sum(b[:n+1])
            (torch.cumsum(rank_hist[: max(self.hits_at_k_s)], dim=0) / n).tolist()
            if n > 0.0
            else [0.0] * max(self.hits_at_k_s)
        )

        for i, k in enumerate(self.hits_at_k_s):
            metrics["hits_at_{}{}".format(k, suffix)] = hits_at_k[k - 1]

        return metrics

    def _candidates_to_str(self, batch_scores: torch.Tensor) -> str:
        values, indices = torch.sort(batch_scores, descending=True)
        positive_mask = values > 0
        values = torch.sigmoid(values[positive_mask] * 0.1).tolist()
        indices = indices[positive_mask].tolist()
        cand_pair_list = []
        for cidx in range(len(indices)):
            cand_pair_list.append([indices[cidx], values[cidx]])
        cand_pair_list.sort(key=lambda x: x[1], reverse=True)
        cand_str_list = []
        for cidx in range(len(indices)):
            cand_str_list.append('{}*{:.6f}'.format(cand_pair_list[cidx][0], cand_pair_list[cidx][1]))

        return ';'.join(cand_str_list)