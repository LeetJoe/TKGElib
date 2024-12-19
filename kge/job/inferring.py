import math
import time
import re
import os
import json

import torch
from kge.job import Job
from kge import Config, Dataset


def _get_annotation_from_trace(exp_dir):
    trace_file = os.path.join(exp_dir, 'trace.yaml')
    save_file = os.path.join(exp_dir, 'annotation.tsv')

    save_dict = {}
    with open(trace_file, 'r') as fr:
        for line in fr:
            line = re.sub('\s', '', line)
            line = re.sub(',', '\",\"', line)
            line = re.sub(':', '\":\"', line)
            line = re.sub('{}', '', line)
            line = re.sub('\[\]', '', line)
            line = re.sub('\"\[', '[\"', line)
            line = re.sub('\]\"', '\"]', line)
            line = re.sub('{', '{\"', line)
            line = re.sub('}', '\"}', line)
            try:
                line_data = json.loads(line)
            except Exception as e:
                continue
            if ('event' in line_data) and (line_data['event'] == 'query_score'):
                s = int(line_data['s'])
                p = int(line_data['p'])
                o = int(line_data['o'])
                t = int(line_data['t'])
                score = float(line_data['score'])

                key = (s, p, o)
                # if (key not in save_dict) or (save_dict[key][1] < score):
                save_dict[key] = [t, score]
        fr.close()

    with open(save_file, 'w') as fw:
        for key, value in save_dict.items():
            fw.write('{}\t{}\t{}\t{}\t{}\n'.format(
                key[0], key[1], key[2], value[0], torch.sigmoid(torch.Tensor([value[1] * 0.1])).item())
            )
        fw.close()


class InferringJob(Job):
    """ Entity ranking evaluation protocol """

    def __init__(self, config: Config, dataset: Dataset, parent_job, model):
        super().__init__(config, dataset, parent_job)

        self.config = config
        self.dataset = dataset
        self.model = model
        self.batch_size = config.get("infer.batch_size")
        self.device = self.config.get("job.device")
        self.config.check("train.trace_level", ["example", "batch", "epoch"])
        self.trace_examples = self.config.get("eval.trace_level") == "example"
        self.trace_batch = (
            self.trace_examples or self.config.get("train.trace_level") == "batch"
        )
        self.infer_split = self.config.get("infer.split")
        self.epoch = -1

        #: Hooks run after training for an epoch.
        #: Signature: job, trace_entry
        self.post_epoch_hooks = []

        #: Hooks run before starting a batch.
        #: Signature: job
        self.pre_batch_hooks = []

        #: Hooks run before outputting the trace of a batch. Can modify trace entry.
        #: Signature: job, trace_entry
        self.post_batch_trace_hooks = []

        #: Hooks run before outputting the trace of an epoch. Can modify trace entry.
        #: Signature: job, trace_entry
        self.post_epoch_trace_hooks = []

        #: Signature: job, trace_entry
        self.post_infer_hooks = [
            self._save_annotation
        ]

        self.is_prepared = False

        # all done, run job_created_hooks if necessary
        if self.__class__ == InferringJob:
            for f in Job.job_created_hooks:
                f(self)

    def _prepare(self):
        """Construct all indexes needed to run."""

        if self.is_prepared:
            return

        # create data and precompute indexes
        # 配置文件里的 eval.split 使用的是 valid
        # 使用 np.load() 得到 shape = (n, 4) 的数据再使用 torch.from_numpy() 转化成 tensor
        self.triples = self.dataset.split(self.config.get("infer.split"))

        # and data loader
        # DataLoader 输入的第一个参数是 Dataset 类型的数据，但是这里使用的 self.triples 其实就是一个 tensor 类型的变量；
        # 而且 collate_fn 里使用的 batch 也并非是这个 tensor 的一段，需要进一步研究下。
        self.loader = torch.utils.data.DataLoader(
            self.triples,  # 看起来只包含 infer.split 数据，默认是 infer 数据，这个是四元组
            collate_fn=self._collate,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.config.get("infer.num_workers"),
            pin_memory=self.config.get("infer.pin_memory"),
        )
        # let the model add some hooks, if it wants to do so
        self.model.prepare_job(self)
        self.is_prepared = True

    def _collate(self, batch):

        # todo !!!!! 这里注意 batch 格式，DataLoader 里输入的 dataset 是 shape=[n, 4] 的一个 tensor，但是 batch 并不是一个 shape
        # todo =[b, 4] 的 tensor, 而是一个长度为 b 的 tuple，其中每一项是一个 shape=[4] 的 tensor

        # batch 原来是由 batch_size 个 shape=[4] 的 tensor 组成，使用 cat() 后变成一个 shape=[batch_size * 4] 的 tensor；
        # 再使用 reshape((-1, 4)) 的作用是将转换成一个 shape=[batch_size, 4] 的 tensor，注意新旧 batch 的差别。
        batch = torch.cat(batch).reshape((-1, 4))
        return batch

    @torch.no_grad()
    def run(self) -> dict:
        self._prepare()

        was_training = self.model.training
        self.model.eval()
        self.config.log(
            "Inferring on "
            + self.infer_split
            + " data (epoch {})...".format(self.epoch)
        )
        num_entities = self.dataset.num_entities()

        # let's go
        epoch_time = -time.time()
        for batch_number, batch in enumerate(self.loader):
            # construct a sparse label tensor of shape batch_size x 2*num_entities
            # entries are either 0 (false) or infinity (true)
            # TODO add timing information
            # todo 从 _collate() 函数的实现来看，这里的 batch_coords 包括三部分：batch, label_coords 和 test_label_coords
            batch = batch.to(self.device)
            s, p, o, t = batch[:, 0], batch[:, 1], batch[:, 2], batch[:, 3]  # 这里终于有 t 了

            # compute true scores beforehand, since we can't get them from a chunked
            # score table, 最后一个 "o"/"s" 表示 direction
            # ECEformer 使用的是点积相似度
            o_true_scores = self.model("score_spo", s, p, o, t, "o").view(-1)
            s_true_scores = self.model("score_spo", s, p, o, t, "s").view(-1)

            # optionally: trace ranks of each example
            # 这个设置为 true 之后，所有的 query 的相关信息都会 trace 出来，在 eval job 里会执行，输出在 trace.yaml 文件里
            # 输出有 s, p, o, t, task(sp, po), split(test), filter([train,valid,test]),rank, rank_filtered
            # todo: 还缺个 score
            if self.trace_examples:
                entry = {
                    "type": "hidden_inference",
                    "scope": "example",
                    "split": self.infer_split,
                    "size": len(batch),
                    "batches": len(self.loader),
                    "epoch": self.epoch,
                }
                for i in range(len(batch)):
                    entry["batch"] = i
                    entry["s"], entry["p"], entry["o"], entry["t"] = (
                        s[i].item(),
                        p[i].item(),
                        o[i].item(),
                        t[i].item(),
                    )
                    self.trace(
                        event="query_score",
                        task="sp",
                        score=o_true_scores[i].item(),
                        **entry,
                    )
                    self.trace(
                        event="query_score",
                        task="po",
                        score=s_true_scores[i].item(),
                        **entry,
                    )

            # optionally: trace batch metrics
            if self.trace_batch:  # 这里在配置文件里是 true —— 如果 trace level 是 example，这个就是 true。
                self.trace(
                    event="batch_completed",
                    type="hidden_inference",
                    scope="batch",
                    split=self.infer_split,
                    epoch=self.epoch,
                    batch=batch_number,
                    size=len(batch),
                    batches=len(self.loader),
                )

            # output batch information to console
            self.config.print(
                (
                    "\r"  # go back
                    + "{}  batch:{: "
                    + str(1 + int(math.ceil(math.log10(len(self.loader)))))
                    + "d}/{} \033[K"  # clear to right
                ).format(
                    self.config.log_prefix,
                    batch_number,
                    len(self.loader) - 1,
                ),
                end="",
                flush=True,
            )

        # we are done; compute final metrics
        self.config.print("\033[2K\r", end="", flush=True)  # clear line and go back
        epoch_time += time.time()

        # compute trace
        trace_entry = dict(
            type="hidden_inference",
            scope="epoch",
            split=self.infer_split,
            epoch=self.epoch,
            batches=len(self.loader),
            size=len(self.triples),
            epoch_time=epoch_time,
            event="infer_completed"
        )
        for f in self.post_epoch_trace_hooks:
            f(self, trace_entry)

        # write out trace
        trace_entry = self.trace(**trace_entry, echo=True, echo_prefix="  ", log=True)

        # reset model and return metrics
        if was_training:
            self.model.train()  # todo: 这个操作是 reset model？？
        # self.config.log("Finished inferring on " + self.infer_split + " split.")  # todo 这个要解除注释

        for f in self.post_infer_hooks:
            f()

        return trace_entry

    def _save_annotation(self):
        _get_annotation_from_trace(self.config.folder)
