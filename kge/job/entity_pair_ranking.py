from kge.job import EvaluationJob, Job


# todo 这个文件是空的？是本就如此还是因为用不到被作者清空了？

class EntityPairRankingJob(EvaluationJob):
    """ Entity-pair ranking evaluation protocol """

    def __init__(self, config, dataset, parent_job, model):
        super().__init__(config, dataset, parent_job, model)

        if self.__class__ == EntityPairRankingJob:
            for f in Job.job_created_hooks:
                f(self)
