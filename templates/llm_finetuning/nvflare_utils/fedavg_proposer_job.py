import torch.nn as nn
from .workflows.fedavg_proposer_workflow import FedAvgProposerWorkflow
from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob

from loguru import logger

class ProposerFedAvgJob(BaseFedJob):
    def __init__(
        self,
        initial_model: nn.Module = None,
        name: str = "fed_job",
        key_metric: str = "accuracy",
    ):
        super().__init__(initial_model=initial_model, name=name, key_metric=key_metric)

        self.controller = FedAvgProposerWorkflow()

        self.to_server(self.controller)

