from typing import List, Optional

import torch.nn as nn

from templates.llm_finetuning.nvflare_utils.workflows.fedavg_voter_workflow import FedAvgVoterWorkflow

from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob

class VoterFedAvgJob(BaseFedJob):
    def __init__(
        self,
        initial_model: nn.Module = None,
        name: str = "fed_job",
        key_metric: str = "accuracy",
    ):
        super().__init__(initial_model=initial_model, name=name, key_metric=key_metric)

        controller = FedAvgVoterWorkflow()

        self.to_server(controller)