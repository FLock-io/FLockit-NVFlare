from nvflare.app_common.workflows.base_fedavg import BaseFedAvg
from nvflare.app_common.abstract.fl_model import FLModel

class FedAvgVoterWorkflow(BaseFedAvg):
    """Controller for FedAvg Workflow. *Note*: This class is based on the `ModelController`.
    Implements [FederatedAveraging](https://arxiv.org/abs/1602.05629).

    Provides the implementations for the `run` routine, controlling the main workflow:
        - def run(self)

    The parent classes provide the default implementations for other routines.

    Args:
        num_clients (int, optional): The number of clients. Defaults to 3.
        num_rounds (int, optional): The total number of training rounds. Defaults to 5.
        start_round (int, optional): The starting round number.
        persistor_id (str, optional): ID of the persistor component. Defaults to "persistor".
    """

    def run(self) -> None:
        self.info("Start Evaluation.")

        client = self.sample_clients(1)

        empty_model = FLModel(params=None, metrics=None, meta={})

        self.send_model_and_wait(data=empty_model, targets=client)

        self.info("Finished FedAvg Evaluation.")