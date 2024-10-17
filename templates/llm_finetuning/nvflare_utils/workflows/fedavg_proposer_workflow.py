from nvflare.app_common.workflows.base_fedavg import BaseFedAvg
from nvflare.app_common.abstract.fl_model import FLModel

class FedAvgProposerWorkflow(BaseFedAvg):
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

    def __init__(
            self,
            *args,
            num_clients: int = 1,
            num_rounds: int = 1,
            start_round: int = 0,
            **kwargs,
    ):
        super().__init__(*args, num_clients=num_clients, num_rounds=num_rounds, start_round=start_round, **kwargs)


    def run(self):
        self.info("Start Training.")

        # model = self.load_model()

        client = self.sample_clients(1)

        empty_model = FLModel(params=None, metrics=None, meta={})
        self.send_model_and_wait(data=empty_model, targets=client)

        # self.send_and_wait(targets=client)
        # self.save_model(model)

        self.info("Finished FedAvg training.")
