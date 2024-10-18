import os
import io
import sys

import torch
from loguru import logger
from ..utils.device import DeviceManager

from .model_loader import ModelLoader
from .nvflare_utils.fedavg_proposer_job import ProposerFedAvgJob
from .nvflare_utils.fedavg_voter_job import VoterFedAvgJob
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

class FLockModelLLMFinetune():
    """
    This class is responsible for fine-tuning the language model.
    """
    def __init__(self, args, verbose: bool = False):
        super().__init__()

        self.args = args

        # Configure logger based on verbose level
        log_level = "DEBUG" if verbose else "INFO"
        logger.remove()  # Remove default logger
        logger.add(sys.stdout, level=log_level)

        # Parameters Configuration
        data_args = args.data_args
        self.data_path = data_args.data_path

        # Model and Adapter Configuration
        model_args = args.model_args
        self.foundation_model_name = model_args.foundation_model_name
        finetune_adapter = model_args.finetune_adapter
        if finetune_adapter.lower() == "lora":
            lora_r = 16
            lora_alpha = 16
        elif finetune_adapter.lower() == "qlora":
            lora_r = 4
            lora_alpha = 4
        else:
            raise ValueError(f"Adapter type {finetune_adapter} not recognized, only support lora or qlora.")
        lora_dropout = model_args.lora_dropout
        lora_target_modules = model_args.lora_target_modules

        # Training Configuration
        train_args = args.train_args
        self.proposer_train_batch_size = train_args.proposer_train_batch_size
        self.proposer_train_micro_batch_size = train_args.proposer_train_micro_batch_size
        self.proposer_num_epochs = train_args.proposer_num_epochs
        self.proposer_learning_rate = train_args.proposer_learning_rate
        self.proposer_val_set_size = train_args.proposer_val_set_size
        self.proposer_save_steps = train_args.proposer_save_steps
        self.proposer_train_group_by_length = train_args.proposer_train_group_by_length
        self.proposer_train_optimizer_name = train_args.proposer_train_optimizer_name
        self.proposer_train_lr_scheduler_type = train_args.proposer_train_lr_scheduler_type
        self.proposer_train_warmup_steps = train_args.proposer_train_warmup_steps
        self.proposer_train_weight_decay = train_args.proposer_train_weight_decay
        self.block_size = train_args.block_size
        self.federated_optimizer_name = train_args.federated_optimizer_name
        self.cutoff_len = train_args.cutoff_len

        # Evaluation Configuration
        eval_args = args.evaluation_args
        self.voter_val_set_size = eval_args.voter_val_set_size

        # Tracking Configuration
        tracking_args = args.tracking_args
        self.out_put_root = tracking_args.out_put_root
        self.finetune_adapter_checkpoint_save_dir = tracking_args.finetune_adapter_checkpoint_save_dir
        self.proposer_train_gradient_checkpointing = tracking_args.proposer_train_gradient_checkpointing
        self.proposer_train_logging_steps = tracking_args.proposer_train_logging_steps
        self.proposer_model_save_steps = tracking_args.proposer_model_save_steps
        self.save_total_limit = tracking_args.save_total_limit

        # Log all parameters
        all_params_str = (
            f"Data Path: {self.data_path}\n"
            f"Foundation Model Name: {self.foundation_model_name}\n"
            f"Finetune Adapter: {finetune_adapter}\n"
            f"LoRA R: {lora_r}\n"
            f"LoRA Alpha: {lora_alpha}\n"
            f"LoRA Dropout: {lora_dropout}\n"
            f"LoRA Target Modules: {lora_target_modules}\n"
            f"Proposer Train Batch Size: {self.proposer_train_batch_size}\n"
            f"Proposer Train Micro Batch Size: {self.proposer_train_micro_batch_size}\n"
            f"Proposer Num Epochs: {self.proposer_num_epochs}\n"
            f"Proposer Learning Rate: {self.proposer_learning_rate}\n"
            f"Proposer Val Set Size: {self.proposer_val_set_size}\n"
            f"Proposer Save Steps: {self.proposer_save_steps}\n"
            f"Proposer Train Group By Length: {self.proposer_train_group_by_length}\n"
            f"Proposer Train Optimizer Name: {self.proposer_train_optimizer_name}\n"
            f"Proposer Train LR Scheduler Type: {self.proposer_train_lr_scheduler_type}\n"
            f"Proposer Train Warmup Steps: {self.proposer_train_warmup_steps}\n"
            f"Proposer Train Weight Decay: {self.proposer_train_weight_decay}\n"
            f"Block Size: {self.block_size}\n"
            f"Federated Optimizer Name: {self.federated_optimizer_name}\n"
            f"Cutoff Length: {self.cutoff_len}\n"
            f"Voter Val Set Size: {self.voter_val_set_size}\n"
            f"Overall Output Root Path: {self.out_put_root}\n"
            f"Finetune Adapter Checkpoint Save Dir: {self.finetune_adapter_checkpoint_save_dir}\n"
            f"Proposer Train Gradient Checkpointing: {self.proposer_train_gradient_checkpointing}\n"
            f"Proposer Train Logging Steps: {self.proposer_train_logging_steps}\n"
            f"Proposer Model Save Steps: {self.proposer_model_save_steps}\n"
            f"Save Total Limit: {self.save_total_limit}\n"
        )

        logger.info(f"All Parameters:\n{all_params_str}")

        # Device Manager Setup
        self.device_manager = DeviceManager()
        device_map, self.ddp, gradient_accumulation_steps = self.device_manager.setup_device(self.foundation_model_name, self.proposer_train_batch_size, self.proposer_train_micro_batch_size)

        # Model Preparation
        model_loader = ModelLoader(model_name=self.foundation_model_name,
                                   finetune_adapter=finetune_adapter,
                                   device_map=device_map,
                                   lora_r=lora_r,
                                   lora_alpha=lora_alpha,
                                   lora_dropout=lora_dropout,
                                   lora_target_modules=lora_target_modules,
                                   verbose=verbose,
                                   ddp=self.ddp)
        _, self.lora_config = model_loader.load_model()

        self.nvflare_key_metric = "accuracy"

        self.nvflare_job_script_proposer = "templates/llm_finetuning/proposer.py"
        self.nvflare_job_script_voter = "templates/llm_finetuning/voter.py"

        self.nvflare_job_script_args = f"--conf /app/templates/llm_finetuning/configs/conf.yaml"
        # self.nvflare_job_script_args = f"--conf /lab/github/FLockit/templates/llm_finetuning/configs/conf.yaml"
        self.nvflare_job_launch_command = f"python -u"
        self.nvflare_job_launch_process = False
        self.nvflare_job_port = "7777"

        self.comm_round_idx = 0

    def init_dataset(self, dataset_path: str):
        return None, None

    def _get_output_dir(self, *subdirs) -> str:
        """
        Helper method to construct output directories.
        """
        return os.path.join(self.out_put_root, *subdirs)

    def _get_output_dir_under_workspace(self, *subdirs) -> str:
        """
        Helper method to construct output directories.
        """
        return os.path.join(self.args.nvflare_args.nvflare_output_dir, self.out_put_root, *subdirs)

    def train(self, parameters=None) -> bytes:
        """
        Handles the training process of the model, including loading parameters, training locally,
        and preparing the trained model parameters to be sent back.
        """

        nvflare_job_script_args = "{} --comm {}".format(self.nvflare_job_script_args, str(self.comm_round_idx))

        if parameters:
            logger.info("Loading latest global adapter model parameters into the local model...")
            buffer_model = io.BytesIO(parameters)
            loaded_model_state_dict = torch.load(buffer_model)
            global_rece_dir = self._get_output_dir_under_workspace(str(self.comm_round_idx), "global_receiving")
            os.makedirs(global_rece_dir, exist_ok=True)
            try:
                torch.save(loaded_model_state_dict, os.path.join(global_rece_dir, "pytorch_global_model_lora.bin"))
            except Exception as e:
                raise IOError(f"Failed to save the model weights: {e}")
            logger.info(f"Model saved at {os.path.join(global_rece_dir, 'pytorch_global_model_lora.bin')}")

        proposer_job = ProposerFedAvgJob(
            name="FLock_Alliance_Proposer",
            key_metric=self.nvflare_key_metric,
        )

        logger.debug(f"nvflare_job_launch_command {self.nvflare_job_launch_command}")
        proposer_executor = ScriptRunner(
            script=self.nvflare_job_script_proposer,
            script_args=nvflare_job_script_args,
            launch_external_process=self.nvflare_job_launch_process,
            command=self.nvflare_job_launch_command.replace("{PORT}", self.nvflare_job_port),
            framework=FrameworkType.PYTORCH,
        )

        proposer_job.to(proposer_executor, self.args.nvflare_args.nvflare_executor_name)

        logger.info("Start simulating the proposer job...")

        proposer_job.simulator_run(self.args.nvflare_args.nvflare_workspace, gpu="0")

        logger.info("Wrapping up the local model parameters and saving the current model state...")

        single_output_dir = self._get_output_dir_under_workspace(str(self.comm_round_idx), "local_output")

        model_save_path = os.path.join(single_output_dir, "pytorch_local_model_lora.bin")
        updated_local_model = torch.load(model_save_path)

        buffer = io.BytesIO()
        torch.save(updated_local_model, buffer)

        self.comm_round_idx += 1

        return buffer.getvalue()

    def evaluate(self, parameters: bytes = None) -> float:
        """
        Evaluate the model using the provided parameters.

        Args:
            parameters (bytes): Serialized model parameters to be loaded before evaluation.

        Returns:
            float: The evaluation loss of the model.
        """

        logger.info("Evaluating the model...")

        if parameters:
            logger.info("Loading latest global adapter model parameters into the local model...")
            buffer_model = io.BytesIO(parameters)
            loaded_model_state_dict = torch.load(buffer_model)
            global_rece_dir = self._get_output_dir_under_workspace(str(self.comm_round_idx), "aggregate_receiving")
            os.makedirs(global_rece_dir, exist_ok=True)
            try:
                torch.save(loaded_model_state_dict, os.path.join(global_rece_dir, "pytorch_aggregated_model_lora.bin"))
            except Exception as e:
                raise IOError(f"Failed to save the model weights: {e}")
            logger.info(f"Model saved at {os.path.join(global_rece_dir, 'pytorch_aggregated_model_lora.bin')}")
            nvflare_job_script_args = "{} --comm {}".format(self.nvflare_job_script_args, str(self.comm_round_idx))
        else:
            # The first round do not have global model to evaluate
            nvflare_job_script_args = "{} --comm {}".format(self.nvflare_job_script_args, str(-1))

        voter_job = VoterFedAvgJob(
            name="FLock_Alliance_Proposer",
            key_metric=self.nvflare_key_metric,
        )

        logger.debug(f"nvflare_job_launch_command {self.nvflare_job_launch_command}")
        voter_executor = ScriptRunner(
            script=self.nvflare_job_script_voter,
            script_args=nvflare_job_script_args,
            launch_external_process=self.nvflare_job_launch_process,
            command=self.nvflare_job_launch_command.replace("{PORT}", self.nvflare_job_port),
            framework=FrameworkType.PYTORCH,
        )
        voter_job.to(voter_executor, self.args.nvflare_args.nvflare_executor_name)

        logger.info("Start simulating the voter job...")

        voter_job.simulator_run(self.args.nvflare_args.nvflare_workspace, gpu="0")

        if parameters:
            loss_file_path = os.path.join(self._get_output_dir_under_workspace(str(self.comm_round_idx), "global_eval_res"), "eval_loss.txt")
        else:
            loss_file_path = os.path.join(
                self._get_output_dir_under_workspace(str(-1), "global_eval_res"), "eval_loss.txt")

        # Check if the file exists
        if os.path.exists(loss_file_path):
            with open(loss_file_path, "r") as file:
                eval_loss_content = float(file.read().split(":")[1].strip())
        else:
            raise FileNotFoundError(f"Loss file not found at {loss_file_path}")

        logger.info(f"Global adapter model loss: {eval_loss_content}")

        self.comm_round_idx += 1

        return eval_loss_content

    def aggregate(self, parameters_list: list[bytes]) -> bytes:
        """
        Aggregate the model parameters from multiple local models.

        Args:
            parameters_list (list[bytes]): A list of serialized model parameters from multiple local models.

        Returns:
            bytes: The serialized aggregated model parameters.
        """

        # Handle DDP alignment problem: relocate the model weights to unified device
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # Load parameters from the list, and skip those that fail to load or are incomplete

        logger.info("Aggregating model parameters...")
        valid_parameters_list = []
        for parameters in parameters_list:
            try:
                buffer_model = io.BytesIO(parameters)
                model_state_dict = torch.load(buffer_model)
                valid_parameters_list.append(model_state_dict)
            except Exception as e:
                logger.warning(f"Failed to load model parameters: {str(e)}")
                continue

        # Ensure that there are valid parameters to aggregate
        if not valid_parameters_list:
            raise ValueError("No valid model parameters to aggregate.")

        logger.info("Aggregating all valid local model parameters...")

        if self.federated_optimizer_name.lower() == "fedavg":
            averaged_params_template = valid_parameters_list[0]
            for key in averaged_params_template.keys():
                for i in range(1, len(valid_parameters_list)):
                    averaged_params_template[key] += valid_parameters_list[i][key]
                averaged_params_template[key] /= len(valid_parameters_list)
        else:
            raise ValueError(f"Federated optimizer {self.federated_optimizer_name} not recognized, only support fedavg.")

        # Ensure output directory exists
        target_path = os.path.join(self.out_put_root, str(self.comm_round_idx))
        os.makedirs(target_path, exist_ok=True)

        # Save the averaged parameters to the file
        global_model_output_path = os.path.join(target_path, "pytorch_local_model_lora.bin")
        logger.info(f"Saving the global adapter model parameters to {global_model_output_path}...")
        torch.save(averaged_params_template, global_model_output_path)
        self.lora_config.save_pretrained(self.out_put_root)

        logger.info("Wrapping up the global adapter model parameters and sending to all Proposers...")

        # Create a buffer and save the averaged parameters
        buffer = io.BytesIO()
        torch.save(averaged_params_template, buffer)

        return buffer.getvalue()