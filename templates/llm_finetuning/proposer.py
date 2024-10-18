import os
import torch
import argparse
from omegaconf import OmegaConf
from trl import SFTTrainer
from loguru import logger
from transformers import TrainingArguments
from worker import Worker
from peft import set_peft_model_state_dict, get_peft_model_state_dict

class Proposer(Worker):
    def __init__(self, args, verbose: bool = False):
        super().__init__(args, verbose)
        self.proposer_train_dataset = self.get_dataset(self.data_path)

    def build_local_trainer(self, tokenizer, local_micro_batch_size: int, gradient_accumulation_steps: int,
                            local_num_epochs: int, local_learning_rate: float, logging_steps: int,
                            optim: str, lr_scheduler_type: str, warmup_steps: int, weight_decay: float,
                            save_total_limit: int, block_size: int, gradient_checkpointing: bool,
                            group_by_length: bool, ddp: bool):
        bf16_supported = torch.cuda.is_bf16_supported()
        self.train_args = TrainingArguments(
            do_train=True,
            do_eval=True,
            output_dir=self.local_output_dir,
            dataloader_drop_last=False,
            save_strategy="steps",
            logging_strategy="steps",
            num_train_epochs=local_num_epochs,
            save_steps=self.proposer_model_save_steps,
            logging_steps=logging_steps,
            per_device_train_batch_size=local_micro_batch_size,
            per_device_eval_batch_size=local_micro_batch_size * 2,
            optim=optim,
            learning_rate=local_learning_rate,
            lr_scheduler_type=lr_scheduler_type,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            weight_decay=weight_decay,
            save_total_limit=save_total_limit,
            bf16=bf16_supported,
            fp16=not bf16_supported,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
        )

        self.local_trainer = SFTTrainer(
            model=self.model,
            args=self.train_args,
            train_dataset=self.proposer_train_dataset,
            dataset_text_field="text",
            max_seq_length=block_size,
            tokenizer=tokenizer,
        )

    def initiate_local_training(self, global_adapter_model=None):
        if global_adapter_model is not None:
            set_peft_model_state_dict(self.local_trainer.model, global_adapter_model)
        self.model.config.use_cache = False
        self.adapter_state_dict = {
            name: param.detach() for name, param in self.model.named_parameters() if "default" in name
        }
        self.model.state_dict = (
            lambda instance, *_, **__: get_peft_model_state_dict(
                instance, self.adapter_state_dict
            )
        ).__get__(self.model, type(self.model))

    def train(self, comm_round_idx):
        global_rece_dir = self._get_output_dir_under_workspace(str(comm_round_idx), "global_receiving")
        os.makedirs(global_rece_dir, exist_ok=True)
        model_path = os.path.join(global_rece_dir, "pytorch_global_model_lora.bin")
        if int(comm_round_idx) > 0:
            if os.path.exists(model_path):
                global_adapter_model = torch.load(model_path)
                set_peft_model_state_dict(self.model, global_adapter_model)
            else:
                logger.warning(f"Model path does not exist: {model_path}")
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
        else:
            global_adapter_model = None

        self.build_local_trainer(
            tokenizer=self.tokenizer,
            local_micro_batch_size=self.proposer_train_micro_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            local_num_epochs=self.proposer_num_epochs,
            local_learning_rate=self.proposer_learning_rate,
            logging_steps=self.proposer_train_logging_steps,
            optim=self.proposer_train_optimizer_name,
            lr_scheduler_type=self.proposer_train_lr_scheduler_type,
            warmup_steps=self.proposer_train_warmup_steps,
            weight_decay=self.proposer_train_weight_decay,
            save_total_limit=self.save_total_limit,
            block_size=self.block_size,
            gradient_checkpointing=self.proposer_train_gradient_checkpointing,
            group_by_length=self.proposer_train_group_by_length,
            ddp=self.ddp,
        )

        logger.info("Initiating the local training...")
        self.initiate_local_training(global_adapter_model)

        logger.info("Local training starts...")
        self.local_trainer.train()

        logger.info("Terminating the local training and saving the model...")
        self.terminate_local_training(comm_round_idx)

    def terminate_local_training(self, comm_round_idx: int) -> torch.nn.Module:
        single_output_dir = self._get_output_dir_under_workspace(str(comm_round_idx), "local_output")
        os.makedirs(single_output_dir, exist_ok=True)

        try:
            logger.info("Saving the updated model weights...")
            model_save_path = os.path.join(single_output_dir, "pytorch_local_model_lora.bin")
            if not os.access(single_output_dir, os.W_OK):
                raise PermissionError(f"No write permission for directory: {single_output_dir}")
            torch.save(self.model.state_dict(), model_save_path)

            current_working_directory = os.getcwd()

            logger.info(f"Current working directory is：{current_working_directory}")
            logger.info(f"Model saved at {model_save_path}")

        except Exception as e:
            logger.error(f"Failed to save the model weights: {e}")
            raise IOError(f"Failed to save the model weights: {e}")

def add_args():
    parser = argparse.ArgumentParser(description="FLockit")
    parser.add_argument("--yaml_config_file", "--conf", help="Templates configuration file (.yaml)", type=str, default="")
    parser.add_argument("--comm_round_idx", "--comm", help="Communication round index", type=str, default="")
    args, unknown = parser.parse_known_args()
    return args

def load_yaml_config(yaml_file):
    if os.path.exists(yaml_file):
        yaml_config = OmegaConf.load(yaml_file)
    else:
        yaml_config = OmegaConf.create()
    return yaml_config

def load_arguments():
    cmd_args = add_args()
    yaml_args = load_yaml_config(cmd_args.yaml_config_file)
    cmd_args_dict = {k: v for k, v in vars(cmd_args).items() if v is not None}
    cmd_args_omega = OmegaConf.create(cmd_args_dict)
    final_args = OmegaConf.merge(yaml_args, cmd_args_omega)
    return final_args

if __name__ == '__main__':
    args = load_arguments()
    proposer = Proposer(args, verbose=True)
    proposer.train(args.comm_round_idx)
    logger.info("Local training completed.")
