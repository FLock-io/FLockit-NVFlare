import os
import sys
from loguru import logger
from transformers import AutoTokenizer
from datasets import load_dataset
from templates.llm_finetuning.prompters.prompter_hub import get_prompter
from templates.llm_finetuning.model_loader import ModelLoader
from templates.utils.device import DeviceManager

class Worker:
    def __init__(self, args, verbose: bool = False):
        self.args = args
        log_level = "DEBUG" if verbose else "INFO"
        logger.remove()
        logger.add(sys.stdout, level=log_level)

        data_args = args.data_args
        self.data_path = data_args.data_path

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

        eval_args = args.evaluation_args
        self.voter_val_set_size = eval_args.voter_val_set_size

        tracking_args = args.tracking_args
        self.out_put_root = tracking_args.out_put_root
        self.local_output_dir = self._get_local_output_dir("local_trainer_saved", "local_output")
        self.finetune_adapter_checkpoint_save_dir = tracking_args.finetune_adapter_checkpoint_save_dir
        self.proposer_train_gradient_checkpointing = tracking_args.proposer_train_gradient_checkpointing
        self.proposer_train_logging_steps = tracking_args.proposer_train_logging_steps
        self.proposer_model_save_steps = tracking_args.proposer_model_save_steps
        self.save_total_limit = tracking_args.save_total_limit

        self.prompter = get_prompter()
        self.gradient_accumulation_steps = self.proposer_train_batch_size // self.proposer_train_micro_batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.foundation_model_name,
                                                       trust_remote_code=False,
                                                       use_fast=True)

        pad_token_id = 0
        self.tokenizer.pad_token_id = pad_token_id
        self.tokenizer.pad_token = self.tokenizer.convert_ids_to_tokens(pad_token_id)
        self.tokenizer.padding_side = "right"

        self.device_manager = DeviceManager()
        device_map, self.ddp, gradient_accumulation_steps = self.device_manager.setup_device(self.foundation_model_name,
                                                                                             self.proposer_train_batch_size,
                                                                                             self.proposer_train_micro_batch_size,
                                                                                             device_map= train_args.device_map if hasattr(train_args, "device_map") else None)

        model_loader = ModelLoader(model_name=self.foundation_model_name,
                                   finetune_adapter=finetune_adapter,
                                   device_map=device_map,
                                   lora_r=lora_r,
                                   lora_alpha=lora_alpha,
                                   lora_dropout=lora_dropout,
                                   lora_target_modules=lora_target_modules,
                                   verbose=verbose,
                                   ddp=self.ddp)
        self.model, self.lora_config = model_loader.load_model()

    def get_dataset(self, dataset_path: str, is_train: bool = True):
        logger.info("\nPreparing the local training and validation dataset")

        try:
            local_data = load_dataset("json", data_files=dataset_path)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

        def generate_and_tokenize_prompt(data_point):
            instruction = data_point.get("instruction", "")
            context = data_point.get("context", "")
            response = data_point.get("response", "")

            if not instruction and not context and not response:
                logger.warning(f"Empty data point found: {data_point}")
                return None

            full_prompt = self.prompter.generate_prompt(instruction, context, response)
            return {"text": full_prompt}

        if self.voter_val_set_size > 0:
            if self.voter_val_set_size >= len(local_data["train"]):
                logger.warning(
                    "Validation set size is greater than or equal to the dataset size. Adjusting to a smaller value.")
                self.voter_val_set_size = int(len(local_data["train"]) * 0.1)

        split_params = {
            "test_size": self.voter_val_set_size,
            "shuffle": True
        }

        local_train_val = local_data["train"].train_test_split(**split_params)

        if is_train:
            split_dataset = (
                local_train_val["train"]
                .shuffle()
                .map(generate_and_tokenize_prompt, num_proc=1)
            )
        else:
            split_dataset = (
                local_train_val["test"]
                .shuffle()
                .map(generate_and_tokenize_prompt, num_proc=1)
            )

        return split_dataset

    def _get_local_output_dir(self, *subdirs) -> str:
        return os.path.join(self.out_put_root, *subdirs)

    def _get_output_dir_under_workspace(self, *subdirs) -> str:
        return os.path.join(self.args.nvflare_args.nvflare_output_dir, self.out_put_root, *subdirs)

    def train(self):
        pass

    def evaluate(self):
        pass