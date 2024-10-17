import argparse
from omegaconf import OmegaConf
import os
import torch
from trl import SFTTrainer
from loguru import logger
from transformers import TrainingArguments
from worker import Worker
from peft import set_peft_model_state_dict

class Voter(Worker):
    def __init__(self, args, verbose: bool = False):
        super().__init__(args, verbose)
        self.voter_eval_dataset = self.get_dataset(self.data_path, is_train=False)

    def evaluate(self, comm_round_idx):
        comm_id = int(comm_round_idx)
        if comm_id != -1:
            global_rece_dir = self._get_output_dir_under_workspace(str(comm_round_idx), "aggregate_receiving")
            current_working_directory = os.getcwd()
            logger.info(f"Current working directory is：{current_working_directory}")

            model_path = os.path.join(global_rece_dir, "pytorch_aggregated_model_lora.bin")
            if os.path.exists(model_path):
                global_adapter_model = torch.load(model_path)
                set_peft_model_state_dict(self.model, global_adapter_model)
            else:
                logger.warning(f"Model path does not exist: {model_path}")
                raise FileNotFoundError(f"Model path does not exist: {model_path}")

        eval_args = TrainingArguments(do_train=False, do_eval=True, output_dir=self.out_put_root)

        trainer = SFTTrainer(
            model=self.model,
            args=eval_args,
            eval_dataset=self.voter_eval_dataset,
            dataset_text_field="text",
            max_seq_length=self.block_size,
            tokenizer=self.tokenizer,
        )

        logger.info("Starting global adapter model evaluation...")

        try:
            eval_result = trainer.evaluate()
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

        eval_loss = round(eval_result.get('eval_loss', float('inf')), 6)

        output_result_dir = self._get_output_dir_under_workspace(str(comm_round_idx), "global_eval_res")
        os.makedirs(output_result_dir, exist_ok=True)
        loss_file_path = os.path.join(output_result_dir, "eval_loss.txt")

        try:
            with open(loss_file_path, "w") as loss_file:
                loss_file.write(f"Eval Loss: {eval_loss}\n")
            logger.info(f"Evaluation result (loss) saved to {loss_file_path}")
        except Exception as e:
            logger.error(f"Failed to save evaluation result: {str(e)}")
            raise

        return eval_loss

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
    voter = Voter(args, verbose=True)
    voter.evaluate(args.comm_round_idx)
    logger.info("Local evaluation completed.")
