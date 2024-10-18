import os
from loguru import logger
from templates.llm_finetuning.flock_model_llm_finetune import FLockModelLLMFinetune
from huggingface_hub.hf_api import HfFolder

def llm_finetuning_init(args):
    if hasattr(args.tracking_args, "report_to"):
        raise NotImplementedError("Report logs to cloud service currently not is supported")

    HF_TOKEN = os.getenv("HF_TOKEN")
    if HF_TOKEN:
        HfFolder.save_token(HF_TOKEN)
    else:
        logger.error("HF_TOKEN environment variable is not set.")

    prepare_pretrained_model(args)

    task_model = FLockModelLLMFinetune(args, verbose=True)

    return task_model

def prepare_pretrained_model(args):
    if args.model_args.foundation_model_source == "huggingface":
        args.model_args.foundation_model_path = args.model_args.foundation_model_name
    else:
        raise ValueError(f"Invalid foundation_model_pre_trained_weights_source {args.model_args.foundation_model_source}")
