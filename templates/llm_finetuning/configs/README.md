<div align="center">

# Configuration YAML File Parameters Explanation

</div>

|       Category       |               Parameter               |             Default             |     Type     |                             Options                             |                 Description                 |
|:--------------------:|:-------------------------------------:|:-------------------------------:|:------------:|:--------------------------------------------------------------:|:-------------------------------------------:|
|    common_args       |           project_name                |     "FLockLLM_finetune"         |     str      |                               -                                |             Overall project name            |
|                      |           template_name               |        "llm_finetuning"         |     str      |                               -                                |             Template name                   |
|     data_args        |             data_path                 |        "/dataset.json"          |     str      |                               -                                |           Path to the dataset file           |
|    model_args        |      foundation_model_name            | "meta-llama/Meta-Llama-3.1-8B"  |     str      | "meta-llama/Meta-Llama-3.1-8B", "Qwen/Qwen2-7B", "google/gemma-2-2b", "mistralai/Mistral-7B-v0.1" |        Name of the foundation model         |
|                      |      foundation_model_source          |         "huggingface"           |     str      |                 "huggingface", "flock_s3"                      |           Source of the foundation model    |
|                      |          finetune_adapter             |            "qlora"              |     str      |                      "lora", "qlora"                           |        Type of adapter used for fine-tuning |
|                      |            lora_dropout               |             0.05                |    float     |                               -                                |           Dropout rate for LoRA             |
|                      |         lora_target_modules           |               []                |  list(str)   |        "q_proj", "k_proj", "v_proj", "o_proj"                  |           Target modules for LoRA           |
|    train_args        |     proposer_train_batch_size         |              32                 |     int      |                               -                                |            Batch size for training          |
|                      | proposer_train_micro_batch_size       |               8                 |     int      |                               -                                |           Micro-batch size for training     |
|                      |        proposer_num_epochs            |               1                 |     int      |                               -                                |            Number of epochs for training    |
|                      |      proposer_learning_rate           |            0.0003               |    float     |                               -                                |             Learning rate for training      |
|                      |      proposer_val_set_size            |              16                 |     int      |                               -                                |             Validation set size             |
|                      |         proposer_save_steps           |               3                 |     int      |                               -                                |          Number of steps between saves      |
|                      |   proposer_train_group_by_length      |             false               |    bool      |                      true, false                               |         Whether to group data by length     |
|                      |   proposer_train_optimizer_name       |      "paged_adamw_8bit"         |     str      |            "adamw", "paged_adamw_8bit"                         |          Optimizer used for training        |
|                      | proposer_train_lr_scheduler_type      |           "constant"            |     str      |            "constant", "linear", "cosine"                      |          Learning rate scheduler type       |
|                      |    proposer_train_warmup_steps        |               1                 |     int      |                               -                                |           Warmup steps for learning rate    |
|                      |    proposer_train_weight_decay        |             0.05                |    float     |                               -                                |            Weight decay for training        |
|                      |            block_size                 |               8                 |     int      |                               -                                |          Block size for sequence training   |
|                      |            cutoff_len                 |             1024                |     int      |                               -                                |            Cutoff length for sequences      |
|                      |     federated_optimizer_name          |           "fedavg"              |     str      |                 "fedavg", "fedsgd"                             |           Federated optimizer name          |
|                      |             device_map                |            "cuda:0"             |     str      |                               -                                |           Device mapping for computation    |
|  evaluation_args     |        voter_val_set_size             |               5                 |     int      |                               -                                |         Validation set size for evaluation  |
|   tracking_args      |           out_put_root                |           "output"              |     str      |                               -                                |            Root directory for output        |
|                      | finetune_adapter_checkpoint_save_dir  | "checkpoints/llama-3.1-8b"      |     str      |                               -                                |        Directory for saving checkpoints     |
|                      | proposer_train_gradient_checkpointing |             true                |    bool      |                      true, false                               |         Whether to use gradient checkpointing |
|                      |     proposer_train_logging_steps      |              10                 |     int      |                               -                                |          Logging steps during training      |
|                      |      proposer_model_save_steps        |              40                 |     int      |                               -                                |          Model save steps                   |
|                      |           save_total_limit            |               3                 |     int      |                               -                                |        Limit for total saved checkpoints    |
|    nvflare_args      |          nvflare_workspace            | "/tmp/nvflare/jobs/workdir"     |     str      |                               -                                |         NVFlare workspace directory         |
|                      |        nvflare_executor_name          |           "flockit"             |     str      |                               -                                |          NVFlare executor name              |
|                      |          nvflare_output_dir           |        "/tmp/nvflare/"          |     str      |                               -                                |          NVFlare output directory           |

# Parameter Descriptions

- **common_args**
    - `project_name`: Overall project name.
    - `template_name`: Template name for the project.

- **data_args**
    - `data_path`: Path to the dataset file.

- **model_args**
    - `foundation_model_name`: Name of the foundation model to use.
    - `foundation_model_source`: Source of the foundation model.
    - `finetune_adapter`: Type of adapter used for fine-tuning.
    - `lora_dropout`: Dropout rate for LoRA.
    - `lora_target_modules`: Target modules for LoRA.

- **train_args**
    - `proposer_train_batch_size`: Batch size for training.
    - `proposer_train_micro_batch_size`: Micro-batch size for training.
    - `proposer_num_epochs`: Number of epochs for training.
    - `proposer_learning_rate`: Learning rate for training.
    - `proposer_val_set_size`: Validation set size.
    - `proposer_save_steps`: Number of steps between saves.
    - `proposer_train_group_by_length`: Whether to group data by length.
    - `proposer_train_optimizer_name`: Optimizer used for training.
    - `proposer_train_lr_scheduler_type`: Learning rate scheduler type.
    - `proposer_train_warmup_steps`: Warmup steps for learning rate.
    - `proposer_train_weight_decay`: Weight decay for training.
    - `block_size`: Block size for sequence training.
    - `cutoff_len`: Cutoff length for sequences.
    - `federated_optimizer_name`: Federated optimizer name.
    - `device_map`: Device mapping for computation.

- **evaluation_args**
    - `voter_val_set_size`: Validation set size for evaluation.

- **tracking_args**
    - `out_put_root`: Root directory for output.
    - `finetune_adapter_checkpoint_save_dir`: Directory for saving checkpoints.
    - `proposer_train_gradient_checkpointing`: Whether to use gradient checkpointing.
    - `proposer_train_logging_steps`: Logging steps during training.
    - `proposer_model_save_steps`: Model save steps.
    - `save_total_limit`: Limit for total saved checkpoints.

- **nvflare_args**
    - `nvflare_workspace`: NVFlare workspace directory.
    - `nvflare_executor_name`: NVFlare executor name.
    - `nvflare_output_dir`: NVFlare output directory.
