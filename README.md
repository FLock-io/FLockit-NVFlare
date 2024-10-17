# FLockit-NVFlare

This repository provides a FLock kit template integrated with the NVFlare framework for decentralized LLM fine-tuning tasks. It serves as a template for task creators to customize the code and implement their own business logic.

By integrating NVFlare as the backend, you can:

1. **Manage your computational units across different processes**: Efficiently control and coordinate computational resources.
2. **On-demand LLM loading**: Pre-trained LLMs are loaded into GPU memory only when a task arrives, optimizing resource usage.
3. **Manage your own federation for hierarchical and hybrid federated learning**: Use NVFlare to manage a centralized federation with your trusted collaborators to leverage their data and improve your local model's performance. Simultaneously, utilize your competitive local model with the FLock platform to achieve decentralized federated learning and gain more rewards.

## Quick Start

By default, the foundation model for fine-tuning is set to **LLaMa 3.1 8B**. However, you can easily modify the configuration file [`conf.yaml`](templates/llm_finetuning/configs/conf.yaml) based on the parameter descriptions provided within the same file.

Once you have received your AWS credentials from the FLock team, export them into your system:

```bash
export AWS_ACCESS_KEY_ID=<Get this from FLock team>
export AWS_SECRET_ACCESS_KEY=<Get this from FLock team>
```

Then, to submit your logic code, simply run:
```bash
python build_and_upload_S3.py
```

## Customize Your Own Flow

If you are familiar with NVFlare, you can customize your own proposer/voter jobs or their workflows under templates/llm_finetuning/nvflare_utils/.