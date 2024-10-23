# FLockit-NVFlare

This repository provides a FLock kit template integrated with the NVFlare framework for decentralized LLM fine-tuning tasks. It serves as a template for task creators to customize the code and implement their own business logic.

By integrating NVFlare as the backend, you can:

1. **Manage your computational units across different processes**: Efficiently control and coordinate computational resources.
2. **On-demand LLM loading**: Pre-trained LLMs are loaded into GPU memory only when a task arrives, optimizing resource usage.
3. **Manage your own federation for hierarchical and hybrid federated learning**: Use NVFlare to manage a centralized federation with your trusted collaborators to leverage their data and improve your local model's performance. Simultaneously, utilize your competitive local model with the FLock platform to achieve decentralized federated learning and gain more rewards.

## Use Cases

### Healthcare and Medical Research (Two-Tier Federated Learning)

In the medical field, privacy, security, and collaboration are paramount, especially when dealing with sensitive patient data. FLockit-NVFlare’s two-tier federated learning architecture enables hospitals and medical institutions to securely collaborate at multiple levels.

**First Tier: Decentralized Federated Learning Across Institutions (Blockchain-Based)**

At the highest level, hospitals or medical institutions can collaborate across regions or countries without relying on a central authority. This decentralized federated learning is secured through blockchain technology, ensuring trust, transparency, and immutability. Each hospital maintains its own data locally and shares only model updates, which are aggregated in a decentralized manner via the FLock platform.

For example, several hospitals from different countries can collaborate to fine-tune a large language model (LLM) for diagnostic support, using local patient data. The blockchain ensures that the collaboration process is transparent and auditable, without the need for a central party. This prevents single points of failure and reinforces the privacy of sensitive data across institutions.

**Second Tier: Internal Federated Learning (Centralized Aggregation)**

Within each institution or between closely collaborating hospitals that have a high level of trust, a traditional centralized federated learning setup can be used. In this layer, a central server aggregates model updates from multiple departments, labs, or clinics within the same organization, or between trusted partner hospitals. This allows for more efficient training and coordination while still keeping the raw data local.

For instance, within a hospital network that spans multiple campuses, the institutions may have established trust relationships that enable them to fine-tune models jointly using internal federated learning. The central server coordinates model updates from different campuses without the need for blockchain, allowing for faster iteration while maintaining privacy within the network. These updates can then be shared with the broader decentralized network via the first tier, contributing to the global model improvement.

This two-tier structure ensures both internal efficiency and external security, combining the trust of centralized learning within trusted collaborators and the transparency and security of decentralized learning between institutions.

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