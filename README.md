# FLockit-NVFlare

This repository provides a FLock kit template integrated with the NVFlare framework for decentralized LLM fine-tuning tasks. It serves as a template for task creators to customize the code and implement their own business logic.

By integrating NVFlare as the backend, you can:

1. **Manage your computational units across different processes**: Efficiently control and coordinate computational resources.
2. **On-demand LLM loading**: Pre-trained LLMs are loaded into GPU memory only when a task arrives, optimizing resource usage.
3. **Manage your own federation for hierarchical and hybrid federated learning**: Use NVFlare to manage a centralized federation with your trusted collaborators to leverage their data and improve your local model's performance. Simultaneously, utilize your competitive local model with the FLock platform to achieve decentralized federated learning and gain more rewards.

## Use Cases

### Peer-to-Peer (P2P) Blockchain-Enabled Federated Learning for Decentralized AI Marketplaces

FLockit and NVFlare enable decentralized AI marketplaces where participants can collaboratively fine-tune LLMs. Blockchain ensures transparency and secure transactions, while federated learning allows entities to contribute data and computing power without exposing raw data. This marketplace fosters decentralized AI model improvements through collective contributions. For example, in decentralized finance (DeFi), platforms can collaborate on fraud detection models while maintaining privacy and ensuring trust through blockchain validation.

### B2B Blockchain-Based Federated Learning for Supply Chain Optimization

In supply chain optimization, FLockit and NVFlare create a decentralized, blockchain-based platform for businesses to fine-tune models collaboratively. Companies share encrypted model updates, improving operations such as demand forecasting and inventory management. Blockchain provides transparency and accountability, while federated learning protects proprietary data, ensuring efficient collaboration without exposing sensitive information. This integrated solution optimizes supply chain operations and enhances trust, innovation, and predictive capabilities.

### Healthcare and Medical Research (Two-Tier Federated Learning)

FLockit-NVFlare’s two-tier architecture allows hospitals to collaborate securely at different levels. **First Tier:** Institutions can engage in decentralized federated learning across regions, using blockchain for trust and transparency. Hospitals collaborate on fine-tuning LLMs using local patient data, ensuring privacy through decentralized aggregation. **Second Tier:** Within trusted networks, hospitals use centralized aggregation for more efficient internal collaboration, with model updates shared across campuses. This two-tier structure ensures secure, efficient collaboration for model improvement, combining decentralized and centralized learning.

### FLock Platform in Glucose Prediction for Diabetes Patients

We developed the Multi-Continental Glucose Prediction (MCGP) framework to overcome the critical challenges of data privacy and sharing in healthcare using blockchain-enabled federated learning technology. MCGP enables healthcare institutions across continents to collaboratively train a global predictive model without directly sharing sensitive patient data. By combining privacy-preserving federated learning with a blockchain-based incentive mechanism, MCGP ensures data security and privacy, promotes honest participation, and protects against malicious interference. Through real-world trials conducted in multiple countries, Toronto, Canada; Shanghai and Shandong, China; and Newcastle and London, UK, MCGP has demonstrated predictive accuracy comparable to centralized models, excelling in personalization, generalization, and resilience against malicious behavior. As an innovative approach to global healthcare data collaboration, MCGP enhances both the accuracy and reliability of glucose prediction, offering a scalable, secure, and privacy-respecting solution for international healthcare partnerships. For further details, please refer to our research paper: [Multi-Continental Healthcare Modelling Using Blockchain-Enabled Federated Learning](https://arxiv.org/abs/2410.17933)

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