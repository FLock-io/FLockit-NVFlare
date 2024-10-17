import torch
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from loguru import logger
import bitsandbytes
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

class ModelLoader:
    """
    This class is responsible for loading the model with the specified configuration.
    """
    def __init__(self, model_name, finetune_adapter, device_map=None, lora_name=None, lora_r=16, lora_alpha=16, lora_dropout=0.1,
                 lora_target_modules=None, verbose=False, ddp=False):
        self.model_name = model_name
        self.finetune_adapter = finetune_adapter.lower()
        self.device_map = device_map if device_map is not None else {'': 0}  # Default to single GPU
        self.lora_name = lora_name  # LoRA or QLoRA name parameter
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules if lora_target_modules is not None else []
        self.verbose = verbose
        self.ddp = ddp

    def load_model(self, return_only_lora_config=False):
        """
        Load the model with the specified configuration.
        """
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        if return_only_lora_config:
            lora_config = self._create_lora_config()
            return lora_config

        if self.lora_name:
            model, lora_config = self._load_pretrained_lora_model()
        else:
            if self.finetune_adapter == "lora":
                model, lora_config = self._load_lora_model()
            elif self.finetune_adapter == "qlora":
                model, lora_config = self._load_qlora_model(torch_dtype)
            else:
                raise ValueError(f"Adapter type {self.finetune_adapter} not recognized")

        if self.ddp and torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True

        self._log_model_parameters(model)

        return model, lora_config

    def _create_lora_config(self):
        """
        Create LoRA config without loading the model.
        """
        if not self.lora_target_modules:
            raise ValueError(
                "Please specify 'lora_target_modules' when 'return_only_lora_config' is True to avoid loading the model.")

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        if self.verbose:
            logger.info("Created LoRA config without loading the model.")

        return lora_config

    def _load_pretrained_lora_model(self):
        """
        Load LoRA/QLoRA model from Hugging Face with name.
        """
        model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                     trust_remote_code=False,
                                                     device_map=self.device_map)

        model = PeftModel.from_pretrained(model, self.lora_name)
        lora_config = None  # LoRA config is part of the loaded model

        if self.verbose:
            logger.info(f"Loaded LoRA/QLoRA model from Hugging Face with name: {self.lora_name}")

        return model, lora_config

    def _load_lora_model(self):
        """
        Load LoRA model.
        """
        model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                     load_in_8bit=True,
                                                     trust_remote_code=False,
                                                     device_map=self.device_map)
        if self.verbose:
            logger.info("LoRA: Preparing model for int8 training.")
        model = prepare_model_for_kbit_training(model)
        return self._configure_peft_model(model)

    def _load_qlora_model(self, torch_dtype):
        """
        Load QLoRA model.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )

        config = AutoConfig.from_pretrained(self.model_name)
        config.use_cache = False  # Disable cache
        config.gradient_checkpointing = True

        if self.verbose:
            logger.info("QLoRA: Preparing model with 4-bit quantization and gradient checkpointing.")

        model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                     config=config,
                                                     quantization_config=bnb_config,
                                                     trust_remote_code=False,
                                                     torch_dtype=torch_dtype,
                                                     device_map=self.device_map)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        return self._configure_peft_model(model)

    def _configure_peft_model(self, model):
        """
        Configure model with PEFT using LoRA.
        """
        if len(self.lora_target_modules) == 0:
            self.lora_target_modules = self._find_all_linear_names(model)

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.lora_target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        if self.verbose:
            logger.info("Configuring model with PEFT using LoRA.")

        return get_peft_model(model, lora_config), lora_config

    def _find_all_linear_names(self, model, add_lm_head=True):
        """
        Find all linear names.
        """
        cls = bitsandbytes.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if add_lm_head and "lm_head" not in lora_module_names:
            lora_module_names.add("lm_head")

        return list(lora_module_names)

    def _log_model_parameters(self, model):
        """
        Log the number of trainable parameters in the model.
        """
        if self.verbose:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Number of trainable parameters: {trainable_params}")