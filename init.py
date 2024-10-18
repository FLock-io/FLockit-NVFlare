from constants import FLockitTemplates

def init(args):

    if args.common_args.template_name == FLockitTemplates.llm_finetuning:
        # from templates.llm_finetuning.init import llm_finetuning_init as template_init
        from templates.llm_finetuning.init import llm_finetuning_init as template_init
    else:
        raise ValueError(f"Template {args.common_args.template_name} not supported, please check your configuration file.")

    task_model = template_init(args)

    return task_model

