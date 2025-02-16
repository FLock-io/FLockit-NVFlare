from .basic_prompter import BasicPrompter
from .template_prompter import TemplatePrompter


# def get_prompter(prompter_type: str, template_name: str = "", verbose: bool = False):
def get_prompter(template_name: str = "", verbose: bool = False):
    if template_name == "":
        return BasicPrompter()
    else:
        return TemplatePrompter(template_name=template_name, verbose=verbose)
