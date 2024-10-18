import json
import os.path as osp
from typing import Optional, Dict, Any
from loguru import logger


class TemplatePrompter(object):

    def __init__(self, template_name: str = "", template_path: str = "", verbose: bool = False):
        self._verbose = verbose

        if template_path:
            file_name = template_path
        else:
            if not template_name:
                # Set default template name
                template_name = "alpaca"
            # Use absolute path based on the current script location
            file_name = osp.join(osp.dirname(__file__), "prompt_templates", f"{template_name}.json")

        if not osp.exists(file_name):
            raise ValueError(f"Template file not found: {file_name}")

        try:
            with open(file_name) as fp:
                self.template = json.load(fp)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {file_name}: {str(e)}")

        if self._verbose:
            logger.info(
                f"Using prompt template from {file_name}: {self.template.get('description', 'No description available')}")

    def generate_prompt(self, **kwargs: Any) -> str:
        """
        Generates a prompt based on the template and provided keyword arguments.

        The template can include any number of placeholders that are dynamically filled
        with the values provided in kwargs.

        Args:
            **kwargs: Any keyword arguments that correspond to the placeholders in the template.

        Returns:
            str: The generated prompt.
        """
        try:
            if "prompt_input" in self.template and "input" in kwargs:
                template_str = self.template["prompt_input"]
            elif "prompt_no_input" in self.template:
                template_str = self.template["prompt_no_input"]
            else:
                raise ValueError("No valid template string found in the template.")

            # Format the template string with provided keyword arguments
            res = template_str.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing key during prompt generation: {str(e)}")

        if "label" in kwargs:
            res += f"{kwargs['label']}"  # Append label if provided

        if self._verbose:
            logger.info(f"Generated prompt: {res}")

        return res

    def get_response(self, output: str) -> str:
        """
        Extracts the response part from the output based on the template's response split key.

        Args:
            output (str): The full output from which to extract the response.

        Returns:
            str: The extracted response.
        """
        try:
            response_split_key = self.template.get("response_split", "### Response:")
            parts = output.split(response_split_key)
            if len(parts) < 2:
                raise ValueError("Response split did not yield expected parts")
            response = parts[1].strip()
        except (IndexError, KeyError) as e:
            raise ValueError(f"Failed to extract response: {str(e)}")

        return response