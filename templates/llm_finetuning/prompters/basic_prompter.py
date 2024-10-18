from typing import Optional

class BasicPrompter:

    def generate_prompt(
            self,
            instruction: str,
            input: Optional[str] = None,
            label: Optional[str] = None,
    ) -> str:
        """
        Generates a prompt for LLM finetuning, including the instruction, optional input context,
        and optionally a label (expected output).

        Parameters:
            instruction (str): The instruction or task description for the model.
            input (Optional[str]): An optional input context that provides additional information.
            label (Optional[str]): An optional label that represents the expected output.

        Returns:
            str: A formatted prompt ready for LLM finetuning.
        """
        # Prepare the context section only if input is provided
        context = f"Here is some context: {input}" if input else ""

        # Construct the full prompt
        full_prompt = (
            f"<s>[INST] {instruction}\n"
            f"{context}\n"
            f"[/INST]{f' {label}' if label else ''}\n"
            f"</s>"
        )

        # Return the prompt ready for finetuning
        return full_prompt