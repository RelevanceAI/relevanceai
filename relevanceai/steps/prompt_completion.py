import requests
from relevanceai.steps._base import StepBase


class PromptCompletion(StepBase):
    """Generate text using LLMs
    Generate text from a large language model like GPT.

    Args:
        prompt (str): The prompt that is fed to the model.
        model ((Optional) str): The model to use for completion. If using gpt3.5, if you do not set your own API Key you will be charged 1.33 credits per 1000 characters of input and output. For other models, Make sure to set an API key.
        history ((Optional) list): Conversation history to be passed into the prompt. For example, [{role: 'user', message: 'Hello, my name is Bob.'}, {role: 'ai', message: 'Hello Bob, how are you?'}].
        system_prompt ((Optional) str): System prompt to be passed into the GPT chat completion prompts.
        strip_linebreaks ((Optional) bool): Whether to strip linebreaks from the output.
        temperature ((Optional) int): Temperature of the selected model. Typically, higher temperature means more random output.
        validators ((Optional) list): Validate that the LLM produces output in the expected format, and re-prompt to fix issues if not.

    Returns:
        answer ((Optional) str): {'type': 'string'}
        prompt (str): {'type': 'string'}
        user_key_used ((Optional) bool): {'type': 'boolean'}
        validation_history ((Optional) list): {'type': 'array', 'items': {'type': 'object', 'properties': {'role': {'type': 'string', 'enum': ['user', 'ai']}, 'message': {'type': 'string'}}, 'required': ['role', 'message'], 'additionalProperties': False}, 'metadata': {'advanced': True, 'title': 'Conversation history', 'description': "Conversation history to be passed into the prompt. For example, [{role: 'user', message: 'Hello, my name is Bob.'}, {role: 'ai', message: 'Hello Bob, how are you?'}]."}}
    """

    def __init__(
        self,
        prompt: str,
        model: str = None,
        history: list = None,
        system_prompt: str = None,
        strip_linebreaks: bool = None,
        temperature: int = None,
        validators: list = None,
        step_name: str = "prompt_completion",
        *args,
        **kwargs,
    ) -> None:
        self.prompt = prompt
        self.model = model
        self.history = history
        self.system_prompt = system_prompt
        self.strip_linebreaks = strip_linebreaks
        self.temperature = temperature
        self.validators = validators
        self.step_name = step_name
        self._outputs = ["answer", "prompt", "user_key_used", "validation_history"]
        self.outputs = [f"steps.{self.step_name}.output.{a}" for a in self._outputs]
        super().__init__(*args, **kwargs)

    @property
    def steps(self):
        step_params = {
            "prompt": self.prompt,
        }
        if self.model is not None:
            step_params["model"] = self.model
        if self.history is not None:
            step_params["history"] = self.history
        if self.system_prompt is not None:
            step_params["system_prompt"] = self.system_prompt
        if self.strip_linebreaks is not None:
            step_params["strip_linebreaks"] = self.strip_linebreaks
        if self.temperature is not None:
            step_params["temperature"] = self.temperature
        if self.validators is not None:
            step_params["validators"] = self.validators
        return [
            {
                "transformation": "prompt_completion",
                "name": self.step_name,
                "foreach": "",
                "output": {output: f"{{{{ {output} }}}}" for output in self._outputs},
                "params": step_params,
            }
        ]
