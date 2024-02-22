import os
from enum import Enum

from llama_cpp import Llama
from pydantic import BaseModel, ValidationError
from loguru import logger

from huggingface_hub import hf_hub_download

from zero_shot_classification.mixin import format_preprompt, format_example_output

SYSTEM_PROMPT_GEMMA = """<start_of_turn>user
{system_prompt}
Message: '{example_input}'
Response: {example_output}
Message: '{prompt}'<end_of_turn>
<start_of_turn>model
Response :"""


class LocalGemma:

    def __init__(self, model_name: str) -> None:
        """Initialize Gemma Llama.cpp model."""
        model_paths = model_name.split("/")
        self.system_prompt = SYSTEM_PROMPT_GEMMA

        if not os.path.exists("temp"):
            os.makedirs("temp")

        if not os.path.exists(os.path.join("temp", model_paths[2])):
            hf_hub_download(repo_id=model_paths[0] + "/" + model_paths[1],
                            filename=model_paths[2],
                            repo_type="model",
                            local_dir="temp",
                            local_dir_use_symlinks=False)

        self.model = Llama(
            model_path=os.path.join("temp", model_paths[2]),
            n_ctx=2048,
            # The max sequence length to use - note that longer sequence lengths require much more resources
            n_threads=1,
            # The number of CPU threads to use, tailor to your system and the resulting performance
            n_gpu_layers=-1,
            # The number of layers to offload to GPU, if you have GPU acceleration available
            # Set to 0 if no GPU acceleration is available on your system.
        )

    def tokenize(self, text: str) -> list[int]:
        """Tokenize text.

        Args:
            text (str): text to tokenize

        Returns:
            list[int]: list of tokens
        """
        return self.model.tokenize(text.encode("utf-8"))

    def token_eos(self) -> int:
        """Get end of sentence token.

        Returns:
            int: index of end of sentence token
        """
        return self.model.token_eos()

    def n_vocab(self) -> int:
        """Get vocabulary size.

        Returns:
            int: vocabulary size
        """
        return self.model.n_vocab()

    def generate(self,
                 preprompt: str,
                 prompt: str,
                 labels: list[str],
                 predict_labels_index: bool,
                 example_input: str,
                 example_output: str,
                 validation_error_label: str = "VALIDATION_ERROR",
                 response_blocked_error_label: str = "RESPONSE_BLOCKED_ERROR",
                 logit_bias: dict[int, float] = None,
                 max_retries: int = 0) \
            -> tuple[str, int, int]:
        """Generate predictions for a single row.

        Args:
            model (str): model name
            preprompt (str): preprompt
            prompt (str): prompt
            labels (list[str]): list of possible labels
            predict_labels_index (bool): whether the model must predict labels index (less costly)
            example_input (str): example input
            example_output (str): example output
            validation_error_label (str): validation error label
            response_blocked_error_label (str): response blocked error label
            logit_bias (dict[int, float]): logit bias to force the model to predict only a subset of tokens
            max_retries (int): max number of retries

        Returns:
            label, completion_tokens, prompt_tokens (tuple[str, int, int])
        """

        class Label(BaseModel):
            label: Enum("Labels", {l: l for l in labels})

        query = self.system_prompt.format(system_prompt=format_preprompt(preprompt=preprompt,
                                                                         labels=labels,
                                                                         with_index=predict_labels_index),
                                          example_input=example_input,
                                          example_output=format_example_output(example_output=example_output,
                                                                               labels=labels,
                                                                               predict_labels_index=predict_labels_index),
                                          prompt=prompt)

        logger.debug(f"Query: {query}")

        max_tokens = len(str(len(labels))) + 1 if predict_labels_index else max([len(label) for label in labels]) / 2

        response = self.model(query,
                              temperature=0.0,
                              # max_tokens=max_tokens,
                              stop=["<end_of_turn>"],
                              # logit_bias=logit_bias,
                              echo=False)

        response_text = response["choices"][0]["text"]
        logger.debug(f"Response: {response_text}")

        try:
            if predict_labels_index:
                response_clean = "".join([i for i in response_text.split() if i.isdigit()])
                label = Label(label=labels[int(response_clean) - 1]).label.value
            else:
                response_clean = "".join([label if response_text.find(label) > -1 else "" for label in labels])
                label = Label(label=response_clean).label.value


        except IndexError:
            logger.warning(f'Index error: {response_text} is not a valid label index.')
            label = validation_error_label


        except ValidationError:
            logger.warning(f'Validation error: {response_text} is not a valid label.')
            label = validation_error_label

        completion_tokens = response["usage"]["completion_tokens"]
        prompt_tokens = response["usage"]["prompt_tokens"]

        return label, completion_tokens, prompt_tokens
