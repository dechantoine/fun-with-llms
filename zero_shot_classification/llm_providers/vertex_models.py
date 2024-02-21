import os
from enum import Enum

from vertexai.generative_models import GenerativeModel, GenerationConfig, ResponseValidationError
from vertexai.language_models import ChatModel, InputOutputTextPair
import vertexai
from pydantic import BaseModel, ValidationError
from zero_shot_classification.mixin import format_preprompt

from loguru import logger


class VertexChat:

    def __init__(self) -> None:
        """Initialize Vertex AI."""
        vertexai.init(project=os.environ.get('PROJECT_ID'),
                      location=os.environ.get('PROJECT_LOCATION'))

    @logger.catch
    def format_message_gemini(self, system_prompt: str, user_message: str, example_input: str,
                              example_output: str) -> str:
        """Format message for Gemini model.
        Args:
            system_prompt (str): system prompt
            user_message (str): user message
            example_input (str): example input
            example_output (str): example output

        Returns:
            prompt (str): formatted prompt for Gemini
        """
        prompt = f"""{system_prompt}
        Message: '{example_input}'
        Label: {example_output}
        Message: '{user_message}'
        Label: """
        return prompt

    @logger.catch
    def generate(self,
                 model: str,
                 preprompt: str,
                 prompt: str,
                 labels: list[str],
                 predict_labels_index: bool,
                 example_input: str,
                 example_output: str,
                 validation_error_label: str = "VALIDATION_ERROR",
                 response_blocked_error_label: str = "RESPONSE_BLOCKED_ERROR",
                 max_retries: int = 0,
                 ) \
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
            max_retries (int): max number of retries

        Returns:
            label, completion_tokens, prompt_tokens (tuple[str, int, int])
        """

        class Label(BaseModel):
            label: Enum("Labels", {l: l for l in labels})

        max_tokens = (len(str(len(labels))) + 1
                      if predict_labels_index
                      else max([len(label) for label in labels]) // 2)

        try:

            if model.find("gemini") > -1:
                model = GenerativeModel(model)
                message = self.format_message_gemini(
                    system_prompt=format_preprompt(preprompt, labels),
                    user_message=prompt,
                    example_input=example_input,
                    example_output=example_output
                )

                chat = model.start_chat()

                response = chat.send_message(
                    content=message,
                    generation_config=GenerationConfig(
                        temperature=0.,
                        # top_p=0.95,
                        # #top_k=20,
                        candidate_count=1,
                        max_output_tokens=max_tokens
                        # #stop_sequences=["\n\n\n"],
                    )
                )
                response_text = response.candidates[0].text
                logger.info(f"Response: {response_text}")

            else:
                model = ChatModel.from_pretrained(model_name=model)
                chat = model.start_chat(
                    context=format_preprompt(preprompt, labels),
                    examples=[
                        InputOutputTextPair(
                            input_text=example_input,
                            output_text=example_output,
                        ),
                    ],
                )

                response = chat.send_message(
                    message=prompt,
                    temperature=0,
                    max_output_tokens=max_tokens
                )
                response_text = response.text

            if predict_labels_index:
                response_clean = "".join([i for i in response_text.split() if i.isdigit()])
                label = Label(label=labels[int(response_clean) - 1]).label.value
            else:
                response_clean = "".join([label if response_text.find(label) > -1 else "" for label in labels])
                label = Label(label=response_clean).label.value

        except IndexError:
            logger.warning(f'Index error: {response_text} is not a valid label index.')
            label = validation_error_label
            completion_tokens = len(response_text)

        except ValidationError:
            logger.warning(f'Validation error: {response_text} is not a valid label.')
            label = validation_error_label
            completion_tokens = len(response_text)

        except ResponseValidationError:
            logger.warning(f'Response blocked for harmful content by API with the prompt: {prompt}')
            label = response_blocked_error_label
            completion_tokens = 0

        else:
            completion_tokens = len(response_text)

        finally:
            prompt_tokens = len(prompt)

        return label, completion_tokens, prompt_tokens
