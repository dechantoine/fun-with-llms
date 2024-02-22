# This file contains mixin functions for the infer module.

import re

def format_preprompt(preprompt: str, labels: list[str], with_index: bool = False) -> str:
    """Format labels in the preprompt for the model.
    Args:
        preprompt (str): preprompt
        labels (list[str]): list of possible labels
        with_index (bool): whether to include index in labels
    """
    if with_index:
        labels = [f"{i + 1}. {label}" for i, label in enumerate(labels)]
    return f"""{preprompt}[{', '.join(labels)}]"""


def clean_model_name(model_name: str) -> str:
    """Clean model name for use in file names.
    Args:
        model_name (str): model name
    """
    return re.sub(pattern=r'[-@/.]', repl='_', string=model_name)


def format_example_output(example_output: str, predict_labels_index: bool, labels: list[str]) -> str:
    """Format example output for the model.
    Args:
        example_output (str): example output
        predict_labels_index (bool): whether the model must predict labels index (less costly)
        labels (list[str]): list of possible labels
    """
    if predict_labels_index:
        return str(labels.index(example_output) + 1)
    return example_output