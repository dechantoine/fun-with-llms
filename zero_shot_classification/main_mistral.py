from google.cloud import bigquery
from more_itertools import ichunked
import asyncio
from ast import literal_eval
from collections import deque
from collections.abc import Coroutine
import json
from loguru import logger
import os

import numpy as np
from .mixin import clean_model_name
from .llm_providers.mistral_llama_models import LocalMistral

PROJECT_ID = os.environ.get('PROJECT_ID')
DATASET_ID = os.environ.get('DATASET_ID')
TASK_FILENAME = os.environ.get('TASK_FILENAME')
BQ_BATCH_SIZE = os.environ.get('BQ_BATCH_SIZE', 100)
MAX_RETRIES = os.environ.get('MAX_RETRIES', 0)

task = json.load(open(f"zero_shot_classification/tasks/{TASK_FILENAME}", "r"))

models = task["MODELS"]["LLAMACPP"]

schema = [
    bigquery.SchemaField(task["ID_COLUMN_NAME"], task["ID_COLUMN_TYPE"], mode="NULLABLE"),
    bigquery.SchemaField(task["LABEL_COLUMN_NAME"], "STRING", mode="NULLABLE"),
    bigquery.SchemaField(task["COMPLETION_TOKENS_COLUMN_NAME"], "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField(task["PROMPT_TOKENS_COLUMN_NAME"], "INTEGER", mode="NULLABLE"),
    bigquery.SchemaField("model", "STRING", mode="NULLABLE"),
]

bq_client = bigquery.Client(project=PROJECT_ID)

LogitBiases = dict[str, dict[int, float]]
ListInterleavedJobs = list[dict[str, list[Coroutine]]]


@logger.catch
def init_tables():
    """Create BQ tables for results if not exist."""
    for model in models:
        # create table if not exists
        model_name = model.split("/")[2].replace("-", "_").replace(".", "_")
        table_id = f"{PROJECT_ID}.{DATASET_ID}.{model_name}_{task['TASK_ID']}"
        try:
            bq_client.get_table(table_id)

        except Exception:
            logger.info(f"Model {model} : table {table_id} does not exist. Creating it.")
            table = bigquery.Table(table_id, schema=schema)
            bq_client.create_table(table)

        else:
            logger.info(f"Model {model} : table {table_id} already exists.")


@logger.catch
def write_to_bq(model_name: str, bq_rows: list[dict]) -> None:
    """Write rows to BQ table.

    Args:
        model_name (str): model name
        bq_rows (list[dict]): list of records to insert
    """
    table_id = f"{PROJECT_ID}.{DATASET_ID}.{model_name}_{task['TASK_ID']}"
    bq_client.insert_rows_json(table=table_id,
                               json_rows=bq_rows)
    logger.info(f"Model {model_name} : inserted {len(bq_rows)} rows to {table_id}.")


@logger.catch
def init_models() -> dict[str, LocalMistral]:
    """Initialize Mistral models.

    Returns:
        dict: dictionary of models
    """
    dict_models = {}
    for model in models:
        model_name = clean_model_name(model.split("/")[2])
        dict_models[model_name] = LocalMistral(model_name=model)
    return dict_models


@logger.catch
def init_logit_bias(dict_models: dict[str, LocalMistral]) -> LogitBiases:
    """Initialize logit bias for each model.

    Args:
        dict_models (dict[str, LocalMistral]): dictionary of models

    Returns:
        dict_logit_bias: dictionary of logit biases for each model
    """
    dict_logit_bias = {}
    for model_name, model in dict_models.items():
        logit_bias = {i: -np.inf for i in range(model.n_vocab())}
        for i, label in enumerate(task["LABELS"]):
            if literal_eval(task["PREDICT_LABELS_INDEX"]):
                tokens_label = model.tokenize(f"{i + 1}")
            else:
                tokens_label = model.tokenize(label)

            for token in tokens_label:
                logit_bias[token] = 0

        dict_logit_bias[model_name] = logit_bias

    return dict_logit_bias


@logger.catch
async def generate(model: LocalMistral,
                   model_name: str,
                   input_row: dict,
                   max_retries: int = 0,
                   logit_bias: dict[int, float] = None) -> dict:
    """Generate predictions for a single row.

    Args:
        model (LocalMistral): model
        model_name (str): model name
        input_row (dict): row to predict
        max_retries (int): max number of retries
        logit_bias (dict[int, float]): logit bias to force the model to predict only a subset of tokens

    Returns:
        insert_row (dict): row record to insert into BQ
    """
    label, completion_tokens, prompt_tokens = model.generate(
        preprompt=task["PREPROMPT"],
        prompt=input_row[task["TEXT_COLUMN_NAME"]],
        labels=task["LABELS"],
        predict_labels_index=literal_eval(task["PREDICT_LABELS_INDEX"]),
        example_input=task["EXAMPLES"]["INPUT"],
        example_output=task["EXAMPLES"]["OUTPUT"],
        max_retries=max_retries,
        logit_bias=logit_bias,
    )

    # insert into BQ table
    insert_row = {
        task["ID_COLUMN_NAME"]: input_row[task["ID_COLUMN_NAME"]],
        "label": label,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "model": model_name,
    }

    return insert_row


@logger.catch
def prepare_jobs(dict_models: dict[str, LocalMistral], dict_logit_bias: LogitBiases) -> ListInterleavedJobs:
    """Prepare jobs for parallel computing.

    Args:
        dict_models (dict[str, LocalMistral]): dictionary of models
        dict_logit_bias (LogitBiases): dictionary of logit biases for each model

    Returns:
        interleaved_jobs (list[dict[str, list[Coroutine]]]): list of dict(model_name: batch[jobs])
    """
    # import data
    data = bq_client.query(f"SELECT * FROM {PROJECT_ID}.{DATASET_ID}.{task['SOURCE_TABLE']}").to_dataframe()
    logger.info(f"Loaded {len(data)} rows to predict from BigQuery.")

    jobs = []

    for model_name, model in dict_models.items():
        table_id = f"{PROJECT_ID}.{DATASET_ID}.{model_name}_{task['TASK_ID']}"
        predicted_data = bq_client.query(f"SELECT {task['ID_COLUMN_NAME']} FROM {table_id}").to_dataframe()
        data_to_predict = data[~data[task['ID_COLUMN_NAME']].isin(predicted_data[task['ID_COLUMN_NAME']])]
        logger.info(f"Model {model} : predicting {len(data_to_predict)} rows.")

        model_jobs = []
        # append job to list for given models
        for row in data_to_predict.to_dict(orient="records"):
            model_jobs.append(generate(model=model,
                                       model_name=model_name,
                                       input_row=row,
                                       max_retries=MAX_RETRIES,
                                       logit_bias=dict_logit_bias[model_name]))

        # split into batches
        jobs.append(deque(ichunked(model_jobs, BQ_BATCH_SIZE)))

    # return list of dict(model_name: batch[jobs])
    interleaved_jobs = []
    while max([len(j) for j in jobs]) > 0:
        dict_jobs = {}
        for i, (model_name, _) in enumerate(dict_models.items()):
            try:
                dict_jobs[model_name] = list(jobs[i].popleft())
            except IndexError:
                pass
        interleaved_jobs.append(dict_jobs)

    return interleaved_jobs


@logger.catch
async def main(model_jobs: dict[str, list[Coroutine]]) -> None:
    """Run all jobs for a given batch.

    Args:
        model_jobs (dict[str, list[asyncio.Task]]): dict(model_name: list[asyncio.Task])
    """
    # for a batch, first obtain results from different models in parallel
    models_results = {model_name: [] for model_name in model_jobs.keys()}
    async with asyncio.TaskGroup() as tg:
        for model_name, model_jobs in model_jobs.items():
            for j in model_jobs:
                models_results[model_name].append(tg.create_task(j))

    # then write to BQ
    for model_name, results in models_results.items():
        rows_to_bq = [r.result() for r in results]
        write_to_bq(model_name, rows_to_bq)

if __name__ == '__main__':
    init_tables()
    dict_models = init_models()
    dict_logit_bias = init_logit_bias(dict_models)
    logger.info(f"Batch size for parallel computing: {BQ_BATCH_SIZE}")
    jobs = prepare_jobs(dict_models=dict_models,
                        dict_logit_bias=dict_logit_bias)
    # run each batch
    for i, job in enumerate(jobs):
        logger.info(f"Computing batch {i+1} out of {len(jobs)}")
        asyncio.run(main(job))
