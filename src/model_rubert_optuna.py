from typing import Dict

import optuna
from datasets import DatasetDict
from transformers import TrainingArguments, Trainer
from transformers import PreTrainedModel, TFPreTrainedModel
from utils.evaluation import compute_f1


def objective(
        trial, 
        model: PreTrainedModel | TFPreTrainedModel, 
        dataset : DatasetDict
    ) -> Dict[str, float]:
    args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        fp16=True,

        # Optimize hyperparameters
        learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        per_device_train_batch_size=trial.suggest_categorical("batch_size", [16, 32]),
        weight_decay=trial.suggest_float("weight_decay", 1e-1, 1e-3, log=True),
        num_train_epochs=5,
        label_smoothing_factor=0.1,

        # Warmup и scheduler
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",

        # best model
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_f1
    )

    result = trainer.evaluate()

    return result["eval_accuracy"]

def hyper_search() -> None:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)