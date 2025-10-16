from __future__ import annotations

import logging
import os
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

import spacy
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataPreprocess:
    """
    Data preparation pipeline for multiple targets (LogReg, ruBERT, Qwen).

    Parameters
    ----------
    dataset_name : str
        Hugging Face datasets identifier (e.g., 'yours/dataset').
    dataset_language : str, optional
        Configuration name (subset) for `load_dataset`, by default None.
    save_dir : str, optional
        Base directory to save prepared datasets, by default './saved_dataset'.
    load_spacy : bool, optional
        Whether to load spaCy model at init for text normalization, by default False.
    spacy_model : str, optional
        spaCy model name, by default 'ru_core_news_sm'.

    Notes
    -----
    - Avoid loading spaCy if only transformer-based pipelines are used.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_language: Optional[str] = None,
        save_dir: str = "./saved_dataset",
        load_spacy: bool = False,
        spacy_model: str = "ru_core_news_sm",
    ) -> None:
        self.dataset: DatasetDict = load_dataset(dataset_name, dataset_language)
        self.save_dir = save_dir
        self._nlp: Optional["spacy.Language"] = None
        self._spacy_model_name = spacy_model

        if load_spacy:
            self._ensure_spacy()

    def _ensure_spacy(self) -> None:
        """Lazily load spaCy model if not loaded."""
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self._spacy_model_name)
                logger.info("spaCy model '%s' loaded.", self._spacy_model_name)
            except OSError as e:
                logger.error("Failed to load spaCy model '%s': %s", self._spacy_model_name, e)
                raise

    @staticmethod
    def delete_data(path: str = "./saved_dataset") -> None:
        """
        Delete saved dataset directory recursively.

        Parameters
        ----------
        path : str, optional
            Directory to remove, by default './saved_dataset'.
        """
        if os.path.exists(path):
            shutil.rmtree(path)
            logger.info("Removed directory: %s", path)
        else:
            logger.info("Path does not exist: %s", path)
    
    def _save_dataset(self, splits_xy: Dict[str, Tuple[List[str], List[int]]], subdir: str) -> None:
        """
        Save classic (X,y) splits as DatasetDict to disk.

        Parameters
        ----------
        splits_xy : Dict[str, Tuple[List[str], List[int]]]
            Mapping split->(texts, labels).
        subdir : str
            Subdirectory under self.save_dir to save into.
        """
        out_dir = os.path.join(self.save_dir, subdir)
        os.makedirs(out_dir, exist_ok=True)

        ds_splits: Dict[str, Dataset] = {}
        for name, (X, y) in splits_xy.items():
            ds_splits[name] = Dataset.from_dict({"text": X, "label": y})

        DatasetDict(ds_splits).save_to_disk(out_dir)
        logger.info("Saved dataset to %s", out_dir)

    def _safe_remove_columns(self, ds: Dataset | DatasetDict, cols: Iterable[str]) -> Dataset | DatasetDict:
        """
        Remove columns if they exist.

        Parameters
        ----------
        ds : Dataset or DatasetDict
            Dataset or DatasetDict.
        cols : Iterable[str]
            Column names to remove.

        Returns
        -------
        Dataset or DatasetDict
            Dataset with missing columns skipped.
        """
        if isinstance(ds, DatasetDict):
            present_cols = {name for split in ds for name in ds[split].column_names}
            to_remove = [c for c in cols if c in present_cols]
            return DatasetDict({s: ds[s].remove_columns([c for c in to_remove if c in ds[s].column_names]) for s in ds})
        else:
            to_remove = [c for c in cols if c in ds.column_names]
            return ds.remove_columns(to_remove) if to_remove else ds

    def _encode_data(self) -> Tuple[DatasetDict, List[str]]:
        """
        Encode categorical labels and drop technical columns if present.

        Returns
        -------
        encoded_splits : DatasetDict
            DatasetDict with encoded 'category' column.
        categories : List[str]
            List of category names in training split.
        """
        ds = self._safe_remove_columns(self.dataset, ["index_id"])
        # class_encode_column доступен в Dataset(Dict)
        encoded: DatasetDict = ds.class_encode_column("category")
        categories = encoded["train"].features["category"].names
        return encoded, categories

    def _normalize_texts(self, texts: List[str]) -> List[str]:
        """
        Normalize and lemmatize texts using spaCy.

        Parameters
        ----------
        texts : List[str]
            Raw input texts.

        Returns
        -------
        List[str]
            Normalized texts (lemmatized, lowercased, punctuation removed, numbers blanked).
        """
        self._ensure_spacy()
        assert self._nlp is not None

        lowered = [t.lower() for t in texts]
        results: List[str] = []

        for doc in self._nlp.pipe(lowered, batch_size=128):
            lemmas = [
                "" if token.like_num else token.lemma_
                for token in doc
                if not token.is_punct and not token.is_space
            ]
            results.append(" ".join(lemmas).lower().strip())

        return results

    def process_logistic_regression(
        self,
        save: bool = False,
        subdir: str = "logreg",
    ) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], Tuple[List[str], List[int]], List[str]]:
        """
        Prepare lemmatized texts and integer labels for classical ML (e.g., Logistic Regression).

        Parameters
        ----------
        save : bool, optional
            Whether to save the prepared dataset, by default False.
        subdir : str, optional
            Subdirectory name for saving, by default 'logreg'.

        Returns
        -------
        train : Tuple[List[str], List[int]]
            (X_train, y_train).
        valid : Tuple[List[str], List[int]]
            (X_valid, y_valid).
        test : Tuple[List[str], List[int]]
            (X_test, y_test).
        categories : List[str]
            Label names in order of encoding.
        """
        encoded_splits, categories = self._encode_data()

        processed: Dict[str, Tuple[List[str], List[int]]] = {}
        for split in ["train", "validation", "test"]:
            X = self._normalize_texts(encoded_splits[split]["text"])
            y = list(encoded_splits[split]["category"])
            processed[split] = (X, y)

        if save:
            self._save_dataset(processed, subdir=subdir)

        return processed["train"], processed["validation"], processed["test"], categories

    @staticmethod
    def _prepare_message(texts: List[str], categories: List[str]) -> List[Dict[str, str]]:
        """
        Build LLM user messages for single-label classification.

        Parameters
        ----------
        texts : List[str]
            Input texts for inference.
        categories : List[str]
            Category names to choose from.

        Returns
        -------
        List[Dict[str, str]]
            List of role/content dicts compatible with chat models.
        """
        prompt_head = (
            "Прочтите текст ниже и выберите одну из следующих категорий, "
            f"которая лучше всего соответствует содержанию: {', '.join(categories)}.\n"
            "Верните только название подходящей категории.\n"
        )
        messages: List[Dict[str, str]] = []
        for msg in texts:
            messages.append(
                {
                    "role": "user",
                    "content": f"{prompt_head}Текст: {msg}",
                }
            )
        return messages

    def process_qwen2_7b(
        self,
        save: bool = False,
        subdir: str = "qwen",
        split: str = "test",
    ) -> Dataset:
        """
        Prepare a split with LLM-style 'llm_message' column for Qwen2-7B style inference.

        Parameters
        ----------
        save : bool, optional
            Whether to save the prepared dataset, by default False.
        subdir : str, optional
            Subdirectory name for saving, by default 'qwen'.
        split : str, optional
            Dataset split to prepare, by default 'test'.

        Returns
        -------
        Dataset
            Dataset with 'llm_message' column.
        """
        encoded_splits, categories = self._encode_data()
        ds = encoded_splits[split]
        llm_messages = self._prepare_message(list(ds["text"]), categories)
        ds = ds.add_column(name="llm_message", column=llm_messages)

        if save:
            out_dir = os.path.join(self.save_dir, subdir)
            os.makedirs(out_dir, exist_ok=True)
            ds.flatten_indices().save_to_disk(out_dir)
            logger.info("Saved Qwen split to %s", out_dir)

        return ds

    def process_rubert(
        self,
        save: bool = False,
        model_name: str = "DeepPavlov/rubert-base-cased",
        max_length: int = 512,
        subdir: str = "bert",
    ) -> Tuple[DatasetDict, List[str]]:
        """
        Tokenize dataset for ruBERT fine-tuning.

        Parameters
        ----------
        save : bool, optional
            Whether to save the tokenized dataset, by default False.
        model_name : str, optional
            Hugging Face tokenizer checkpoint, by default 'DeepPavlov/rubert-base-cased'.
        max_length : int, optional
            Sequence max length for padding/truncation, by default 512.
        subdir : str, optional
            Subdirectory name for saving, by default 'bert'.

        Returns
        -------
        tokenized : DatasetDict
            Tokenized dataset with 'labels' column.
        labels : List[str]
            Category label names (order consistent with encoding).
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def tokenize_function(examples: Dict[str, List[str]]) -> Dict[str, List[int]]:
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors=None,
            )

        # Encode labels and remove technical columns safely
        encoded: DatasetDict = self._safe_remove_columns(self.dataset, ["index_id"]).class_encode_column("category")
        labels = encoded["train"].features["category"].names

        tokenized = encoded.map(
            tokenize_function,
            batched=True,
            remove_columns=[c for c in ["text"] if c in encoded["train"].column_names],
        )

        # Rename 'category' -> 'labels' for Trainer API
        tokenized = tokenized.rename_column("category", "labels")

        if save:
            out_dir = os.path.join(self.save_dir, subdir)
            os.makedirs(out_dir, exist_ok=True)
            for split in tokenized:
                tokenized[split] = tokenized[split].flatten_indices()
            tokenized.save_to_disk(out_dir)
            logger.info("Saved tokenized dataset to %s", out_dir)

        return tokenized, labels