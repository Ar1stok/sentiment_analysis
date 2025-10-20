from __future__ import annotations

import logging
import os
import shutil
from typing import Dict, Iterable, List, Optional, Tuple

import spacy
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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
        preload_dataset: bool = True,
        save_dir: str = "./saved_dataset",
        load_spacy: bool = False,
        spacy_model: str = "ru_core_news_sm",
    ) -> None:
        self.dataset: Optional["DatasetDict | Dataset"] = None
        self.save_dir = save_dir
        self._nlp: Optional["spacy.Language"] = None
        self._spacy_model_name = spacy_model

        if preload_dataset:
            self._ensure_dataset(dataset_name, dataset_language)
        if load_spacy:
            self._ensure_spacy()

    def _ensure_dataset(
            self,
            dataset_name: str,
            dataset_language: Optional[str] = None,
        ) -> None:
        """Lazily load dataset if not loaded."""
        if self.dataset is None:
            try:
                self.dataset = load_dataset(dataset_name, dataset_language)
                logger.info("Dataset '%s' loaded.", dataset_name)
            except OSError as e:
                logger.error("Failed to load dataset '%s': %s", dataset_name, e)
                raise

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
    def check_dataset(out_dir: str) -> DatasetDict | Dataset | None:
        """
        Tries to load Dataset from disk, returns None if loading failed.

        Parameters
        ----------
        out_dir : str
            Path to saved dataset dict.

        Returns
        -------
        ds : Dataset or DatasetDict or None
            Loaded dataset or None if not found/corrupted.
        """
        try:
            ds = load_from_disk(out_dir)
            logger.info("Dataset loaded from %s", out_dir)
            return ds
        except Exception:
            logger.info("Dataset not found in %s", out_dir)
            return None

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
        
        # class_encode_column avaible in Dataset(Dict)
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
    ) -> Tuple[DatasetDict, List[str]]:
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
        dataset_dict : DatasetDict
            Prepared dataset for logreg.
            
        categories : List[str]
            Label names in order of encoding.
        """
        out_dir = os.path.join(self.save_dir, subdir)
        dataset_dict = self.check_dataset(out_dir)
        if dataset_dict:
            return dataset_dict

        encoded_splits, categories = self._encode_data()

        dataset_dict = DatasetDict({
            split: Dataset.from_dict({
                "text": self._normalize_texts(encoded_splits[split]["text"]),
                "label": list(encoded_splits[split]["category"])
            })
            for split in ["train", "validation", "test"]
        })

        if save:
            dataset_dict.save_to_disk(out_dir)
            logger.info("Saved dataset to %s", out_dir)

        return dataset_dict

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
        messages: List[List[Dict[str, str]]] = []
        for msg in texts:
            messages.append(
                [{
                    "role": "user",
                    "content": f"{prompt_head} Текст: {msg}",
                }]
            )
        return {'llm_messages': messages}

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
            Dataset with 'llm_messages' column.
        """
        out_dir = os.path.join(self.save_dir, subdir)
        ds = self.check_dataset(out_dir)
        if ds:
            return ds

        encoded_splits, categories = self._encode_data()
        ds = encoded_splits[split]
        ds = ds.add_column(name="llm_messages", column=self._prepare_message(ds["text"], categories)["llm_messages"])

        if save:
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
        out_dir = os.path.join(self.save_dir, subdir)
        tokenized = self.check_dataset(out_dir)
        if tokenized:
            return tokenized

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

        return tokenized