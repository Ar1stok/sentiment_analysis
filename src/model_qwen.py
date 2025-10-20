import logging

from rapidfuzz import process
from sklearn.metrics import classification_report
from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast, TFPreTrainedModel, pipeline
from tqdm.auto import tqdm
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def qwen_evaluate(
        model: PreTrainedModel | TFPreTrainedModel, 
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, 
        dataset: Dataset
    ) -> str | dict:
    """
    Evaluate LLM model on classification task with batch processing.
    
    Parameters:
    -----------
    model: PreTrainedModel | TFPreTrainedModel
        The model that will be used by the pipeline to make predictions
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
        The tokenizer that will be used by the pipeline to encode data for the model.
    dataset: Dataset
        HuggingFace dataset with 'test' split
    
    Returns:
    --------
    report : str | dict
        Return classification report from sklearn.metrics
    """
    
    # Get category names
    categories = dataset.features['category'].names
    
    # Initialize pipeline with optimizations
    llm = pipeline(
        "text-generation", 
        model=model,
        tokenizer=tokenizer,
        return_full_text=False,
        max_new_tokens=256, 
        device_map='auto', 
        torch_dtype=torch.float16,
    )
    
    test_messages = dataset['llm_messages']
    
    logger.info(f"Evaluating {len(test_messages)} samples...")
    try:
        y_pred_test = list(map(lambda x: llm(x)[0]['generated_text'], test_messages))
    except Exception as e:
        logger.warning(f"Error processing: {e}\nFailed predictions will be empty")
        y_pred_test = [''] * len(test_messages)

    # Map predictions to category indices with rapidfuzz
    logger.info('Start matching categories...')
    test_pred = []
    for pred in tqdm(y_pred_test, desc="Matching categories"):
        if pred.strip():  # Skip empty predictions
            match = process.extractOne(
                pred, 
                categories, 
                processor=None,  # Already preprocessed
                score_cutoff=50  # Minimum similarity threshold
            )
            test_pred.append(categories.index(match[0]) if match else 0)
        else:
            test_pred.append(0)  # Default category for failed predictions
    
    # Generate classification report with category names
    logger.info('Generating report...')
    report = classification_report(
        dataset['category'], 
        test_pred,
        target_names=categories,
        digits=3,
    )
    
    return report