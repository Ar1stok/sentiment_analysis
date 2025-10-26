## Sentiment Analysis Benchmark

### Objective
The purpose of this project is to perform a comparative analysis of the performance and characteristics of different approaches for multi-class text classification in Russian, using the Dalvan/sib200 dataset.

The following models are evaluated:
- Classic ML-method (Logistic Regression)
- Pre-trained transformer ruBERT (DeepPavlov/rubert-base-cased)
- LLM Qwen2-7B-Instruct

### Description
1. Classic ML (Logistic Regression)
    - Preprocessing: Lemmatization (spaCy), conversion of categories to numeric labels
    - Goal: Baseline performance with basic features (TF-IDF)
    - Evaluation: Classification accuracy/metrics


2. Qwen2-7B (LLM)
    - Form prompt:
        ```    
        "role": "user",
        "content":  "Прочтите текст ниже и выберите одну из следующих категорий, "
                    f"которая лучше всего соответствует содержанию: {categories}.\n"
                    "Верните только название подходящей категории.\n"
                    "Текст: {text}",
        ```
    - Evaluation: Compare by F1-Score (using string matching and convert to num)


3. ruBERT (Transformer)
    - Preprocessing: PreUses HuggingFace tokenizer 
    - Train: Model fine-tuned for classification
    - Evaluation: Final results compared to baseline and LLM (F1-Score)


Модель                   |  Стек реализации                      |  F1-score  |  Особенности                                                       
-------------------------|---------------------------------------|------------|--------------------------------------------------------------------
Logistic Regression      |  TF-IDF + LogisticRegression          |  0.67      |  Классический ML (baseline), низкая точность    
Qwen2-7B-Instruct (LLM)  |  HuggingFace                          |  0.83      |  Современный LLM, требует много ресурсов, чувствителен к запросам
ruBERT (Transformer)     |  DeepPavlov/rubert-base-cased         |  0.90      |  Предобученная модель, лучший результат

> [!NOTE]
> Something might change in the future

