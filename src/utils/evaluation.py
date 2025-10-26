import evaluate
import numpy as np

# init metric F1 from evaluate
f1_score = evaluate.load("f1")

def compute_f1(eval_pred) -> dict | None:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score.compute(predictions=predictions, references=labels, average='macro')
    return f1