from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
NUM_LABELS = 5  # Number of labels for Yelp Review Full dataset


def get_model():
    """Get a pretrained language model."""
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)


def get_tokenizer():
    """Get a pretrained tokenizer."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)
