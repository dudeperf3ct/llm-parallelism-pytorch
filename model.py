from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "HuggingFaceTB/SmolLM2-360M-Instruct"
PP_MODEL_NAME = "distilbert/distilbert-base-uncased"
NUM_LABELS = 5  # Number of labels for Yelp Review Full dataset


def get_model():
    """Get the default model used for DDP and sharding experiments."""
    return AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)


def get_tokenizer():
    """Get the tokenizer matching the default model."""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def get_pp_model():
    """Get the pipeline-parallel friendly model."""
    return AutoModelForSequenceClassification.from_pretrained(PP_MODEL_NAME, num_labels=NUM_LABELS)


def get_pp_tokenizer():
    """Get the tokenizer matching the pipeline-parallel model."""
    return AutoTokenizer.from_pretrained(PP_MODEL_NAME)
