from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler

from model import get_tokenizer

DATASET_NAME = "Yelp/yelp_review_full"


def get_dataset():
    """Get the yelp review dataset."""
    return load_dataset(DATASET_NAME)


def tokenize_function(examples):
    tokenizer = get_tokenizer()
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def tokenize_data(dataset):
    """Tokenize the raw dataset.

    Args:
        dataset: The raw dataset to prepare.

    Returns:
        The tokenized dataset.
    """
    return dataset.map(tokenize_function, batched=True)


def split_dataset(tokenized_dataset):
    """Get train and test splits from tokenized dataset.

    Args:
        tokenized_dataset: The tokenized dataset.

    Returns:
        train_ds: The training dataset.
        eval_ds: The evaluation dataset.
    """
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    # model expects the argument to be named labels
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")
    train_ds = tokenized_dataset["train"].shuffle(seed=42)
    eval_ds = tokenized_dataset["test"].shuffle(seed=42)
    return train_ds, eval_ds


def prepare_data(batch_size: int, rank: int, world_size: int):
    """Prepare the dataset for training and evaluation.

    Returns:
        train_loader: DataLoader for training dataset.
        eval_loader: DataLoader for evaluation dataset.
    """
    raw_dataset = get_dataset()
    tokenized_dataset = tokenize_data(raw_dataset)
    train_ds, eval_ds = split_dataset(tokenized_dataset)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_ds, shuffle=False, batch_size=batch_size, sampler=train_sampler, pin_memory=True
    )
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, pin_memory=True)
    return train_loader, eval_loader
