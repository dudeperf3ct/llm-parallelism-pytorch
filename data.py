import os

from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import DataCollatorWithPadding

from model import get_tokenizer

DATASET_NAME = "Yelp/yelp_review_full"

tokenizer = get_tokenizer()
TOKENIZE_BATCH_SIZE = 1000
DEFAULT_NUM_PROC = max(1, os.cpu_count())


def get_data_collator():
    """Pad dynamically at batch time instead of during preprocessing."""
    return DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")


def get_dataset():
    """Get the yelp review dataset."""
    return load_dataset(DATASET_NAME)


def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=True)


def tokenize_data(dataset):
    """Tokenize the raw dataset.

    Args:
        dataset: The raw dataset to prepare.

    Returns:
        The tokenized dataset.
    """
    return dataset.map(
        tokenize_function, batched=True, batch_size=TOKENIZE_BATCH_SIZE, num_proc=DEFAULT_NUM_PROC
    )


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
    raw_dataset["train"] = raw_dataset["train"].shuffle(seed=42).select(range(100))
    raw_dataset["test"] = raw_dataset["test"].shuffle(seed=42).select(range(50))
    tokenized_dataset = tokenize_data(raw_dataset)
    train_ds, eval_ds = split_dataset(tokenized_dataset)

    if rank == 0:
        print(
            f"Dataset sizes -> train: {len(train_ds)} samples, "
            f"eval: {len(eval_ds)} samples (world size={world_size})"
        )
    collator = get_data_collator()
    num_workers = min(8, os.cpu_count() // max(1, world_size))
    use_workers = num_workers > 0
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(
        train_ds,
        shuffle=False,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=collator,
        num_workers=num_workers,
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        pin_memory=True,
        collate_fn=collator,
        num_workers=num_workers,
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
    )
    return train_loader, eval_loader
