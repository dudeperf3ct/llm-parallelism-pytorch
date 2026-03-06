import os

from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from transformers import DataCollatorWithPadding

from model import get_tokenizer

DATASET_NAME = "Yelp/yelp_review_full"

tokenizer = get_tokenizer()
TOKENIZE_BATCH_SIZE = 1000
DEFAULT_NUM_PROC = max(1, os.cpu_count())


def get_data_collator(static_shapes: bool = False, fixed_seq_len: int | None = None):
    """Build a collator for either dynamic or fixed-shape batching.

    Args:
        static_shapes: If True, enforce a fixed sequence length per batch.
        fixed_seq_len: Target sequence length when `static_shapes=True`.
    """
    if static_shapes:
        if fixed_seq_len is None or fixed_seq_len <= 0:
            raise ValueError("fixed_seq_len must be a positive int when static_shapes=True")
        return DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding="max_length",
            max_length=fixed_seq_len,
        )
    # Dynamic pad-to-longest (default) gives variable seq_len across steps.
    return DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")


def get_dataset():
    """Get the yelp review dataset."""
    return load_dataset(DATASET_NAME)


def tokenize_function(examples, max_length: int | None = None):
    return tokenizer(
        examples["text"],
        padding=False,
        truncation=True,
        max_length=max_length,
    )


def tokenize_data(dataset, max_length: int | None = None):
    """Tokenize the raw dataset.

    Args:
        dataset: The raw dataset to prepare.
        max_length: Optional truncation length.

    Returns:
        The tokenized dataset.
    """
    return dataset.map(
        tokenize_function,
        batched=True,
        batch_size=TOKENIZE_BATCH_SIZE,
        num_proc=DEFAULT_NUM_PROC,
        fn_kwargs={"max_length": max_length},
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


def prepare_data(
    batch_size: int,
    rank: int,
    world_size: int,
    shard_data: bool = True,
    static_shapes: bool = False,
    fixed_seq_len: int | None = None,
    drop_last_train: bool = False,
):
    """Prepare the dataset for training and evaluation.

    Args:
        batch_size: Per-process batch size used by the dataloaders.
        rank: Global rank of current process.
        world_size: Number of distributed processes.
        shard_data: Whether to shard dataset across ranks (DDP style).
        static_shapes: Whether to enforce fixed sequence length in collator.
        fixed_seq_len: Sequence length for static-shape mode.
        drop_last_train: Whether to drop the last incomplete train batch.

    Returns:
        train_loader: DataLoader for training dataset.
        eval_loader: DataLoader for evaluation dataset.
    """
    raw_dataset = get_dataset()
    raw_dataset["train"] = raw_dataset["train"].shuffle(seed=42).select(range(32))
    raw_dataset["test"] = raw_dataset["test"].shuffle(seed=42).select(range(16))
    tokenize_max_length = fixed_seq_len if static_shapes else None
    tokenized_dataset = tokenize_data(raw_dataset, max_length=tokenize_max_length)
    train_ds, eval_ds = split_dataset(tokenized_dataset)

    if rank == 0:
        print(
            f"Dataset sizes -> train: {len(train_ds)} samples, "
            f"eval: {len(eval_ds)} samples (world size={world_size})"
        )
    collator = get_data_collator(static_shapes=static_shapes, fixed_seq_len=fixed_seq_len)
    num_workers = min(8, os.cpu_count() // max(1, world_size))
    use_workers = num_workers > 0
    if shard_data:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True
        )
        eval_sampler = DistributedSampler(
            eval_ds, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        # PP mode: all stages must consume the same samples in the same order.
        train_sampler = DistributedSampler(train_ds, num_replicas=1, rank=0, shuffle=True)
        eval_sampler = DistributedSampler(eval_ds, num_replicas=1, rank=0, shuffle=False)
    train_loader = DataLoader(
        train_ds,
        shuffle=False,
        batch_size=batch_size,
        sampler=train_sampler,
        drop_last=drop_last_train,
        pin_memory=True,
        collate_fn=collator,
        num_workers=num_workers,
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        sampler=eval_sampler,
        pin_memory=True,
        collate_fn=collator,
        num_workers=num_workers,
        persistent_workers=use_workers,
        prefetch_factor=2 if use_workers else None,
    )
    return train_loader, eval_loader
