from collections import Counter
from functools import partial
import random
import datasets

# Use smaller writer batch size for large datasets to avoid OOM.
LARGE_DATASET_WRITER_BATCH_SIZE = 1000

# Define the dataset columns
DS_COLUMNS = {"question", "solution", "cot_type", "source_type", "metadata"}

### Load functions ###
def load_math():
    ds = datasets.load_dataset("simplescaling/s1K-1.1", trust_remote_code=True)["train"]
    ds = ds.map(lambda x: {"question": x.pop("question"), "solution": x.pop("solution"), "cot_type": "math", "source_type": "simplescaling/s1K-1.1", "metadata": str(x)},
                writer_batch_size=LARGE_DATASET_WRITER_BATCH_SIZE)
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_numinamath():
    ds = datasets.load_dataset("AI-MO/NuminaMath-1.5", trust_remote_code=True)["train"]
    ds = ds.map(lambda x: {"question": x.pop("problem"), "solution": x.pop("solution"), "cot_type": "math", "source_type": "AI-MO/NuminaMath-1.5/"+ x["source"], "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

### Dataset processing ###
DS_TO_SELECTION = {
    "MATH": [load_math, None, None],
    "NuminaMath-1.5": [load_numinamath, None, None]  
}

if __name__ == "__main__":
    random.seed(42)

    ds_all = []
    for ds_name, (load_fn, selection_fn, n_samples) in DS_TO_SELECTION.items():
        print(f"Processing {ds_name}...")
        ds = load_fn()
        ds_all.append(ds)

    ds = datasets.concatenate_datasets(ds_all)
    ds = ds.map(lambda x: {"cot": None, **x})

    # Deduplication
    memory = set()
    def is_unique(elem, column, memory):
        if elem[column] in memory:
            return False
        memory.add(elem[column])
        return True

    ds = ds.filter(partial(is_unique, column="question", memory=memory))
    ds.push_to_hub("BeastGokul/s2")

