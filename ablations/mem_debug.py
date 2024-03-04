import psutil
import os
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader, Dataset
import pyarrow

BATCH_SIZE=1
NUM_TRIES = 10000


class PreTokenizedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = iter(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return next(self.dataset)

# dataset = load_from_disk('/weka/home-griffin/clinical_pile/v1/tokenized/dataset_hf_pubmed', keep_in_memory=True)
dataset = load_dataset('medarc/pubmed', split='train', streaming=True)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE)
mem_before = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
count = 0
for batch in enumerate(dataset):
    mem_after = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(count, mem_before, mem_after - mem_before)
    count += 1
