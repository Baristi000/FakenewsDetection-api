import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from config import settings


def tokenizer_data(text: str):
    return AutoTokenizer.from_pretrained(settings.ai_config['model_name'], torchscript=True).encode_plus(
        text=text,
        truncation=True,
        max_length=128,
        return_tensors="pt",
        return_attention_mask=True,
        padding="max_length",
    )


class NewsDataset(Dataset):

    "Custom Dataset class to create the torch dataset"

    def __init__(self, text_datas):
        self.text = text_datas
        self.label = []
        [self.label.append(1) for i in range(len(self.text))]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.label[idx]

        input_encoding = tokenizer_data(text)

        return {
            "input_ids": input_encoding['input_ids'].squeeze().clone().to(settings.device).detach(),
            "attention_mask": input_encoding['attention_mask'].squeeze().clone().to(settings.device).detach(),
            "label": torch.tensor([label], dtype=torch.float).to(settings.device)
        }


class FakeNewsDataModule(pl.LightningDataModule):

    """Lightning Data Module to detach data from model"""

    def __init__(self, config, text):
        """
            config: a dicitonary containing data configuration such as batch size, split_size etc
        """
        super().__init__()
        self.config = config
        self.dataset = NewsDataset(text)

    def train_dataloader(self):
        return DataLoader(dataset=self.dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'])
