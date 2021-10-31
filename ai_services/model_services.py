from pickle import NONE
import torch.nn as nn
from transformers import AutoModel
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, f1_score
import torch

from config import settings
from ai_services.data_services import tokenizer_data


class Model(nn.Module):

    """ 
        Fake News Classifier Model
        A pretrained model is used as for contextualized embedding and a classifier on top of that. 

    """

    def __init__(self, model_name, num_classes=1):
        """
            model_name:  What base model to use from hugginface transformers
            num_classes: Number of classes to classify. This is simple binary classification hence 2 classes
        """
        super().__init__()

        # pretrained transformer model as base
        self.base = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name)

        # nn classifier on top of base model
        self.classfier = nn.Sequential(*[
            nn.Linear(in_features=768, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=num_classes),
            nn.Sigmoid()
        ])

    def forward(self, input_ids, attention_mask=None):
        """
            input_ids: input ids tensors for tokens  shape = [batch_size, max_len]
            attention_mask: attention for input ids, 0 for pad tokens and 1 for non-pad tokens [batch_size, max_len]

            returns: logits tensors as output, shape = [batch, num_classes]
        """
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        pooler = outputs[1]
        logits = self.classfier(pooler)
        return logits


class LightningModel(pl.LightningModule):

    """
        LightningModel as trainer model
    """

    def __init__(self, config):

        super(LightningModel, self).__init__()
        self.config = config
        self.model = Model(
            model_name=self.config['model_name'], num_classes=self.config['num_classes'])

    def forward(self, input_ids, attention_mask=None):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return logits.squeeze()

    def configure_optimizers(self):
        return optim.AdamW(params=self.parameters(), lr=self.config['lr'])

    def training_step(self, batch, batch_idx):

        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['label'].squeeze(
        )
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.mse_loss(logits, targets)

        pred_labels = logits.cpu() > 0.5  # logits.argmax(dim=1).cpu() for non-sigmoid
        acc = accuracy_score(targets.cpu(), pred_labels)
        f1 = f1_score(targets.cpu(), pred_labels,
                      average=self.config['average'])
        return {"loss": loss, "accuracy": acc, "f1_score": f1}


def create_model():
    return LightningModel(settings.ai_config)


def load_weight(model):
    l = torch.load(f=settings.retrain_dir)
    return model.load_state_dict(l['state_dict'])


def save_weight(model):
    torch.save(model.state_dict(), settings.retrain_dir)


def predict(model, text: str):
    token = tokenizer_data(text)
    input_ids = token['input_ids']
    attention_mask = token['attention_mask']
    result = model(input_ids=input_ids, attention_mask=attention_mask)
    return result


def init_trainer():
    trainer = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.manual_seed(42)
        trainer = pl.Trainer(
            gpus=[0],
            max_epochs=settings.ai_config["epochs"],
            precision=settings.ai_config["precision"]
        )
        print("\tCuda core is available")
    else:
        device = torch.device("cpu")
        torch.manual_seed(42)
        trainer = pl.Trainer(
            max_epochs=settings.ai_config["epochs"],
            precision=settings.ai_config["precision"]
        )
        print("\tCuda core is unavailable, use cpu instead")

    return trainer
