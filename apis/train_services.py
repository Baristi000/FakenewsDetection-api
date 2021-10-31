from config import settings
from ai_services.model_services import init_trainer, create_model, load_weight, save_weight, predict
from ai_services.data_services import FakeNewsDataModule, tokenizer_data


def setup_ai_services():
    settings.trainer = init_trainer()
    settings.mode = load_weight(create_model())


def train(data_list):
    processed_data = FakeNewsDataModule(settings.ai_config, data_list)
    settings.trainer(
        model=settings.model,
        datamodule=processed_data
    )
    save_weight(settings.model)
    return {"message": "train succeed"}


def predict(sentence: str):
    token = tokenizer_data(sentence)
    result = settings.model(
        input_ids=token["input_ids"],
        attention_mask=token["attention_mask"])
    return{
        "status": True if result.item() > settings.threshold else False,
        "percent": result.item()
    }
