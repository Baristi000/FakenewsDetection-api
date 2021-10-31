from config import settings
from ai_services.model_services import init_trainer, create_model, load_weight, save_weight, predict
from ai_services.data_services import FakeNewsDataModule, tokenizer_data


def setup_ai_services():
    settings.trainer = init_trainer()
    if settings.ckpt_load_dir == "":
        settings.mode = create_model()
    else:
        settings.mode = load_weight(create_model())


def train(data_list):
    settings.trainer.fit(
        model=settings.model,
        datamodule=FakeNewsDataModule(settings.ai_config, data_list)
    )
    save_weight()
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
