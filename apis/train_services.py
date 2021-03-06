from config import settings
from ai_services.model_services import init_trainer, create_model, load_weight, save_weight
from ai_services.data_services import FakeNewsDataModule, tokenizer_data


def setup_ai_services():
    settings.ckpt_load_dir = settings.get_retrain_model_dir()
    settings.trainer = init_trainer()
    if settings.ckpt_load_dir == "":
        settings.model = create_model()
    else:
        settings.model = load_weight(create_model())


def train(data_list):
    settings.trainer.fit(
        model=settings.model,
        datamodule=FakeNewsDataModule(settings.ai_config, data_list)
    )
    return {"message": "train succeed"}


def backup_checkpoint():
    save_weight()
    return {"message": "train succeed"}


def predict(sentence: str):
    token = tokenizer_data(sentence)
    try:
        settings.model = settings.model.to(settings.device)
    except:
        pass
    result = settings.model(
        input_ids=token["input_ids"].to(settings.device),
        attention_mask=token["attention_mask"].to(settings.device))
    return{
        "status": True if result.item() > settings.threshold else False,
        "percent": result.item()
    }
