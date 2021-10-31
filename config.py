class Setting:
    ai_config = {
        "model_name": "roberta-base",
        "num_classes": 1,
        "max_len": 128,
        "batch_size": 32,
        "num_workers": 8,
        "average": "macro",
        "lr": 2e-5,
        "precision": 32,
        "epochs": 1,
    }
    threshold = 0.8
    trainer = None
    model = None
    retrain_dir = "ai_services/checkpoints/epoch_1.ckpt"
    init_model_dir = "ai_services/checkpoints/epoch_1.ckpt"
    host = "0.0.0.0"
    port = 8001


settings = Setting()
