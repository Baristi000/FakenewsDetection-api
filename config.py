import os
import multiprocessing


class Setting:
    ai_config = {
        "model_name": "roberta-base",
        "num_classes": 1,
        "max_len": 128,
        "batch_size": 32,
        "num_workers": multiprocessing.cpu_count(),
        "average": "macro",
        "lr": 2e-5,
        "precision": 32,
        "epochs": 10,
    }
    threshold = 0.75
    trainer = None
    model = None
    ckpt_save_dir = "ai_services/checkpoints/current.ckpt"
    ckpt_load_dir = None
    host = "0.0.0.0"
    port = 8001

    def get_retrain_model_dir(self):
        dir = "ai_services/checkpoints"
        checkpoints = (os.listdir(dir))
        for f in checkpoints:
            if ".ckpt" in f:
                dir += "/"+f
        if ".ckpt" not in dir:
            dir = ""
            print("No initialized model")
        else:
            print("Existed initialized model")
        return dir
    device = None


settings = Setting()
