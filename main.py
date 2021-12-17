from fastapi import FastAPI
import uvicorn
import torch
import nest_asyncio
from pyngrok import ngrok, conf

from apis.__init__ import api_router
from apis.train_services import setup_ai_services
from config import settings


app = FastAPI()


@app.on_event("startup")
def init():
    setup_ai_services()


app.include_router(api_router)


@app.get("/")
def hello_world():
    return {"message": "Hello from fake new detetction server"}


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    # Tunnel
    ''' conf.get_default().region = "ap"
    ngrok.set_auth_token("22NQbE6s24xQQlAier9ASODJyKs_4TbPDyYZfz8d1iDWnoPbg")
    ngrok_tunnel = ngrok.connect(settings.port)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply() '''
    uvicorn.run(app, host=settings.host, port=settings.port)
