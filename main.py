from fastapi import FastAPI
import uvicorn
import nest_asyncio
from pyngrok import ngrok

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
    ngrok_tunnel = ngrok.connect(settings.port)
    print('Public URL:', ngrok_tunnel.public_url)
    nest_asyncio.apply()
    uvicorn.run("main:app", host=settings.host, port=settings.port)
