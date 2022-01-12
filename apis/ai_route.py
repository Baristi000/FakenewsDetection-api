from fastapi import APIRouter, Body

from apis.response_base import ResponseDto
from apis.train_services import train, predict, backup_checkpoint


router = APIRouter()
response = ResponseDto()


@router.post(
    "/train",
    description="Feed data to model"
    # responses=exampleResponse.post_test_crawler
)
def trainer(data: list = Body(..., embed=True)):
    try:
        return response.success(train(data))
    except:
        return response.bad_request("train error")


@router.get(
    "/backup-checkpoint",
    description="Save checkpoint to \"ai_services/checkpoints/current.ckpt\"")
def checkpoint():
    return response.success(backup_checkpoint())


@router.post(
    "/predict",
    description="Detect if data is true  or not")
def predicter(sentence: str = Body(..., embed=True)):
    try:
        return response.success(predict(sentence))
    except:
        return response.bad_request("predict error")
