from fastapi import APIRouter, Body

from apis.response_base import ResponseDto
from apis.train_services import train, predict


router = APIRouter()
response = ResponseDto()


@router.post(
    "/train",
    description="Test crawling one url"
    # responses=exampleResponse.post_test_crawler
)
def trainer(data: list = Body(..., embed=True)):
    # try:
    return response.success(train(data))
    # except:
    #    return response.bad_request("train error")


@router.post("/predict")
def predicter(sentence: str = Body(..., embed=True)):
    # try:
    return response.success(predict(sentence))
    ''' except:
        return response.bad_request("predict error") '''
