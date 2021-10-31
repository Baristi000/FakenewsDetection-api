from fastapi import APIRouter
from apis import ai_route

api_router = APIRouter()

api_router.include_router(
    ai_route.router,
    prefix='/FakenewsDetection',
    tags=['Fake news Detection']
)
