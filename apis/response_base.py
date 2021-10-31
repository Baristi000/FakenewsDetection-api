from datetime import datetime


class ResponseDto():
    def success(self, data: dict):
        return{
            "status": 200,
            "time": datetime.now(),
            "data": data
        }

    def bad_request(self, message: str):
        return{
            "status": 400,
            "time": datetime.now(),
            "message": message
        }

    def create(self, data: dict):
        return{
            "status": 201,
            "time": datetime.now(),
            "data": data
        }

    def genExampleResponse(self, succeed_body: dict = None, error_body: dict = None, create_body=None):
        base_response = {
            200: {
                "description": "success",
                "content": {
                    "application/json": {
                        "examples": {}
                    }
                }
            },
        }
        if succeed_body != None:
            base_response[200]["content"]["application/json"]["examples"].update(
                {
                    "success": {
                        "summary": "success",
                        "value": succeed_body
                    }
                }
            )
        if error_body != None:
            base_response[200]["content"]["application/json"]["examples"].update(
                {
                    "error": {
                        "summary": "error",
                        "value": error_body
                    }
                }
            )
        if create_body != None:
            base_response[200]["content"]["application/json"]["examples"].update(
                {
                    "create": {
                        "summary": "create",
                        "value": create_body
                    }
                }
            )
        return base_response
