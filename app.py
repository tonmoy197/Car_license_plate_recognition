from fastapi import FastAPI, File, UploadFile
from contextlib import asynccontextmanager
import base64
from service import NumberPlateService, NumberPlateRequest

service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = NumberPlateService()
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/recognize")
async def recognize_plate(image: UploadFile = File(...)):
    image_bytes = await image.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    request = NumberPlateRequest(image_base64=image_base64)
    result = service.recognize_plate(request)
    return {"area": result.area, "number": result.number}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8010, reload=True)