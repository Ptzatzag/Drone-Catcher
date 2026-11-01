from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from huggingface_hub import hf_hub_download
import torch
from .inference import predict
from model.maskrcnn import load_model
from contextlib import asynccontextmanager
from fastapi import Request
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil

from fastapi import HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from .video import predict_video



@asynccontextmanager
async def lifespan(app: FastAPI):
    weight_path = hf_hub_download(
        repo_id="Ptzatzag/drone-catcher",
        filename="drone_weights.pth/Best_Model.pth"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.model = load_model(weight_path, device)

    yield  

    del app.state.model 


app = FastAPI(
    title="Drone Catcher",
    lifespan=lifespan,
)

@app.get('/')
def health_check():
    return {'status': 'ok'}

@app.post('/predict')
async def predict_damage(request: Request, file: UploadFile = File(...)):
    
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = predict(image, request.app.state.model)
    return result

@app.post("/predict/video")
def predict_uploaded_video(
    request: Request,
    file: UploadFile = File(...),
):
    temporary_directory = TemporaryDirectory()
    directory = Path(temporary_directory.name)

    input_path = directory / "input_video"
    output_path = directory / "predictions.mp4"

    try:
        with input_path.open("wb") as destination:
            shutil.copyfileobj(file.file, destination)

        predict_video(
            input_path,
            output_path,
            request.app.state.model,
        )

        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename="predictions.mp4",
            background=BackgroundTask(temporary_directory.cleanup),
        )

    except ValueError as error:
        temporary_directory.cleanup()
        raise HTTPException(status_code=400, detail=str(error))

    except Exception:
        temporary_directory.cleanup()
        raise

    finally:
        file.file.close()

        