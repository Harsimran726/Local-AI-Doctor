"""LOCAL AI DOCTOR """


from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates 
from pydantic import BaseModel, Field, ValidationError , WrapSerializer
from typing import List, Optional
import uvicorn 
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import json 
import os 
import re 
from datetime import datetime 
from fastapi.middleware.cors import CORSMiddleware
# from fastapi_utilities.cache import cache 
# from fastapi_utilities.timer import timer


# APP SETUP 
BASE_DIR = Path(__file__).resolve().parent 

app = FastAPI(
    title="Local AI Doctor",
    description="A local AI doctor that can diagnose and provide medical advice based on symptoms.",
    version="1.0.0",
)

# mount static files 
app.mount("/static",StaticFiles(directory=BASE_DIR / "static"), name="static")

# load jinja2 templates 
templates = Jinja2Templates(directory=BASE_DIR / "templates")



@app.get("/", response_class=HTMLResponse)
async def home(request:Request):

    return templates.TemplateResponse("index.html",{"request":request})