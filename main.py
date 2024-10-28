from fastapi import FastAPI, UploadFile, File, Query
from typing import List
from indexer import index_documents  
from fastapi.middleware.cors import CORSMiddleware


from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

uploaded_files = []

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost:8000",
    "http://172.132.45.185:5173",
    "https://2622-114-143-119-218.ngrok-free.app/",
    "https://73d1-114-143-119-218.ngrok-free.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload/")
def upload_files(files: List[UploadFile] = File(...), index_name: str = Query(...)):
    global uploaded_files
    results = []
    for file in files:
        results.append({"filename": file.filename})
        uploaded_files.append({"file": file, "index_name": index_name})
        index_documents(file, index_name)
    return {"files": results}

@app.get("/get_uploaded_files/")
def get_uploaded_files():
    global uploaded_files
    results = []
    for item in uploaded_files:
        results.append({"filename": item["file"].filename})
    return {"files": results}
