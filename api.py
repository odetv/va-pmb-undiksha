import json
import os
import shutil
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security import APIKeyHeader
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.status import HTTP_404_NOT_FOUND, HTTP_405_METHOD_NOT_ALLOWED
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from typing_extensions import List
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from main import build_graph
from utils.scrapper_rss import scrap_news


load_dotenv()
bearer_token = os.getenv("VA_BEARER_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
DATASETS_DIR = os.getenv("APP_DATASETS_DIR")
VECTORDB_DIR = os.getenv("APP_VECTORDB_DIR")
bearer_token_header = APIKeyHeader(name="Authorization", auto_error=False)


#  Autherization bearer token API
def verify_bearer_token(token: str = Depends(bearer_token_header)):
    if token is None or not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Eitss... mau ngapain? Akses terbatas!")
    actual_token = token[7:]
    if actual_token != bearer_token:
        raise HTTPException(status_code=401, detail="Invalid Token")


# Class API model
class QuestionRequest(BaseModel):
    question: str
class QuestionResponse(BaseModel):
    question: str
    answer: str
class DeleteDatasetsRequest(BaseModel):
    filenames: List[str]
class ProcessRequest(BaseModel):
    embbeder: str
    model: str
    chunksize: int
    chunkoverlap: int


# Metadata API
tags_metadata = [
    {
        "name": "root",
        "description": "Status kondisi service API dengan timestamp."
    },
    {
        "name": "datasets",
        "description": "Proses menyiapkan datasets (CRUD)."
    },
    {
        "name": "setup",
        "description": "Proses load dokumen, chunking, dan embedding."
    },
    {
        "name": "news",
        "description": "Hasil scrapping news."
    },
    {
        "name": "graph",
        "description": "Gambar alur graph terbentuk."
    },
    {
        "name": "chat",
        "description": "Percakapan pengguna dengan virtual assistant."
    },
]


# Initialize FastAPI
app = FastAPI(
    openapi_tags=tags_metadata,
    title="PMB Undiksha",
    summary="API PMB Undiksha.",
    version="0.0.1",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapispmb.json"
)


# Format API response
def api_response(status_code: int, success: bool, message: str, data=None):
    return JSONResponse(
        status_code=status_code,
        content={
            "statusCode": status_code,
            "success": success,
            "message": message,
            "data": data
        }
    )


# Enpoint untuk base url root API request
@app.get("/", tags=["root"])
async def root(token: str = Depends(verify_bearer_token)):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "timestamp": current_time,
        "message": "API Virtual Assistant Undiksha"
    }


# Endpoint untuk melihat daftar file (List)
@app.get("/datasets/list", tags=["datasets"])
async def list_datasets(token: str = Depends(verify_bearer_token)):
    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)
        return api_response(
            status_code=404,
            success=False,
            message="Folder datasets dibuat, tetapi masih kosong.",
            data=None
        )
    datasets = [
        file for file in os.listdir(DATASETS_DIR)
        if file.lower().endswith(('.pdf', '.doc', '.docx'))
    ]
    if not datasets:
        return api_response(
            status_code=404,
            success=False,
            message="Folder datasets kosong atau tidak ada file PDF/Word.",
            data=None
        )
    return api_response(
        status_code=200,
        success=True,
        message="Datasets ditemukan.",
        data=datasets
    )


# Endpoint untuk membaca konten file tertentu (Read)
@app.get("/datasets/read/{filename}", tags=["datasets"])
async def read_datasets(filename: str, token: str = Depends(verify_bearer_token)):
    file_path = os.path.join(DATASETS_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File tidak ditemukan.")
    if filename.lower().endswith(".pdf"):
        def iter_file():
            with open(file_path, "rb") as file:
                yield from file
        return StreamingResponse(iter_file(), media_type="application/pdf", headers={"Content-Disposition": f"inline; filename={filename}"})
    elif filename.lower().endswith((".doc", ".docx")):
        return FileResponse(file_path, media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document", headers={"Content-Disposition": f"attachment; filename={filename}"})
    else:
        raise HTTPException(status_code=400, detail="Format file tidak didukung untuk ditampilkan atau diunduh.")


# Endpoint untuk mengunggah file PDF/Word single atau multiple files (Create)
@app.post("/datasets/upload", tags=["datasets"])
async def upload_datasets(files: List[UploadFile] = File(...), token: str = Depends(verify_bearer_token)):
    uploaded_files = []
    unsupported_files = []
    for file in files:
        if file.content_type not in ["application/pdf", "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            unsupported_files.append(file.filename)
            continue
        file_location = os.path.join(DATASETS_DIR, file.filename)
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)
        uploaded_files.append(file.filename)
    if not uploaded_files:
        raise HTTPException(status_code=400, detail="Tidak ada file yang diunggah atau file yang diunggah tidak didukung.")
    if unsupported_files:
        return api_response(
            status_code=207,
            success=True,
            message="Beberapa file tidak didukung.",
            data={"uploaded_files": uploaded_files, "unsupported_files": unsupported_files}
        )
    else:
        return api_response(
            status_code=201,
            success=True,
            message="Semua file berhasil diunggah.",
            data={"uploaded_files": uploaded_files}
        )


# Endpoint untuk memperbarui (Update) file
@app.put("/datasets/update", tags=["datasets"])
async def update_dataset(
    target: str = Form(...),
    file: UploadFile = File(...),
    token: str = Depends(verify_bearer_token)
):
    old_filename = target
    new_filename = file.filename
    original_file_path = os.path.join(DATASETS_DIR, target)
    if not os.path.isfile(original_file_path):
        raise HTTPException(status_code=404, detail="File tidak ditemukan.")
    new_file_path = os.path.join(DATASETS_DIR, new_filename)
    os.remove(original_file_path)
    with open(new_file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return api_response(
        status_code=200,
        success=True,
        message="File berhasil diperbarui.",
        data={
            "target": old_filename,
            "old_filename": old_filename,
            "new_filename": new_filename
        }
    )


# Endpoint untuk menghapus (Delete) file
@app.delete("/datasets/delete", tags=["datasets"])
async def delete_datasets(request: DeleteDatasetsRequest, token: str = Depends(verify_bearer_token)):
    deleted_files = []
    not_found_files = []
    for filename in request.filenames:
        file_path = os.path.join(DATASETS_DIR, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            deleted_files.append(filename)
        else:
            not_found_files.append(filename)
    if not_found_files:
        return api_response(
            status_code=207,
            success=True,
            message="Beberapa file tidak ditemukan.",
            data={"deleted_files": deleted_files, "not_found_files": not_found_files}
        )
    if not deleted_files:
        raise HTTPException(status_code=400, detail="Tidak ada file yang dipilih.")
    else:
        return api_response(
            status_code=200,
            success=True,
            message="Semua file berhasil dihapus.",
            data={"deleted_files": deleted_files}
        )


# Endpoint setup awal untuk raw process vectordb (load dokumen, chunking, dan embedding)
@app.post("/setup", tags=["setup"])
async def raw_process(request: ProcessRequest, token: str = Depends(verify_bearer_token)):
    # Tambahkan log untuk melihat nilai request yang diterima
    print("Request diterima:", request)
    def get_embbeder():
        if request.embbeder.lower() == "openai":
            return OpenAIEmbeddings(api_key=openai_api_key, model=request.model)
        elif request.embbeder.lower() == "ollama":
            return OllamaEmbeddings(base_url=ollama_base_url, model=request.model)
        else:
            raise HTTPException(status_code=400, detail="Embbeder tidak valid. Gunakan 'openai' atau 'ollama'.")
    if not request.embbeder:
        raise HTTPException(status_code=400, detail="Embbeder harus diisi dengan sesuai.")
    if not request.model:
        raise HTTPException(status_code=400, detail="Model harus diisi dengan sesuai.")
    if request.chunksize < 0:
        raise HTTPException(status_code=400, detail="Chunksize harus diisi nilai positif.")
    if request.chunkoverlap < 0:
        raise HTTPException(status_code=400, detail="Chunkoverlap harus diisi nilai positif.")

    EMBEDDER = get_embbeder()
    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)
    if not os.path.exists(VECTORDB_DIR):
        os.makedirs(VECTORDB_DIR)
    try:
        loader = PyPDFDirectoryLoader(DATASETS_DIR)
        documents = loader.load()
        if not documents:
            return api_response(
                status_code=404,
                success=False,
                message="Tidak ada dokumen PDF di direktori datasets.",
                data=None
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat memuat dokumen: {str(e)}")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=request.chunksize,
            chunk_overlap=request.chunkoverlap
        )
        chunks = text_splitter.split_documents(documents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat melakukan chunking: {str(e)}")
    try:
        vectordb = FAISS.from_documents(chunks, EMBEDDER)
        vectordb.save_local(VECTORDB_DIR)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat menyimpan embeddings: {str(e)}")
    return api_response(
        status_code=200,
        success=True,
        message="Proses dokumen berhasil diselesaikan dan embeddings disimpan.",
        data={
            "embbeder": request.embbeder,
            "model": request.model,
            "chunksize": request.chunksize,
            "chunkoverlap": request.chunkoverlap,
            "total_chunks": len(chunks)
        }
    )


# Endpoint untuk scrapping berita
@app.get("/news", tags=["news"])
async def scrapping_news(token: str = Depends(verify_bearer_token)):
    news_data = scrap_news()
    news_data_json = json.loads(news_data)
    try:
        return api_response(
                status_code=200,
                success=True,
                message="OK",
                data=news_data_json
            )
    except HTTPException as e:
        return api_response(
            status_code=e.status_code,
            success=False,
            message=f"{e.detail}",
            data=None
        )
    except Exception as e:
        return api_response(
            status_code=500,
            success=False,
            message=f"{str(e)}",
            data=None
        )


# Endpoint untuk menampilkan graph
@app.get("/graph", tags=["graph"])
async def visualize_graph(token: str = Depends(verify_bearer_token)):
    file_path = "src/graph/graph-va-pmb-undiksha.png"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    else:
        return api_response(
                status_code=404,
                success=False,
                message="Tidak ditemukan file graph.",
                data=None
            )


# Enpoint untuk chat
@app.post("/chat", tags=["chat"])
async def chat_conversation(request: QuestionRequest, token: str = Depends(verify_bearer_token)):
    question = request.question
    if not request.question:
        raise HTTPException(status_code=400, detail="Pertanyaan tidak boleh kosong.")
    try:
        answer = build_graph(question)
        return api_response(
            status_code=200,
            success=True,
            message="OK",
            data=[{
                "question": question,
                "answer": answer
            }]
        )
    except HTTPException as e:
        return api_response(
            status_code=e.status_code,
            success=False,
            message=f"{e.detail}",
            data=None
        )
    except Exception as e:
        return api_response(
            status_code=500,
            success=False,
            message=f"{str(e)}",
            data=None
        )


# Custom handler untuk 404 Not Found
@app.exception_handler(HTTP_404_NOT_FOUND)
async def not_found_handler(request: Request, exc: StarletteHTTPException):
    return api_response(
        status_code=404,
        success=False,
        message="Not Found",
        data=None
    )


# Custom handler untuk 405 Method Not Allowed
@app.exception_handler(HTTP_405_METHOD_NOT_ALLOWED)
async def method_not_allowed_handler(request: Request, exc: StarletteHTTPException):
    return api_response(
        status_code=405,
        success=False,
        message="Method Not Allowed",
        data=None
    )


# General handler untuk HTTP Exception lain
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return api_response(
        status_code=exc.status_code,
        success=False,
        message=f"{exc.detail}",
        data=None
    )


# General handler untuk validasi error
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    error_messages = "; ".join([f"{err['loc']}: {err['msg']}" for err in errors])
    return api_response(
        status_code=422,
        success=False,
        message=f"{error_messages}",
        data=None
    )


# RUNNING API
# uvicorn api:app --reload --port 1014