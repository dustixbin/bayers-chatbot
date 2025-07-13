from fastapi import FastAPI, File, UploadFile, Depends
from sqlmodel import Session, select
from dotenv import load_dotenv
from db import init_db
from typing import List, Dict
import tempfile
import shutil
from helpers.retriver import Retriver
import uuid
from models import PDFS, Image
from helpers.generator import generate_response
from db import get_session
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
from starlette.concurrency import run_in_threadpool
from helpers.webpage_parser import extract_webpage

load_dotenv(".env_pgvector")

class RAGRequest(BaseModel):
    query: str
    document_ids: List[str]
    chad_history: List[Dict]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/upload_pdfs", tags=["pdf"])
async def upload_pdfs(files: List[UploadFile] = File(...), session: Session = Depends(get_session)):
    response = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            print("temp_file")
            shutil.copyfileobj(file.file, tmp)
            temp_file_path = tmp.name
        try:
            tmp_id = "".join(str(uuid.uuid4()).split("-"))
            obj = Retriver(document_id=tmp_id, path=temp_file_path)
            await obj.upsert(session)
            pdf = PDFS(pdf_file_name=file.filename, pdf_uuid=tmp_id)
            await run_in_threadpool(session.add, pdf)
            await run_in_threadpool(session.commit)
            response.append({"filename": file.filename, "file_uuid": tmp_id})
        except ValueError as e:
            print(e)
        finally:
            await run_in_threadpool(os.remove, temp_file_path)
    return response

@app.get("/get_all_pdfs", tags=["pdf"])
def get_all_pdfs(session: Session = Depends(get_session)):
    all_pdfs = session.execute(select(PDFS))
    all_pdfs = all_pdfs.scalars().all()
    return [PDFS(pdf_file_name = pdf.pdf_file_name, pdf_uuid=pdf.pdf_uuid) for pdf in all_pdfs]


@app.post("/upload_webpage", tags=["webpage"])
async def upload_webpage(url: str, session: Session = Depends(get_session)):
    data = extract_webpage(url)
    tmp_id = "".join(str(uuid.uuid4()).split("-"))
    docs = text_n_images(data, tmp_id, session)
    obj = Retriver(document_id=tmp_id)
    storage_context = StorageContext.from_defaults(vector_store=obj.vector_store)
    VectorStoreIndex.from_documents(
        docs,
        embed_model=obj.embedding_model,
        vector_store=obj.vector_store,
        storage_context=storage_context,
        show_progress=True
    )
    pdf = PDFS(pdf_file_name=url, pdf_uuid=tmp_id)
    session.add(pdf)
    session.commit()
    return {"url": url, "file_uuid": tmp_id}

@app.delete("/delete_pdf/{pdf_uuid}", tags=["pdf"])
def delete_pdf(pdf_uuid: str, session: Session = Depends(get_session)):
    pdf_to_delete = session.exec(select(PDFS).where(PDFS.pdf_uuid == pdf_uuid)).first()
    if pdf_to_delete:
        # 🔥 Delete vector collection from PGVector
        retriever = Retriver(document_id=pdf_uuid)
        retriever.delete_collection()

        # 🗑️ Delete DB record
        session.delete(pdf_to_delete)
        session.commit()
        return {"message": f"PDF '{pdf_uuid}' and its embeddings deleted successfully."}
    return {"error": f"No PDF found with UUID '{pdf_uuid}'."}

@app.delete("/delete_all_pdfs", tags=["pdf"])
def delete_all_pdfs(session: Session = Depends(get_session)):
    all_pdfs = session.exec(select(PDFS)).all()
    for pdf in all_pdfs:
        retriever = Retriver(document_id=pdf.pdf_uuid)
        retriever.delete_collection()  # delete from PGVector
        session.delete(pdf)
    session.commit()
    return {"message": "All PDFs and their embeddings deleted successfully."}


@app.post("/get_response", tags=["rag"])
def get_response(request: RAGRequest, session: Session = Depends(get_session)):
    print("Query:", request.query)
    print("Document IDs:", request.document_ids)

    relevant_docs = []

    # === CASE 1: GENERAL CHAT ===
    if not request.document_ids:
        # General LLM mode — no documents
        ans = generate_response(docs=[], query=request.query, chad_history=request.chad_history)
        return {
            "response": ans,
            "relevant_docs": [],
            "image_list": []
        }


    # === CASE 2: DOCUMENT CHAT ===
    for doc_id in request.document_ids:
        obj = Retriver(document_id=doc_id)
        relevant_docs += obj.similarity_search(request.query)

    relevant_docs = [doc for doc in relevant_docs if doc.score >= 0.15]
    relevant_docs.sort(key=lambda x: x.score, reverse=True)

    ans = generate_response(docs=relevant_docs, query=request.query, chad_history=request.chad_history)
    ans_json = json.loads(ans)

    image_list = []
    if ans_json.get("image"):
        image_uuid = ans_json["image"]
        image = session.exec(select(Image).where(Image.image_id == image_uuid)).first()
        if image:
            image_list.append(image.image_b64)

    citation_list = ans_json.get("citation", [])
    final_relevant_docs = []

    if relevant_docs:
        scores = [doc.score for doc in relevant_docs]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score if max_score != min_score else 1

        for citation in citation_list:
            for doc in relevant_docs:
                if doc.node_id == citation:
                    normalized_score = (doc.score - min_score) / score_range
                    final_relevant_docs.append({
                        "text": doc.text,
                        "score": normalized_score
                    })

    return {
        "response": ans,
        "relevant_docs": final_relevant_docs,
        "image_list": image_list
    }