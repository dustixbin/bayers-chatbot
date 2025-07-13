from sqlmodel import Field, SQLModel
from typing import Optional

class PDFS(SQLModel, table=True):
    pdf_file_name: str = Field(index=True, primary_key=True)
    pdf_uuid: str = Field(index=True)

class Image(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    document_id: str = Field(index=True)
    image_id: str = Field(index=True)
    image_b64: str
