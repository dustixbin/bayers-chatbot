from llama_index.core import Document
from llama_index.core.node_parser.text.token import TokenTextSplitter
from llama_index.core.llms import ChatMessage, ImageBlock
from llama_index.llms.openai import OpenAI
import base64
from models import Image
import uuid
from sqlmodel import Session

# load_dotenv()
 
def text_n_images(data, document_id,session :Session):
    documents = []
    splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=100)
    for component in data:
        # Assume 'text' is the field with the content
        text = data[component]['text']
        chunks = splitter.split_text(text)
        for chunk in chunks:
            doc = Document(text=chunk, metadata={'page_label':component, "type":"text"})  # Optionally attach metadata
            documents.append(doc)
 
    img_docs = []
    for component in data:
        # Assume 'text' is the field with the content
        images = data[component]['images']
        # chunks = splitter.split_text(text)
        # print(images)
        # for chunk in chunks:
        for img in images:
            if not img.get("base64"):
                continue
            msg = ChatMessage("Please capture every detail in the image and describe in atmost 200 words. It should be concise and should contain the semantic meaning of the image")
            image_bytes = base64.b64decode(img['base64'])  # decode base64 to bytes
            detected_type = "jpg"
            # if detected_type not in ["png", "jpeg", "jpg"]:
            #     print(f"Skipping unsupported image type: {detected_type}")
            #     continue
            mime_type = f"image/{'jpeg' if detected_type == 'jpg' else detected_type}"
            msg.blocks.append(ImageBlock(image=image_bytes, mime_type=img["mime"]))
            desc = OpenAI(model="gpt-4o").chat([msg])
            image_uuid = str(uuid.uuid4())
            doc = Document(text=desc.message.content, metadata={'page_label': component, "type": "image", "image_uuid": image_uuid})
            img_docs.append(doc)
            img_record = Image(document_id=document_id,image_id= image_uuid, image_b64=img["base64"])
            session.add(img_record)
    documents.extend(img_docs)
    session.commit()
 
    table_docs = []
    for component in data:
        # Assume 'text' is the field with the content
        table_base64 = data[component]['tables']
        # print(table_base64)
        if table_base64:
            # Suppose table_base64 is your base64 string
            for table in table_base64:
                md_str = table["md"]
                table_docs.append(Document(text=str(md_str), metadata={'page_label': component, "type": "table"}))
    documents.extend(table_docs)
    return documents
 
if __name__ == "__main__":
    with open('extracted.json', 'r') as f:
        data = json.load(f)
    docs = text_n_images(data)