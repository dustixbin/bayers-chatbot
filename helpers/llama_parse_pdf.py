import re
from llama_cloud_services import LlamaParse
from dotenv import load_dotenv
import base64
import os
load_dotenv(".env_pgvector")

async def extract_pdf_llamaparse(filename: str):
    parser = LlamaParse(
        api_key=os.getenv("LLAMAPARSE_API_KEY"),
        num_workers=1,
        verbose=True,
        language="en",
        result_type="json",
    )
    num_tables = 0
    num_images = 0
    result = await parser.aparse(filename)
    json_data = {}
    for page in result.pages:
        page_key = "page_" + str(page.page)
        text = page.text
        text = re.sub(r"\s{2,}", " ", text)

        image_list = []
        for image in page.images:
            try:
                image_data = await result.aget_image_data(image.name)
                print(type(image_data))
                image_b64 = base64.b64encode(image_data).decode("utf-8")
                image_list.append({"filename": image.name, "mime": "image/"+str(image.name).split(".")[-1], "base64":image_b64})
                num_images += 1
            except Exception as e:
                print(f"Failed to fetch image {image.name}: {e}")

        table_list = []
        for item in page.items:
            if item.type == "table":
                table_list.append({"md": item.md})
                num_tables += 1

        json_data[page_key] = {"text": text, "images": image_list, "tables": table_list}

    return json_data, num_tables, num_images

if __name__ == "__main__":
    data, num_tables, num_images = extract_pdf_llamaparse("/Users/gagan/Documents/bayers-usecase/data/s41598-024-83090-3 (1).pdf")
    print(data)
    print(f"Total tables: {num_tables}")
    print(f"Total images: {num_images}")