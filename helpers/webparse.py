import requests
from bs4 import BeautifulSoup
import base64

def extract_webpage(url: str):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    data = {}
    text = soup.get_text(separator=" ", strip=True)
    images = []
    for img in soup.find_all("img"):
        src = img.get("src")
        # Handle relative URLs
        if src and not src.startswith("http"):
            src = requests.compat.urljoin(url, src)
        try:
            img_resp = requests.get(src)
            image_bytes = img_resp.content
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            # Guess MIME type from extension
            mime = "image/jpeg" if src.lower().endswith((".jpg", ".jpeg")) else "image/png" if src.lower().endswith(".png") else "image"
        except Exception:
            image_b64 = None
            mime = "image"
        images.append({"filename": src, "mime": mime, "base64": image_b64})
    tables = []
    for table in soup.find_all("table"):
        tables.append({"md": str(table)})
    data["page_1"] = {"text": text, "images": images, "tables": tables}
    return data