import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import os

def extract_text(pdf_path):
    """Extract text from a PDF file"""
    document = fitz.open(pdf_path)
    text = ""
    # iterate through each page and extract
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# Function to extract and save each PDF page as an image
def extract_page_images(pdf_path, pages = True, output_folder = "images"):
    document = fitz.open(pdf_path)
    os.makedirs(output_folder, exist_ok=True)  # Ensure the output folder exists
    image_paths = []

    # Convert each page to an image and save as PNG
    if pages:
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            pix = page.get_pixmap()
            image = Image.open(io.BytesIO(pix.tobytes("png")))

            # Save the image as a PNG file
            output_path = os.path.join(output_folder, f"page_{page_num + 1}.png")
            image.save(output_path, format="PNG")

            # Store the file path
            image_paths.append(output_path)
    else:
        # Iterate over each page in the PDF
        for page_num in range(len(document)):
            page = document.load_page(page_num)
            image_list = page.get_images(full=True)

            # Iterate over each image on the page
            for img_index, img in enumerate(image_list):
                xref = img[0]  # XREF of the image
                base_image = document.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))

                # Save the image as a PNG file
                output_path = os.path.join(output_folder, f"page_{page_num + 1}_image_{img_index + 1}.png")
                image.save(output_path, "PNG")

                # Store the file path
                image_paths.append(output_path)
            
    return image_paths

# Function to convert image at a given path to base64 encoding
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_string

def parse_pdf(pdf_path, pages = True):
    # Extract text
    text = extract_text(pdf_path)
    images = extract_page_images(pdf_path, pages = pages)

    return text, images




