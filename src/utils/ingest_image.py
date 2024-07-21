import os
import fitz 
import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction # type: ignore

def extract_and_store_images(pdf_path, db_path='image_vdb', images_dir='extracted_images'):
    # Step 1: Extract images from PDF
    pdf_document = fitz.open(pdf_path)
    os.makedirs(images_dir, exist_ok=True)

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)

        for image_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image_filename = f"{images_dir}/page_{page_num+1}_img_{image_index+1}.{image_ext}"

            with open(image_filename, "wb") as image_file:
                image_file.write(image_bytes)
            print(f"Saved: {image_filename}")

    print("Image extraction complete.")

    # Step 2: Add extracted images to ChromaDB
    chroma_client = chromadb.PersistentClient(path=db_path)
    image_loader = ImageLoader()
    CLIP = OpenCLIPEmbeddingFunction()
    image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function=CLIP, data_loader=image_loader)

    ids = []
    uris = []

    for i, filename in enumerate(sorted(os.listdir(images_dir))):
        if filename.endswith('.jpeg') or filename.endswith('.png'):
            file_path = os.path.join(images_dir, filename)
            ids.append(str(i))
            uris.append(file_path)

    image_vdb.add(ids=ids, uris=uris)
    print("Images added to the database.")

    return image_vdb

# Example usage
