import fitz
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import marqo
import json
from dotenv import load_dotenv

load_dotenv()

def index_documents(file, index_name, marqo_url="http://3.111.51.0:8883", embedding_model="open_clip/ViT-H-14/laion2b_s32b_b79k", chunk_size=1024, chunk_overlap=200, split_length=1, split_overlap=0, fresh_index=False):
    index_settings = {
        "index_defaults": {
            "model": embedding_model,
            "normalize_embeddings": True,
            "text_preprocessing": {
                "split_length": split_length,
                "split_overlap": split_overlap,
                "split_method": "sentence"
            }
        }
    }

    marqo_client = marqo.Client(url=marqo_url)

    if fresh_index:
        try:
            marqo_client.index(index_name).delete()
            print("Existing Index successfully deleted.")
        except marqo.errors.MarqoWebError as e:
            if e.error_code == "index_not_found":
                print("Index does not exist.")
            else:
                print(f"Error deleting index: {e}")
                return
        except Exception as e:
            print(f"Error deleting index: {e}")
            return

    try:
        marqo_client.create_index(
            index_name, settings_dict=index_settings)
        print(f"Index {index_name} created.")
    except Exception as e:
        print(f"Error creating index: {e}")
        return

    print("Loading document...")
    documents = load_documents(file, chunk_size, chunk_overlap)

    print("Total Documents ===>", len(documents))

    f = open("indexed_documents.txt", "w")
    f.write(str(documents))
    f.close()

    print(f"Indexing document...")
    formatted_documents = get_formatted_documents(documents)
    tensor_fields = ['text']
    _document_batch_size = 50
    chunks = list(chunk_list(formatted_documents, _document_batch_size))
    for chunk in chunks:
        try:
            marqo_client.index(index_name).add_documents(
                documents=chunk, client_batch_size=_document_batch_size, tensor_fields=tensor_fields)
        except marqo.errors.MarqoWebError as e:
            print(f"Error adding documents to index: {e}")

    print("============ INDEX DONE =============")

def load_documents(file, input_chunk_size, input_chunk_overlap):
    source_chunks = []

    with fitz.open(file.file) as pdf:
        for page_number, page in enumerate(pdf):
            text = page.get_text() 
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=input_chunk_size, chunk_overlap=input_chunk_overlap
            )
            for chunk in splitter.split_text(text):
                source_chunks.append(Document(page_content=chunk, metadata={
                    "file_name": file.filename
                }))
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"] 
                image_filename = f"{file.filename}_page_{page_number + 1}_image_{img_index + 1}.png"

                # Save the image
                with open(images/image_filename, "wb") as img_file:
                    img_file.write(image_bytes)

    return source_chunks

def get_formatted_documents(documents):
    docs = []
    for d in documents:
        doc = {
            "text": d.page_content,
            "metadata": json.dumps(d.metadata) if d.metadata else json.dumps({})
        }
        docs.append(doc)
    return docs

def chunk_list(document, batch_size):
    """Return a list of batch sized chunks from document."""
    return [document[i: i + batch_size] for i in range(0, len(document), batch_size)]
