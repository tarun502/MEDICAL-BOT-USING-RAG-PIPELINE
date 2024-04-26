from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS 

DATA_PATH = r"C:\Users\tarun\llama\The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf"  # Update this to the directory containing your PDF files

DB_FAISS_PATH = "vectorstores/df_faiss"

def create_vector_db():
    loader = PyPDFLoader(file_path=DATA_PATH)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == '__main__':
    create_vector_db()


