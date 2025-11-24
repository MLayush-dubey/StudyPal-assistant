import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


#loading env variables
load_dotenv()
CLASS_SUBJECT_NAME = os.getenv("CLASS_SUBJECT_NAME")
DEVICE = os.getenv("DEVICE", 'cuda')

#in 2 steps--> We are basically moving one level up the directory
working_dir = os.path.dirname(os.path.abspath(__file__)) #dirname removes the last component. basically we grab the folder based on our current file and then cut out this file
parent_dir = os.path.dirname(working_dir) #now remove the last component which is the folder which is saving this file. now this is the main parent directory
data_dir = f"{parent_dir}/data"
vector_db_dir = f"{parent_dir}/vector_db"
chapters_vector_db_dir = f"{parent_dir}/chapters_vector_db"

embedding = HuggingFaceEmbeddings(model_kwargs={'device': DEVICE})
text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=500)


def vectorize_book_and_store_to_db(class_subject_name, vector_db_name):
    book_dir = f"{data_dir}/{class_subject_name}" #points to the directory containing PDF and adds subj name
    vector_db_path = f"{vector_db_dir}/{vector_db_name}" #where the vector db is saved
    loader = DirectoryLoader(path=book_dir, glob="./*.pdf", loader_cls=UnstructuredFileLoader)
    documents = loader.load()
    text_chunks = text_splitter.split_documents(documents)
    Chroma.from_documents(documents=text_chunks, embedding=embedding, persist_directory=vector_db_path)
    print(f"{class_subject_name} saved to vector db: {vector_db_name}")
    return 0


def vectorize_chapters(class_subject_name):
    book_dir = f"{data_dir}/{class_subject_name}"
    for chapter in os.listdir(book_dir):
        if not chapter.endswith('.pdf'):
            continue
        chapter_name = chapter[:-4]
        chapter_pdf_path = f"{book_dir}/{chapter}"
        loader = UnstructuredFileLoader(chapter_pdf_path)
        documents = loader.load()
        texts = text_splitter.split_documents(documents)
        Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=f"{chapters_vector_db_dir}/{chapter_name}")
        print(f"{chapter_name} chapter vectorized")
    return 0
