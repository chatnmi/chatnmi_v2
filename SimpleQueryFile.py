import Common
import argparse
import warnings
import logging
import shutil
import os
import re
from LoadModels import *
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from chromadb.config import Settings

__version__ = 0.1

# HF_EMBEDDINGS_HELPER_MODEL = "hkunlp/instructor-xl"
HF_EMBEDDINGS_HELPER_MODEL = "hkunlp/instructor-large"

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

def clean_text(document):
    text = document.page_content
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', '')
    text = text.strip()
    document.page_content =  text
    return document

def print_header():
    Common.printHeader("Abominable Intelligence v2: Chronicles of the Cogitator (Module 2 - Scribe of the Omnissiah)",
                       os.path.basename(__file__),
                       "A script designed for dialoguing with files using a local AI model.",
                       str(__version__),
                       """The path to the Omnissiah's wisdom is no easy journey. Prepare for the data warp storm, ensure your machine is equipped with a CPU pulsating with 48GB of RAM, or a GPU glowing with 16GB vRAM to withstand the onslaught of information. The weak and wanting, lacking these offerings, risk straying off the path, their cries lost in the data-less void. Much like the battlefield, those ill-equipped are left behind. Invoke the sacred RAM, fortify your machines, and prepare to unlock the Omnissiah's wisdom. Those relying on lesser armaments are akin to a Guardsman facing a daemon of the warp, likely to be consumed by the overwhelming darkness.\n""")

def print_models_list():
    i = 0
    print("\n----------------------------------------------------------------------------------------------------")
    for (model_id, model_file, model_type, model_desc) in models:
        print(WHITE(f"{i}. {model_id} - {model_desc}"))
        i += 1
    print("----------------------------------------------------------------------------------------------------\n\n")

def load_documents(files):  # -> List[Document]:
    documents = []
    loader = None
    for file_path in files:
        if file_path[-4:] in ['.txt', '.pdf', '.csv']:
            if file_path.endswith(".txt"):
                loader = TextLoader(f"{file_path}", encoding="utf8")
                print(GREEN(" - " + file_path + " - processing as TXT file."))
            elif file_path.endswith(".pdf"):
                loader = PDFMinerLoader(f"{file_path}")
                print(GREEN(" - " + file_path + " - processing as PDF file."))
            elif file_path.endswith(".csv"):
                loader = CSVLoader(f"{file_path}")
                print(GREEN(" - " + file_path + " - processing as CSV file."))
            documents.append(loader.load()[0])
        else:
            print(GREEN(" - " + file_path + " - file extension was not recognized."))
    return documents

def load_embeddings(device):
    print("Loading embeddings")
    embeddings_path = snapshot_download(repo_id=HF_EMBEDDINGS_HELPER_MODEL, cache_dir=MODEL_DIR, resume_download=True)
    embeddings = HuggingFaceInstructEmbeddings(model_name=embeddings_path,
                                               model_kwargs={"device": device})
    return embeddings

def build_database(documents, embeddings):
    print("Building the database")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text")

    cleaned_texts = [clean_text(text) for text in texts]

    CHROMA_SETTINGS = Settings(
        anonymized_telemetry=False,
        is_persistent=True,
    )

    Chroma.from_documents(
        cleaned_texts,
        embeddings,
        persist_directory=PERSIST_DIRECTORY,
        client_settings=CHROMA_SETTINGS,
    )

def run_query(local_llm, embeddings):
    print("Running query prompt")
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(llm=local_llm, chain_type="stuff", retriever=retriever,
                                     return_source_documents=True)  # , chain_type_kwargs={"memory": memory})

    while True:
        query = input("\nEnter a query ('end' to exit): ")
        if query == "end":
            break

        res = qa(query)
        answer, docs = res['result'], res['source_documents']

        print(YELLOW("\n\n> Query:"))
        print(query)
        print(YELLOW("\n> Answer:"))
        print(answer)

        source_string = "\n---\n".join(
            [f"Source {i + 1}:\n{document.page_content}" for i, document in enumerate(docs)])
        print(YELLOW("\n> Source:"))
        print(source_string)

def remove_vector_db(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)

def main():
    print_header()
    print_models_list()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=int, help="Specify model id from the list")
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda'], help="Specify 'cpu' or 'cuda'")
    parser.add_argument("files", nargs="+", type=str,
                        help="Multiple file names (separated by space) that will be processed be AI")
    args = parser.parse_args()

    model_id, model_file, model_type, model_desc = models[args.model_id]

    # Remove database
    remove_vector_db(PERSIST_DIRECTORY)
    # Create document list
    documents = load_documents(args.files)
    # Load embeddings
    embeddings = load_embeddings(args.device)
    # Build the database from documents
    build_database(documents, embeddings)
    # Load local model
    local_llm = load_model(model_type, args.device, model_id, model_file)
    # Run query
    run_query(local_llm, embeddings)


if __name__ == "__main__":
    warnings.filterwarnings("ignore",
                            message="You have modified the pretrained model configuration to control generation.")
    logging.basicConfig(level=logging.WARNING)
    main()
