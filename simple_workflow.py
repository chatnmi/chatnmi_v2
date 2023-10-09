from langchain.embeddings import HuggingFaceInstructEmbeddings
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from huggingface_hub import snapshot_download
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PDFMinerLoader
from langchain.prompts import PromptTemplate
import logging
from chromadb.config import Settings

logging.basicConfig(level=logging.INFO)
MODEL_DIR = r".\models"

# Loading file
loader = PDFMinerLoader("war-and-peace.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Building Chroma database
embeddings_path = snapshot_download(repo_id="hkunlp/instructor-large", cache_dir=MODEL_DIR, resume_download=True)
embeddings = HuggingFaceInstructEmbeddings(model_name=embeddings_path, model_kwargs={"device": "cuda"})

db = Chroma.from_documents(
    texts,
    embeddings,
    client_settings=Settings(anonymized_telemetry=False)
)
retriever = db.as_retriever()

# Loading model and creating pipeline
tokenizer = AutoTokenizer.from_pretrained("TheBloke/vicuna-13B-v1.5-GPTQ", cache_dir=MODEL_DIR, device_map="auto")
model = AutoModelForCausalLM.from_pretrained("TheBloke/vicuna-13B-v1.5-GPTQ", cache_dir=MODEL_DIR, device_map="auto")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=4096,
)
local_llm = HuggingFacePipeline(pipeline=pipe)

# Creating querying chain
prompt_template = '''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful and detailed answers to the user's questions. 
Context: {context}

USER: {question} 

ASSISTANT:'''

prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
qa = RetrievalQA.from_chain_type(llm=local_llm, chain_type="stuff", retriever=db.as_retriever(),
                                 return_source_documents=True, chain_type_kwargs={"prompt": prompt})

# Running query
query = "Tell me the story of Pierre."
res = qa(query)

# Getting answer and sources
answer, docs = res['result'], res['source_documents']

print(answer)
print("\n---\n".join([f"Source {i + 1}:\n{document.page_content}" for i, document in enumerate(docs)]))
