
import os
import glob
import docx2txt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def load_documents_from_docs_folder(folder_path):
    documents = []
    for filepath in glob.glob(os.path.join(folder_path, "*.docx")):
        text = docx2txt.process(filepath)
        documents.append(text)
    return documents

def split_texts(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.create_documents(texts)

def load_and_embed():
    docs_path = "docs"
    texts = load_documents_from_docs_folder(docs_path)
    documents = split_texts(texts)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local("faiss_index")
    print("ベクトルDBの作成が完了しました。")

def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    load_and_embed()
