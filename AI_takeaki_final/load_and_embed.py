import os
import glob
import docx2txt
import zipfile # 追加
import requests # 追加
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

# OpenAI APIキーを環境変数から取得
# Streamlit CloudのSecretsで設定している場合、os.getenv()で自動的に読み込まれます。
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# FAISSインデックスのパス
FAISS_INDEX_PATH = "faiss_index"

# GitHub Releases のダウンロードURL (ここにあなたのURLを正確に貼り付けてください！)
DOWNLOAD_URL = "https://github.com/Takeaki920/AI-Takeaki-2/releases/download/v1.0.0/faiss_index.zip"


def load_documents_from_docs_folder(folder_path):
    documents = []
    for filepath in glob.glob(os.path.join(folder_path, "*.docx")):
        try:
            text = docx2txt.process(filepath)
            documents.append(text)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    return documents

def split_texts(texts):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    # create_documentsはリストのリストを返す可能性があるため、フラットにする
    flat_documents = []
    for text in texts:
        flat_documents.extend(splitter.create_documents([text]))
    return flat_documents

def download_and_extract_faiss_index():
    # faiss_index フォルダが存在し、かつ中身が空でないかを確認
    if os.path.exists(FAISS_INDEX_PATH) and os.path.isdir(FAISS_INDEX_PATH) and os.listdir(FAISS_INDEX_PATH):
        print(f"'{FAISS_INDEX_PATH}' already exists and is not empty. Skipping download.")
        return

    print(f"'{FAISS_INDEX_PATH}' not found or is empty. Downloading from {DOWNLOAD_URL}...")
    
    try:
        # ZIPファイルをダウンロード
        response = requests.get(DOWNLOAD_URL, stream=True)
        response.raise_for_status() # HTTPエラー (4xx, 5xx) があれば例外を発生

        zip_path = "faiss_index.zip"
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {zip_path}")

        # ZIPファイルを解凍
        # 解凍先は現在の作業ディレクトリ (AI_takeaki_final)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".") # 現在のディレクトリに解凍
        print(f"Extracted {zip_path} to {os.getcwd()}")

        # ダウンロードしたZIPファイルを削除
        os.remove(zip_path)
        print(f"Removed {zip_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading FAISS index: {e}")
        raise
    except zipfile.BadZipFile as e:
        print(f"Error extracting ZIP file (bad zip): {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during download/extraction: {e}")
        raise


def load_and_embed():
    # この関数は、FAISSインデックスを最初から作成し直す場合にのみ使用します。
    # Streamlit Cloudでは、通常は download_and_extract_faiss_index() を通して既存のインデックスをロードします。
    docs_path = "docs"
    texts = load_documents_from_docs_folder(docs_path)
    documents = split_texts(texts)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key) # APIキーを渡す
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH, index_name="faiss_index") # index_nameも指定
    print("ベクトルDBの作成が完了しました。")

def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key) # APIキーを渡す
    # FAISSインデックスが存在しない場合にダウンロードと展開を実行
    download_and_extract_faiss_index()
    # 既存のFAISSインデックスをロード
    return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    # このスクリプトを単独で実行してベクトルDBを再作成したい場合に、以下の行をコメントアウト解除してください。
    # load_and_embed()
    # 通常は、Streamlit Cloudでこのファイルが起動された際にload_vectorstore()が呼ばれ、
    # その中で必要に応じてダウンロードが実行されます。
    print("load_and_embed.py executed. This script typically acts as a module for Streamlit.")
