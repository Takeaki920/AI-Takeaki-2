import os
import glob
import docx2txt
import zipfile
import requests
import shutil # 追加
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

# 環境変数をロード
load_dotenv()

# OpenAI APIキーを環境変数から取得
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
    flat_documents = []
    for text in texts:
        flat_documents.extend(splitter.create_documents([text]))
    return flat_documents

def download_and_extract_faiss_index():
    target_faiss_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), FAISS_INDEX_PATH)

    # 最初に、ターゲットディレクトリをクリーンアップします。
    # これにより、前回のデプロイで作成された可能性のある不完全な/間違ったフォルダが削除されます。
    if os.path.exists(target_faiss_dir):
        print(f"Clearing existing FAISS directory: {target_faiss_dir}")
        import shutil
        shutil.rmtree(target_faiss_dir)
    
    # フォルダが存在しない場合のみダウンロード (shutil.rmtree で削除したので、このチェックはほぼ常にTrueになりますが、安全のため残す)
    if not os.path.exists(target_faiss_dir) or not os.listdir(target_faiss_dir):
        print(f"'{target_faiss_dir}' not found or is empty. Downloading from {DOWNLOAD_URL}...")
        
        try:
            # ZIPファイルをダウンロード
            response = requests.get(DOWNLOAD_URL, stream=True)
            response.raise_for_status() 

            zip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "faiss_index.zip")
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {zip_path}")

            # ZIPファイルを解凍
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # 一時ディレクトリに解凍し、中身を移動させる堅牢な方法
                temp_extract_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_faiss_extract")
                os.makedirs(temp_extract_dir, exist_ok=True)
                zip_ref.extractall(temp_extract_dir) # 一時ディレクトリに解凍

                # ターゲットディレクトリを最終的に作成
                os.makedirs(target_faiss_dir, exist_ok=True)

                # 解凍されたフォルダ（通常はfaiss_index）を正しい位置に移動
                # temp_extract_dir の中身を確認し、適切なフォルダを移動
                extracted_content = os.listdir(temp_extract_dir)
                
                # ZIPの中にfaiss_indexフォルダが丸ごと入っている場合（最も一般的なケース）
                if len(extracted_content) == 1 and os.path.isdir(os.path.join(temp_extract_dir, extracted_content[0])):
                    print(f"Moving extracted directory '{extracted_content[0]}' to '{target_faiss_dir}'")
                    # 例: temp_faiss_extract/faiss_index/ を AI_takeaki_final/faiss_index/ に移動
                    for item_in_inner_dir in os.listdir(os.path.join(temp_extract_dir, extracted_content[0])):
                        shutil.move(os.path.join(temp_extract_dir, extracted_content[0], item_in_inner_dir), target_faiss_dir)
                    shutil.rmtree(os.path.join(temp_extract_dir, extracted_content[0])) # 空になった中間ディレクトリを削除
                else:
                    # ZIPの中に直接index.faissなどが入っている場合
                    print(f"Moving extracted files directly to '{target_faiss_dir}'")
                    for item in extracted_content:
                        shutil.move(os.path.join(temp_extract_dir, item), target_faiss_dir)

                # 一時ディレクトリを削除
                shutil.rmtree(temp_extract_dir)

            print(f"Extracted {zip_path} to {target_faiss_dir}")

            # ダウンロードしたZIPファイルを削除
            os.remove(zip_path)
            print(f"Removed {zip_path}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading FAISS index: {e}")
            raise
        except zipfile.BadZipFile as e:
            print(f"Error extracting ZIP file (bad zip): {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
                print(f"Removed potentially corrupted {zip_path}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during download/extraction: {e}")
            raise
    else:
        # このパスは、クリーンアップロジックがあるので、通常は実行されないはずですが、念のため
        print(f"'{target_faiss_dir}' already exists and is not empty. Skipping download (should have been cleared).")


def load_and_embed():
    docs_path = "docs"
    texts = load_documents_from_docs_folder(docs_path)
    documents = split_texts(texts)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(FAISS_INDEX_PATH, index_name="faiss_index") 
    print("ベクトルDBの作成が完了しました。")

def load_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    download_and_extract_faiss_index()
    return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    print("load_and_embed.py executed. This script typically acts as a module for Streamlit.")
