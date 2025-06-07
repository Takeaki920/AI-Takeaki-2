
# langchain-community からの正しいインポートパスに変更
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA # langchain.chains はまだ変更不要な場合も
from langchain_openai import ChatOpenAI # OpenAI のモデルは langchain_openai に移動
from langchain_core.prompts import PromptTemplate # プロンプトは langchain_core.prompts に移動

from load_and_embed import load_vectorstore
from dotenv import load_dotenv
import os
import streamlit as st

# 環境変数をロード（ローカル用）
load_dotenv()

def create_qa_chain():
    # APIキーの取得（ローカル優先、なければStreamlit Cloud用のsecretsから）
    openai_api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY"))

    llm = ChatOpenAI(temperature=0.7, openai_api_key=openai_api_key) # ★ここを0.7に変更★

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
あなたは「AIたけあき」です。ユーザーの分身として、フレンドリーでありながらも、丁寧な言葉遣いで回答してください。
回答は会話形式で、難しい専門用語は避け、分かりやすく説明することを心がけてください。
過度にフランクになったり、不自然な繰り返しをしたりすることは避けてください。
著作の内容をもとに回答するようにしてください。
回答の締めくくりは、自然で親しみやすい表現にしてください。
ユーザーの話し方の例としては、語尾が「というわけです」や「なんです」となることが多いです。

文脈: {context}

質問: {question}

回答：
"""
    )

    vectorstore = load_vectorstore()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

def ask_ai(query):
    qa = create_qa_chain()
    return qa.run(query)
