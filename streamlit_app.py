import streamlit as st
import os
from google import genai
from google.genai.errors import APIError
from pypdf import PdfReader
from langchain_community.embeddings import GoogleGenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.prompts import PromptTemplate # QAチェーンは後で実装
# from langchain.chains.question_answering import load_qa_chain # QAチェーンは後で実装

# --- 1. 初期設定とAPIクライアントの初期化 ---

st.set_page_config(page_title="PDF参照型チャットボット by Gemini", layout="wide")
st.title("📄 PDF参照型チャットボット")
st.subheader("最低限の実装：PDFをアップロードし、知識ベースを構築します。")

# secrets.tomlからAPIキーを取得
try:
    # 環境変数に設定している場合は os.environ.get("GEMINI_API_KEY")
    api_key = st.secrets["GEMINI_API_KEY"] 
except KeyError:
    st.error("⚠️ GEMINI_API_KEYが`.streamlit/secrets.toml`に設定されていません。先に設定してください。")
    st.stop()

# Gemini APIクライアントの初期化 (ここでは埋め込み用のみ)
try:
    client = genai.Client(api_key=api_key)
except Exception as e:
    st.error(f"Gemini APIクライアントの初期化に失敗しました: {e}")
    st.stop()


# --- 2. ユーティリティ関数 ---

@st.cache_resource(show_spinner=False)
def get_pdf_text(pdf_docs):
    """アップロードされたPDFドキュメントからテキストを抽出し、結合します。"""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

@st.cache_resource(show_spinner=False)
def get_text_chunks(text):
    """抽出したテキストを、埋め込みに適したサイズに分割します。"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_resource(show_spinner=False)
def get_vector_store(text_chunks):
    """テキストチャンクから埋め込みを生成し、FAISSベクトルストアに保存します。"""
    if not text_chunks:
        st.warning("処理するテキストチャンクがありません。")
        return None
        
    with st.spinner("🔄 ベクトルストアを構築中..."):
        # LangChainのGoogleGenAIEmbeddingsを使用
        embeddings = GoogleGenAIEmbeddings(model="embedding-001", google_api_key=api_key)
        # FAISS (インメモリ)に保存
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        st.success(f"✅ ベクトルストア構築完了！ {len(text_chunks)}個のチャンクを知識ベースに追加しました。")
    return vector_store


# --- 3. Streamlit UIとメインロジック ---

# サイドバーでファイルアップロードと処理を実行
with st.sidebar:
    st.subheader("ファイルのアップロード")
    pdf_docs = st.file_uploader(
        "PDFファイルをアップロードしてください。",
        accept_multiple_files=True,
        type=['pdf']
    )
    
    # PDFがアップロードされ、ボタンが押されたら処理を開始
    if pdf_docs and st.button("知識ベースの構築 (最低機能)"):
        try:
            # 1. PDFからテキストを抽出
            raw_text = get_pdf_text(pdf_docs)
            
            if not raw_text.strip():
                st.error("アップロードされたPDFファイルからテキストを抽出できませんでした。ファイル形式を確認してください。")
            else:
                # 2. テキストをチャンクに分割
                text_chunks = get_text_chunks(raw_text)
                
                # 3. ベクトルストアを生成し、セッションステートに保存
                vector_store = get_vector_store(text_chunks)
                st.session_state.vector_store = vector_store
                st.session_state.messages = [] # チャット履歴をリセット
                
        except Exception as e:
            st.error(f"ファイルの処理中にエラーが発生しました: {e}")

# --- 4. チャットインターフェース (最低限の表示のみ) ---

# 最低機能の実装として、チャット入力はまだ動作しません。
st.warning("質問応答機能は未実装です。次はチャット機能を追加します。")

# チャット履歴をセッションステートで管理
if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去のメッセージを表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# ユーザーからの入力を受け付ける (今回はまだ処理しない)
if prompt := st.chat_input("質問を入力してください (現在は非アクティブ)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    st.chat_message("assistant").write("ベクトルストアの構築は完了しましたが、質問応答機能はまだ実装されていません。")
