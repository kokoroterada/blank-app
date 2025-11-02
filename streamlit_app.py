import streamlit as st
from google import genai
from google.genai.errors import APIError

# --- ページ設定 ---
st.set_page_config(
    page_title="Gemini Chatbot",
    page_icon="🤖"
)
st.title("🤖 Gemini チャットボット")

# --- APIキーの読み込みと初期化 ---

# StreamlitのsecretsからAPIキーを取得
try:
    gemini_api_key = st.secrets["gemini_api_key"]
except KeyError:
    st.error("🚨 `.streamlit/secrets.toml` に `gemini_api_key` が設定されていません。")
    st.stop()

# Geminiクライアントの初期化
try:
    client = genai.Client(api_key=gemini_api_key)
except Exception as e:
    st.error(f"🚨 Geminiクライアントの初期化に失敗しました: {e}")
    st.stop()

# --- モデル設定 ---
# 使用するモデル。マルチターンチャットに対応しているモデルを選択
MODEL_NAME = "gemini-2.5-flash"

# --- チャット履歴の初期化 ---
if "chat" not in st.session_state:
    try:
        # 新しいチャットセッションを作成
        st.session_state.chat = client.chats.create(model=MODEL_NAME)
    except APIError as e:
        st.error(f"🚨 チャットセッションの作成に失敗しました: {e}")
        st.session_state.chat = None
        st.stop()

# Streamlitのセッションステートにメッセージ履歴を初期化
if "messages" not in st.session_state:
    # 最初の挨拶
    st.session_state.messages = [
        {"role": "assistant", "content": "こんにちは！私はGeminiを搭載したチャットボットです。何をお手伝いしましょうか？"}
    ]

# --- 既存のメッセージの表示 ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- ユーザー入力の処理 ---
if prompt := st.chat_input("メッセージを入力してください..."):
    # ユーザーのメッセージを履歴に追加して表示
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.chat:
        # アシスタント（Gemini）の応答を生成
        with st.chat_message("assistant"):
            try:
                # Geminiにメッセージを送信し、ストリーミングで応答を取得
                response = st.session_state.chat.send_message(prompt, stream=True)
                
                # ストリーミングされた応答を表示
                full_response = st.write_stream(response)
                
                # 完全な応答を履歴に追加
                st.session_state.messages.append({"role": "assistant", "content": full_response})

            except APIError as e:
                error_message = f"Gemini APIエラーが発生しました: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
            except Exception as e:
                error_message = f"予期せぬエラーが発生しました: {e}"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
