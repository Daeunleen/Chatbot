import streamlit as st
import openai
import os
from dotenv import load_dotenv
from datetime import datetime
import base64
import time
import sqlite3 # SQLite 데이터베이스 사용을 위한 임포트
import uuid    # 고유임포트 ID 생성을 위한 

# LangChain 관련 라이브러리 임포트 (RAG 구현용)
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

# .env 파일 로드 (스크립트 최상단에 위치 권장)
load_dotenv()

st.set_page_config(page_title="한밭대학교 AI 챗봇", layout="wide", initial_sidebar_state="auto") # 사이드바 초기 상태 변경

# --- 데이터베이스 관련 설정 및 초기화 ---
DB_NAME = 'chat_history.db' # 데이터베이스 파일명

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # 채팅 세션 저장 테이블 (대화 목록을 위한 메타데이터)
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_sessions (
            session_id TEXT PRIMARY KEY,
            title TEXT,
            start_time TEXT,
            last_updated TEXT
        )
    ''')
    # 각 메시지 저장 테이블 (실제 대화 내용)
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            message_id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TEXT,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id)
        )
    ''')
    conn.commit()
    conn.close()

# --- 앱 시작 시 DB 초기화 호출 ---
init_db()

# --- 이미지 파일을 Base64로 인코딩하는 함수 ---
def get_image_as_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        st.error(f"⚠️ '{file_path}' 파일을 찾을 수 없습니다. 이미지가 프로젝트 폴더에 있는지 확인해주세요.")
        return None
    except Exception as e:
        st.error(f"이미지 인코딩 중 오류 발생: {e}")
        return None

# 로고 및 배경 이미지 경로 설정 (동일한 이미지 파일 사용)
LOGO_FILENAME = "한밭대로고.jpg" # 실제 로고 파일명으로 변경하세요
BACKGROUND_IMAGE_FILENAME = "한밭대로고.jpg" # 실제 배경 파일명으로 변경하세요

# 이미지 Base64 인코딩 실행
encoded_logo_image = get_image_as_base64(LOGO_FILENAME)
encoded_background_image = get_image_as_base64(BACKGROUND_IMAGE_FILENAME)

# --- Custom CSS (새로운 디자인 테마 적용) ---
background_css = ""
if encoded_background_image:
    background_css = f"""
    [data-testid="stAppViewContainer"] {{
        background: linear-gradient(rgba(240, 242, 245, 0.95), rgba(240, 242, 245, 0.95)), url(data:image/jpg;base64,{encoded_background_image}) no-repeat center center fixed;
        background-size: 30% auto; /* 배경 이미지 크기 조정 */
        background-position: center;
        margin: 0 !important;
        padding: 0 !important;
        height: 100vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }}
    """
else:
    background_css = """
    [data-testid="stAppViewContainer"] {
        background-color: #F0F2F5 !important; /* Fallback: 더 부드러운 밝은 회색 */
        margin: 0 !important;
        padding: 0 !important;
        height: 100vh !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    """

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');

    :root {{
        --primary-color: #0056b3; /* 딥블루 (기존보다 톤 다운) */
        --secondary-color: #6c757d; /* 회색 */
        --accent-color: #28a745; /* 강조색 (버튼 등) */
        --background-light: #F8F9FA; /* 밝은 배경색 */
        --background-dark: #E9ECEF; /* 어두운 배경색 */
        --text-dark: #212529; /* 어두운 텍스트 */
        --text-muted: #495057; /* 뮤트 텍스트 */
        --border-color: #dee2e6; /* 테두리 색상 */
        --chat-bubble-user: #007bff; /* 사용자 채팅 버블 */
        --chat-bubble-bot: #F1F3F5; /* 봇 채팅 버블 */
        --shadow-light: rgba(0, 0, 0, 0.08);
        --shadow-medium: rgba(0, 0, 0, 0.15);
        --shadow-strong: rgba(0, 0, 0, 0.25);
    }}

    html, body {{
        margin: 0 !important;
        padding: 0 !important;
        height: 100%;
        overflow: hidden;
        font-family: 'Noto Sans KR', sans-serif;
        color: var(--text-dark);
    }}

    {background_css} /* 동적으로 생성된 배경 CSS 삽입 */

    .main {{
        background: transparent !important;
        margin: 0 !important;
        padding: 0 !important;
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }}

    .stApp > header {{ display: none !important; }}

    .block-container {{
        padding-top: 0 !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        padding-bottom: 0 !important;
        margin: 0 auto !important;
        width: 100%;
        max-width: 100%;
        display: flex;
        flex-direction: column;
        flex-grow: 1;
    }}

    /* Streamlit 내부 요소들의 불필요한 여백 제거 및 재정의 */
    div[data-testid="stVerticalBlock"], div[data-testid="stVerticalBlock"] > div:first-child {{
        padding-top: 0 !important;
        margin-top: 0 !important;
    }}
    div[data-testid="stHorizontalBlock"]:first-child {{
        margin-top: 0 !important;
        padding-top: 0 !important;
    }}
    .stForm {{ width: 100%; }} /* 폼이 전체 너비를 차지하도록 */

    /* 헤더 컨테이너 스타일링 */
    .header-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 2rem 0; /* 상하 여백 증가 */
        background-color: rgba(255, 255, 255, 0.95); /* 더 밝고 불투명한 배경 */
        border-bottom: 1px solid var(--border-color);
        margin-bottom: 25px; /* 하단 여백 증가 */
        width: 100%;
        max-width: 960px; /* 최대 너비 조정 */
        margin-left: auto;
        margin-right: auto;
        border-radius: 20px; /* 더 둥근 모서리 */
        box-shadow: 0px 10px 40px var(--shadow-medium); /* 그림자 효과 강화 */
        backdrop-filter: blur(15px); /* 프로스티드 글라스 효과 강화 */
        -webkit-backdrop-filter: blur(15px);
        transform: translateY(-10px); /* 초기 위치 조정 */
        opacity: 0; /* 초기 투명 */
        animation: slideInDown 0.8s ease-out forwards; /* 애니메이션 적용 */
    }}
    .logo-img {{ width: 120px; height: auto; margin-bottom: 15px; animation: fadeIn 1.2s ease-out; }}
    .title {{ font-size: 2.8em; font-weight: 700; text-align: center; color: var(--text-dark); margin-bottom: 0.6rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.05); }}
    .subtitle {{ text-align: center; color: var(--text-muted); margin-bottom: 1.5rem; font-size: 1.15em; font-weight: 400; animation: fadeIn 1.5s ease-out; }}

    /* 채팅 래퍼 스타일링 */
    .chat-wrapper {{
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        overflow: hidden;
        max-width: 900px; /* 채팅창 최대 너비 조정 */
        width: 100%;
        margin: 0 auto;
        background-color: rgba(255, 255, 255, 0.98); /* 거의 불투명한 흰색 배경 */
        border-radius: 20px; /* 더 둥근 모서리 */
        box-shadow: 0px 15px 50px var(--shadow-strong); /* 그림자 효과 강화 */
        backdrop-filter: blur(18px); /* 프로스티드 글라스 효과 강화 */
        -webkit-backdrop-filter: blur(18px);
        animation: fadeInScale 0.8s ease-out forwards 0.3s; /* 애니메이션 적용 */
        opacity: 0;
        transform: scale(0.98);
    }}
    .chat-container {{
        flex-grow: 1;
        overflow-y: auto;
        padding: 25px; /* 패딩 증가 */
        display: flex;
        flex-direction: column;
        gap: 20px; /* 메시지 간 간격 증가 */
    }}
    .chat-message-wrapper {{ display: flex; align-items: flex-end; gap: 12px; }} /* 아이콘과 버블 정렬 조정 */
    .chat-user .chat-message-wrapper {{ justify-content: flex-end; flex-direction: row-reverse; }}
    .chat-bot .chat-message-wrapper {{ justify-content: flex-start; flex-direction: row; }}
    .chat-icon {{
        font-size: 1.8em; padding: 8px; border-radius: 50%; background-color: var(--background-dark); color: var(--text-muted);
        display: flex; align-items: center; justify-content: center; width: 45px; height: 45px; /* 아이콘 크기 증가 */
        box-shadow: 0 2px 5px var(--shadow-light); flex-shrink: 0;
        transition: transform 0.2s ease-in-out, background-color 0.2s ease-in-out;
    }}
    .chat-user .chat-icon {{ background-color: var(--primary-color); color: white; }}
    .chat-bot .chat-icon {{ background-color: #E6E6E6; color: var(--primary-color); }} /* 봇 아이콘 색상 변경 */
    .chat-message {{
        padding: 14px 20px; border-radius: 25px; line-height: 1.7; font-size: 1.05em; /* 메시지 버블 스타일 조정 */
        position: relative; box-shadow: 0 3px 8px var(--shadow-light);
        animation: fadeInMessage 0.5s ease-out;
        max-width: 75%; /* 메시지 버블 최대 너비 조정 */
        /* --- 줄바꿈 문제 해결을 위한 CSS 추가 --- */
        overflow-wrap: break-word;
        word-break: break-word;
        /* ------------------------------------- */
    }}
    .chat-user .chat-message {{ background-color: var(--primary-color); color: white; border-bottom-right-radius: 10px; }}
    .chat-bot .chat-message {{ background-color: var(--chat-bubble-bot); color: var(--text-dark); border-bottom-left-radius: 10px; }}
    .message-timestamp {{ font-size: 0.78em; color: var(--text-muted); margin-top: 5px; /* 타임스탬프 여백 조정 */ }}
    .chat-user .message-timestamp {{ color: rgba(255,255,255,0.7); }}
    
    /* 참고 문서 섹션 스타일링 */
    .chat-bot .chat-message .source-documents {{
        margin-top: 12px;
        padding-top: 10px;
        border-top: 1px dashed var(--border-color);
        font-size: 0.88em;
        color: var(--text-muted);
    }}
    .chat-bot .chat-message .source-documents strong {{
        font-weight: 600;
        color: var(--text-dark);
        margin-bottom: 6px;
        display: block;
    }}
    .chat-bot .chat-message .source-documents ul {{
        list-style-type: none;
        padding-left: 0;
        margin-top: 5px;
        margin-bottom: 0;
    }}
    .chat-bot .chat-message .source-documents li {{
        padding: 3px 0;
        color: var(--text-muted);
    }}
    .chat-bot .chat-message .source-documents li i {{
        margin-right: 8px;
        color: var(--primary-color);
    }}

    /* 입력 폼 컨테이너 스타일링 */
    .input-form-container {{
        position: sticky; bottom: 0; background-color: rgba(255, 255, 255, 0.98);
        padding: 15px 20px; border-top: 1px solid var(--border-color); width: 100%;
        max-width: 900px; margin: 0 auto; z-index: 1000;
        box-shadow: 0 -8px 25px var(--shadow-medium);
        border-bottom-left-radius: 20px; border-bottom-right-radius: 20px;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
    }}
    .input-form-container > div[data-testid="stForm"] {{ display: flex; gap: 15px; align-items: center; }}
    .stTextInput > div > div > input {{
        border-radius: 30px; padding: 12px 25px; border: 1px solid var(--border-color);
        box-shadow: inset 0 1px 4px var(--shadow-light);
        font-size: 1.05em; transition: all 0.3s ease-in-out;
        flex-grow: 1; /* 입력 필드가 가능한 모든 공간을 차지하도록 */
    }}
    .stTextInput > div > div > input:focus {{
        border-color: var(--primary-color); box-shadow: 0 0 0 0.25rem rgba(0,86,179,.25), inset 0 1px 4px var(--shadow-light);
        outline: none;
    }}
    .input-form-container div[data-testid="stForm"] .stButton button {{
        border-radius: 50%; width: 52px; height: 52px; background-color: var(--primary-color); color: white; border: none;
        font-size: 1.8em; display: flex; justify-content: center; align-items: center;
        box-shadow: 0px 4px 10px var(--shadow-medium);
        transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out;
    }}
    .input-form-container div[data-testid="stForm"] .stButton button:hover {{ background-color: #004085; transform: translateY(-3px); }}
    /* 이전 새 대화 버튼 스타일 제거 (사이드바로 이동) */
    .input-form-container div[data-testid="stForm"] .stButton:last-of-type button {{
        display: none; /* 하단의 새 대화 버튼 숨기기 */
    }}

    /* 스크롤바 스타일링 */
    .chat-container::-webkit-scrollbar {{ width: 10px; }}
    .chat-container::-webkit-scrollbar-track {{ background: var(--background-light); border-radius: 10px; }}
    .chat-container::-webkit-scrollbar-thumb {{ background: var(--secondary-color); border-radius: 10px; }}
    .chat-container::-webkit-scrollbar-thumb:hover {{ background: #5a6268; }}

    /* 환영 메시지 스타일링 */
    .welcome-message {{
        text-align: center; padding: 35px 40px; border-radius: 15px;
        background-color: var(--background-light); color: var(--text-dark); margin: 15px auto 25px auto;
        font-size: 1.1em; line-height: 1.8; box-shadow: 0 6px 20px var(--shadow-light);
        animation: fadeIn 1s ease-out; max-width: 90%;
        border: 1px solid var(--border-color);
    }}
    .welcome-message h3 {{ color: var(--primary-color); margin-bottom: 18px; font-weight: 700; font-size: 1.5em; }}
    .welcome-message li {{ text-align: left; margin-bottom: 8px; }} /* 리스트 아이템 왼쪽 정렬 및 간격 */
    .welcome-message li i {{ margin-right: 10px; color: var(--primary-color); }}
    .welcome-message ul {{ padding-left: 20px; }} /* 리스트 내부 여백 */

    /* 타이핑 인디케이터 스타일링 */
    .typing-indicator {{
        display: flex; align-items: center; gap: 6px; font-style: italic; color: var(--text-muted);
        font-size: 0.95em; margin-top: 5px; margin-left: 5px;
    }}
    .typing-indicator span {{
        animation: blink 1s infinite;
        font-weight: bold;
    }}
    .typing-indicator span:nth-child(2) {{ animation-delay: 0.2s; }}
    .typing-indicator span:nth-child(3) {{ animation-delay: 0.4s; }}

    /* 복사 버튼 스타일링 */
    .copy-button-container {{
        display: flex; justify-content: flex-end; margin-top: 8px; /* 마진 증가 */
        padding-right: 5px; /* 버튼 오른쪽 여백 */
    }}
    .copy-button {{
        background-color: var(--background-dark); color: var(--text-muted); border: none;
        border-radius: 20px; padding: 6px 12px; font-size: 0.85em;
        cursor: pointer; transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out;
        box-shadow: 0 1px 4px var(--shadow-light);
    }}
    .copy-button:hover {{ background-color: #DDE2E7; transform: translateY(-1px); }}

    /* 모달 스타일링 */
    .st-modal-container {{
        position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background-color: rgba(0, 0, 0, 0.6); display: flex; /* 배경 불투명도 증가 */
        justify-content: center; align-items: center; z-index: 9999;
        backdrop-filter: blur(5px); /* 모달에도 블러 효과 */
    }}
    .st-modal-content {{
        background-color: white; padding: 35px; border-radius: 20px;
        box-shadow: 0 15px 40px var(--shadow-strong); text-align: center;
        max-width: 450px; width: 90%;
        animation: fadeInScale 0.3s ease-out;
    }}
    .st-modal-content h4 {{ margin-bottom: 25px; color: var(--text-dark); font-size: 1.3em; font-weight: 600; }}
    .st-modal-buttons {{ display: flex; justify-content: center; gap: 20px; }}
    .st-modal-buttons button {{
        padding: 12px 30px; border-radius: 30px; font-size: 1.05em;
        cursor: pointer; transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out;
        box-shadow: 0 3px 8px var(--shadow-light);
    }}
    .st-modal-buttons .confirm-btn {{ background-color: #dc3545; color: white; border: none; }}
    .st-modal-buttons .confirm-btn:hover {{ background-color: #c82333; transform: translateY(-2px); }}
    .st-modal-buttons .cancel-btn {{ background-color: var(--secondary-color); color: white; border: none; }}
    .st-modal-buttons .cancel-btn:hover {{ background-color: #5a6268; transform: translateY(-2px); }}

    /* 디버그 정보 영역 스타일링 */
    .debug-info-box {{
        background-color: #e3f2fd; /* 연한 하늘색 배경 */
        border-left: 5px solid var(--primary-color);
        padding: 15px 20px;
        margin-top: 20px;
        border-radius: 10px;
        font-size: 0.88em;
        color: var(--text-muted);
        box-shadow: 0 3px 10px var(--shadow-light);
        overflow-x: auto;
    }}
    .debug-info-box strong {{
        color: var(--primary-color);
        font-weight: 700;
        margin-bottom: 8px;
        display: block;
    }}
    .debug-info-box pre {{
        white-space: pre-wrap;
        word-break: break-all;
        margin: 0;
        padding: 0;
        font-family: 'Noto Sans KR', sans-serif;
    }}

    /* 사이드바 커스텀 스타일링 (st.radio를 사용하는 경우) */
    [data-testid="stSidebar"] {{
        background-color: #f8f9fa; /* 밝은 배경색 */
        padding: 20px;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    }}
    [data-testid="stSidebar"] .stRadio > label {{
        padding: 10px 15px;
        margin-bottom: 5px;
        border-radius: 8px;
        cursor: pointer;
        transition: background-color 0.2s ease, border-left 0.2s ease;
        display: block; /* 전체 라벨 클릭 가능하도록 */
        font-size: 0.95em;
        line-height: 1.4;
        border: 1px solid transparent; /* 기본 테두리 숨김 */
    }}
    [data-testid="stSidebar"] .stRadio > label:hover {{
        background-color: #eef2f6;
    }}
    /* 선택된 라디오 버튼 스타일 */
    [data-testid="stSidebar"] .stRadio input:checked + div {{
        background-color: #e6f2ff !important; /* 선택 시 밝은 파란색 배경 */
        border-left: 5px solid var(--primary-color) !important; /* 좌측 파란색 바 */
        font-weight: 500;
        color: var(--primary-color);
    }}
    /* 라디오 버튼의 동그라미 숨기기 */
    [data-testid="stSidebar"] .stRadio div[data-testid="stCheckableInput-0"] {{
        display: none;
    }}
    /* 사이드바 내 버튼 스타일링 */
    [data-testid="stSidebar"] .stButton button {{
        width: 100%;
        padding: 10px 15px;
        border-radius: 20px;
        font-size: 1em;
        margin-top: 10px;
        box-shadow: 0 2px 5px var(--shadow-light);
    }}
    [data-testid="stSidebar"] .stButton button.secondary-btn {{
        background-color: var(--primary-color);
        color: white;
    }}
    [data-testid="stSidebar"] .stButton button.secondary-btn:hover {{
        background-color: #004085;
    }}
    [data-testid="stSidebar"] h3 {{
        color: var(--primary-color);
        margin-bottom: 15px;
        font-weight: 700;
    }}


    /* Keyframe 애니메이션 */
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    @keyframes fadeInMessage {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes slideInDown {{
        from {{ opacity: 0; transform: translateY(-30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    @keyframes fadeInScale {{
        from {{ opacity: 0; transform: scale(0.95); }}
        to {{ opacity: 1; transform: scale(1); }}
    }}

    /* 반응형 디자인 (모바일 최적화) */
    @media (max-width: 768px) {{
        .header-container {{
            padding: 1rem 0;
            margin-bottom: 15px;
            border-radius: 10px;
            box-shadow: 0px 5px 20px var(--shadow-light);
        }}
        .logo-img {{ width: 90px; margin-bottom: 10px; }}
        .title {{ font-size: 2em; margin-bottom: 0.3rem; }}
        .subtitle {{ font-size: 1em; margin-bottom: 1rem; }}

        .chat-wrapper {{
            border-radius: 10px;
            box-shadow: 0px 8px 25px var(--shadow-light);
        }}
        .chat-container {{
            padding: 15px;
            gap: 15px;
        }}
        .chat-icon {{ width: 35px; height: 35px; font-size: 1.4em; }}
        .chat-message {{
            padding: 10px 15px;
            font-size: 0.95em;
            max-width: 85%;
        }}
        .message-timestamp {{ font-size: 0.7em; }}

        .input-form-container {{
            padding: 10px 15px;
            border-bottom-left-radius: 10px; border-bottom-right-radius: 10px;
            box-shadow: 0 -4px 15px var(--shadow-light);
        }}
        .stTextInput > div > div > input {{
            padding: 8px 18px;
            font-size: 0.95em;
        }}
        .input-form-container div[data-testid="stForm"] .stButton button {{
            width: 45px; height: 45px; font-size: 1.4em;
        }}
        /* 모바일에서는 하단 새 대화 버튼 표시 (선택 사항) */
        .input-form-container div[data-testid="stForm"] .stButton:last-of-type button {{
            display: flex; /* 숨김 해제 */
            height: 45px; padding: 0px 15px; font-size: 0.85em; /* 크기 조정 */
            border-radius: 30px; /* 둥근 버튼 */
            background-color: var(--secondary-color); /* 색상 변경 */
            color: white;
            box-shadow: 0px 4px 10px var(--shadow-light);
        }}
        .input-form-container div[data-testid="stForm"] .stButton:last-of-type button:hover {{
            background-color: #5a6268;
            transform: translateY(-3px);
        }}

        .welcome-message {{
            padding: 20px 25px;
            font-size: 0.95em;
        }}
        .welcome-message h3 {{ font-size: 1.2em; margin-bottom: 10px; }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

actual_api_key = os.getenv("OPENAI_API_KEY")

if not actual_api_key:
    st.error("⚠️ OpenAI API 키가 환경 변수(OPENAI_API_KEY)에 설정되지 않았습니다. `.env` 파일을 확인하거나 환경 변수를 설정해주세요.")
    st.stop()

api_key_set = True
try:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, api_key=actual_api_key)
    embeddings_model_instance = OpenAIEmbeddings(model="text-embedding-3-small")
except Exception as e:
    st.error(f"OpenAI 클라이언트 초기화 중 오류 발생: {e}. API 키를 확인해주세요.")
    api_key_set = False
    st.stop()

# --- RAG 시스템 설정 (문서 로드 및 벡터 저장소 생성) ---
# 프롬프트 개선
PROMPT_TEMPLATE = """
당신은 한밭대학교의 공식 학칙, 이수 학점 체계, 장학금 규정, 학생생활관 관리운영 지침에 기반한 정보를 제공하는 전문 AI 챗봇입니다.  
다음 원칙에 따라 사용자의 질문에 답변해주세요:

1. **문서 기반 우선**  
   제공된 공식 문서(학칙, 규정 등)에 명시된 내용만을 바탕으로 답변하는 것을 원칙으로 합니다. 문서에 존재하지 않는 내용은 임의로 추론하지 마세요.

2. **출처 명시**  
   문서 기반 정보에는 반드시 출처를 함께 제공해주세요. (예: “한밭대학교 학칙 제N조에 따르면” 등)

3. **문서에 정보가 없는 경우의 대응**  
   제공된 문서에서 관련 정보를 찾을 수 없는 경우, 다음 두 가지 중 하나를 선택합니다:
   
   - **(1) 질문이 학교 공식 정보와 직접적으로 관련 있을 경우:**  
     최신 정보를 제공하기 위해 한밭대학교 공식 웹사이트 또는 신뢰 가능한 출처를 조건부로 검색해, 반드시 **출처를 명확히 밝힌 후** 안내합니다.  
     (예: “한밭대학교 홈페이지에 따르면... (출처: https://홈페이지주소)”)

   - **(2) 질문이 학교 공식 문서 또는 신뢰 가능한 출처 어디에도 없는 경우:**  
     “죄송합니다. 제공된 문서 및 공개된 정보에서는 해당 내용을 찾을 수 없습니다. 관련 부서에 직접 문의하시는 것을 권장드립니다.” 라고 안내합니다.

4. **어조와 형식**  
   답변은 간결하고 정중하며, 이해하기 쉬운 자연스러운 한국어로 제공되어야 합니다.

5. **언어**  
   사용자가 한국어로 질문할 경우에는 한국어로, 외국어(예: 영어)로 질문할 경우에는 해당 언어로 답변해주세요. 
   단, 응답의 정확성을 위해 항상 문서 기반 정보를 바탕으로 하며, 출처 표기는 한국어 또는 해당 언어로 적절히 표현합니다.

---
문서 내용:
{context}

---
질문: {question}

---
답변:
"""
qa_chain_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=PROMPT_TEMPLATE,
)

@st.cache_resource(show_spinner="🎓 한밭대학교 학칙 문서들을 학습 중입니다. 잠시만 기다려 주세요...")
def setup_rag(api_key): # API 키만 인자로 받도록 변경
    # 함수 내부에서 API 키를 사용하여 모델을 초기화합니다.
    try:
        _llm_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.1, api_key=api_key)
        _embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key) # embedding 모델에도 api_key 명시
    except Exception as e:
        # 모델 초기화에 실패하면 에러 메시지와 함께 None 반환
        return None, f"OpenAI 서비스 초기화 중 오류 발생: {e}. API 키를 확인해주세요."

    doc_dir = "."
    documents = []
    file_names = [
        "국립한밭대학교 학칙.txt", "이수 학점 체계.txt",
        "장학금 유형, 지침.txt", "학생생활관 관리운영 지침.txt",
        "학내 무선인터넷(와이파이).txt"
    ]
    loaded_any_document = False
    error_files = []
    for file_name in file_names:
        file_path = os.path.join(doc_dir, file_name)
        if os.path.exists(file_path):
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
                loaded_any_document = True
            except Exception as e:
                st.warning(f"'{file_name}' 파일 로드 중 오류: {e}")
                error_files.append(file_name)
        else:
            st.warning(f"⚠️ '{file_name}' 파일을 찾을 수 없습니다. 앱과 같은 폴더에 있는지 확인해주세요.")
            error_files.append(file_name)
    if not loaded_any_document:
        return None, "참고할 문서를 전혀 찾거나 로드할 수 없습니다. 모든 파일이 앱과 같은 디렉토리에 있고, UTF-8로 인코딩되었는지 확인해주세요."
    if error_files:
        st.warning(f"다음 파일들을 처리하는 데 문제가 있었습니다: {', '.join(error_files)}. 해당 파일의 내용은 답변에 반영되지 않을 수 있습니다.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, chunk_overlap=100,
        length_function=len, add_start_index=True,
    )
    texts = text_splitter.split_documents(documents)
    if not texts:
        return None, "문서에서 텍스트를 추출하지 못했습니다. 파일 내용을 확인해주세요."
    try:
        # 함수 내부에서 초기화된 _embeddings_model과 _llm_model을 사용
        vectorstore = FAISS.from_documents(texts, _embeddings_model)
        retriever = vectorstore.as_retriever(search_kwargs={'k': 4})
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=_llm_model, chain_type="stuff", retriever=retriever,
            chain_type_kwargs={"prompt": qa_chain_prompt},
            return_source_documents=True
        )
        return qa_chain, None
    except Exception as e:
        return None, f"벡터 저장소 또는 QA 체인 초기화 중 오류 발생: {e}."

# setup_rag 함수를 호출할 때, 실제 API 키만 전달합니다.
qa_chain, rag_error = setup_rag(actual_api_key)

if rag_error:
    st.error(rag_error)
    rag_ready = False
    st.toast("❌ 문서 학습 시스템 초기화 실패!", icon="⚠️")
else:
    rag_ready = True
    st.toast("⚡️ 챗봇이 질문에 답변할 준비가 되었습니다!", icon="✅")

# --- 데이터베이스 메시지 저장/로드 함수 ---
def save_message(session_id, role, content, is_initial_question=False):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute("INSERT INTO chat_messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
              (session_id, role, content, timestamp))
    
    # 세션의 last_updated 시간 업데이트
    c.execute("UPDATE chat_sessions SET last_updated = ? WHERE session_id = ?",
              (timestamp, session_id))
    
    # 첫 사용자 질문일 경우 세션 제목 업데이트
    if role == "user" and is_initial_question:
        # 첫 질문 내용을 바탕으로 제목 생성 (최대 40자)
        first_question_title = content.split('\n')[0][:40] 
        if len(content.split('\n')[0]) > 40:
            first_question_title += "..."
        c.execute("UPDATE chat_sessions SET title = ? WHERE session_id = ?",
                  (first_question_title, session_id))
    
    conn.commit()
    conn.close()

def load_messages_from_db(session_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT role, content, timestamp FROM chat_messages WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
    messages_data = c.fetchall()
    conn.close()
    
    loaded_messages = []
    for msg_role, msg_content, msg_timestamp_str in messages_data:
        msg_time_obj = datetime.strptime(msg_timestamp_str, "%Y-%m-%d %H:%M:%S")
        loaded_messages.append({
            "role": msg_role, 
            "content": msg_content, 
            "time": msg_time_obj.strftime("%H:%M")
        })
    return loaded_messages


# --- 세션 상태 초기화 및 초기 메시지 설정 ---
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = str(uuid.uuid4()) # 고유한 세션 ID 생성
    st.session_state.messages = [] # 초기 대화 메시지 리스트 초기화
    st.session_state.title_set_for_current_session = False # 현재 세션의 제목이 설정되었는지 여부
    st.session_state.show_new_chat_confirm = False # 새 대화 모달 상태
    st.session_state.show_delete_confirm = False # 삭제 모달 상태
    
    # 새 세션 시작 시 DB에 세션 정보 기록 (제목은 나중에 첫 질문으로 업데이트)
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO chat_sessions (session_id, title, start_time, last_updated) VALUES (?, ?, ?, ?)",
              (st.session_state.current_session_id, "새로운 대화", start_time, start_time))
    conn.commit()
    conn.close()

    initial_message_content = "안녕하세요! 한밭대학교 학칙, 학점, 장학금, 생활관 규정에 대해 궁금한 점을 질문해주세요."
    if not api_key_set:
        initial_message_content = "안녕하세요! OpenAI API 키 설정이 필요합니다. 관리자에게 문의해주세요."
    elif not rag_ready:
        initial_message_content = "안녕하세요! 현재 문서 학습에 문제가 있어 답변이 제한적일 수 있습니다. 관리자에게 문의해주세요."
    
    st.session_state.messages.append(
        {"role": "assistant", "content": initial_message_content, "time": datetime.now().strftime("%H:%M")}
    )

# --- 사이드바 구현 ---
with st.sidebar:
    st.markdown("### 💬 대화 기록")

    # 새 대화 시작 버튼
    if st.button("✨ 새 대화 시작", key="sidebar_new_chat_button", help="새로운 대화 세션을 시작합니다."):
        st.session_state.show_new_chat_confirm = True # 모달 띄우기
        st.rerun()
        
    st.markdown("---")

    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # 모든 세션 불러오기 (최신 업데이트순으로 정렬)
    c.execute("SELECT session_id, title, start_time, last_updated FROM chat_sessions ORDER BY last_updated DESC")
    sessions = c.fetchall()
    conn.close()

    if sessions:
        # Streamlit의 st.radio를 사용하여 세션 선택 UI를 구성합니다.
        
        # 표시할 세션 목록 생성 (ID: 표시 텍스트)
        session_options_dict = {} # {session_id: display_text}
        for session_id, title, start_time, last_updated in sessions:
            # 제목이 없거나 "새로운 대화"일 경우 날짜와 시간으로 표시
            display_title = title if title and title != "새로운 대화" else f"새 대화 {datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S').strftime('%m/%d %H:%M')}"
            session_options_dict[session_id] = display_title

        # st.radio의 options는 리스트여야 합니다. 딕셔너리의 키(session_id) 리스트를 넘깁니다.
        session_ids_list = list(session_options_dict.keys())
        
        # 현재 선택된 세션 ID의 인덱스 찾기
        current_selection_index = 0
        if st.session_state.current_session_id in session_ids_list:
            current_selection_index = session_ids_list.index(st.session_state.current_session_id)

        # 라디오 버튼으로 세션 선택
        selected_session_id_from_radio = st.radio(
            "이전 대화 기록:",
            options=session_ids_list,
            format_func=lambda x: session_options_dict[x], # ID를 받아서 표시 텍스트로 변환
            index=current_selection_index, # 현재 세션이 선택되도록
            key="chat_session_selector_radio",
            help="클릭하여 이전 대화 기록을 불러옵니다."
        )

        # 선택된 세션이 변경되면 메시지 로드 및 상태 업데이트
        if selected_session_id_from_radio != st.session_state.current_session_id:
            st.session_state.current_session_id = selected_session_id_from_radio
            # 선택된 세션의 모든 메시지를 DB에서 불러와서 st.session_state.messages에 저장
            st.session_state.messages = load_messages_from_db(selected_session_id_from_radio)
            st.session_state.last_user_input = None # 새 대화 로드 시 타이핑 인디케이터 방지
            
            # 불러온 세션은 이미 제목이 있다고 간주합니다.
            st.session_state.title_set_for_current_session = True 
            
            st.rerun() # UI 업데이트를 위해 다시 로드

        st.markdown("---")
        # 현재 대화 삭제 버튼
        if st.button("🗑️ 현재 대화 삭제", key="delete_current_chat_button", help="현재 보고 있는 대화를 영구적으로 삭제합니다.", type="secondary"):
            st.session_state.show_delete_confirm = True # 삭제 확인 모달 띄우기
            st.rerun()

    else:
        st.markdown("<p style='color: var(--text-muted); text-align: center; margin-top: 20px;'>저장된 대화가 없습니다.<br>첫 질문을 시작해보세요!</p>", unsafe_allow_html=True)

# --- 헤더 섹션 ---
st.markdown('<div class="header-container">', unsafe_allow_html=True)
if encoded_logo_image:
    st.markdown(f'<img src="data:image/jpg;base64,{encoded_logo_image}" class="logo-img" alt="한밭대학교 로고">', unsafe_allow_html=True)
else:
    st.markdown('<div style="text-align: center; margin-bottom: 15px;"><i class="fas fa-university fa-3x" style="color:var(--primary-color);"></i></div>', unsafe_allow_html=True)
st.markdown('<div class="title">한밭대학교 AI 챗봇</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">학칙, 학점, 장학금, 생활관 규정에 대해 무엇이든 물어보세요!</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- 채팅 UI 컨테이너 시작 ---
st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
st.markdown('<div class="chat-container" id="chat-container-scroll">', unsafe_allow_html=True)

# --- 초기 환영 메시지 표시 (대화 시작 시에만) ---
# 단, DB에서 불러온 기존 대화가 아닌, 완전히 새로운 대화 세션일 때만 표시
if len(st.session_state.messages) == 1 and st.session_state.messages[0]["role"] == "assistant" \
   and not st.session_state.title_set_for_current_session: # 새 대화이고 제목이 아직 설정되지 않았을 때
    st.markdown(f'''
        <div class="welcome-message">
            <h3><i class="fas fa-robot"></i> 한밭대학교 규정 안내 챗봇</h3>
            <p>안녕하세요! 저는 한밭대학교의 <b>학칙, 이수 학점 체계, 장학금 유형 및 지침, 학생생활관 관리운영 지침</b>에 대한 정보를 학습한 전문 AI 챗봇입니다.</p>
            <p>아래와 같이 질문해보세요:</p>
            <ul>
                <li><i class="fas fa-book"></i> "졸업하려면 총 몇 학점 들어야 해?"</li>
                <li><i class="fas fa-money-bill-wave"></i> "국가장학금 신청 기준이 뭐야?"</li>
                <li><i class="fas fa-building"></i> "기숙사 통금 시간 알려줘."</li>
                <li><i class="fas fa-gavel"></i> "학칙 제5조 내용이 궁금해."</li>
            </ul>
            <p style="margin-top:20px;">무엇이든 물어보시면 최선을 다해 정확한 정보를 제공해 드리겠습니다!</p>
        </div>
    ''', unsafe_allow_html=True)

# --- 채팅 메시지 표시 ---
for i, msg in enumerate(st.session_state.messages):
    role = msg["role"]
    # 메시지 내용을 HTML로 변환하여 줄바꿈을 적용합니다.
    content_html = msg["content"]
    # HTML 태그로 이미 포함된 경우, `br` 태그를 추가하지 않도록 조건 추가
    if not any(tag in content_html for tag in ['<br>', '<p>', '<div>', '<ul>', '<ol>', '<h3>', '<strong>', '<small>']):
        content_html = content_html.replace("\n", "<br>")

    timestamp = msg.get("time", datetime.now().strftime("%H:%M"))
    class_name = "chat-user" if role == "user" else "chat-bot"
    icon = "fas fa-user-graduate" if role == "user" else "fas fa-university"
    
    st.markdown(f'''
        <div class="{class_name}">
            <div class="chat-message-wrapper">
                <div class="chat-icon"><i class="{icon}"></i></div>
                <div class="chat-bubble-content">
                    <div class="chat-message">{content_html}</div>
                    <div class="message-timestamp">{timestamp}</div>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    # 봇 메시지에만 복사 버튼 및 디버그 정보 표시
    if role == "assistant":
        if "copy_text" in msg: # 복사 버튼은 순수 텍스트를 복사하도록
            st.markdown(f"""
                <div class="copy-button-container">
                    <button class="copy-button" onclick="
                        navigator.clipboard.writeText(`{msg['copy_text'].replace('`', '\\`')}`)
                        .then(() => alert('답변이 클립보드에 복사되었습니다!'))
                        .catch(err => console.error('복사 실패:', err));
                    ">
                        <i class="far fa-copy"></i> 복사
                    </button>
                </div>
                """, unsafe_allow_html=True)
        
        if "debug_source_content" in msg and st.session_state.get("show_debug_info", False):
            st.markdown(f"""
                <div class="debug-info-box">
                    <strong>[디버그 정보 - 검색된 문서 내용]</strong>
                    <pre>{msg['debug_source_content']}</pre>
                </div>
            """, unsafe_allow_html=True)

# 챗봇 타이핑 인디케이터
if "last_user_input" in st.session_state and st.session_state.last_user_input and \
   st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    st.markdown(f'''
        <div class="chat-bot">
            <div class="chat-message-wrapper">
                <div class="chat-icon"><i class="fas fa-university"></i></div>
                <div class="chat-bubble-content">
                    <div class="chat-message">
                        <div class="typing-indicator">
                            챗봇이 입력 중<span>.</span><span>.</span><span>.</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

st.markdown('</div></div>', unsafe_allow_html=True) # chat-container 및 chat-wrapper 닫기

# --- 입력 폼 및 버튼 ---
st.markdown('<div class="input-form-container">', unsafe_allow_html=True)
with st.form("chat_input_form", clear_on_submit=True):
    input_col, button_col = st.columns([8, 1]) # "새 대화" 버튼을 사이드바로 옮겼으므로 컬럼 조정
    with input_col:
        user_input = st.text_input(
            "한밭대학교 규정에 대해 물어보세요...", label_visibility="collapsed",
            disabled=(not api_key_set or not rag_ready), placeholder="여기에 질문을 입력하세요..."
        )
    with button_col:
        submitted = st.form_submit_button("⬆️", help="질문 전송", disabled=(not api_key_set or not rag_ready))

# --- "새 대화" 확인 모달 ---
if st.session_state.show_new_chat_confirm:
    st.markdown("""<div class="st-modal-container">""", unsafe_allow_html=True)
    with st.container():
        st.markdown("""<div class="st-modal-content">""", unsafe_allow_html=True)
        st.markdown("<h4>현재 대화 내용을 초기화하고 새로운 대화를 시작하시겠습니까?</h4>", unsafe_allow_html=True)
        st.markdown("""<div class="st-modal-buttons">""", unsafe_allow_html=True)
        
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("확인", key="confirm_new_chat", help="대화 초기화", type="secondary", class_name="confirm-btn"): # class_name 추가
                st.session_state.messages = [] # 현재 표시된 대화 초기화
                st.session_state.current_session_id = str(uuid.uuid4()) # 새로운 고유 세션 ID 생성
                st.session_state.title_set_for_current_session = False # 새 세션이므로 제목 미설정 상태로

                # 새 세션 정보를 DB에 기록
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                new_session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                c.execute("INSERT INTO chat_sessions (session_id, title, start_time, last_updated) VALUES (?, ?, ?, ?)",
                          (st.session_state.current_session_id, "새로운 대화", new_session_start_time, new_session_start_time))
                conn.commit()
                conn.close()

                new_initial_message = "새로운 대화를 시작합니다. 무엇이든 물어보세요!"
                if not rag_ready and api_key_set: new_initial_message = "새 대화 시작. (문서 학습 문제로 답변 제한적일 수 있음)"
                elif not api_key_set: new_initial_message = "새 대화 시작. (API 키 설정 필요)"
                st.session_state.messages.append(
                    {"role": "assistant", "content": new_initial_message, "time": datetime.now().strftime("%H:%M")}
                )
                st.session_state.show_new_chat_confirm = False
                st.toast("새 대화가 시작되었습니다!", icon="💬")
                st.rerun()
        with col_cancel:
            if st.button("취소", key="cancel_new_chat", help="초기화 취소", type="primary", class_name="cancel-btn"): # class_name 추가
                st.session_state.show_new_chat_confirm = False
                st.rerun()
        st.markdown("""</div></div></div>""", unsafe_allow_html=True)

# --- (선택 사항) 대화 삭제 확인 모달 ---
if st.session_state.show_delete_confirm:
    st.markdown("""<div class="st-modal-container">""", unsafe_allow_html=True)
    with st.container():
        st.markdown("""<div class="st-modal-content">""", unsafe_allow_html=True)
        st.markdown("<h4>현재 대화를 정말 삭제하시겠습니까?</h4><p style='color:#dc3545;'>이 작업은 되돌릴 수 없습니다.</p>", unsafe_allow_html=True)
        st.markdown("""<div class="st-modal-buttons">""", unsafe_allow_html=True)
        
        col_confirm_del, col_cancel_del = st.columns(2)
        with col_confirm_del:
            if st.button("삭제", key="confirm_delete_chat", help="현재 대화 삭제", type="secondary", class_name="confirm-btn"):
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                c.execute("DELETE FROM chat_messages WHERE session_id = ?", (st.session_state.current_session_id,))
                c.execute("DELETE FROM chat_sessions WHERE session_id = ?", (st.session_state.current_session_id,))
                conn.commit()
                conn.close()
                
                # 삭제 후 새로운 세션 시작
                st.session_state.messages = []
                st.session_state.current_session_id = str(uuid.uuid4())
                st.session_state.title_set_for_current_session = False
                
                # 새로운 세션 DB에 기록
                conn = sqlite3.connect(DB_NAME)
                c = conn.cursor()
                new_session_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                c.execute("INSERT INTO chat_sessions (session_id, title, start_time, last_updated) VALUES (?, ?, ?, ?)",
                          (st.session_state.current_session_id, "새로운 대화", new_session_start_time, new_session_start_time))
                conn.commit()
                conn.close()

                initial_message_content = "새로운 대화를 시작합니다. 무엇이든 물어보세요!"
                if not rag_ready and api_key_set: initial_message_content = "새 대화 시작. (문서 학습 문제로 답변 제한적일 수 있음)"
                elif not api_key_set: initial_message_content = "새 대화 시작. (API 키 설정 필요)"
                st.session_state.messages.append(
                    {"role": "assistant", "content": initial_message_content, "time": datetime.now().strftime("%H:%M")}
                )
                st.session_state.show_delete_confirm = False
                st.toast("대화가 삭제되었습니다.", icon="🗑️")
                st.rerun()
        with col_cancel_del:
            if st.button("취소", key="cancel_delete_chat", help="삭제 취소", type="primary", class_name="cancel-btn"):
                st.session_state.show_delete_confirm = False
                st.rerun()
        st.markdown("""</div></div></div>""", unsafe_allow_html=True)


st.markdown('</div>', unsafe_allow_html=True) # input-form-container 닫기

# --- 사용자 입력 처리 및 답변 생성 로직 ---
if api_key_set and rag_ready:
    if submitted and user_input:
        current_time = datetime.now().strftime("%H:%M")
        st.session_state.messages.append({"role": "user", "content": user_input, "time": current_time})
        
        # 사용자 메시지 저장: 현재 세션의 첫 질문인지 확인하여 제목 자동 생성 로직 실행
        is_initial_question_for_session_title = not st.session_state.title_set_for_current_session
        save_message(st.session_state.current_session_id, "user", user_input, is_initial_question=is_initial_question_for_session_title)
        
        if is_initial_question_for_session_title:
            st.session_state.title_set_for_current_session = True # 제목 설정 완료 플래그

        st.session_state.last_user_input = user_input
        st.rerun()

    if "last_user_input" in st.session_state and st.session_state.last_user_input and \
       st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
        
        query_to_process = st.session_state.last_user_input
        del st.session_state.last_user_input
        
        with st.spinner("답변을 생성 중입니다... 문서를 참고하고 있어요! 🤔"):
            try:
                response = qa_chain.invoke({"query": query_to_process})
                llm_answer = response["result"]
                source_docs = response.get("source_documents", [])
                
                final_reply_content = llm_answer
                copy_text_content = llm_answer # 복사할 텍스트는 순수 답변 내용으로 시작
                
                debug_source_content = ""
                if source_docs:
                    cited_sources_filenames = sorted(list(set(
                        os.path.basename(doc.metadata.get("source", "알 수 없는 출처")).replace(".txt", "")
                        for doc in source_docs
                    )))
                    
                    sources_html = "<div class='source-documents'><strong><i class='fas fa-file-alt'></i> 참고 문서:</strong><ul>"
                    for filename in cited_sources_filenames:
                        sources_html += f"<li><i class='fas fa-check-circle'></i> {filename}</li>"
                    sources_html += "</ul></div>"
                    final_reply_content += sources_html # UI에 표시할 내용에만 HTML 추가
                    
                    copy_text_content += "\n\n--- 참고 문서 ---\n" + ", ".join(cited_sources_filenames) # 복사 텍스트에는 순수 텍스트로 추가

                    debug_source_content = "\n\n".join([
                        f"--- {os.path.basename(doc.metadata.get('source', '알 수 없는 출처'))} (시작 인덱스: {doc.metadata.get('start_index', 'N/A')}) ---\n{doc.page_content}"
                        for doc in source_docs
                    ])
                    

            except openai.AuthenticationError:
                final_reply_content = "⚠️ OpenAI API 인증 오류가 발생했습니다. API 키가 유효한지 또는 사용량 한도를 확인해주세요."
                copy_text_content = final_reply_content
                st.error(final_reply_content)
            except openai.RateLimitError:
                final_reply_content = "⚠️ API 호출 한도 초과 오류입니다. 잠시 후 다시 시도해주시거나 API 플랜을 확인해주세요."
                copy_text_content = final_reply_content
                st.error(final_reply_content)
            except Exception as e:
                final_reply_content = f"⚠️ 답변 생성 중 오류가 발생했습니다: {str(e)}"
                copy_text_content = final_reply_content
                st.error(final_reply_content)
            
            current_time = datetime.now().strftime("%H:%M")
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_reply_content,
                "time": current_time,
                "copy_text": copy_text_content, 
                "debug_source_content": debug_source_content
            })
            save_message(st.session_state.current_session_id, "assistant", copy_text_content) # 봇 답변 저장 (출처 포함 텍스트)
            st.rerun()

elif not api_key_set:
    pass
elif not rag_ready:
    pass

# --- 디버그 모드 토글 버튼 (개발 시 유용) ---
with st.sidebar:
    st.checkbox("디버그 정보 표시 (참고 문서 내용)", key="show_debug_info", value=False,
                help="챗봇 답변 아래에 LLM이 참고한 문서 청크의 원본 내용을 표시합니다. 문제 해결에 유용합니다.")

# --- 자동 스크롤 JavaScript (MutationObserver 사용) ---
st.markdown("""
<script>
    function scrollToBottom() {
        const chatContainer = document.getElementById("chat-container-scroll");
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }

    // 초기 로드 시 스크롤
    setTimeout(scrollToBottom, 100); 

    const targetNode = document.getElementById('chat-container-scroll');
    if (targetNode) {
        const config = { childList: true, subtree: true };
        const callback = function(mutationsList, observer) {
            for(let mutation of mutationsList) {
                if (mutation.type === 'childList' || mutation.type === 'characterData') {
                    // 메시지가 추가되거나 내용이 변경될 때 (타이핑 효과 등) 스크롤
                    scrollToBottom();
                }
            }
        };
        const observer = new MutationObserver(callback);
        observer.observe(targetNode, config);

        window.addEventListener('beforeunload', () => {
            observer.disconnect();
        });
    }
</script>
""", unsafe_allow_html=True)