import streamlit as st
from rag_core import BiddingAgent

# 페이지 설정
st.set_page_config(page_title="Bidding Mate", layout="wide")
st.title("입찰 공고 분석 AI")

# 사이드바
with st.sidebar:
    st.header("System Info")
    st.success("System Status: Online")
    st.info("Module: LangGraph + OOP Applied")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 에이전트 로딩
@st.cache_resource
def load_agent():
    return BiddingAgent() # rag_core.py의 클래스 실행

try:
    agent = load_agent()
except Exception as e:
    st.error(f"시스템 초기화 오류: {e}")
    st.stop()

# 채팅 인터페이스
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("궁금한 점을 물어보세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("분석 중..."):
            try:
                # 에이전트에게 질문하기
                answer, docs = agent.get_answer(prompt)
                
                st.markdown(answer)
                
                # 근거 문서 보여주기
                with st.expander("📚 참고 문서 보기"):
                    for i, doc in enumerate(docs):
                        st.markdown(f"**[문서 {i+1}]**")
                        st.text(doc[:200] + "...") # 200자 미리보기

                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"오류 발생: {e}")