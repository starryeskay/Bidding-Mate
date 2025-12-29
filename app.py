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
    return BiddingAgent()

try:
    agent = load_agent()
except Exception as e:
    st.error(f"시스템 초기화 오류: {e}")
    st.stop()

# 대화 히스토리 출력 루프
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "docs" in message and message["docs"]:
            with st.expander("📚 참고 문서 보기"):
                for i, doc in enumerate(message["docs"]):
                    st.markdown(f"**[문서 {i+1}]**")
                    st.text(doc[:500] + "...")

# 채팅 입력 및 처리
if prompt := st.chat_input("궁금한 점을 물어보세요..."):
    # 1. 사용자 질문 추가 및 화면 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 어시스턴트 답변 생성 및 화면 표시
    with st.chat_message("assistant"):
        with st.spinner("분석 중..."):
            try:
                # 에이전트에게 질문하여 답변과 문서 리스트를 받아옴
                answer, docs = agent.get_answer(prompt)
                
                # 답변 텍스트 먼저 출력
                st.markdown(answer)
                
                # docs가 존재할 때만(라우터가 yes일 때만) expander 생성
                if docs and len(docs) > 0:
                    with st.expander("📚 참고 문서 보기"):
                        for i, doc in enumerate(docs):
                            st.markdown(f"**[문서 {i+1}]**")
                            st.text(doc[:500] + "...")

                # 3. 세션 상태에 답변과 문서를 함께 저장 (docs 포함이 핵심!)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer, 
                    "docs": docs
                })
            except Exception as e:
                st.error(f"오류 발생: {e}")