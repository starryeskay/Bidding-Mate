import os
from typing import TypedDict, List
from dotenv import load_dotenv

# LangChain & LangGraph 관련 임포트
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

# 환경변수 로드
load_dotenv()

# [OOP 적용] RAG 시스템을 하나의 클래스로 정의
class BiddingAgent:
    def __init__(self, db_path="./chroma_db_final", model_name="gpt-5"):
        """
        초기화: DB 로드, LLM 설정, 그래프(Workflow) 빌드
        """
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 1. DB 연결
        if not os.path.exists(db_path):
            print(f"경고: {db_path}를 찾을 수 없습니다. 현재 위치: {os.getcwd()}")
            
        self.vectorstore = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}    # 문서 5개 가져오기
        )
        
        # 2. LangGraph 워크플로우 빌드
        self.app_workflow = self._build_graph()

    # LangGraph 상태(State) 정의
    class GraphState(TypedDict):
        question: str
        context: List[str]
        answer: str
        relevance: str # yes/no

    # 노드 함수들
    def _retrieve(self, state):
        """문서 검색 단계"""
        print(f"검색 중: {state['question']}")
        docs = self.retriever.invoke(state['question'])
        context = [doc.page_content for doc in docs]
        return {"context": context}

    def _grade_documents(self, state):
        """문서 품질 평가 (LangGraph 핵심)"""
        context = state['context']
        # 문서가 하나라도 있으면 통과
        if not context:
            return {"relevance": "no"}
        return {"relevance": "yes"}

    def _generate(self, state):
        """답변 생성 단계"""
        question = state['question']
        context = "\n\n".join(state['context'])
        
        prompt = ChatPromptTemplate.from_template(
            """
            당신은 공공 입찰(Bidding) 전문가입니다. 
            아래 제공된 [참고 문서]를 바탕으로 질문에 대해 명확하게 답변하세요.
            
            [참고 문서]
            {context}
            
            질문: {question}
            답변:
            """
        )
        
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "question": question})
        return {"answer": response}

    def _rewrite_query(self, state):
        """검색 실패 시 처리"""
        return {"answer": "죄송합니다. 관련 문서를 찾을 수 없어 답변하기 어렵습니다."}

    # 그래프 조립
    def _build_graph(self):
        workflow = StateGraph(self.GraphState)
        
        # 노드 추가
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade", self._grade_documents)
        workflow.add_node("generate", self._generate)
        workflow.add_node("fallback", self._rewrite_query)
        
        # 엣지 연결
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade")
        
        # 조건부 분기
        def decide_next(state):
            return "generate" if state["relevance"] == "yes" else "fallback"

        workflow.add_conditional_edges("grade", decide_next, {"generate": "generate", "fallback": "fallback"})
        workflow.add_edge("generate", END)
        workflow.add_edge("fallback", END)
        
        return workflow.compile()

    # 외부 호출 함수
    def get_answer(self, question: str):
        inputs = {"question": question}
        result = self.app_workflow.invoke(inputs)
        return result['answer'], result.get('context', [])