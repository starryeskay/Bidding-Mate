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
        self.llm = ChatOpenAI(model=model_name)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # 1. DB 연결
        if not os.path.exists(db_path):
            print(f"경고: {db_path}를 찾을 수 없습니다. 현재 위치: {os.getcwd()}")
            
        self.vectorstore = Chroma(persist_directory=db_path, embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 10,
                "fetch_k" : 30,
                "lambda_mult": 0.6
            } 
        )
        
        # 2. LangGraph 워크플로우 빌드
        self.app_workflow = self._build_graph()

    # LangGraph 상태(State) 정의
    class GraphState(TypedDict):
        question: str
        context: List[str]
        answer: str
        relevance: str # yes/no

    def _route_question(self, state):
        print(f"---의도 파악 중: {state['question']}---")
        router_prompt = ChatPromptTemplate.from_template(
            "당신은 질문의 의도를 분류하는 라우터입니다.\n"
            "질문이 공공 입찰, 사업 공고, 제안서 작성 등 '사업 관련 질문'이면 'bid'라고 답하세요.\n"
            "그 외의 일상적인 질문이나 잡담(음식, 연예, 일반 상식 등)이면 'off-topic'이라고 답하세요.\n\n"
            "질문: {question}\n"
            "답변(한 단어):"
        )
        chain = router_prompt | self.llm | StrOutputParser()
        category = chain.invoke({"question": state['question']}).lower()
        
        # 'bid'면 통과(yes), 아니면 차단(no)
        return {"relevance": "yes" if "bid" in category else "no"}

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
            당신은 공공 입찰 및 사업 공고 분석 전문가입니다. 
            당신의 임무는 오로지 제공된 [참고 문서]의 내용을 바탕으로 질문에 답하는 것입니다.

            [엄격 준수 지침]
            1. 주제 제한: 질문이 공공 입찰, 사업 내용, 제안서 작성 등 본 사업과 관련 없는 내용(예: 일상 대화, 음식 추천, 일반 상식 등)일 경우, 
               "죄송합니다. 저는 공고문 분석 전문가로서 해당 질문에 대해서는 답변을 드릴 수 없습니다."라고만 답변하세요.
            
            2. 근거 기반 답변: 반드시 제공된 [참고 문서]에 명시된 사실만 답변하세요. 
            
            3. 정보 부재 시: 질문이 사업과 관련은 있지만 [참고 문서]에 관련 내용이 없는 경우, 
               "제공된 문서에서 관련 정보를 찾을 수 없습니다."라고 답변하세요.

            4. 사업명 일치 확인: 질문에서 언급한 '사업명'과 [참고 문서]의 '사업명'이 다를 경우, 
                해당 문서는 무시하고 "관련 정보를 찾을 수 없습니다"라고 답하세요.

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
        """잡담이거나 검색 실패 시 처리"""
        return {"answer": "죄송합니다. 저는 공고문 분석 전문가로서 사업 및 입찰과 관련된 질문에만 답변을 드릴 수 있습니다."}

    # 그래프 조립
    def _build_graph(self):
        workflow = StateGraph(self.GraphState)
        
        # (1) 노드 추가 - router 추가됨
        workflow.add_node("router", self._route_question) 
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("grade", self._grade_documents)
        workflow.add_node("generate", self._generate)
        workflow.add_node("fallback", self._rewrite_query)
        
        # (2) 시작점을 retrieve에서 router로 변경!
        workflow.set_entry_point("router")
        
        # (3) 라우터 결과에 따른 조건부 분기 추가
        workflow.add_conditional_edges(
            "router",
            lambda x: "proceed" if x["relevance"] == "yes" else "reject",
            {
                "proceed": "retrieve", # 사업 질문이면 검색으로
                "reject": "fallback"   # 잡담이면 바로 거절로
            }
        )
        
        # (4) 나머지 연결 (기존과 동일)
        workflow.add_edge("retrieve", "grade")
        
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
        
        # 라우터가 'no'라고 했거나, 답변이 거절 멘트인 경우 컨텍스트를 비움
        answer = result.get('answer', '')
        # "죄송합니다"로 시작하는 거절 답변일 경우 참고 문서를 강제로 비웁니다.
        if "죄송합니다" in answer or result.get('relevance') == 'no':
            return answer, [] 
            
        return answer, result.get('context', [])
    
    def ask_with_context(self, question):
        """
        Ragas 평가를 위해 질문, 답변, 근거문서(Context)를 딕셔너리로 반환하는 함수
        """
        # 1. 실제 우리가 쓰는 get_answer 함수를 호출해서 답변과 근거를 가져옵니다.
        answer, contexts = self.get_answer(question)
        
        # 2. Ragas가 좋아하는 포맷으로 포장해서 줍니다.
        return {
            "question": question,
            "answer": answer,
            "contexts": contexts  # 리스트 형태 ([문서1, 문서2...])
        }