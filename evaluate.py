import os
from dotenv import load_dotenv
from rag_core import BiddingAgent
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.run_config import RunConfig

# 1. 환경변수 로드
load_dotenv()

class GPT5ChatOpenAI(ChatOpenAI):
    # 1. 동기 호출 방어
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._fix_temperature(kwargs)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    # 2. 비동기(Async) 호출 방어
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self._fix_temperature(kwargs)
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

    # 온도 수정 함수 (무조건 1.0으로 만듦)
    def _fix_temperature(self, kwargs):
        if "temperature" in kwargs:
            del kwargs["temperature"]
        self    .temperature = 1.0
        if self.model_kwargs and "temperature" in self.model_kwargs:
             del self.model_kwargs["temperature"]

# 2. RAG 시스템 불러오기
rag_system = BiddingAgent()

# 3. 채점관 설정
judge_llm = GPT5ChatOpenAI(model="gpt-5") 
judge_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 4. 테스트 데이터
# 이곳에 테스트 질문과 답변 넣기
test_data = [
    # 1. [예산]
    {
        "question": "벤처확인종합관리시스템 기능 고도화 용역사업의 소요 예산은 얼마야?",
        "ground_truth": "352,000,000원(부가가치세 포함)입니다."
    },

    # 2. [기간]
    {
        "question": "벤처확인종합관리시스템 기능 고도화 용역사업의 사업 기간은 어떻게 돼?",
        "ground_truth": "계약일로부터 150일입니다."
    },

    # 3. [평가]
    {
        "question": "벤처확인종합관리시스템 기능 고도화 용역사업의 제안서 평가 방법과 배점 기준을 알려줘.",
        "ground_truth": "기술평가 90%와 가격평가 10%를 합산하여 종합평가점수를 산출합니다."
    }

]

print("시험을 치고 있습니다...")

questions = []
answers = []
contexts = []
ground_truths = []

for item in test_data:
    result = rag_system.ask_with_context(item["question"])
    questions.append(result["question"])
    answers.append(result["answer"])
    contexts.append(result["contexts"])
    ground_truths.append(item["ground_truth"])

# 5. 데이터셋 변환
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
}
dataset = Dataset.from_dict(data)

# 6. 채점 시작
print("채점 중입니다...")

my_run_config = RunConfig(timeout=360)

result = evaluate(
    dataset=dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ],
    llm=judge_llm,
    embeddings=judge_embeddings,
    run_config=my_run_config
)

# 7. 결과 출력
print(result)

# DataFrame으로 변환
df = result.to_pandas()

# CSV 저장
df.to_csv("ragas_score.csv", index=False)
print("상세 결과가 'ragas_score.csv'로 저장되었습니다.")

print("오답 노트")

def get_col(row, candidates):
    for col in candidates:
        if col in row:
            return row[col]
    return "N/A"

for i, row in df.iterrows():
    q_text = get_col(row, ['question', 'user_input'])
    a_text = get_col(row, ['answer', 'response'])
    gt_text = get_col(row, ['ground_truth', 'ground_truths'])
    ctx_text = get_col(row, ['contexts', 'retrieved_contexts'])
    
    print(f"\nQ{i+1}: {q_text}")
    print(f"정답: {gt_text}")
    print(f"AI 답변: {a_text}")
    
    if isinstance(ctx_text, list) and len(ctx_text) > 0:
        print(f"가져온 문서(Context): {ctx_text[0][:500]}...") 
    else:
        print("가져온 문서(Context): (없음)")