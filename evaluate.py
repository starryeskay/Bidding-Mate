import os
import json
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

# GPT-5 등 미래 모델명 대응을 위한 커스텀 클래스
class GPT5ChatOpenAI(ChatOpenAI):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # Ragas가 무슨 값을 보내든 상관없이 temperature를 1로 강제 고정
        kwargs["temperature"] = 1
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        # 비동기 호출 시에도 temperature를 1로 강제 고정
        kwargs["temperature"] = 1
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)

# 2. RAG 시스템 불러오기
rag_system = BiddingAgent()

# 3. 채점관 설정 (온도 1로 초기화)
judge_llm = GPT5ChatOpenAI(
    model="gpt-5",  # 실제 사용 가능한 모델명으로 변경 필요 (예: o1-preview, gpt-4o 등)
    temperature=1   # 초기값도 1로 설정
)
judge_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 4. 테스트 데이터 로드 (JSON 파일 불러오기)
json_file_path = "test_data.json"

questions = []
answers = []
contexts = []
ground_truths = []

try:
    with open(json_file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
        
    print("시험을 치고 있습니다...")

    # Case A: {"question": [...], "ground_truth": [...]} 형태 (Dict of Lists)
    if isinstance(raw_data, dict) and "question" in raw_data and isinstance(raw_data["question"], list):
        print(f"'{json_file_path}'에서 {len(raw_data['question'])}개의 데이터를 불러옵니다. (Dict of Lists 구조)")
        
        # zip으로 묶어서 순회
        target_questions = raw_data["question"]
        target_gts = raw_data["ground_truth"]
        
        for q_text, gt_text in zip(target_questions, target_gts):
            # RAG 시스템에 질문 던지기
            result = rag_system.ask_with_context(q_text)
            
            questions.append(result["question"])
            answers.append(result["answer"])
            contexts.append(result["contexts"])
            ground_truths.append(gt_text)

    # Case B: [{"question": "...", "ground_truth": "..."}, ...] 형태 (List of Dicts)
    elif isinstance(raw_data, list):
        print(f"'{json_file_path}'에서 {len(raw_data)}개의 데이터를 불러옵니다. (List of Dicts 구조)")
        for item in raw_data:
            q_text = item.get("question")
            gt_text = item.get("ground_truth")
            
            result = rag_system.ask_with_context(q_text)
            
            questions.append(result["question"])
            answers.append(result["answer"])
            contexts.append(result["contexts"])
            ground_truths.append(gt_text)
            
    else:
        print("지원하지 않는 데이터 형식입니다.")
        exit()

except FileNotFoundError:
    print(f"오류: '{json_file_path}' 파일을 찾을 수 없습니다.")
    exit()
except json.JSONDecodeError:
    print(f"오류: '{json_file_path}' 파일 형식이 올바르지 않습니다.")
    exit()

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

# 7. 결과 출력 및 저장
print(result)

# DataFrame으로 변환
df = result.to_pandas()

# CSV 저장
df.to_csv("ragas_score.csv", index=False)
print("상세 결과가 'ragas_score.csv'로 저장되었습니다.")

# 오답 노트 출력
print("\n=== 오답 노트 및 상세 확인 ===")

def get_col(row, candidates):
    for col in candidates:
        if col in row and row[col] is not None:
            try:
                import pandas as pd
                if pd.isna(row[col]): continue
            except: pass
            return row[col]
    return "N/A"

for i, row in df.iterrows():
    # 1. 질문 (Ragas가 user_input으로 바꿈)
    q_text = get_col(row, ['user_input', 'question'])
    
    # 2. AI 답변 (Ragas가 response로 바꿈)
    a_text = get_col(row, ['response', 'answer'])
    
    # 3. 정답 (Ragas가 reference로 바꿈)
    gt_text = get_col(row, ['reference', 'ground_truth', 'ground_truths'])
    
    # 4. 참고 문서
    ctx_text = get_col(row, ['retrieved_contexts', 'contexts'])
    
    print(f"\nQ{i+1}: {q_text}")
    print(f"정답(GT): {gt_text}")
    print(f"AI 답변: {a_text}")
    
    if isinstance(ctx_text, list) and len(ctx_text) > 0:
        print(f"가져온 문서(Context): {str(ctx_text[0])[:100]}...") 
    elif isinstance(ctx_text, str):
        print(f"가져온 문서(Context): {ctx_text[:100]}...")
    else:
        print("가져온 문서(Context): (없음)")