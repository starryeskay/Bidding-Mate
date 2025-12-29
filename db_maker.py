import os
import re
import shutil
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pdfminer.pdfparser import PDFSyntaxError

# 0. 환경변수 로드 (API KEY 확인용)
load_dotenv()

# 1. 설정
PDF_FOLDER = "./data/raw/100_PDF"   # PDF 원본 폴더
DB_PATH = "./chroma_db_final"      # 최종 DB 저장 경로

# 기존 DB 폴더가 있다면 삭제하고 새로 시작
if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)
    print(f"기존 DB 폴더({DB_PATH})를 삭제하고 새로 만듭니다.")

# 2. 텍스트 청소 함수 (앵무새 죽이기 & 선 제거)
def clean_text(text):
    if not text:
        return ""
    
    # 1. 줄바꿈, 탭 정리
    text = text.replace('\r\n', '\n').replace('\t', ' ')
    
    # 2. 무의미한 문구 제거
    text = text.replace('- 이 하 여 백 -', '').replace('이 하 여 백', '')
    
    # 3. 점선, 실선 등 특수문자 반복 제거 (......, -----)
    text = re.sub(r'[\.\-\=_]{3,}', '', text)

    # 4. [핵심] 단어 반복 제거 (띄어쓰기 있는 경우: "특성화 특성화")
    # 같은 단어가 2번 이상 반복되면 1번만 남김
    text = re.sub(r'(\b\w+\b)( \1){2,}', r'\1', text)

    # 5. [핵심] 단어 반복 제거 (띄어쓰기 없는 경우: "고도화고도화")
    text = re.sub(r'(\w{2,})(\1){2,}', r'\1', text)
    
    # 6. 다중 공백 및 줄바꿈 정리
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n\n', text)
    
    return text.strip()

# 3. 문서 로드 및 전처리
documents = []
print(f"'{PDF_FOLDER}' 폴더에서 PDF 로딩 시작...")

if not os.path.exists(PDF_FOLDER):
    print(f"오류: 폴더가 없습니다! 경로를 확인해주세요: {PDF_FOLDER}")
    exit()

files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
print(f"   -> 총 {len(files)}개의 PDF 파일을 찾았습니다.")

for i, file in enumerate(files):
    file_path = os.path.join(PDF_FOLDER, file)
    
    try:
        # PDFPlumber로 로드 (표 인식에 강함)
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        
        # 전처리 및 메타데이터 정리
        for doc in docs:
            # A. 텍스트 청소 (반복 제거 등)
            doc.page_content = clean_text(doc.page_content)
            
            # B. 메타데이터 다이어트 (불필요한 정보 삭제)
            # '텍스트' 키가 있으면 삭제 (용량 절약)
            if "텍스트" in doc.metadata: 
                del doc.metadata["텍스트"]
            
            # 파일 형식 정보 삭제
            if "file_type" in doc.metadata: 
                del doc.metadata["file_type"]
                
            # 출처 명확하게 저장
            doc.metadata["source"] = file 
            
        documents.extend(docs)
        # 진행 상황 출력 (10개 단위로 로그)
        if (i + 1) % 10 == 0:
            print(f"   [{i+1}/{len(files)}] 처리 중...")
        
    except PDFSyntaxError:
        print(f"   [{i+1}/{len(files)}] [Skip] 손상된 파일 건너뜀: {file}")
    except Exception as e:
        print(f"   [{i+1}/{len(files)}] [Fail] 알 수 없는 오류: {file} ({e})")

print(f"\n총 {len(documents)} 페이지 로드 완료!")

# 4. 청킹
print("텍스트 분할 시작...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 한 덩어리 크기
    chunk_overlap=200,    # 문맥 유지를 위해 겹치는 구간
    separators=["\n\n", "\n", " ", ""] # 자르는 우선순위
)

split_docs = text_splitter.split_documents(documents)
print(f"   -> 총 {len(split_docs)}개의 청크로 분할되었습니다.")

# 5. 벡터 DB 저장
print("벡터 DB 저장 중...")

# 임베딩 모델 준비
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# ChromaDB 생성 및 저장
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding_model,
    persist_directory=DB_PATH
)

print(f"\n DB 생성이 끝났습니다!")
print(f"저장 경로: {DB_PATH}")