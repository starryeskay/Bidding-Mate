import os

# 멘토님 조언 + 팀원 협업을 위한 디렉토리 구조 정의
folders = [
    "configs",              # 설정 파일 (yaml 등)
    "data/raw",             # 원본 데이터 (hwp, pdf) - 깃허브 업로드 제외됨
    "data/processed",       # 전처리된 데이터
    "shared/utils",         # 공용 유틸리티 함수
    "shared/models",        # 공용 모델 관련 코드
    "team_members/obj",     # 팀장 폴더
    "team_members/member2", # 팀원 2 (나중에 이름 변경)
    "team_members/member3", # 팀원 3 (나중에 이름 변경)
    "team_members/member4", # 팀원 4 (나중에 이름 변경)
    "team_members/member5", # 팀원 5 (나중에 이름 변경)
]

# 폴더 생성
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, ".gitkeep"), "w") as f:
        pass

# .gitignore 생성 (중요: API Key, 큰 데이터 유출 방지)
gitignore_content = """
# 환경 변수 및 키 파일 (절대 올리지 말 것)
.env
.env.*
keys/
*.key

# 파이썬 자동 생성 파일
__pycache__/
*.pyc
venv/
.venv/

# 대용량 데이터 (GitHub 용량 제한 방지)
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep
*.hwp
*.pdf

# 시스템 파일
.DS_Store
"""

with open(".gitignore", "w", encoding="utf-8") as f:
    f.write(gitignore_content)

# README.md 생성
readme_content = """
# Team 4: RFP RAG Project

## 프로젝트 소개
입찰 제안요청서(RFP) 기반의 질의응답 시스템을 구축하는 프로젝트입니다.

## 폴더 구조
- `configs/`: 학습 및 실행 설정 파일 관리
- `data/`: 데이터셋 (Raw 데이터는 업로드 되지 않음)
- `shared/`: 공통으로 사용할 모듈 및 유틸리티
- `team_members/`: 팀원별 개인 작업 공간 (Sandbox)

## 협업 규칙
1. 본인 폴더(`team_members/본인이름`)에서 자유롭게 작업
2. `main` 브랜치에는 검증된 코드만 병합(Merge)
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme_content)