# 로봇팔 작업 상태 (Pickee, Packee 공통)

int32 robot_id
int32 order_id
int32 product_id
string status              # in_progress, completed, failed
string current_phase       # 현재 작업 단계
float32 progress           # 진행률 (0.0~1.0)
string message             # 상세 메시지
string arm_side            # Packee만 사용 (left/right), Pickee는 빈 문자열 ""
