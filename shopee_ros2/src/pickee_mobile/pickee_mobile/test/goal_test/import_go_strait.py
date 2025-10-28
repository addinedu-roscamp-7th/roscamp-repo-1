import time
from pickee_mobile.module.module_go_strait import run

def main():
    print("🚀 0.5m 전진 시작!")
    run(0.17)
    wait_seconds = 2
    print(f"⏳ {wait_seconds}초 대기...")
    
    time.sleep(wait_seconds)
    print("✅ 전진 완료!")
    run(-0.17)

if __name__ == '__main__':
    main()
