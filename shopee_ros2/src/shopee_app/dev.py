import sys
import time
import signal
import subprocess
from pathlib import Path

from watchdog.observers import Observer
from watchdog.events import FileSystemEvent
from watchdog.events import FileSystemEventHandler

ROOT = Path(__file__).parent.resolve()
MODULE_ROOT = ROOT / "shopee_app"
UI_FILES = {
    MODULE_ROOT / "ui" / "main_window.ui": MODULE_ROOT / "ui_gen" / "main_window.py",
    MODULE_ROOT / "ui" / "layout_user.ui": MODULE_ROOT / "ui_gen" / "layout_user.py",
    MODULE_ROOT / "ui" / "layout_admin.ui": MODULE_ROOT / "ui_gen" / "layout_admin.py",
    MODULE_ROOT
    / "ui"
    / "promoded_class.ui": MODULE_ROOT
    / "ui_gen"
    / "promoded_class.py",
    MODULE_ROOT / "ui" / "cart_item.ui": MODULE_ROOT / "ui_gen" / "cart_item.py",
    MODULE_ROOT
    / "ui"
    / "dialog_profile.ui": MODULE_ROOT
    / "ui_gen"
    / "dialog_profile.py",
    MODULE_ROOT
    / "ui"
    / "cart_select_item.ui": MODULE_ROOT
    / "ui_gen"
    / "cart_select_item.py",
}

OUT_DIR = MODULE_ROOT / "ui_gen"


def log(msg: str):
    print(f"[dev] {msg}", flush=True)


def build_ui(
    src=MODULE_ROOT / "ui" / "main_window.ui",
    dest=MODULE_ROOT / "ui_gen" / "main_window.py",
):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not (OUT_DIR / "__init__.py").exists():
        (OUT_DIR / "__init__.py").write_text("")  # 패키지 보장
    cmd = [sys.executable, "-m", "PyQt6.uic.pyuic", "-o", str(dest), str(src)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"pyuic6 실패\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
    log(f".ui -> .py 변환 완료: {dest.relative_to(MODULE_ROOT)}")


class DebouncedHandler(FileSystemEventHandler):
    def __init__(self, on_change, debounce_ms=150):
        self.on_change = on_change
        self.debounce = debounce_ms / 1000.0
        self._last = 0.0
        self._ui_names = {path.name: path for path in UI_FILES.keys()}
        self.last_changed = None

    def maybe(self, path: str) -> bool:
        # 임시파일 → 원본으로 move 되는 경우까지 커버
        name = Path(path).name
        if name in self._ui_names:
            self.last_changed = self._ui_names[name]
            return True
        return False

    def debounce_call(self):
        now = time.time()
        if now - self._last >= self.debounce:
            self._last = now
            self.on_change()

    def on_modified(self, event: FileSystemEvent):
        if not event.is_directory and self.maybe(event.src_path):
            self.debounce_call()

    def on_created(self, event: FileSystemEvent):
        if not event.is_directory and self.maybe(event.src_path):
            self.debounce_call()

    def on_moved(self, event: FileSystemEvent):
        if not event.is_directory and self.maybe(event.dest_path):
            self.debounce_call()


def start_app():
    # 환경에 따라 python 경로가 다를 수 있어 sys.executable 사용
    return subprocess.Popen(
        [sys.executable, "-m", "shopee_app.launcher"], cwd=str(ROOT)
    )


def main():
    missing = [path.name for path in UI_FILES if not path.exists()]
    existing = {src: dest for src, dest in UI_FILES.items() if src.exists()}
    if missing:
        missing_list = ", ".join(missing)
        log(f"경고: 다음 .ui 파일을 찾을 수 없습니다: {missing_list}")
    if not existing:
        log(".ui 파일이 없어 변환을 수행할 수 없습니다.")
        sys.exit(1)

    # 최초 빌드
    try:
        for src, dest in existing.items():
            build_ui(src, dest)
    except Exception as e:
        log(f"초기 변환 실패: {e}")
        sys.exit(1)

    # 앱 실행
    proc = start_app()
    log("앱 실행 시작")

    # 파일 감시 시작
    observer = Observer()

    def on_change():
        nonlocal proc
        log("변경 감지 → 변환 → 앱 재시작")
        try:
            changed = handler.last_changed
            if changed and changed in UI_FILES:
                build_ui(changed, UI_FILES[changed])
            else:
                for src, dest in UI_FILES.items():
                    build_ui(src, dest)
        except Exception as e:
            log(f"변환 실패: {e}")
            return
        # 앱 재시작
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass
        proc = start_app()

    handler = DebouncedHandler(on_change)
    observer.schedule(handler, str((MODULE_ROOT / "ui")), recursive=False)
    observer.start()

    # SIGINT/SIGTERM 처리
    def cleanup(signum=None, frame=None):
        log("종료 중...")
        observer.stop()
        observer.join()
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # 메인 루프
    try:
        while True:
            time.sleep(0.5)
            # 자식 프로세스가 비정상 종료 시 자동 재시작(선택)
            if proc and proc.poll() is not None:
                log("앱이 종료됨 → 자동 재시작")
                proc = start_app()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
