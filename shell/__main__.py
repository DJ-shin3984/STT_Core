import argparse

from .app import main

ap = argparse.ArgumentParser(prog="python -m shell")
ap.add_argument("--config", default="config.toml")
ap.add_argument("--smoke", action="store_true", help="FakeEngine으로 전체 흐름을 자동 점검하고 종료")
a = ap.parse_args()
main(a.config, a.smoke)
