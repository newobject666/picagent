# backend/manage.py

import os
import sys
from pathlib import Path


def main():
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent
    sys.path.insert(0, str(BASE_DIR))
    # 让 Django 可以 import 项目根目录下的 figure_agent
    sys.path.insert(0, str(PROJECT_ROOT))

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()