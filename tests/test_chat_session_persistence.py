import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django

django.setup()

from apps.chat.views import has_complete_chat_turn


def test_complete_chat_turn_requires_user_and_assistant_content():
    assert has_complete_chat_turn("hello", "hi")
    assert not has_complete_chat_turn("", "hi")
    assert not has_complete_chat_turn("hello", "")
    assert not has_complete_chat_turn("hello", "   ")
    assert not has_complete_chat_turn("   ", "hi")
