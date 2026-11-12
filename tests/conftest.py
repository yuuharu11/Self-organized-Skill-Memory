"""Make this checkout's SkillNet openpi sources importable for every test.

Several test modules import example scripts that insert their own directory,
but none of them puts ``skill_moe/skillnet/src`` on ``sys.path``. Without
this, ``import openpi`` falls through to whatever copy the environment
provides (for example an editable install of another checkout), so a test
file passes or fails depending on which test module ran first. Inserting the
sources here keeps every test on this checkout's code.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "skill_moe/skillnet/src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
