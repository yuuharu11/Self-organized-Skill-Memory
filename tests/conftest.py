"""Make the SkillNet openpi sources importable for every test."""
from tests.support import REPO, add_path

add_path(REPO / "skill_moe/skillnet/src")
