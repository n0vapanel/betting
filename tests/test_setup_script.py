"""deploy/setup_vps.sh: checked as text and by running its small helpers in bash (nothing is installed)."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "deploy" / "setup_vps.sh"
BASH = shutil.which("bash")


def helper(name: str) -> str:
    """One shell function from the script (with the constants it reads), as text to run on its own."""
    text = SCRIPT.read_text()
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", text, re.M | re.S)
    assert match, name
    constants = "\n".join(re.findall(r"^MIN_SAFE_PASSWORD=.*$", text, re.M))
    return f"{constants}\n{match.group(0)}"


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_the_script_parses() -> None:
    assert subprocess.run([BASH, "-n", str(SCRIPT)], capture_output=True).returncode == 0


@pytest.mark.skipif(BASH is None, reason="needs bash")
@pytest.mark.parametrize("password, warned", [("abcd1234", True), ("abcdefgh123", True), ("abcdefgh1234", False),
                                              ("Xy7-Xy7-Xy7-Xy7-Xy7", False)])
def test_a_short_dashboard_password_gets_a_warning_but_still_works(password, warned) -> None:
    run = subprocess.run([BASH, "-c", f'{helper("warn_short_password")}\nwarn_short_password "$1"', "-", password],
                         capture_output=True, text=True)
    assert run.returncode == 0  # a warning, never a refusal: 8-11 character passwords keep working
    assert ("WARNING" in run.stdout) is warned
    if warned:
        assert f"only {len(password)} characters" in run.stdout and "12 or more" in run.stdout
    assert password not in run.stdout + run.stderr  # the password itself is never printed


def test_new_and_changed_passwords_are_checked_and_the_pattern_still_takes_8() -> None:
    text = SCRIPT.read_text()
    assert "[DASHBOARD_PASSWORD]='^[A-Za-z0-9_.@%+=:-]{8,64}$'" in text
    # Typed at the prompt (not generated), and given on the command line when it differs from the saved one
    assert text.count('warn_short_password "$DASHBOARD_PASSWORD"') == 2
    assert '[ "$DASHBOARD_PASSWORD" != "$saved_password" ]' in text


def test_the_ai_chat_allowances_can_be_set_from_the_command_line() -> None:
    text = SCRIPT.read_text()
    for name in ("AI_CHAT_DAILY_CENTS", "AI_CHAT_WEB_DAILY_CENTS", "AI_CHAT_DAILY_TOTAL_CENTS"):
        assert re.search(rf"^  \[{name}\]='\^\[0-9\]\{{1,\d\}}\$'$", text, re.M), name
        assert re.search(rf"^#   {name}=\d+ ", text, re.M), name  # documented in the header
