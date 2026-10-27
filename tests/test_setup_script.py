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


# --- Python packages: installed from requirements.lock, every hash checked ------------------------------------

ROOT = SCRIPT.parent.parent
LOCK = ROOT / "requirements.lock"
LOCK_COMMAND = "uv pip compile requirements.txt --universal --generate-hashes --python-version 3.10 -o requirements.lock"


STUB = """#!/bin/sh
# Stands in for "runuser -u USER -- env HOME=... COMMAND...", as strict about runuser's own options as the real one
[ "$1" = -u ] && [ -n "$2" ] || { echo "runuser: expected -u USER" >&2; exit 1; }
shift 2
[ "$1" = -- ] || { echo "runuser: invalid option -- '$1'" >&2; exit 1; }
shift
[ "$1" = env ] || { echo "stand-in: expected env after --" >&2; exit 1; }
shift
while [ "${1#*=}" != "$1" ]; do shift; done
echo "$*" >> "$CALLS"
if [ "$1" = sed ]; then exec "$@"; fi
case "$*" in *"$FAILS"*) exit 1;; esac
case "$*" in *--require-hashes*) if [ -n "$HASH_MISMATCH" ]; then
  echo "ERROR: THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS FILE." >&2; exit 1; fi;; esac
case "$*" in "python3 -I -m venv"*)
  for last in "$@"; do :; done
  mkdir -p "$last/bin" && printf '#!/bin/sh\\nexit 0\\n' > "$last/bin/python" && chmod +x "$last/bin/python"
  printf '#!%s/bin/python3\\nimport pip\\n' "$last" > "$last/bin/pip"
  printf 'home = /usr/bin\\ncommand = /usr/bin/python3 -m venv %s\\n' "$last" > "$last/pyvenv.cfg"
  echo built > "$last/built";;
esac
"""


def run_install(tmp_path: Path, *, lock: bool = True, fails: str = "", allow: str = "", old_venv: bool = False,
                marker: str = "", hash_mismatch: bool = False) -> subprocess.CompletedProcess:
    """install_packages called the way setup_vps.sh calls it (... runuser -u bettingbot -- env HOME=...), with a
    stand-in for runuser that records each command, makes a pretend venv for "python3 -I -m venv", runs sed for real,
    and fails any command containing ``fails``."""
    import grp
    import os
    import pwd
    tmp_path.mkdir(parents=True, exist_ok=True)
    app, venv, mark = tmp_path / "app", tmp_path / "venv", tmp_path / "checked"
    app.mkdir()
    (app / "requirements.txt").write_text("requests\n")
    if lock:
        (app / "requirements.lock").write_text("requests==1.0 \\\n    --hash=sha256:" + "0" * 64 + "\n")
    if old_venv:  # an earlier venv: its own interpreter must never run
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "python").write_text("#!/bin/sh\necho OLD VENV RAN >&2\nexit 0\n")
        (venv / "bin" / "python").chmod(0o755)
        (venv / "old").write_text("")
    if marker:
        mark.write_text(marker)
    stub = tmp_path / "as"
    stub.write_text(STUB)
    stub.chmod(0o755)
    owner = f"{pwd.getpwuid(os.getuid()).pw_name}:{grp.getgrgid(os.getgid()).gr_name}"
    env = {"PATH": "/usr/bin:/bin", "ALLOW_UNPINNED": allow, "CALLS": str(tmp_path / "calls"),
           "FAILS": fails or "NEVER MATCHES", "HASH_MISMATCH": "1" if hash_mismatch else ""}
    script = "\n".join(helper(name) for name in ("install_packages", "new_venv"))
    run = subprocess.run([BASH, "-c", f'set -euo pipefail\n{script}\n'
                          'install_packages "$1" "$2" "$3" "$4" "$5" -u bettingbot -- env HOME="$6" || exit 1', "-",
                          str(app), str(venv), str(mark), owner, str(stub), str(tmp_path)],
                         capture_output=True, text=True, env=env)
    calls = tmp_path / "calls"
    run.calls = calls.read_text().splitlines() if calls.exists() else []  # type: ignore[attr-defined]
    run.venv, run.mark = venv, mark  # type: ignore[attr-defined]
    return run


def built_for(run) -> str:
    return run.mark.read_text().strip() if run.mark.exists() else ""


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_a_new_venv_is_built_from_the_system_python_with_every_hash_checked(tmp_path) -> None:
    run = run_install(tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    make, install, rewrite = run.calls
    assert make == f"python3 -I -m venv --clear --prompt venv {run.venv}.new"  # -I: never a module from the current folder
    assert install.startswith(f"{run.venv}.new/bin/python -I -m pip install")
    assert "--require-hashes" in install and "--only-binary :all:" in install
    assert install.endswith("app/requirements.lock")
    assert rewrite.startswith("sed -i ")
    assert (run.venv / "built").exists() and not Path(f"{run.venv}.new").exists()
    import hashlib
    lock_hash = hashlib.sha256((tmp_path / "app" / "requirements.lock").read_bytes()).hexdigest()
    assert re.fullmatch(rf"python 3\.\d+ lock {lock_hash}", built_for(run))


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_the_swapped_in_venv_names_its_own_folder(tmp_path) -> None:
    """Scripts pip and venv write (pip, activate, uvicorn) name the folder they were built in: after the swap they
    must point at the venv, not at venv.new, or they stop working."""
    run = run_install(tmp_path)
    assert run.returncode == 0, run.stdout + run.stderr
    pip, cfg = (run.venv / "bin" / "pip").read_text(), (run.venv / "pyvenv.cfg").read_text()
    assert pip.startswith(f"#!{run.venv}/bin/python3\n") and f"-m venv {run.venv}\n" in cfg
    assert ".new" not in pip + cfg


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_an_older_venv_is_replaced_and_its_interpreter_never_runs(tmp_path) -> None:
    """A venv from before the lock (or an ALLOW_UNPINNED run) may hold unchecked packages, and its own python would
    run their code: it is swapped out for a new one, never asked to clean itself."""
    run = run_install(tmp_path, old_venv=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert not any(call.startswith(f"{run.venv}/") for call in run.calls)
    assert "OLD VENV RAN" not in run.stderr
    assert not (run.venv / "old").exists() and (run.venv / "built").exists()
    assert not Path(f"{run.venv}.old").exists()


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_an_unchanged_lock_on_the_same_python_installs_nothing(tmp_path) -> None:
    first = run_install(tmp_path / "a")
    again = run_install(tmp_path / "b", old_venv=True, marker=built_for(first))
    assert again.returncode == 0 and again.calls == [] and (again.venv / "old").exists()
    changed = run_install(tmp_path / "c", old_venv=True, marker=built_for(first).replace("lock ", "lock 0"))
    assert changed.returncode == 0 and len(changed.calls) == 3 and not (changed.venv / "old").exists()


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_a_failed_checked_install_changes_nothing_unless_allowed(tmp_path) -> None:
    stopped = run_install(tmp_path / "a", fails="requirements.lock", old_venv=True, marker="python 3.0 lock old")
    assert stopped.returncode == 1  # never falls back to unchecked packages on its own
    assert "Nothing was changed or restarted" in stopped.stdout and "ALLOW_UNPINNED=1" in stopped.stdout
    assert (stopped.venv / "old").exists() and not Path(f"{stopped.venv}.new").exists()
    assert built_for(stopped) == "python 3.0 lock old"
    allowed = run_install(tmp_path / "b", fails="requirements.lock", allow="1", old_venv=True, marker="x")
    assert allowed.returncode == 0 and "WARNING" in allowed.stdout
    # A new venv (the old one may be for another Python), filled from requirements.txt without hashes
    assert allowed.calls[2] == f"python3 -I -m venv --clear --prompt venv {allowed.venv}.new"
    assert allowed.calls[3] == (f"{allowed.venv}.new/bin/python -I -m pip install --quiet --no-cache-dir "
                                "--disable-pip-version-check -r " + str(tmp_path / "b" / "app" / "requirements.txt"))
    assert not (allowed.venv / "old").exists() and (allowed.venv / "built").exists()
    assert not allowed.mark.exists()  # the next normal run builds a checked venv again


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_a_package_that_fails_its_hash_check_is_never_installed_unchecked(tmp_path) -> None:
    """A download that doesn't match its pinned hash may have been tampered with: ALLOW_UNPINNED doesn't skip that."""
    for name, allow in (("a", ""), ("b", "1")):
        run = run_install(tmp_path / name, hash_mismatch=True, old_venv=True, marker="python 3.0 lock old",
                          allow=allow)
        assert run.returncode == 1 and "DO NOT MATCH THE HASHES" in run.stderr
        assert "doesn't skip this check" in run.stdout and "WARNING" not in run.stdout
        assert not any("requirements.txt" in call for call in run.calls)
        assert (run.venv / "old").exists() and built_for(run) == "python 3.0 lock old"


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_without_a_lock_nothing_is_installed_unless_allowed(tmp_path) -> None:
    stopped = run_install(tmp_path / "a", lock=False)
    assert stopped.returncode == 1 and stopped.calls == []
    assert "requirements.lock is missing" in stopped.stdout and "Nothing was changed or restarted" in stopped.stdout
    allowed = run_install(tmp_path / "b", lock=False, allow="1", marker="x")
    assert allowed.returncode == 0 and "WARNING" in allowed.stdout and not allowed.mark.exists()
    assert "Nothing was changed" not in allowed.stdout
    assert allowed.calls[0] == f"python3 -I -m venv --clear --prompt venv {allowed.venv}.new"
    assert allowed.calls[1].endswith("app/requirements.txt") and "--require-hashes" not in allowed.calls[1]


def test_the_install_step_uses_the_checked_install_as_the_bot_user() -> None:
    text = SCRIPT.read_text()
    assert ('install_packages "$APP" "$VENV" "$BASE/.packages-hash-checked" "$BOT_USER:$BOT_USER" \\\n'
            '    runuser -u "$BOT_USER" -- env HOME="$BASE" || exit 1') in text
    assert text.count('-r "$APP/requirements.txt"') == 0  # the only plain install is inside install_packages
    assert 'python3 -m venv "$VENV"' not in text  # venvs are made inside install_packages only


def test_python_never_runs_from_the_folder_the_script_was_started_in() -> None:
    """Python looks for modules in the current folder first: the script moves to / before anything runs, and its
    own Python calls use isolated mode (-I)."""
    text = SCRIPT.read_text()
    main = text[text.index("\nmain() {"):]
    assert main.index("\n  cd /\n") < main.index("python3")
    # (the services themselves run "python -m" from the app's own folder, which root's checkout owns)
    script_lines = [line for line in text.splitlines()
                    if not line.lstrip().startswith(("#", "ExecStart="))]
    body = "\n".join(script_lines)
    calls = re.findall(r"python3 (-\S+ )?-[cm] ", body) + re.findall(r'bin/python" (-\S+)', body)
    assert len(calls) >= 6 and all(flag.strip() == "-I" for flag in calls), calls


def test_the_deploy_notification_runs_as_the_bot_user() -> None:
    """The venv belongs to the bot user, so root never runs its Python."""
    text = SCRIPT.read_text()
    assert 'runuser -u "$BOT_USER" -- "$VENV/bin/python" -I "$APP/brain/notify.py"' in text
    assert not re.search(r'^\s*"\$VENV/bin/python"', text, re.M)


def _locked() -> dict:
    """{name: [(version, marker, hash count)]} from requirements.lock."""
    pins: dict = {}
    for block in re.split(r"\n(?=[A-Za-z0-9])", LOCK.read_text()):
        head = re.match(r"([A-Za-z0-9_.-]+)==([^ ;\\\n]+)(?: ; ([^\\\n]+))?", block)
        if head:
            name = re.sub(r"[-_.]+", "-", head.group(1)).lower()
            pins.setdefault(name, []).append((head.group(2), (head.group(3) or "").strip(),
                                              len(re.findall(r"--hash=sha256:[0-9a-f]{64}", block))))
    return pins


def test_the_lock_pins_every_requirement_with_hashes() -> None:
    """requirements.lock must match requirements.txt: every requirement pinned at a version its line allows, every
    pin with at least one sha256 hash (pip's --require-hashes refuses anything else)."""
    from packaging.requirements import Requirement
    pins = _locked()
    assert pins and all(count >= 1 for entries in pins.values() for _, _, count in entries)
    for line in (ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        req = Requirement(line)
        # Extras and markers would need their own checks here: make the lock again and extend this test
        assert not req.extras and req.marker is None, line
        name = re.sub(r"[-_.]+", "-", req.name).lower()
        assert name in pins, f"{req.name} is in requirements.txt but not in requirements.lock: make the lock again"
        for version, _, _ in pins[name]:
            assert req.specifier.contains(version, prereleases=True), \
                f"requirements.lock pins {req.name}=={version}, which requirements.txt ({line}) doesn't allow"


def test_the_lock_covers_every_python_the_bot_runs_on() -> None:
    """Made for any platform and Python 3.10+ (Ubuntu 22.04 has 3.10): a lock made for one Python stops every
    deploy on another, and the script's way round that is an unchecked install."""
    text = LOCK.read_text()
    assert f"#    {LOCK_COMMAND}\n" in text
    assert any("python_full_version < '3.11'" in marker for entries in _locked().values() for _, marker, _ in entries)
