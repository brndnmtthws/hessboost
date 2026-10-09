#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["tomlkit>=0.13"]
# ///
"""Check, bump, and publish a hessboost release.

`./release.py bump {major|minor|patch|X.Y.Z[-alpha.N|-beta.N|-rc.N]}` prepares,
pushes, and opens the release-bump PR. After merging it, `./release.py` creates
and pushes the annotated release tag. The no-subcommand flow remains tagging.

A bump starts from the manifest version, or from the latest crates.io release
when the manifest is an unreleased bump made by hand (see `bump_base`).
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import tomlkit

ROOT = Path(__file__).resolve().parent
REPO = "brndnmtthws/hessboost"
USER_AGENT = f"hessboost release.py (https://github.com/{REPO})"
_ID = r"(?:0|[1-9]\d*|\d*[A-Za-z-][0-9A-Za-z-]*)"
SEMVER = re.compile(rf"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-({_ID}(?:\.{_ID})*))?")
PEP440_PRE = {"alpha": "a", "beta": "b", "rc": "rc"}
SemverKey = tuple[int, int, int, int, tuple[tuple[int, int, str], ...]]


class CheckError(Exception):
    """An expected release-check failure."""


def execute(*cmd: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        raise CheckError(f"{cmd[0]} is not installed") from None


def run(*cmd: str, cwd: Path = ROOT) -> str:
    proc = execute(*cmd, cwd=cwd)
    if proc.returncode:
        lines = (proc.stderr or proc.stdout).strip().splitlines()
        raise CheckError(f"`{' '.join(cmd)}` failed" + (f": {lines[-1]}" if lines else ""))
    return proc.stdout.strip()


def http_get(url: str) -> tuple[int, bytes]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    try:
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=headers), timeout=15
        ) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as err:
        return err.code, b""
    except (urllib.error.URLError, TimeoutError) as err:
        raise CheckError(f"GET {url} failed: {getattr(err, 'reason', err)}") from None


def semver_key(version: str) -> SemverKey:
    """Return SemVer 2.0 precedence (build metadata is ignored)."""
    match = SEMVER.fullmatch(version.split("+", 1)[0])
    if match is None:
        raise CheckError(f"{version!r} is not a SemVer version (MAJOR.MINOR.PATCH[-PRERELEASE])")
    major, minor, patch, pre = match.groups()
    ids = (
        tuple((0, int(item), "") if item.isdigit() else (1, 0, item) for item in pre.split("."))
        if pre
        else ()
    )
    return int(major), int(minor), int(patch), 0 if pre else 1, ids


def pep440(version: str) -> str:
    """Return the PEP 440 form maturin derives from the Cargo version."""
    base, _, pre = version.partition("-")
    if not pre:
        return base
    match = re.fullmatch(r"(alpha|beta|rc)(?:\.(0|[1-9]\d*))?", pre)
    if match is None:
        raise CheckError(
            f"prerelease -{pre} has no clean PEP 440 form; use -alpha.N, -beta.N or -rc.N"
        )
    return f"{base}{PEP440_PRE[match[1]]}{match[2] or 0}"


def package_version(manifest: str) -> str:
    with (ROOT / manifest).open("rb") as manifest_file:
        return str(tomllib.load(manifest_file)["package"]["version"])


def require_tools(names: tuple[str, ...]) -> str:
    missing = [tool for tool in names if shutil.which(tool) is None]
    if missing:
        raise CheckError(f"not on PATH: {', '.join(missing)}")
    return ", ".join(names)


def check_git() -> str:
    problems = []
    branch = run("git", "branch", "--show-current")
    if branch != "main":
        problems.append(f"on branch {branch or '(detached HEAD)'}, not main")
    if run("git", "status", "--porcelain", "--untracked-files=all"):
        problems.append("working tree has uncommitted or untracked changes")
    run("git", "fetch", "--quiet", "origin", "main", "--tags")
    head, upstream = run("git", "rev-parse", "HEAD", "origin/main").split()
    if head != upstream:
        problems.append(f"HEAD {head[:12]} != origin/main {upstream[:12]}")
    if problems:
        raise CheckError("; ".join(problems))
    return f"main, clean, HEAD == origin/main ({head[:12]})"


def check_manifests(version: str) -> str:
    root, python = package_version("Cargo.toml"), package_version("python/Cargo.toml")
    if not root == python == version:
        raise CheckError(f"Cargo.toml {root}, python/Cargo.toml {python}, release {version}")
    return f"Cargo.toml and python/Cargo.toml are {version}"


def latest_release() -> str | None:
    """Return the newest SemVer version of hessboost on crates.io, or None if there is none."""
    status, body = http_get("https://crates.io/api/v1/crates/hessboost")
    if status == 404:
        return None
    if status != 200:
        raise CheckError(f"crates.io returned HTTP {status}")
    published = [
        item["num"]
        for item in json.loads(body)["versions"]
        if SEMVER.fullmatch(item["num"].split("+", 1)[0])
    ]
    return max(published, key=semver_key, default=None)


def check_newer(version: str) -> str:
    key, wheel = semver_key(version), pep440(version)
    newest = latest_release()
    if newest is None:
        return f"{version} (PyPI {wheel}); hessboost has no release on crates.io yet"
    if key <= semver_key(newest):
        raise CheckError(f"{version} is not greater than {newest} on crates.io")
    return f"{version} (PyPI {wheel}) > {newest} on crates.io"


def check_locks() -> str:
    problems = []
    if execute(
        "cargo",
        "metadata",
        "--locked",
        "--format-version",
        "1",
        "--manifest-path",
        "python/Cargo.toml",
    ).returncode:
        problems.append(
            "python/Cargo.lock is stale: run "
            "`cargo update -p hessboost --manifest-path python/Cargo.toml`"
        )
    if execute("uv", "lock", "--check", cwd=ROOT / "python").returncode:
        problems.append("python/uv.lock is stale: run `uv lock` in python/")
    if problems:
        raise CheckError("; ".join(problems))
    return "python/Cargo.lock and python/uv.lock are up to date"


def check_saved_models(version: str) -> str:
    directory = f"tests/data/saved/{version}"
    if not (ROOT / directory).is_dir():
        raise CheckError(f"{directory}/ does not exist; save this release's models and commit them")
    tracked = run("git", "ls-files", "--", f"{directory}/**")
    if not tracked:
        raise CheckError(
            f"{directory}/ exists but contains no tracked files; commit the saved models"
        )
    return f"{directory}/ exists with tracked saved models"


def check_tag(tag: str) -> str:
    if execute("git", "rev-parse", "-q", "--verify", f"refs/tags/{tag}").returncode == 0:
        raise CheckError(f"{tag} already exists locally")
    if run("git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"):
        raise CheckError(f"{tag} already exists on origin")
    return f"{tag} exists neither locally nor on origin"


def check_unpublished(version: str) -> str:
    semver_key(version)
    wheel = pep440(version)
    status, _ = http_get(f"https://crates.io/api/v1/crates/hessboost/{version}")
    if status == 200:
        raise CheckError(f"hessboost {version} is already on crates.io")
    if status != 404:
        raise CheckError(f"crates.io returned HTTP {status} for hessboost {version}")
    status, body = http_get("https://pypi.org/pypi/hessboost/json")
    if status == 404:
        return (
            "not on crates.io; PyPI project does not exist yet, configure pending trusted "
            f"publisher ({REPO}, publish.yml, environment pypi)"
        )
    if status != 200:
        raise CheckError(f"PyPI returned HTTP {status}")
    if wheel in json.loads(body)["releases"]:
        raise CheckError(f"hessboost {wheel} is already on PyPI")
    return f"hessboost {version} is on neither crates.io nor PyPI ({wheel})"


def check_ci() -> str:
    sha = run("git", "rev-parse", "HEAD")
    out = run(
        "gh",
        "run",
        "list",
        "--commit",
        sha,
        "--workflow",
        "ci.yml",
        "--branch",
        "main",
        "--limit",
        "1",
        "--json",
        "conclusion,status,url",
    )
    runs: list[dict[str, str]] = json.loads(out)
    if not runs:
        raise CheckError(f"no CI run for {sha[:12]} on main")
    ci = runs[0]
    if ci["status"] != "completed":
        raise CheckError(f"CI is {ci['status']}; wait for CI: {ci['url']}")
    if ci["conclusion"] != "success":
        raise CheckError(f"CI concluded {ci['conclusion']}: {ci['url']}")
    return f"CI passed: {ci['url']}"


def publish_run_url(tag: str) -> str:
    for _ in range(10):
        time.sleep(3)
        proc = execute(
            "gh",
            "run",
            "list",
            "--workflow",
            "publish.yml",
            "--branch",
            tag,
            "--limit",
            "1",
            "--json",
            "url",
        )
        if proc.returncode == 0 and (runs := json.loads(proc.stdout)):
            return str(runs[0]["url"])
    return f"https://github.com/{REPO}/actions/workflows/publish.yml"


VERSION_REFERENCE_FILES = (
    "README.md",
    "python/README.md",
    "src/lib.rs",
    "examples/ranking.rs",
    "examples/bench_compare.rs",
    "examples/binary_classification.rs",
    "examples/budget.rs",
    "examples/compact_model.rs",
    "examples/conformal.rs",
    "examples/constraints.rs",
    "examples/custom_objective.rs",
    "examples/distributional.rs",
    "examples/metal.rs",
    "examples/model_io.rs",
    "examples/multiclass.rs",
    "examples/ordered_target_stats.rs",
    "examples/pfn_boost.rs",
    "examples/shap.rs",
    "examples/train_regression.rs",
    "docs/performance.md",
)
DEPENDENCY_VERSION = re.compile(r'(hessboost\s*=\s*")([0-9]+\.[0-9]+)(")')


def saved_models_dir(version: str) -> Path:
    return ROOT / "tests" / "data" / "saved" / version


def bump_base(current: str, latest: str | None) -> str:
    """Return the version a bump of manifest version `current` starts from.

    That is `current`, unless the manifest is an unreleased bump: newer than
    `latest`, the newest crates.io release, with no saved models (a version bumped
    by hand, not by `bump`). Then it is `latest`: `major|minor|patch` count from
    it, an explicit target may equal `current`, and the dependency snippets
    still name its minor.
    """
    if (
        latest is not None
        and semver_key(current) > semver_key(latest)
        and not saved_models_dir(current).exists()
    ):
        return latest
    return current


def bumped_version(base: str, part: str) -> str:
    """Return `base` bumped by `part`, or `part` itself if it is an explicit version."""
    major, minor, patch, _, _ = semver_key(base)
    match = SEMVER.fullmatch(base)
    assert match is not None
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        return f"{major}.{minor}.{patch}" if match[4] else f"{major}.{minor}.{patch + 1}"
    if SEMVER.fullmatch(part) is None:
        raise CheckError(f"{part!r} is not an accepted SemVer release version")
    semver_key(part)
    pep440(part)
    return part


def check_bump_version(version: str, current: str, base: str, latest: str | None) -> str:
    """Check `version` against manifest `current`, its `bump_base`, and crates.io's `latest`.

    The target must be newer than the manifest, or at least the manifest if that is
    an unreleased bump, and newer than the latest release.
    """
    key = semver_key(version)
    wheel = pep440(version)
    unreleased = base != current
    if key < semver_key(current) or (key == semver_key(current) and not unreleased):
        raise CheckError(
            f"{version} is older than unreleased manifest version {current}"
            if unreleased
            else f"{version} is not newer than current version {current}"
        )
    if latest is not None and key <= semver_key(latest):
        raise CheckError(f"{version} is not greater than {latest} on crates.io")
    manifest = f">= unreleased manifest {current}" if unreleased else f"> current {current}"
    release = f"> {latest} on crates.io" if latest else "no crates.io release yet"
    return f"{version} (PyPI {wheel}) {manifest}; {release}"


def check_bump_targets(version: str) -> str:
    tag = f"v{version}"
    check_tag(tag)
    saved = saved_models_dir(version)
    if saved.exists():
        raise CheckError(f"{saved.relative_to(ROOT)}/ already exists")
    branch = f"release/v{version}"
    if execute("git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}").returncode == 0:
        raise CheckError(f"branch {branch} already exists locally")
    if run("git", "ls-remote", "--heads", "origin", f"refs/heads/{branch}"):
        raise CheckError(f"branch {branch} already exists on origin")
    return f"{tag} and {saved.relative_to(ROOT)}/ do not exist; branch name is available"


def replace_package_version(manifest: str, version: str) -> None:
    path = ROOT / manifest
    document = tomlkit.parse(path.read_text())
    document["package"]["version"] = version
    path.write_text(tomlkit.dumps(document))


def dependency_minor(version: str) -> str:
    """Return the `MAJOR.MINOR` the dependency snippets name for `version`."""
    return ".".join(version.split(".")[:2])


def version_reference_updates(base: str, version: str) -> dict[str, str]:
    """Map each reference file whose dependency snippets name `base`'s minor to its
    text naming `version`'s minor instead."""
    old_minor, new_minor = dependency_minor(base), dependency_minor(version)
    updates = {}
    for name in VERSION_REFERENCE_FILES:
        path = ROOT / name
        if not path.is_file():
            continue
        text = path.read_text()
        updated = DEPENDENCY_VERSION.sub(
            lambda match: match[1] + new_minor + match[3] if match[2] == old_minor else match[0],
            text,
        )
        if updated != text:
            updates[name] = updated
    return updates


def update_version_references(base: str, version: str) -> list[str]:
    updates = version_reference_updates(base, version)
    for name, text in updates.items():
        (ROOT / name).write_text(text)
    return list(updates)


def step(name: str, check: Callable[[], str]) -> bool:
    try:
        print(f"  ✓ {name}: {check()}")
        return True
    except (CheckError, OSError, KeyError, ValueError) as err:
        print(f"  ✗ {name}: {err}")
        return False


def run_checks(checks: list[tuple[str, Callable[[], str]]], failure: str) -> bool:
    failed = 0
    for name, check in checks:
        if not step(name, check):
            failed += 1
    if failed:
        print(f"\n{failure.format(failed=failed)}")
        return False
    return True


def confirm(prompt: str) -> bool:
    try:
        answer = input(prompt)
    except EOFError:
        answer = ""
    return answer.strip().lower() in ("y", "yes")


def refresh_manifest_versions(version: str) -> str:
    replace_package_version("Cargo.toml", version)
    replace_package_version("python/Cargo.toml", version)
    if package_version("Cargo.toml") != version or package_version("python/Cargo.toml") != version:
        raise CheckError("manifest version verification failed")
    return f"Cargo.toml and python/Cargo.toml are {version}"


def refresh_lockfiles() -> str:
    run("cargo", "update", "-p", "hessboost", "--manifest-path", "python/Cargo.toml")
    run("uv", "lock", cwd=ROOT / "python")
    return check_locks()


def save_and_verify_models(version: str) -> str:
    before = run("git", "status", "--porcelain", "--", "tests/data/saved")
    run(
        "cargo",
        "nextest",
        "run",
        "--test",
        "native_format",
        "--run-ignored",
        "only",
        "save_models_of_this_version",
    )
    saved = saved_models_dir(version)
    after = run("git", "status", "--porcelain", "--", "tests/data/saved")
    if not saved.is_dir() or not any(saved.iterdir()):
        raise CheckError(f"{saved.relative_to(ROOT)}/ was not created with files")
    if after.strip() != f"?? tests/data/saved/{version}/" or before:
        raise CheckError("saved-model status changed outside the new version directory")
    return f"{saved.relative_to(ROOT)}/ contains new files only"


def open_pr(branch: str, version: str) -> str:
    body = (
        f"## Release bump\n\nUpdated hessboost to `{version}`, refreshed Rust/Python "
        "lockfiles, and saved this version's native models.\n\n"
        "After merge, on `main`: `./release.py --dry-run` then `./release.py`."
    )
    return run(
        "gh",
        "pr",
        "create",
        "--base",
        "main",
        "--head",
        branch,
        "--title",
        f"chore(release): v{version}",
        "--body",
        body,
    )


def commit_and_push(branch: str, version: str, no_pr: bool) -> None:
    run("git", "add", "-A")
    run("git", "commit", "-m", f"chore(release): v{version}")
    run("git", "push", "-u", "origin", branch)
    print(f"  ✓ push: pushed {branch} to origin")
    if not no_pr:
        print(f"  ✓ pull request: {open_pr(branch, version)}")


def bump(version_arg: str, dry_run: bool, yes: bool, no_pr: bool) -> int:
    try:
        current = package_version("Cargo.toml")
        latest = latest_release()
        base = bump_base(current, latest)
        version = bumped_version(base, version_arg)
    except (CheckError, OSError, tomllib.TOMLDecodeError, KeyError, ValueError) as err:
        print(f"error: cannot compute release version: {err}", file=sys.stderr)
        return 1
    branch = f"release/v{version}"
    checks: list[tuple[str, Callable[[], str]]] = [
        (
            "tools",
            lambda: require_tools(
                ("git", "cargo", "cargo-nextest", "uv") + (() if no_pr else ("gh",))
            ),
        ),
        ("git state", check_git),
        ("version", lambda: check_bump_version(version, current, base, latest)),
        ("release targets", lambda: check_bump_targets(version)),
    ]
    source = current if base == current else f"{base} (unreleased manifest {current})"
    print(f"Checking release bump hessboost {source} → {version}")
    if not run_checks(checks, "Preconditions failed; no changes made."):
        return 1
    old_minor, new_minor = dependency_minor(base), dependency_minor(version)
    snippets = f'hessboost = "{old_minor}" → "{new_minor}" in ' + (
        ", ".join(version_reference_updates(base, version)) or "no files"
    )
    print(
        f"\nPlan: hessboost {source} → {version}\n"
        f"  PyPI:         {pep440(version)}\n"
        f"  branch:       {branch}\n"
        f"  snippets:     {snippets if old_minor != new_minor else 'unchanged'}\n"
        f"  saved models: {saved_models_dir(version).relative_to(ROOT)}/"
    )
    if dry_run:
        print("\nDry run: preconditions passed; no changes made.")
        return 0
    if not yes and not confirm(f"\nCreate and push {branch}? [y/N] "):
        print("Aborted.")
        return 1
    created = False
    try:
        run("git", "switch", "-c", branch)
        created = True
        print(f"  ✓ branch: created {branch}")

        if not step("manifest versions", lambda: refresh_manifest_versions(version)):
            raise CheckError("manifest update failed")
        if not step("lockfiles", refresh_lockfiles):
            raise CheckError("lockfile refresh failed")
        references = update_version_references(base, version)
        print(
            "  ✓ version references: "
            + (", ".join(references) if references else "no current-release snippets found")
        )
        if not step("saved models", lambda: save_and_verify_models(version)):
            raise CheckError("saved-model generation failed")
        commit_and_push(branch, version, no_pr)
        return 0
    except (CheckError, OSError, KeyError, ValueError) as err:
        if created:
            print(
                f"\n✗ Release bump stopped: {err}\nBranch {branch} holds partial changes. "
                f"Discard it with `git checkout main && git branch -D {branch}`.",
                file=sys.stderr,
            )
        else:
            print(f"error: {err}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    bump_parser = commands.add_parser("bump", help="prepare and open a version-bump PR")
    bump_parser.add_argument("version", help="major, minor, patch, or explicit SemVer")
    bump_parser.add_argument("--dry-run", action="store_true", help="show plan without changes")
    bump_parser.add_argument("--yes", action="store_true", help="skip confirmation")
    bump_parser.add_argument(
        "--no-pr", action="store_true", help="push branch without opening a PR"
    )
    parser.add_argument("tag_version", nargs="?", help="tag version; defaults to root Cargo.toml")
    parser.add_argument("--dry-run", action="store_true", help="run tag checks only")
    parser.add_argument("--yes", action="store_true", help="tag and push without asking")
    args = parser.parse_args()
    if args.command == "bump":
        return bump(args.version, args.dry_run, args.yes, args.no_pr)
    try:
        version = (args.tag_version or package_version("Cargo.toml")).removeprefix("v")
    except (OSError, tomllib.TOMLDecodeError, KeyError) as err:
        print(f"error: cannot read version from Cargo.toml: {err}", file=sys.stderr)
        return 1
    tag = f"v{version}"
    checks: list[tuple[str, Callable[[], str]]] = [
        ("tools", lambda: require_tools(("git", "cargo", "uv", "gh"))),
        ("git state", check_git),
        ("manifest versions", lambda: check_manifests(version)),
        ("version", lambda: check_newer(version)),
        ("lockfiles", check_locks),
        ("saved models", lambda: check_saved_models(version)),
        ("tag", lambda: check_tag(tag)),
        ("not published", lambda: check_unpublished(version)),
        ("CI", check_ci),
    ]
    print(f"Checking release hessboost {version} ({tag})")
    if not run_checks(checks, "{failed} check(s) failed; not tagging."):
        return 1
    commit = run("git", "log", "-1", "--format=%h %s")
    prerelease = " (marked prerelease)" if "-" in version else ""
    print(
        f"\nRelease hessboost {version}\n  tag:    {tag} (annotated)\n  commit: {commit}\n"
        f"Pushing {tag} starts Publish: it validates, builds and verifies Python "
        f"distributions, publishes hessboost {version} to crates.io and hessboost "
        f"{pep440(version)} to PyPI, then creates a GitHub release{prerelease}."
    )
    if args.dry_run:
        print("\nDry run: all checks passed; not tagging.")
        return 0
    if not args.yes and not confirm(f"\nCreate and push tag {tag}? [y/N] "):
        print("Aborted.")
        return 1
    run("git", "tag", "-a", tag, "-m", tag)
    try:
        run("git", "push", "origin", f"refs/tags/{tag}")
    except CheckError:
        print(
            f"Push failed; local tag remains (delete with `git tag -d {tag}`).",
            file=sys.stderr,
        )
        raise
    print(f"Pushed {tag}. Publish run: {publish_run_url(tag)}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CheckError as err:
        print(f"error: {err}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
