#!/usr/bin/env bash
set -Eeuo pipefail

# FastqOptiFilter release helper
#
# Usage:
#   ./release_fastqoptifilter.sh 0.1.2
#
# Assumptions:
#   - Run this script from the FastqOptiFilter git repository (normally branch main).
#   - PyPI credentials are already configured for Twine (for example in ~/.pypirc).
#   - The initial FastqOptiFilter Bioconda recipe has already been merged.
#   - A separate conda environment named "bioconda" contains bioconda-utils.
#   - ~/bioconda-recipes is your fork/clone with origin=dnaase/bioconda-recipes
#     and upstream=bioconda/bioconda-recipes. Override with BIOCONDA_REPO.
#   - GitHub CLI (gh) is authenticated.
#
# Optional environment variables:
#   BIOCONDA_REPO=/Users/yaping/Documents/workspace/code/bioconda-recipes
#   BIOCONDA_ENV=bioconda
#   BIOCONDA_GITHUB_USER=dnaase
#   SOURCE_BRANCH=main
#   PYTHON=python3
#   SKIP_TESTS=1
#   SKIP_BIOCONDA=1
#   CREATE_GITHUB_RELEASE=1

PACKAGE="fastqoptifilter"
VERSION_FILE="fastq_optifilter.py"
CITATION_FILE="CITATION.cff"
BIOCONDA_REPO="${BIOCONDA_REPO:-$HOME/bioconda-recipes}"
BIOCONDA_ENV="${BIOCONDA_ENV:-bioconda}"
BIOCONDA_GITHUB_USER="${BIOCONDA_GITHUB_USER:-dnaase}"
SOURCE_BRANCH="${SOURCE_BRANCH:-main}"
PYTHON="${PYTHON:-python3}"
SKIP_TESTS="${SKIP_TESTS:-0}"
SKIP_BIOCONDA="${SKIP_BIOCONDA:-0}"
CREATE_GITHUB_RELEASE="${CREATE_GITHUB_RELEASE:-0}"

info() { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
need_cmd() { command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"; }

usage() {
    cat <<USAGE
Usage: $(basename "$0") VERSION

Example:
  $(basename "$0") 0.1.2

Environment overrides:
  BIOCONDA_REPO=/Users/yaping/Documents/workspace/code/bioconda-recipes
  BIOCONDA_ENV=bioconda
  BIOCONDA_GITHUB_USER=dnaase
  SOURCE_BRANCH=main
  PYTHON=python3
  SKIP_TESTS=1
  SKIP_BIOCONDA=1
  CREATE_GITHUB_RELEASE=1
USAGE
}

[[ $# -eq 1 ]] || { usage; exit 2; }
NEW_VERSION="$1"
[[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([a-zA-Z0-9.+-]*)$ ]] || \
    die "Version '$NEW_VERSION' does not look like a PEP 440-style release version."

need_cmd git
need_cmd "$PYTHON"
need_cmd conda
need_cmd gh

PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || \
    die "Run this script from inside the FastqOptiFilter git repository."
cd "$PROJECT_ROOT"

[[ -f "$VERSION_FILE" ]] || die "Cannot find $VERSION_FILE in $PROJECT_ROOT"
[[ -f pyproject.toml ]] || die "Cannot find pyproject.toml in $PROJECT_ROOT"
[[ -f "$CITATION_FILE" ]] || die "Cannot find $CITATION_FILE in $PROJECT_ROOT"

git remote get-url origin >/dev/null 2>&1 || die "Git remote 'origin' is not configured."
CURRENT_BRANCH="$(git branch --show-current)"
[[ "$CURRENT_BRANCH" == "$SOURCE_BRANCH" ]] || \
    die "Release from '$SOURCE_BRANCH'. Current branch is '$CURRENT_BRANCH'."

if [[ -n "$(git status --porcelain)" ]]; then
    die "FastqOptiFilter working tree is not clean. Commit/stash your changes before releasing."
fi

"$PYTHON" -c 'import build, twine' >/dev/null 2>&1 || \
    die "Python packages 'build' and 'twine' are required. Install with: $PYTHON -m pip install -U build twine"

gh auth status >/dev/null 2>&1 || die "GitHub CLI is not authenticated. Run: gh auth login"

current_version() {
    "$PYTHON" - "$VERSION_FILE" <<'PY'
import pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text()
m = re.search(r'(?m)^VERSION\s*=\s*["\']([^"\']+)["\']\s*$', text)
if not m:
    raise SystemExit("Could not find VERSION in fastq_optifilter.py")
print(m.group(1))
PY
}

pypi_state() {
    "$PYTHON" - "$PACKAGE" "$NEW_VERSION" <<'PY'
import json, sys, urllib.error, urllib.request
package, version = sys.argv[1:]
url = f"https://pypi.org/pypi/{package}/{version}/json"
try:
    with urllib.request.urlopen(url, timeout=20) as r:
        data = json.load(r)
except urllib.error.HTTPError as e:
    if e.code == 404:
        print("missing")
        raise SystemExit(0)
    raise
files = data.get("urls", [])
has_sdist = any(x.get("packagetype") == "sdist" for x in files)
has_wheel = any(x.get("packagetype") == "bdist_wheel" for x in files)
if has_sdist and has_wheel:
    print("complete")
elif files:
    print("partial")
else:
    print("missing")
PY
}

wait_for_pypi() {
    local state="missing"
    for _ in $(seq 1 20); do
        state="$(pypi_state)"
        if [[ "$state" == "complete" ]]; then
            return 0
        fi
        sleep 3
    done
    return 1
}

ORIGINAL_VERSION="$(current_version)"
PYPI_STATE="$(pypi_state)"

if [[ "$PYPI_STATE" == "partial" ]]; then
    die "PyPI already has a partial $PACKAGE $NEW_VERSION release. Inspect PyPI before continuing; distribution filenames cannot be replaced."
fi

if [[ "$PYPI_STATE" == "complete" && "$ORIGINAL_VERSION" != "$NEW_VERSION" ]]; then
    die "PyPI already contains $PACKAGE $NEW_VERSION, but local VERSION is $ORIGINAL_VERSION. Refusing to rewrite local source to an already-published version."
fi

VERSION_EDITED=0
RELEASE_COMMITTED=0
cleanup_on_exit() {
    status=$?
    if [[ $status -ne 0 && "$VERSION_EDITED" == "1" && "$RELEASE_COMMITTED" == "0" ]]; then
        printf '\nRelease stopped before commit; restoring version metadata.\n' >&2
        git restore -- "$VERSION_FILE" "$CITATION_FILE" >/dev/null 2>&1 || true
    fi
    exit "$status"
}
trap cleanup_on_exit EXIT

if [[ "$ORIGINAL_VERSION" != "$NEW_VERSION" ]]; then
    info "Updating FastqOptiFilter version $ORIGINAL_VERSION -> $NEW_VERSION"
    RELEASE_DATE="$(date '+%Y-%m-%d')"
    "$PYTHON" - "$VERSION_FILE" "$CITATION_FILE" "$NEW_VERSION" "$RELEASE_DATE" <<'PY'
import pathlib, re, sys
version_file, citation_file, version, release_date = sys.argv[1:]

p = pathlib.Path(version_file)
text = p.read_text()
text2, n = re.subn(
    r'(?m)^VERSION\s*=\s*["\'][^"\']+["\']\s*$',
    f'VERSION = "{version}"',
    text,
    count=1,
)
if n != 1:
    raise SystemExit("Expected exactly one VERSION assignment")
p.write_text(text2)

p = pathlib.Path(citation_file)
text = p.read_text()
text, n1 = re.subn(r'(?m)^version:\s*.*$', f'version: "{version}"', text, count=1)
text, n2 = re.subn(r'(?m)^date-released:\s*.*$', f'date-released: "{release_date}"', text, count=1)
if n1 != 1 or n2 != 1:
    raise SystemExit("Could not update version/date-released in CITATION.cff")
p.write_text(text)
PY
    VERSION_EDITED=1
else
    info "Source already reports version $NEW_VERSION; treating this as a resumed release."
fi

[[ "$(current_version)" == "$NEW_VERSION" ]] || die "Version update verification failed."

if [[ "$SKIP_TESTS" != "1" ]]; then
    info "Running FastqOptiFilter calibration/accuracy release test"
    "$PYTHON" test/test_calibration.py
else
    info "Skipping tests because SKIP_TESTS=1"
fi

info "Building wheel and source distribution"
rm -rf dist build
"$PYTHON" -m build
"$PYTHON" -m twine check dist/*

if [[ -n "$(git status --porcelain -- "$VERSION_FILE" "$CITATION_FILE")" ]]; then
    info "Committing release metadata"
    git add "$VERSION_FILE" "$CITATION_FILE"
    git commit -m "Release v$NEW_VERSION"
fi
RELEASE_COMMITTED=1

PYPI_STATE="$(pypi_state)"
if [[ "$PYPI_STATE" == "missing" ]]; then
    info "Uploading FastqOptiFilter $NEW_VERSION to PyPI"
    "$PYTHON" -m twine upload --repository pypi --non-interactive dist/*
    info "Waiting for PyPI JSON metadata to expose both sdist and wheel"
    wait_for_pypi || die "Upload returned successfully, but PyPI did not expose a complete release within about one minute. Check the PyPI project page before retrying."
elif [[ "$PYPI_STATE" == "complete" ]]; then
    info "PyPI already contains a complete $PACKAGE $NEW_VERSION release; skipping upload."
else
    die "Unexpected PyPI state: $PYPI_STATE"
fi

TAG="v$NEW_VERSION"
if git rev-parse "$TAG" >/dev/null 2>&1; then
    TAG_COMMIT="$(git rev-list -n 1 "$TAG")"
    HEAD_COMMIT="$(git rev-parse HEAD)"
    [[ "$TAG_COMMIT" == "$HEAD_COMMIT" ]] || \
        die "Tag $TAG already exists but does not point to the current release commit."
else
    info "Creating git tag $TAG"
    git tag -a "$TAG" -m "FastqOptiFilter $NEW_VERSION"
fi

info "Pushing source release commit and tag to origin"
git push origin "$SOURCE_BRANCH"
git push origin "$TAG"

if [[ "$CREATE_GITHUB_RELEASE" == "1" ]]; then
    if gh release view "$TAG" >/dev/null 2>&1; then
        info "GitHub release $TAG already exists; skipping."
    else
        info "Creating GitHub release $TAG"
        gh release create "$TAG" --generate-notes --title "FastqOptiFilter $NEW_VERSION"
    fi
fi

if [[ "$SKIP_BIOCONDA" == "1" ]]; then
    info "Skipping Bioconda because SKIP_BIOCONDA=1"
    info "Release $NEW_VERSION completed."
    exit 0
fi

[[ -d "$BIOCONDA_REPO/.git" ]] || \
    die "Bioconda clone not found at $BIOCONDA_REPO. Set BIOCONDA_REPO=/path/to/bioconda-recipes."

info "Preparing Bioconda update in $BIOCONDA_REPO"
cd "$BIOCONDA_REPO"

if [[ -n "$(git status --porcelain)" ]]; then
    die "Bioconda working tree is not clean: $BIOCONDA_REPO"
fi

git remote get-url origin >/dev/null 2>&1 || die "Bioconda clone has no origin remote."
git remote get-url upstream >/dev/null 2>&1 || die "Bioconda clone has no upstream remote. Add bioconda/bioconda-recipes as upstream."

# If Bioconda's own autobump bot has already opened the update PR, do not duplicate it.
EXISTING_PR="$(gh pr list \
    --repo bioconda/bioconda-recipes \
    --state open \
    --limit 100 \
    --json title,url \
    --jq ".[] | select((.title | ascii_downcase | contains(\"fastqoptifilter\")) and (.title | contains(\"$NEW_VERSION\"))) | .url" \
    | head -n 1 || true)"

if [[ -n "$EXISTING_PR" ]]; then
    info "A Bioconda PR for FastqOptiFilter $NEW_VERSION already exists:"
    printf '%s\n' "$EXISTING_PR"
    info "Nothing more to do."
    exit 0
fi

info "Synchronizing Bioconda master with upstream"
git checkout master
git pull --ff-only upstream master
git push origin master

RECIPE="recipes/$PACKAGE/meta.yaml"
if [[ ! -f "$RECIPE" ]]; then
    info "No merged Bioconda recipe found at $RECIPE."
    info "Your initial Bioconda recipe must be merged before future releases can be autobumped."
    info "PyPI/GitHub release $NEW_VERSION is complete."
    exit 0
fi

BRANCH="update-${PACKAGE}-${NEW_VERSION}-$(date '+%Y%m%d%H%M%S')"
git checkout -b "$BRANCH"

info "Running Bioconda autobump (updates version and source checksum)"
conda run -n "$BIOCONDA_ENV" \
    bioconda-utils autobump recipes/ config.yml --packages "$PACKAGE"

if git diff --quiet -- "$RECIPE"; then
    info "Bioconda autobump made no recipe change. The recipe may already be current."
    git checkout master
    git branch -D "$BRANCH"
    exit 0
fi

info "Linting updated Bioconda recipe"
conda run -n "$BIOCONDA_ENV" \
    bioconda-utils lint recipes/ config.yml --packages "$PACKAGE"

info "Bioconda recipe diff"
git --no-pager diff -- "$RECIPE"

info "Committing and pushing Bioconda update"
git add "$RECIPE"
git commit -m "Update $PACKAGE to $NEW_VERSION"
git push -u origin "$BRANCH"

info "Opening Bioconda pull request"
PR_URL="$(gh pr create \
    --repo bioconda/bioconda-recipes \
    --base master \
    --head "$BIOCONDA_GITHUB_USER:$BRANCH" \
    --title "Update $PACKAGE to $NEW_VERSION" \
    --body "Updates FastqOptiFilter to $NEW_VERSION from PyPI. Version and source checksum were updated with bioconda-utils autobump and the recipe was linted locally. Runtime dependencies should still be reviewed if they changed upstream.")"

printf '\nBioconda PR: %s\n' "$PR_URL"
printf '\nRelease %s complete. Bioconda CI will now build/test the PR.\n' "$NEW_VERSION"
printf 'If CI is green and a maintainer label is needed, comment: @BiocondaBot please add label\n'
