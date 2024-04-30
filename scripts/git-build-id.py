#!/usr/bin/env python3
"""
This script calls git to get the latest commit sha & describe the dirty state of the
repository (if applicable).
"""
import datetime
import os
import subprocess
import sys


ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))


def warn_and_exit(message, exit_code=0):
    print(message, file=sys.stderr)
    print(0, file=sys.stdout)
    sys.exit(exit_code)


def run_subprocess(args, check=True):
    output = subprocess.run(
        args,
        capture_output=True,
        encoding="utf8",
        check=False,
        cwd=ROOT,
    )

    if check and output.returncode != 0:
        raise Exception(
            f"failed to run '{' '.join(args)}' (exit code {output.returncode})\n"
            f"stdout: {output.stdout}\n"
            f"stderr: {output.stderr}\n"
        )

    if output.stderr != "":
        print(output.stderr, file=sys.stderr)

    return output


if __name__ == "__main__":
    if len(sys.argv) != 2:
        warn_and_exit(f"usage: {sys.argv[0]} <include-sha>", exit_code=1)

    if sys.argv[1] == "True":
        include_sha = True
    elif sys.argv[1] == "False":
        include_sha = False
    else:
        warn_and_exit("TODO")

    try:
        result = run_subprocess(["git", "--version"])
    except Exception:
        warn_and_exit("could not run `git --version`, is git installed on your system?")

    result = run_subprocess(["git", "rev-parse", "--show-toplevel"], check=False)

    if result.returncode != 0 or not os.path.samefile(result.stdout.strip(), ROOT):
        warn_and_exit(
            "the git root is not metatensor repository, if you are trying to build "
            "metatensor from source please use a git checkout"
        )

    # get the full list of tags
    if include_sha:
        result = run_subprocess(["git", "rev-parse", "--short", "HEAD"])
        build_id = result.stdout.strip()
    else:
        build_id = ""

    # Get the list of "dirty" files: (1) tracked but modified files
    result = run_subprocess(["git", "diff-index", "HEAD", "--name-only"])
    dirty_files = result.stdout.strip().split("\n")

    # untracked files
    result = run_subprocess(["git", "ls-files", "--exclude-standard", "--others"])
    dirty_files += result.stdout.strip().split("\n")

    modified = []
    for file in dirty_files:
        if file == "":
            continue

        path = os.path.join(ROOT, file)
        if os.path.exists(path):
            modified.append(os.path.getmtime(path))

    if len(modified) > 0:
        last_modified = datetime.datetime.fromtimestamp(max(modified))

        if build_id != "":
            build_id += "."

        build_id += last_modified.strftime("dirty%Y%m%d%H%M%S")

    print(build_id)
