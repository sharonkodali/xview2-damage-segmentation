"""The commands documented in the README must parse with the real CLIs."""

import os
import re
import shlex

import pytest

import evaluate
import train

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARSERS = {"train.py": train.build_parser, "evaluate.py": evaluate.build_parser}


def readme_commands():
    with open(os.path.join(ROOT, "README.md")) as f:
        text = f.read().replace("\\\n", " ")
    cmds = []
    for line in text.splitlines():
        m = re.match(r"\s*python (train\.py|evaluate\.py)\b(.*)", line)
        if m:
            cmds.append((m.group(1), shlex.split(m.group(2))))
    return cmds


def test_readme_documents_train_and_evaluate():
    scripts = {s for s, _ in readme_commands()}
    assert scripts == {"train.py", "evaluate.py"}


@pytest.mark.parametrize("script, argv", readme_commands())
def test_readme_command_parses(script, argv):
    args = PARSERS[script]().parse_args(argv)  # SystemExit on an unknown flag
    assert args is not None


def test_reproduce_command_matches_reported_run():
    """The Reproduce train command uses the settings stored in docs/metrics.json."""
    import json

    with open(os.path.join(ROOT, "docs", "metrics.json")) as f:
        stored = json.load(f)["train_args"]
    argv = [a for s, a in readme_commands() if s == "train.py" and "--events" in a][0]
    args = vars(train.build_parser().parse_args(argv))
    for key in ["images_dir", "labels_dir", "epochs", "batch_size", "lr", "img_size",
                "base", "val_frac", "seed", "events", "weighted_loss", "out"]:
        assert args[key] == stored[key], key
