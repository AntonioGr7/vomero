from pathlib import Path

from vomero.context.corpus import Corpus

CORPUS = Path(__file__).resolve().parents[1] / "examples" / "sample_corpus"


def test_files_and_subset():
    c = Corpus(CORPUS)
    files = c.files()
    assert "employees.md" in files
    assert "projects/P-ATLAS.md" in files
    sub = c.subset(["teams.md"])
    assert sub.files() == ["teams.md"]


def test_grep_finds_dependency():
    c = Corpus(CORPUS)
    hits = c.grep(r"Blocked by")
    assert any("P-BEACON" in h.path for h in hits)


def test_path_escape_blocked():
    c = Corpus(CORPUS)
    try:
        c.read("../../etc/passwd")
    except ValueError:
        return
    raise AssertionError("expected ValueError for path escaping the corpus root")
