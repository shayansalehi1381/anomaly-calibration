import anomcal


def test_version_is_set():
    assert isinstance(anomcal.__version__, str) and anomcal.__version__


def test_core_imports_do_not_need_torch():
    # The cloud/CI side must work without the [gpu] extra.
    import anomcal.repo_guard  # noqa: F401
    import anomcal.seeding  # noqa: F401
