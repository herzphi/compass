import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--remote-data",
        action="store_true",
        default=False,
        help="Run tests that require network access to external services (Gaia, Simbad).",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--remote-data"):
        skip = pytest.mark.skip(reason="Pass --remote-data to run network tests")
        for item in items:
            if item.get_closest_marker("remote_data"):
                item.add_marker(skip)
