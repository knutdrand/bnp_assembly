import pytest


@pytest.fixture(scope="session", autouse=True)
def setup(request):
    pass
