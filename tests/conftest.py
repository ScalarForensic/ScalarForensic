"""Repo-wide test hermeticity.

Settings() loads the CWD's .env via load_dotenv, so a populated operator .env
(e.g. a live case configuration with SFN_FACES_ENABLED=true) leaks into every
test that asserts config defaults.  CI has no .env, which hid this: the suite
passed there and failed on any examiner machine with a configured case.

The fixture neutralises both leak paths for every test: the .env file itself
(load_dotenv becomes a no-op inside scalar_forensic.config — patched where it
is *used*, per the project's mock-target convention) and SFN_* values that an
earlier Settings() call or the shell already folded into the process env.
SFN_TEST_QDRANT_URL survives — it is the opt-in gate for integration tests,
not application config.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _hermetic_settings(monkeypatch):
    monkeypatch.setattr("scalar_forensic.config.load_dotenv", lambda *a, **k: False)
    for var in list(os.environ):
        if var.startswith("SFN_") and var != "SFN_TEST_QDRANT_URL":
            monkeypatch.delenv(var)
