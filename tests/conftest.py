"""Keep tests away from the developer's real builds/ folder and .env settings.

The server loads every stored build when it is imported, so point it at an empty
temporary folder before any test module imports it.
"""
import os
import tempfile

os.environ["NF_BUILDS_DIR"] = tempfile.mkdtemp(prefix="nf-tests-")
os.environ["NF_DEMO_BUILDS"] = "0"
