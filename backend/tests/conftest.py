"""
Pytest Configuration and Fixtures

Shared fixtures and configuration for all tests.
"""

import pytest
import sys
from pathlib import Path

# Add backend directory to Python path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))


@pytest.fixture(autouse=True)
def setup_test_environment():
    """
    Setup test environment before each test
    """
    # Setup code here (if needed)
    yield
    # Cleanup code here (if needed)


@pytest.fixture
def temp_test_dir(tmp_path):
    """
    Create temporary directory for test files
    """
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    return test_dir
