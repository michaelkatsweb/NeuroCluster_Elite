#!/usr/bin/env python3
"""
Test utilities and fixtures
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

@pytest.fixture
def temp_directory():
    """Provide temporary directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def mock_market_data():
    """Provide mock market data for testing"""
    return {
        'symbol': 'AAPL',
        'price': 150.00,
        'change': 2.50,
        'change_percent': 1.69,
        'volume': 1000000
    }

@pytest.fixture
def mock_security_manager():
    """Provide mock security manager"""
    manager = Mock()
    manager.hash_password.return_value = "hashed_password"
    manager.verify_password.return_value = True
    manager.create_jwt_token.return_value = "mock_jwt_token"
    return manager
