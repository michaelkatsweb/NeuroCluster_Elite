#!/usr/bin/env python3
"""
Intrusion detection security tests
"""

import pytest
from src.security.intrusion_detection import IntrusionDetectionSystem

class TestIntrusionDetection:
    """Test intrusion detection system"""
    
    def test_sql_injection_detection(self):
        """Test SQL injection detection"""
        ids = IntrusionDetectionSystem()
        
        malicious_payload = "'; DROP TABLE users; --"
        result = ids.analyze_request("192.168.1.1", "Mozilla/5.0", malicious_payload)
        
        assert result['threat_score'] >= 10
        assert result['threat_level'] in ['MEDIUM', 'HIGH']
        assert any('pattern' in threat for threat in result['threats_detected'])
    
    def test_xss_detection(self):
        """Test XSS attack detection"""
        ids = IntrusionDetectionSystem()
        
        xss_payload = "<script>alert('xss')</script>"
        result = ids.analyze_request("192.168.1.1", "Mozilla/5.0", xss_payload)
        
        assert result['threat_score'] >= 10
        assert result['threat_level'] in ['MEDIUM', 'HIGH']
    
    def test_suspicious_user_agent(self):
        """Test suspicious user agent detection"""
        ids = IntrusionDetectionSystem()
        
        result = ids.analyze_request("192.168.1.1", "python-requests/2.25.1", "normal_data")
        
        assert result['threat_score'] >= 5
        assert any('user agent' in threat for threat in result['threats_detected'])
    
    def test_ip_blocking(self):
        """Test IP blocking functionality"""
        ids = IntrusionDetectionSystem()
        
        # Trigger high threat
        malicious_payload = "'; DROP TABLE users; -- <script>alert('xss')</script>"
        result = ids.analyze_request("192.168.1.100", "curl/7.68.0", malicious_payload)
        
        if result['threat_score'] >= 20:
            assert ids.is_ip_blocked("192.168.1.100")
