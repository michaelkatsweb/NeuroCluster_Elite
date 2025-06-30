#!/usr/bin/env python3
"""
File: security.py
Path: NeuroCluster-Elite/src/utils/security.py
Description: Security management system for NeuroCluster Elite

This module provides comprehensive security features for the NeuroCluster Elite
trading platform, including authentication, authorization, encryption, rate limiting,
and security monitoring.

Features:
- JWT-based authentication with refresh tokens
- Role-based access control (RBAC)
- API key management and rotation
- Rate limiting and DDoS protection
- Request encryption and validation
- Security event logging and monitoring
- Two-factor authentication (2FA)
- Session management

Author: Your Name
Created: 2025-06-28
Version: 1.0.0
License: MIT
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import logging
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import threading
import json
import base64
from pathlib import Path

# Cryptography imports
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Authentication imports
import jwt
from passlib.context import CryptContext
from passlib.totp import TOTP
import qrcode
import io

# FastAPI security imports
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Configure logging
logger = logging.getLogger(__name__)

# ==================== ENUMS AND DATA STRUCTURES ====================

class UserRole(Enum):
    """User roles for access control"""
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"
    API_USER = "api_user"
    GUEST = "guest"

class SecurityEventType(Enum):
    """Security event types for logging"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    TOKEN_REFRESH = "token_refresh"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    PERMISSION_DENIED = "permission_denied"
    PASSWORD_CHANGED = "password_changed"
    TWO_FA_ENABLED = "two_fa_enabled"

@dataclass
class User:
    """User data structure"""
    id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    is_active: bool = True
    is_verified: bool = False
    two_fa_secret: Optional[str] = None
    two_fa_enabled: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    login_attempts: int = 0
    locked_until: Optional[datetime] = None

@dataclass
class APIKey:
    """API key data structure"""
    id: str
    key_hash: str
    name: str
    user_id: str
    permissions: List[str]
    is_active: bool = True
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    usage_count: int = 0

@dataclass
class SecurityEvent:
    """Security event data structure"""
    event_type: SecurityEventType
    user_id: Optional[str]
    ip_address: str
    user_agent: Optional[str]
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    severity: str = "info"  # info, warning, error, critical

@dataclass
class RateLimitRule:
    """Rate limiting rule"""
    name: str
    max_requests: int
    window_seconds: int
    scope: str = "global"  # global, user, ip
    block_duration_seconds: int = 300

# ==================== SECURITY MANAGER ====================

class SecurityManager:
    """Comprehensive security management system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        
        # Initialize cryptography
        self.password_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.encryption_key = self._initialize_encryption()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # JWT configuration
        self.jwt_secret = self.config.get("jwt_secret", secrets.token_urlsafe(32))
        self.jwt_algorithm = "HS256"
        self.jwt_expiry_minutes = self.config.get("jwt_expiry_minutes", 30)
        self.refresh_token_expiry_days = self.config.get("refresh_token_expiry_days", 7)
        
        # Storage (in production, use proper database)
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self.security_events: List[SecurityEvent] = []
        self.active_sessions: Dict[str, Dict] = {}
        self.blocked_ips: Dict[str, datetime] = {}
        
        # Threading locks
        self.security_lock = threading.RLock()
        
        # Initialize default admin user
        self._create_default_admin()
        
        logger.info("SecurityManager initialized")
    
    def _default_config(self) -> Dict:
        """Default security configuration"""
        return {
            "max_login_attempts": 5,
            "account_lockout_minutes": 15,
            "jwt_expiry_minutes": 30,
            "refresh_token_expiry_days": 7,
            "api_key_expiry_days": 90,
            "password_min_length": 8,
            "require_uppercase": True,
            "require_lowercase": True,
            "require_numbers": True,
            "require_special_chars": True,
            "session_timeout_minutes": 60,
            "max_concurrent_sessions": 5
        }
    
    def _initialize_encryption(self) -> bytes:
        """Initialize encryption key"""
        
        key_file = Path("config/.security_key")
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            key_file.parent.mkdir(exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            # Secure the key file
            key_file.chmod(0o600)
            return key
    
    def _create_default_admin(self):
        """Create default admin user"""
        
        admin_username = "admin"
        admin_password = "NeuroCluster2025!"  # Should be changed immediately
        
        if admin_username not in self.users:
            admin_user = User(
                id=secrets.token_urlsafe(16),
                username=admin_username,
                email="admin@neurocluster-elite.com",
                password_hash=self.hash_password(admin_password),
                role=UserRole.ADMIN,
                is_active=True,
                is_verified=True
            )
            
            self.users[admin_username] = admin_user
            logger.warning(f"Default admin user created: {admin_username} / {admin_password}")
            logger.warning("CHANGE DEFAULT PASSWORD IMMEDIATELY!")
    
    # ==================== PASSWORD MANAGEMENT ====================
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self.password_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.password_context.verify(plain_password, hashed_password)
    
    def validate_password_strength(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password strength"""
        
        errors = []
        
        if len(password) < self.config["password_min_length"]:
            errors.append(f"Password must be at least {self.config['password_min_length']} characters")
        
        if self.config["require_uppercase"] and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.config["require_lowercase"] and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.config["require_numbers"] and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        
        if self.config["require_special_chars"] and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    # ==================== USER MANAGEMENT ====================
    
    def create_user(self, username: str, email: str, password: str, role: UserRole = UserRole.TRADER) -> User:
        """Create new user"""
        
        with self.security_lock:
            # Validate inputs
            if username in self.users:
                raise ValueError("Username already exists")
            
            # Validate password strength
            is_strong, errors = self.validate_password_strength(password)
            if not is_strong:
                raise ValueError(f"Password validation failed: {', '.join(errors)}")
            
            # Create user
            user = User(
                id=secrets.token_urlsafe(16),
                username=username,
                email=email,
                password_hash=self.hash_password(password),
                role=role
            )
            
            self.users[username] = user
            
            # Log security event
            self._log_security_event(
                SecurityEventType.LOGIN_SUCCESS,
                user_id=user.id,
                ip_address="127.0.0.1",
                details={"action": "user_created", "username": username}
            )
            
            logger.info(f"User created: {username}")
            return user
    
    def authenticate_user(self, username: str, password: str, ip_address: str) -> Optional[User]:
        """Authenticate user with username/password"""
        
        with self.security_lock:
            user = self.users.get(username)
            
            if not user:
                self._log_security_event(
                    SecurityEventType.LOGIN_FAILURE,
                    user_id=None,
                    ip_address=ip_address,
                    details={"reason": "user_not_found", "username": username}
                )
                return None
            
            # Check if account is locked
            if user.locked_until and datetime.utcnow() < user.locked_until:
                self._log_security_event(
                    SecurityEventType.LOGIN_FAILURE,
                    user_id=user.id,
                    ip_address=ip_address,
                    details={"reason": "account_locked", "username": username}
                )
                return None
            
            # Check if account is active
            if not user.is_active:
                self._log_security_event(
                    SecurityEventType.LOGIN_FAILURE,
                    user_id=user.id,
                    ip_address=ip_address,
                    details={"reason": "account_disabled", "username": username}
                )
                return None
            
            # Verify password
            if not self.verify_password(password, user.password_hash):
                user.login_attempts += 1
                
                # Lock account if too many failed attempts
                if user.login_attempts >= self.config["max_login_attempts"]:
                    user.locked_until = datetime.utcnow() + timedelta(
                        minutes=self.config["account_lockout_minutes"]
                    )
                    logger.warning(f"Account locked due to failed login attempts: {username}")
                
                self._log_security_event(
                    SecurityEventType.LOGIN_FAILURE,
                    user_id=user.id,
                    ip_address=ip_address,
                    details={"reason": "invalid_password", "username": username, "attempts": user.login_attempts}
                )
                return None
            
            # Reset failed attempts on successful login
            user.login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.utcnow()
            
            self._log_security_event(
                SecurityEventType.LOGIN_SUCCESS,
                user_id=user.id,
                ip_address=ip_address,
                details={"username": username}
            )
            
            return user
    
    # ==================== JWT TOKEN MANAGEMENT ====================
    
    def create_access_token(self, user: User, additional_claims: Dict = None) -> str:
        """Create JWT access token"""
        
        claims = {
            "sub": user.id,
            "username": user.username,
            "role": user.role.value,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(minutes=self.jwt_expiry_minutes),
            "type": "access"
        }
        
        if additional_claims:
            claims.update(additional_claims)
        
        return jwt.encode(claims, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token"""
        
        claims = {
            "sub": user.id,
            "username": user.username,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(days=self.refresh_token_expiry_days),
            "type": "refresh"
        }
        
        return jwt.encode(claims, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify and decode JWT token"""
        
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            
            # Check token type
            if payload.get("type") not in ["access", "refresh"]:
                return None
            
            # Check expiration
            if datetime.utcnow() > datetime.fromisoformat(payload["exp"]):
                return None
            
            return payload
            
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Refresh access token using refresh token"""
        
        payload = self.verify_token(refresh_token)
        
        if not payload or payload.get("type") != "refresh":
            return None
        
        user = self.get_user_by_id(payload["sub"])
        if not user or not user.is_active:
            return None
        
        # Create new access token
        new_token = self.create_access_token(user)
        
        self._log_security_event(
            SecurityEventType.TOKEN_REFRESH,
            user_id=user.id,
            ip_address="127.0.0.1",
            details={"username": user.username}
        )
        
        return new_token
    
    # ==================== API KEY MANAGEMENT ====================
    
    def create_api_key(self, user_id: str, name: str, permissions: List[str] = None, 
                      expires_days: int = None) -> Tuple[str, APIKey]:
        """Create new API key"""
        
        with self.security_lock:
            # Generate API key
            api_key_value = f"nce_{secrets.token_urlsafe(32)}"
            key_hash = hashlib.sha256(api_key_value.encode()).hexdigest()
            
            # Set expiration
            expires_at = None
            if expires_days:
                expires_at = datetime.utcnow() + timedelta(days=expires_days)
            elif self.config.get("api_key_expiry_days"):
                expires_at = datetime.utcnow() + timedelta(days=self.config["api_key_expiry_days"])
            
            # Create API key object
            api_key = APIKey(
                id=secrets.token_urlsafe(16),
                key_hash=key_hash,
                name=name,
                user_id=user_id,
                permissions=permissions or ["read"],
                expires_at=expires_at
            )
            
            self.api_keys[key_hash] = api_key
            
            self._log_security_event(
                SecurityEventType.API_KEY_CREATED,
                user_id=user_id,
                ip_address="127.0.0.1",
                details={"api_key_name": name, "permissions": permissions}
            )
            
            logger.info(f"API key created: {name} for user {user_id}")
            return api_key_value, api_key
    
    def verify_api_key(self, api_key_value: str) -> Optional[APIKey]:
        """Verify API key"""
        
        key_hash = hashlib.sha256(api_key_value.encode()).hexdigest()
        api_key = self.api_keys.get(key_hash)
        
        if not api_key:
            return None
        
        # Check if active
        if not api_key.is_active:
            return None
        
        # Check expiration
        if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
            api_key.is_active = False
            return None
        
        # Update usage statistics
        api_key.last_used = datetime.utcnow()
        api_key.usage_count += 1
        
        return api_key
    
    def revoke_api_key(self, key_hash: str, user_id: str) -> bool:
        """Revoke API key"""
        
        with self.security_lock:
            api_key = self.api_keys.get(key_hash)
            
            if not api_key or api_key.user_id != user_id:
                return False
            
            api_key.is_active = False
            
            self._log_security_event(
                SecurityEventType.API_KEY_REVOKED,
                user_id=user_id,
                ip_address="127.0.0.1",
                details={"api_key_name": api_key.name}
            )
            
            logger.info(f"API key revoked: {api_key.name}")
            return True
    
    # ==================== TWO-FACTOR AUTHENTICATION ====================
    
    def enable_two_factor_auth(self, user: User) -> Tuple[str, str]:
        """Enable 2FA for user and return secret and QR code"""
        
        # Generate TOTP secret
        secret = secrets.token_urlsafe(32)
        user.two_fa_secret = secret
        user.two_fa_enabled = True
        
        # Generate QR code
        totp = TOTP(secret)
        qr_uri = totp.provisioning_uri(
            name=user.email,
            issuer_name="NeuroCluster Elite"
        )
        
        # Create QR code image
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        img_buffer = io.BytesIO()
        img.save(img_buffer, format='PNG')
        qr_code_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        self._log_security_event(
            SecurityEventType.TWO_FA_ENABLED,
            user_id=user.id,
            ip_address="127.0.0.1",
            details={"username": user.username}
        )
        
        return secret, qr_code_b64
    
    def verify_two_factor_token(self, user: User, token: str) -> bool:
        """Verify 2FA token"""
        
        if not user.two_fa_enabled or not user.two_fa_secret:
            return False
        
        try:
            totp = TOTP(user.two_fa_secret)
            return totp.verify(token, valid_window=1)
        except Exception as e:
            logger.error(f"2FA verification error: {e}")
            return False
    
    # ==================== RATE LIMITING ====================
    
    def check_rate_limit(self, identifier: str, rule: RateLimitRule) -> bool:
        """Check if request is within rate limits"""
        
        current_time = time.time()
        window_start = current_time - rule.window_seconds
        
        # Get or create request history for identifier
        if not hasattr(self, '_rate_limit_data'):
            self._rate_limit_data = {}
        
        if identifier not in self._rate_limit_data:
            self._rate_limit_data[identifier] = []
        
        request_history = self._rate_limit_data[identifier]
        
        # Remove old requests outside the window
        request_history[:] = [req_time for req_time in request_history if req_time > window_start]
        
        # Check if within limit
        if len(request_history) >= rule.max_requests:
            self._log_security_event(
                SecurityEventType.RATE_LIMIT_EXCEEDED,
                user_id=None,
                ip_address=identifier,
                details={"rule": rule.name, "requests": len(request_history), "limit": rule.max_requests}
            )
            return False
        
        # Add current request
        request_history.append(current_time)
        return True
    
    # ==================== SECURITY EVENT LOGGING ====================
    
    def _log_security_event(self, event_type: SecurityEventType, user_id: Optional[str],
                           ip_address: str, details: Dict[str, Any], user_agent: str = None):
        """Log security event"""
        
        event = SecurityEvent(
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            details=details
        )
        
        # Determine severity
        if event_type in [SecurityEventType.LOGIN_FAILURE, SecurityEventType.RATE_LIMIT_EXCEEDED]:
            event.severity = "warning"
        elif event_type in [SecurityEventType.SUSPICIOUS_ACTIVITY, SecurityEventType.PERMISSION_DENIED]:
            event.severity = "error"
        
        self.security_events.append(event)
        
        # Log to file
        logger.info(f"Security event: {event_type.value} | User: {user_id} | IP: {ip_address} | Details: {details}")
        
        # Keep only recent events (last 10000)
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-10000:]
    
    def get_security_events(self, user_id: str = None, event_type: SecurityEventType = None,
                           hours: int = 24) -> List[SecurityEvent]:
        """Get security events with filtering"""
        
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        events = [
            event for event in self.security_events
            if event.timestamp > cutoff_time
        ]
        
        if user_id:
            events = [event for event in events if event.user_id == user_id]
        
        if event_type:
            events = [event for event in events if event.event_type == event_type]
        
        return events
    
    # ==================== UTILITY METHODS ====================
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        for user in self.users.values():
            if user.id == user_id:
                return user
        return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.users.get(username)
    
    def has_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission"""
        
        # Admin has all permissions
        if user.role == UserRole.ADMIN:
            return True
        
        # Define role permissions
        role_permissions = {
            UserRole.TRADER: ["read", "trade", "portfolio"],
            UserRole.VIEWER: ["read"],
            UserRole.API_USER: ["read", "api"],
            UserRole.GUEST: []
        }
        
        return permission in role_permissions.get(user.role, [])
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()

# ==================== RATE LIMITER ====================

class RateLimiter:
    """Advanced rate limiting system"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.request_counts = {}
        self.blocked_ips = {}
        self.lock = threading.RLock()
        
        # Define rate limiting rules
        self.rules = {
            "api_general": RateLimitRule("API General", 100, 60),
            "api_trading": RateLimitRule("API Trading", 20, 60),
            "login_attempts": RateLimitRule("Login Attempts", 5, 300),
            "password_reset": RateLimitRule("Password Reset", 3, 3600)
        }
    
    def _default_config(self) -> Dict:
        """Default rate limiting configuration"""
        return {
            "enabled": True,
            "default_limit": 100,
            "default_window": 60,
            "block_duration": 300
        }
    
    def allow_request(self, identifier: str, rule_name: str = "api_general") -> bool:
        """Check if request should be allowed"""
        
        if not self.config["enabled"]:
            return True
        
        with self.lock:
            # Check if IP is blocked
            if identifier in self.blocked_ips:
                if time.time() < self.blocked_ips[identifier]:
                    return False
                else:
                    del self.blocked_ips[identifier]
            
            # Get rule
            rule = self.rules.get(rule_name, self.rules["api_general"])
            
            # Check rate limit
            current_time = time.time()
            window_start = current_time - rule.window_seconds
            
            if identifier not in self.request_counts:
                self.request_counts[identifier] = []
            
            # Clean old requests
            self.request_counts[identifier] = [
                req_time for req_time in self.request_counts[identifier]
                if req_time > window_start
            ]
            
            # Check limit
            if len(self.request_counts[identifier]) >= rule.max_requests:
                # Block IP
                self.blocked_ips[identifier] = time.time() + rule.block_duration_seconds
                return False
            
            # Add current request
            self.request_counts[identifier].append(current_time)
            return True
    
    def get_remaining_requests(self, identifier: str, rule_name: str = "api_general") -> int:
        """Get remaining requests for identifier"""
        
        rule = self.rules.get(rule_name, self.rules["api_general"])
        
        with self.lock:
            if identifier not in self.request_counts:
                return rule.max_requests
            
            current_time = time.time()
            window_start = current_time - rule.window_seconds
            
            # Count recent requests
            recent_requests = [
                req_time for req_time in self.request_counts[identifier]
                if req_time > window_start
            ]
            
            return max(0, rule.max_requests - len(recent_requests))

# ==================== FASTAPI SECURITY DEPENDENCIES ====================

class JWTBearer(HTTPBearer):
    """JWT Bearer token authentication for FastAPI"""
    
    def __init__(self, security_manager: SecurityManager, auto_error: bool = True):
        super(JWTBearer, self).__init__(auto_error=auto_error)
        self.security_manager = security_manager
    
    async def __call__(self, request: Request) -> Optional[Dict]:
        credentials: HTTPAuthorizationCredentials = await super(JWTBearer, self).__call__(request)
        
        if credentials:
            if credentials.scheme != "Bearer":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid authentication scheme."
                )
            
            payload = self.security_manager.verify_token(credentials.credentials)
            if not payload:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid or expired token."
                )
            
            return payload
        
        return None

# ==================== MAIN FUNCTION ====================

if __name__ == "__main__":
    # Test security manager
    print("🔒 Testing NeuroCluster Elite Security Manager")
    
    # Initialize security manager
    security_manager = SecurityManager()
    
    # Test user creation
    user = security_manager.create_user("testuser", "test@example.com", "TestPassword123!", UserRole.TRADER)
    print(f"User created: {user.username}")
    
    # Test authentication
    auth_user = security_manager.authenticate_user("testuser", "TestPassword123!", "127.0.0.1")
    print(f"Authentication: {'✅ PASS' if auth_user else '❌ FAIL'}")
    
    # Test JWT tokens
    access_token = security_manager.create_access_token(user)
    payload = security_manager.verify_token(access_token)
    print(f"JWT token: {'✅ PASS' if payload else '❌ FAIL'}")
    
    # Test API key
    api_key_value, api_key = security_manager.create_api_key(user.id, "Test API Key", ["read", "trade"])
    verified_api_key = security_manager.verify_api_key(api_key_value)
    print(f"API key: {'✅ PASS' if verified_api_key else '❌ FAIL'}")
    
    # Test rate limiting
    rate_limiter = RateLimiter()
    for i in range(5):
        allowed = rate_limiter.allow_request("127.0.0.1", "login_attempts")
        if not allowed:
            print(f"Rate limit exceeded after {i} requests")
            break
    
    print("✅ Security manager test completed!")