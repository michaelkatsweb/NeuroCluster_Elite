#!/usr/bin/env python3
"""
File: backup.py
Path: NeuroCluster-Elite/scripts/backup.py
Description: Comprehensive backup and restore utility for NeuroCluster Elite

This utility provides complete backup and restore functionality for:
- Trading database (SQLite)
- Configuration files
- Log files
- User data and exports
- Portfolio data
- Strategy configurations
- Performance metrics
- Custom settings

Features:
- Incremental and full backups
- Compression and encryption
- Cloud storage integration (AWS S3, Google Cloud)
- Automated scheduling
- Backup verification
- Restore point management
- Data integrity checking

Usage:
    python scripts/backup.py backup [options]
    python scripts/backup.py restore [options]
    python scripts/backup.py list [options]
    python scripts/backup.py verify [options]

Author: NeuroCluster Elite Team
Created: 2025-06-30
Version: 1.0.0
License: MIT
"""

import os
import sys
import json
import sqlite3
import zipfile
import tarfile
import shutil
import hashlib
import argparse
import logging
import schedule
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import subprocess
import tempfile
import getpass

# Cloud storage imports (optional)
try:
    import boto3
    from google.cloud import storage as gcs
    CLOUD_AVAILABLE = True
except ImportError:
    CLOUD_AVAILABLE = False

# Encryption imports (optional)
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    ENCRYPTION_AVAILABLE = True
except ImportError:
    ENCRYPTION_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================

class BackupType(Enum):
    """Backup types"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    CONFIGURATION = "config"
    DATABASE = "database"
    LOGS = "logs"

class CompressionType(Enum):
    """Compression types"""
    NONE = "none"
    ZIP = "zip"
    TAR_GZ = "tar.gz"
    TAR_BZ2 = "tar.bz2"

class StorageType(Enum):
    """Storage types"""
    LOCAL = "local"
    AWS_S3 = "s3"
    GOOGLE_CLOUD = "gcs"
    FTP = "ftp"
    SFTP = "sftp"

@dataclass
class BackupConfig:
    """Backup configuration"""
    backup_type: BackupType = BackupType.FULL
    compression: CompressionType = CompressionType.TAR_GZ
    storage_type: StorageType = StorageType.LOCAL
    encryption_enabled: bool = True
    retention_days: int = 30
    max_backups: int = 10
    include_logs: bool = True
    include_cache: bool = False
    verify_backup: bool = True
    backup_schedule: str = "daily"  # daily, weekly, monthly
    
    # Paths
    source_path: str = "."
    backup_path: str = "./backups"
    
    # Cloud storage settings
    aws_bucket: Optional[str] = None
    aws_region: Optional[str] = None
    gcs_bucket: Optional[str] = None
    
    # Encryption settings
    encryption_password: Optional[str] = None

@dataclass
class BackupMetadata:
    """Backup metadata"""
    backup_id: str
    timestamp: datetime
    backup_type: BackupType
    file_count: int
    total_size: int
    compressed_size: int
    checksum: str
    encryption_enabled: bool
    storage_location: str
    source_path: str
    version: str = "1.0.0"

# ==================== BACKUP MANAGER ====================

class BackupManager:
    """Main backup manager class"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.project_root = Path(config.source_path).resolve()
        self.backup_dir = Path(config.backup_path).resolve()
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption if available and enabled
        self.encryption_key = None
        if config.encryption_enabled and ENCRYPTION_AVAILABLE:
            self._setup_encryption()
        
        # Backup patterns
        self.include_patterns = [
            "*.py", "*.yaml", "*.yml", "*.json", "*.env*",
            "*.db", "*.sqlite", "*.sql", "*.csv", "*.txt",
            "config/**/*", "data/**/*", "logs/**/*.log"
        ]
        
        self.exclude_patterns = [
            "__pycache__/", "*.pyc", "*.pyo", "*.pyd",
            ".git/", ".pytest_cache/", "node_modules/",
            "*.tmp", "*.temp", "cache/*" if not config.include_cache else ""
        ]
    
    def _setup_encryption(self):
        """Setup encryption key"""
        if not self.config.encryption_password:
            self.config.encryption_password = getpass.getpass("Enter backup encryption password: ")
        
        # Derive key from password
        password = self.config.encryption_password.encode()
        salt = b'neurocluster_elite_salt'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.encryption_key = Fernet(key)
    
    def create_backup(self) -> BackupMetadata:
        """Create a backup"""
        logger.info(f"🔄 Starting {self.config.backup_type.value} backup...")
        
        # Generate backup ID
        timestamp = datetime.now()
        backup_id = f"neurocluster_backup_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Create temporary working directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            staging_path = temp_path / "staging"
            staging_path.mkdir()
            
            # Collect files to backup
            files_to_backup = self._collect_files()
            logger.info(f"📁 Found {len(files_to_backup)} files to backup")
            
            # Copy files to staging area
            total_size = self._stage_files(files_to_backup, staging_path)
            
            # Create backup archive
            archive_path = self._create_archive(staging_path, backup_id)
            
            # Calculate checksums
            checksum = self._calculate_checksum(archive_path)
            
            # Encrypt if enabled
            if self.config.encryption_enabled and self.encryption_key:
                archive_path = self._encrypt_backup(archive_path)
            
            # Get final file size
            compressed_size = archive_path.stat().st_size
            
            # Move to final location
            final_path = self.backup_dir / archive_path.name
            shutil.move(str(archive_path), str(final_path))
            
            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=timestamp,
                backup_type=self.config.backup_type,
                file_count=len(files_to_backup),
                total_size=total_size,
                compressed_size=compressed_size,
                checksum=checksum,
                encryption_enabled=self.config.encryption_enabled,
                storage_location=str(final_path),
                source_path=str(self.project_root)
            )
            
            # Save metadata
            self._save_metadata(metadata)
            
            # Upload to cloud storage if configured
            if self.config.storage_type != StorageType.LOCAL:
                self._upload_to_cloud(final_path, metadata)
            
            # Verify backup if enabled
            if self.config.verify_backup:
                self._verify_backup(metadata)
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            logger.info(f"✅ Backup completed: {backup_id}")
            logger.info(f"📊 Size: {self._format_size(compressed_size)} ({len(files_to_backup)} files)")
            
            return metadata
    
    def _collect_files(self) -> List[Path]:
        """Collect files to backup based on backup type"""
        files = []
        
        if self.config.backup_type == BackupType.FULL:
            # Full backup - include everything
            patterns = ["**/*"]
        elif self.config.backup_type == BackupType.DATABASE:
            # Database only
            patterns = ["data/**/*.db", "data/**/*.sqlite"]
        elif self.config.backup_type == BackupType.CONFIGURATION:
            # Configuration files only
            patterns = ["config/**/*", "*.env*", "*.yaml", "*.yml", "*.json"]
        elif self.config.backup_type == BackupType.LOGS:
            # Log files only
            patterns = ["logs/**/*", "data/logs/**/*"]
        else:
            patterns = self.include_patterns
        
        # Collect files based on patterns
        for pattern in patterns:
            for file_path in self.project_root.glob(pattern):
                if file_path.is_file() and not self._should_exclude(file_path):
                    files.append(file_path.relative_to(self.project_root))
        
        return sorted(set(files))
    
    def _should_exclude(self, file_path: Path) -> bool:
        """Check if file should be excluded"""
        relative_path = file_path.relative_to(self.project_root)
        path_str = str(relative_path)
        
        for pattern in self.exclude_patterns:
            if pattern and (
                path_str.startswith(pattern.rstrip('*')) or
                pattern.rstrip('/') in path_str or
                file_path.name.endswith(pattern.lstrip('*'))
            ):
                return True
        return False
    
    def _stage_files(self, files: List[Path], staging_path: Path) -> int:
        """Copy files to staging area"""
        total_size = 0
        
        for file_path in files:
            source = self.project_root / file_path
            target = staging_path / file_path
            
            # Create parent directories
            target.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            shutil.copy2(source, target)
            total_size += source.stat().st_size
        
        return total_size
    
    def _create_archive(self, staging_path: Path, backup_id: str) -> Path:
        """Create compressed archive"""
        archive_name = f"{backup_id}.{self.config.compression.value}"
        archive_path = staging_path.parent / archive_name
        
        if self.config.compression == CompressionType.ZIP:
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in staging_path.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(staging_path)
                        zipf.write(file_path, arcname)
        
        elif self.config.compression == CompressionType.TAR_GZ:
            with tarfile.open(archive_path, 'w:gz') as tarf:
                tarf.add(staging_path, arcname='.')
        
        elif self.config.compression == CompressionType.TAR_BZ2:
            with tarfile.open(archive_path, 'w:bz2') as tarf:
                tarf.add(staging_path, arcname='.')
        
        else:  # No compression
            # Create tar without compression
            with tarfile.open(archive_path.with_suffix('.tar'), 'w') as tarf:
                tarf.add(staging_path, arcname='.')
            archive_path = archive_path.with_suffix('.tar')
        
        return archive_path
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _encrypt_backup(self, archive_path: Path) -> Path:
        """Encrypt backup file"""
        encrypted_path = archive_path.with_suffix(archive_path.suffix + '.enc')
        
        with open(archive_path, 'rb') as infile:
            with open(encrypted_path, 'wb') as outfile:
                encrypted_data = self.encryption_key.encrypt(infile.read())
                outfile.write(encrypted_data)
        
        # Remove unencrypted file
        archive_path.unlink()
        
        return encrypted_path
    
    def _save_metadata(self, metadata: BackupMetadata):
        """Save backup metadata"""
        metadata_file = self.backup_dir / f"{metadata.backup_id}_metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
    
    def _upload_to_cloud(self, file_path: Path, metadata: BackupMetadata):
        """Upload backup to cloud storage"""
        if not CLOUD_AVAILABLE:
            logger.warning("Cloud storage libraries not available")
            return
        
        logger.info(f"☁️ Uploading to {self.config.storage_type.value}...")
        
        try:
            if self.config.storage_type == StorageType.AWS_S3:
                self._upload_to_s3(file_path, metadata)
            elif self.config.storage_type == StorageType.GOOGLE_CLOUD:
                self._upload_to_gcs(file_path, metadata)
        except Exception as e:
            logger.error(f"Cloud upload failed: {e}")
    
    def _upload_to_s3(self, file_path: Path, metadata: BackupMetadata):
        """Upload to AWS S3"""
        s3_client = boto3.client('s3', region_name=self.config.aws_region)
        key = f"neurocluster-backups/{metadata.backup_id}/{file_path.name}"
        
        s3_client.upload_file(
            str(file_path),
            self.config.aws_bucket,
            key,
            ExtraArgs={'ServerSideEncryption': 'AES256'}
        )
        
        logger.info(f"✅ Uploaded to S3: s3://{self.config.aws_bucket}/{key}")
    
    def _upload_to_gcs(self, file_path: Path, metadata: BackupMetadata):
        """Upload to Google Cloud Storage"""
        client = gcs.Client()
        bucket = client.bucket(self.config.gcs_bucket)
        blob_name = f"neurocluster-backups/{metadata.backup_id}/{file_path.name}"
        blob = bucket.blob(blob_name)
        
        blob.upload_from_filename(str(file_path))
        
        logger.info(f"✅ Uploaded to GCS: gs://{self.config.gcs_bucket}/{blob_name}")
    
    def _verify_backup(self, metadata: BackupMetadata):
        """Verify backup integrity"""
        logger.info("🔍 Verifying backup integrity...")
        
        backup_file = Path(metadata.storage_location)
        if not backup_file.exists():
            raise Exception(f"Backup file not found: {backup_file}")
        
        # Verify checksum
        current_checksum = self._calculate_checksum(backup_file)
        if current_checksum != metadata.checksum:
            raise Exception("Backup checksum verification failed")
        
        # Test archive extraction (if not encrypted)
        if not metadata.encryption_enabled:
            self._test_extraction(backup_file)
        
        logger.info("✅ Backup verification successful")
    
    def _test_extraction(self, archive_path: Path):
        """Test archive extraction"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                if archive_path.suffix == '.zip':
                    with zipfile.ZipFile(archive_path, 'r') as zipf:
                        zipf.testzip()
                elif '.tar' in archive_path.suffixes:
                    with tarfile.open(archive_path, 'r') as tarf:
                        tarf.extractall(temp_dir)
            except Exception as e:
                raise Exception(f"Archive test extraction failed: {e}")
    
    def _cleanup_old_backups(self):
        """Clean up old backups based on retention policy"""
        logger.info("🧹 Cleaning up old backups...")
        
        # Get all backup metadata files
        metadata_files = list(self.backup_dir.glob("*_metadata.json"))
        
        # Parse and sort by timestamp
        backups = []
        for metadata_file in metadata_files:
            try:
                with open(metadata_file) as f:
                    data = json.load(f)
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    backups.append(data)
            except Exception as e:
                logger.warning(f"Could not parse metadata file {metadata_file}: {e}")
        
        backups.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Determine backups to delete
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)
        backups_to_delete = []
        
        # Keep recent backups within max_backups limit
        for i, backup in enumerate(backups):
            if (i >= self.config.max_backups or 
                backup['timestamp'] < cutoff_date):
                backups_to_delete.append(backup)
        
        # Delete old backups
        for backup in backups_to_delete:
            try:
                # Delete backup file
                backup_file = Path(backup['storage_location'])
                if backup_file.exists():
                    backup_file.unlink()
                
                # Delete metadata file
                metadata_file = self.backup_dir / f"{backup['backup_id']}_metadata.json"
                if metadata_file.exists():
                    metadata_file.unlink()
                
                logger.info(f"🗑️ Deleted old backup: {backup['backup_id']}")
            except Exception as e:
                logger.warning(f"Failed to delete backup {backup['backup_id']}: {e}")
    
    def list_backups(self) -> List[BackupMetadata]:
        """List all available backups"""
        backups = []
        metadata_files = list(self.backup_dir.glob("*_metadata.json"))
        
        for metadata_file in metadata_files:
            try:
                with open(metadata_file) as f:
                    data = json.load(f)
                    data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                    data['backup_type'] = BackupType(data['backup_type'])
                    backups.append(BackupMetadata(**data))
            except Exception as e:
                logger.warning(f"Could not parse metadata file {metadata_file}: {e}")
        
        return sorted(backups, key=lambda x: x.timestamp, reverse=True)
    
    def restore_backup(self, backup_id: str, restore_path: Optional[str] = None):
        """Restore a backup"""
        logger.info(f"🔄 Starting restore of backup: {backup_id}")
        
        # Find backup metadata
        metadata_file = self.backup_dir / f"{backup_id}_metadata.json"
        if not metadata_file.exists():
            raise Exception(f"Backup metadata not found: {backup_id}")
        
        with open(metadata_file) as f:
            metadata_data = json.load(f)
            metadata = BackupMetadata(**{
                **metadata_data,
                'timestamp': datetime.fromisoformat(metadata_data['timestamp']),
                'backup_type': BackupType(metadata_data['backup_type'])
            })
        
        # Determine restore path
        if restore_path is None:
            restore_path = self.project_root
        else:
            restore_path = Path(restore_path)
        
        # Get backup file
        backup_file = Path(metadata.storage_location)
        if not backup_file.exists():
            raise Exception(f"Backup file not found: {backup_file}")
        
        # Verify backup before restore
        if self.config.verify_backup:
            self._verify_backup(metadata)
        
        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Decrypt if necessary
            if metadata.encryption_enabled:
                if not self.encryption_key:
                    self._setup_encryption()
                backup_file = self._decrypt_backup(backup_file, temp_path)
            
            # Extract archive
            extract_path = temp_path / "extracted"
            extract_path.mkdir()
            
            if backup_file.suffix == '.zip':
                with zipfile.ZipFile(backup_file, 'r') as zipf:
                    zipf.extractall(extract_path)
            elif '.tar' in backup_file.suffixes:
                with tarfile.open(backup_file, 'r') as tarf:
                    tarf.extractall(extract_path)
            
            # Copy files to restore location
            self._restore_files(extract_path, restore_path)
        
        logger.info(f"✅ Restore completed to: {restore_path}")
    
    def _decrypt_backup(self, encrypted_file: Path, temp_path: Path) -> Path:
        """Decrypt backup file"""
        decrypted_file = temp_path / encrypted_file.stem
        
        with open(encrypted_file, 'rb') as infile:
            encrypted_data = infile.read()
            decrypted_data = self.encryption_key.decrypt(encrypted_data)
            
            with open(decrypted_file, 'wb') as outfile:
                outfile.write(decrypted_data)
        
        return decrypted_file
    
    def _restore_files(self, source_path: Path, target_path: Path):
        """Restore files from extracted backup"""
        target_path.mkdir(parents=True, exist_ok=True)
        
        for item in source_path.rglob('*'):
            if item.is_file():
                relative_path = item.relative_to(source_path)
                target_file = target_path / relative_path
                
                # Create parent directories
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(item, target_file)
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

# ==================== SCHEDULER ====================

class BackupScheduler:
    """Backup scheduler for automated backups"""
    
    def __init__(self, config: BackupConfig):
        self.config = config
        self.backup_manager = BackupManager(config)
        self.running = False
    
    def start_scheduler(self):
        """Start the backup scheduler"""
        logger.info("⏰ Starting backup scheduler...")
        
        # Schedule based on configuration
        if self.config.backup_schedule == "daily":
            schedule.every().day.at("02:00").do(self._run_scheduled_backup)
        elif self.config.backup_schedule == "weekly":
            schedule.every().sunday.at("02:00").do(self._run_scheduled_backup)
        elif self.config.backup_schedule == "monthly":
            schedule.every(30).days.at("02:00").do(self._run_scheduled_backup)
        
        self.running = True
        
        # Run scheduler loop in separate thread
        scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        scheduler_thread.start()
        
        logger.info(f"✅ Scheduler started ({self.config.backup_schedule})")
    
    def stop_scheduler(self):
        """Stop the backup scheduler"""
        self.running = False
        schedule.clear()
        logger.info("⏹️ Backup scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _run_scheduled_backup(self):
        """Run scheduled backup"""
        try:
            logger.info("🔄 Running scheduled backup...")
            metadata = self.backup_manager.create_backup()
            logger.info(f"✅ Scheduled backup completed: {metadata.backup_id}")
        except Exception as e:
            logger.error(f"❌ Scheduled backup failed: {e}")

# ==================== COMMAND LINE INTERFACE ====================

def create_config_from_args(args) -> BackupConfig:
    """Create backup configuration from command line arguments"""
    return BackupConfig(
        backup_type=BackupType(args.type),
        compression=CompressionType(args.compression),
        storage_type=StorageType(args.storage),
        encryption_enabled=args.encrypt,
        retention_days=args.retention,
        max_backups=args.max_backups,
        include_logs=args.include_logs,
        include_cache=args.include_cache,
        verify_backup=args.verify,
        source_path=args.source,
        backup_path=args.backup_dir,
        aws_bucket=args.aws_bucket,
        aws_region=args.aws_region,
        gcs_bucket=args.gcs_bucket
    )

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="NeuroCluster Elite Backup Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create full backup
    python backup.py backup --type full
    
    # Create database backup with encryption
    python backup.py backup --type database --encrypt
    
    # List all backups
    python backup.py list
    
    # Restore specific backup
    python backup.py restore backup_20250630_120000
    
    # Start automated scheduler
    python backup.py schedule --daily
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Create backup')
    backup_parser.add_argument('--type', choices=['full', 'incremental', 'database', 'config', 'logs'],
                              default='full', help='Backup type')
    backup_parser.add_argument('--compression', choices=['none', 'zip', 'tar.gz', 'tar.bz2'],
                              default='tar.gz', help='Compression type')
    backup_parser.add_argument('--storage', choices=['local', 's3', 'gcs'],
                              default='local', help='Storage type')
    backup_parser.add_argument('--encrypt', action='store_true', help='Encrypt backup')
    backup_parser.add_argument('--no-verify', dest='verify', action='store_false', help='Skip verification')
    backup_parser.add_argument('--include-logs', action='store_true', default=True, help='Include log files')
    backup_parser.add_argument('--include-cache', action='store_true', help='Include cache files')
    backup_parser.add_argument('--retention', type=int, default=30, help='Retention days')
    backup_parser.add_argument('--max-backups', type=int, default=10, help='Maximum backups to keep')
    backup_parser.add_argument('--source', default='.', help='Source directory')
    backup_parser.add_argument('--backup-dir', default='./backups', help='Backup directory')
    backup_parser.add_argument('--aws-bucket', help='AWS S3 bucket name')
    backup_parser.add_argument('--aws-region', help='AWS region')
    backup_parser.add_argument('--gcs-bucket', help='Google Cloud Storage bucket')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore backup')
    restore_parser.add_argument('backup_id', help='Backup ID to restore')
    restore_parser.add_argument('--target', help='Restore target directory')
    restore_parser.add_argument('--backup-dir', default='./backups', help='Backup directory')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List backups')
    list_parser.add_argument('--backup-dir', default='./backups', help='Backup directory')
    
    # Verify command
    verify_parser = subparsers.add_parser('verify', help='Verify backup')
    verify_parser.add_argument('backup_id', help='Backup ID to verify')
    verify_parser.add_argument('--backup-dir', default='./backups', help='Backup directory')
    
    # Schedule command
    schedule_parser = subparsers.add_parser('schedule', help='Start backup scheduler')
    schedule_parser.add_argument('--daily', action='store_const', const='daily', dest='schedule',
                                help='Daily backups')
    schedule_parser.add_argument('--weekly', action='store_const', const='weekly', dest='schedule',
                                help='Weekly backups')
    schedule_parser.add_argument('--monthly', action='store_const', const='monthly', dest='schedule',
                                help='Monthly backups')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == 'backup':
            config = create_config_from_args(args)
            manager = BackupManager(config)
            metadata = manager.create_backup()
            print(f"✅ Backup created: {metadata.backup_id}")
            
        elif args.command == 'restore':
            config = BackupConfig(backup_path=args.backup_dir)
            manager = BackupManager(config)
            manager.restore_backup(args.backup_id, args.target)
            print(f"✅ Backup restored: {args.backup_id}")
            
        elif args.command == 'list':
            config = BackupConfig(backup_path=args.backup_dir)
            manager = BackupManager(config)
            backups = manager.list_backups()
            
            if not backups:
                print("No backups found")
            else:
                print(f"{'Backup ID':<30} {'Type':<12} {'Date':<20} {'Size':<10} {'Files':<8}")
                print("-" * 80)
                for backup in backups:
                    size_str = manager._format_size(backup.compressed_size)
                    date_str = backup.timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"{backup.backup_id:<30} {backup.backup_type.value:<12} "
                          f"{date_str:<20} {size_str:<10} {backup.file_count:<8}")
            
        elif args.command == 'verify':
            config = BackupConfig(backup_path=args.backup_dir)
            manager = BackupManager(config)
            
            # Load metadata
            metadata_file = Path(args.backup_dir) / f"{args.backup_id}_metadata.json"
            with open(metadata_file) as f:
                metadata_data = json.load(f)
                metadata = BackupMetadata(**{
                    **metadata_data,
                    'timestamp': datetime.fromisoformat(metadata_data['timestamp']),
                    'backup_type': BackupType(metadata_data['backup_type'])
                })
            
            manager._verify_backup(metadata)
            print(f"✅ Backup verified: {args.backup_id}")
            
        elif args.command == 'schedule':
            config = BackupConfig(backup_schedule=args.schedule or 'daily')
            scheduler = BackupScheduler(config)
            scheduler.start_scheduler()
            
            print(f"⏰ Backup scheduler started ({config.backup_schedule})")
            print("Press Ctrl+C to stop...")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                scheduler.stop_scheduler()
                print("\n✅ Scheduler stopped")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()