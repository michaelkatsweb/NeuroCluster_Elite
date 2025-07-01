# Backups Directory

This directory contains database and configuration backups.

Backups are created automatically:
- Before major migrations
- Daily at 2:00 AM (if scheduled)
- Manually via backup utility

Use `python scripts/backup.py` to manage backups.
