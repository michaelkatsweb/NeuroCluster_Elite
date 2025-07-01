#!/usr/bin/env python3
"""Fix LogCategory RISK attribute missing"""

import re
from pathlib import Path

def fix_log_category():
    """Add missing RISK attribute to LogCategory enum"""
    
    logger_file = Path("src/utils/logger.py")
    if not logger_file.exists():
        print("❌ logger.py not found")
        return False
    
    try:
        content = logger_file.read_text(encoding='utf-8')
        
        # Find the LogCategory enum and add RISK if missing
        if 'class LogCategory(Enum):' in content and 'RISK = "risk"' not in content:
            # Add RISK to the enum
            pattern = r'(class LogCategory\(Enum\):.*?)(    API = "api")(.*?)(?=\n\n|\nclass|\n@|\Z)'
            replacement = r'\1\2\n    RISK = "risk"\3'
            
            content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            
            logger_file.write_text(content, encoding='utf-8')
            print("✅ Added RISK attribute to LogCategory")
            return True
        else:
            print("ℹ️ LogCategory already has RISK attribute or class not found")
            return True
            
    except Exception as e:
        print(f"❌ Error fixing LogCategory: {e}")
        return False

def alternative_fix():
    """Alternative: Change risk_manager.py to use existing category"""
    
    risk_manager_file = Path("src/trading/risk_manager.py")
    if not risk_manager_file.exists():
        print("❌ risk_manager.py not found")
        return False
    
    try:
        content = risk_manager_file.read_text(encoding='utf-8')
        
        # Replace LogCategory.RISK with LogCategory.TRADING
        if 'LogCategory.RISK' in content:
            content = content.replace('LogCategory.RISK', 'LogCategory.TRADING')
            risk_manager_file.write_text(content, encoding='utf-8')
            print("✅ Changed LogCategory.RISK to LogCategory.TRADING in risk_manager.py")
            return True
            
    except Exception as e:
        print(f"❌ Error fixing risk_manager.py: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Fixing LogCategory RISK attribute...")
    
    # Try method 1: Add RISK to LogCategory
    success = fix_log_category()
    
    if not success:
        # Try method 2: Change the usage
        print("🔄 Trying alternative fix...")
        success = alternative_fix()
    
    if success:
        print("✅ LogCategory fix applied!")
        print("🚀 Try running: python main_console.py")
    else:
        print("❌ Could not apply fix")