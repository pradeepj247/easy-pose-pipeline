"""
Installation Verification Script
Run this after installing requirements to verify everything works
"""

import importlib
import sys

def check_import(package_name, import_name=None):
    """Check if a package can be imported"""
    name = import_name or package_name
    try:
        importlib.import_module(name)
        print(f"✅ {package_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: {e}")
        return False

def main():
    print("🔍 VERIFYING INSTALLATION")
    print("=" * 30)
    
    packages = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("opencv-python", "cv2"),
        ("ultralytics", "ultralytics"),
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
        ("easy_ViTPose", "easy_ViTPose")
    ]
    
    all_good = True
    for package_name, import_name in packages:
        if not check_import(package_name, import_name):
            all_good = False
    
    print("\n" + "=" * 30)
    if all_good:
        print("🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!")
        print("You can now run the demos.")
    else:
        print("⚠️  SOME PACKAGES FAILED TO IMPORT")
        print("Check the errors above and reinstall if needed.")

if __name__ == "__main__":
    main()
