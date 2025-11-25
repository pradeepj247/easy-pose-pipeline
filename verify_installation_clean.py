import importlib
import sys
import os

def check_import(package_name, import_name=None):
    name = import_name or package_name
    try:
        # Suppress stdout for ultralytics to avoid verbose messages
        if name == 'ultralytics':
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        
        module = importlib.import_module(name)
        
        # Restore stdout
        if name == 'ultralytics':
            sys.stdout.close()
            sys.stdout = old_stdout
        
        # Get version if available
        version = 'unknown'
        if hasattr(module, '__version__'):
            version = module.__version__
        elif name == 'cv2':
            version = module.__version__
        elif name == 'PIL':
            version = module.__version__
        
        print(f'✅ {package_name:20} v{version}')
        return True
    except ImportError as e:
        # Restore stdout if there was an error
        if name == 'ultralytics':
            sys.stdout.close()
            sys.stdout = old_stdout
        print(f'❌ {package_name}: {e}')
        return False

def main():
    print('🔍 VERIFYING INSTALLATION WITH VERSIONS')
    print('=' * 45)
    
    packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('opencv-python', 'cv2'),
        ('ultralytics', 'ultralytics'),
        ('numpy', 'numpy'),
        ('Pillow', 'PIL'),
        ('easy_ViTPose', 'easy_ViTPose')
    ]
    
    all_good = True
    for package_name, import_name in packages:
        if not check_import(package_name, import_name):
            all_good = False
    
    print('\n' + '=' * 45)
    if all_good:
        print('🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!')
        print('You can now run the demos.')
    else:
        print('⚠️  SOME PACKAGES FAILED TO IMPORT')
        print('Check the errors above and reinstall if needed.')

if __name__ == '__main__':
    main()