"""
ViTPose-B Model Downloader
Automatically downloads the model from GitHub Releases
"""

import os
import requests
import sys
from pathlib import Path
from tqdm import tqdm

class ModelDownloader:
    def __init__(self):
        self.model_path = Path("models/vitpose-b.pth")
        self.model_size = 343  # MB
    
    def check_existing(self):
        """Check if model already exists"""
        if self.model_path.exists():
            actual_size = self.model_path.stat().st_size / (1024 * 1024)
            if abs(actual_size - self.model_size) < 10:
                print(f"✅ Model already exists: {self.model_path} ({actual_size:.1f} MB)")
                return True
            else:
                print(f"⚠️ Found incomplete model ({actual_size:.1f} MB), re-downloading...")
                self.model_path.unlink()
        return False
    
    def download_with_progress(self, url, destination):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        
        if response.status_code != 200:
            raise Exception(f"Download failed with status {response.status_code}")
            
        with open(destination, "wb") as file, tqdm(
            desc="Downloading ViTPose-B",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                bar.update(size)
    
    def download_from_github_releases(self):
        """Download from GitHub Releases"""
        url = "https://github.com/pradeepj247/easy-pose-pipeline/releases/download/v1.0/vitpose-b.pth"
        print("🔄 Downloading from GitHub Releases...")
        self.download_with_progress(url, self.model_path)
        
    def download(self):
        """Main download method"""
        if self.check_existing():
            return True
            
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        print("📥 Downloading ViTPose-B model (343 MB)...")
        print("💡 This may take a few minutes depending on your internet connection.")
        
        try:
            self.download_from_github_releases()
            print("✅ Download completed successfully!")
            return True
        except Exception as e:
            print(f"❌ GitHub Releases download failed: {e}")
            print("")
            print("🔧 Manual download required:")
            print("   1. Go to: https://github.com/pradeepj247/easy-pose-pipeline/releases")
            print("   2. Download vitpose-b.pth from Assets")
            print("   3. Place it in: models/vitpose-b.pth")
            return False

def download_vitpose_b():
    """Convenience function for easy usage"""
    downloader = ModelDownloader()
    return downloader.download()

if __name__ == "__main__":
    success = download_vitpose_b()
    sys.exit(0 if success else 1)
