# GitHub Releases Setup Instructions

## How to upload the model to GitHub Releases:

1. **Go to your repository**: https://github.com/pradeepj247/easy-pose-pipeline

2. **Create a new release**:
   - Click "Releases" in the right sidebar
   - Click "Create a new release"
   - Tag: `v1.0`
   - Title: `Version 1.0 - Initial Release`
   - Description: Include model details and usage instructions

3. **Upload the model**:
   - Drag and drop `releases/vitpose-b.pth` to the "Assets" section
   - Or click "Attach binaries" and select the file

4. **Publish the release**

## The download script will automatically:
- Check if model exists
- Download from GitHub Releases
- Show progress bar
- Fallback to manual instructions if needed

## File Structure:
```
releases/
└── vitpose-b.pth          # Model for GitHub Releases upload
download_model.py          # Smart downloader
demos/basic_demo.py        # Updated demo with auto-download
```
