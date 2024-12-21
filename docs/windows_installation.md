# Windows Installation Guide

## Prerequisites

1. Python 3.8 or higher (Python 3.11 recommended)
2. Git for Windows
3. Microsoft Visual C++ Build Tools (required for some dependencies)

## Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/SakanaAI/AI-Scientist.git
cd AI-Scientist
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Common Issues and Solutions

### Missing Microsoft Visual C++ Build Tools
If you encounter build errors during installation, you may need to install the Microsoft Visual C++ Build Tools:
1. Download from [Visual Studio Downloads](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Install "Desktop development with C++"
3. Retry the installation

### Path Too Long Errors
Windows has path length limitations. If you encounter path too long errors:
1. Enable long paths in Windows:
   - Run Registry Editor (regedit)
   - Navigate to `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem`
   - Set `LongPathsEnabled` to 1
2. Or use a shorter installation path

### Environment Variables
Make sure these environment variables are set:
- PYTHONPATH: Path to the AI-Scientist directory
- APPDATA: Usually set automatically by Windows
- LOCALAPPDATA: Usually set automatically by Windows

## Running the Software

Use the launcher to run scripts:
```bash
python -m ai_scientist.launcher <script_path> [args...]
```

Example:
```bash
python -m ai_scientist.launcher ai_scientist/perform_writeup.py --config config.json
```

## Getting Help


If you encounter any issues:
1. Check the logs in `%LOCALAPPDATA%\AI-Scientist\logs`
2. Ensure all prerequisites are installed
3. Verify Python version compatibility
4. Open an issue on GitHub with:
   - Windows version
   - Python version
   - Full error message
   - Steps to reproduce
