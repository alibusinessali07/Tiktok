"""
Open Chrome Profile for Extension Installation (Manual, No Automation)

Why this script exists:
The Chrome Web Store blocks extension installations when the browser is launched
by automation tools (Playwright, Chromedriver, etc.). The Web Store detects
automation flags and shows "installation is not enabled".

This script launches Chrome as a completely unmanaged process (no automation
control), allowing normal Web Store installations to work.

Workflow:
1. Run this script to install the extension manually in normal Chrome
2. Then use launch_chrome_with_extension_automation.py for Playwright automation
3. Both scripts use the same PROFILE_DIR, so the extension persists across both

Profile directory:
- Default: ./chrome_profile_tiktok_sorter
- Override with CHROME_EXTENSION_PROFILE environment variable
- This profile persists across runs - install the extension once, and it will
  remain installed for all future automation runs that reuse the same profile directory.
"""

import os
import sys
import subprocess
import platform


def find_chrome_executable():
    """
    Find Chrome executable path, checking environment variable first,
    then platform-specific default locations.
    """
    # Check environment variable override
    chrome_path = os.getenv("CHROME_EXECUTABLE_PATH")
    if chrome_path and os.path.exists(chrome_path):
        return chrome_path
    
    system = platform.system()
    
    if system == "Windows":
        # Windows: check common installation paths
        possible_paths = [
            os.path.join(os.getenv("PROGRAMFILES", ""), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.getenv("PROGRAMFILES(X86)", ""), "Google", "Chrome", "Application", "chrome.exe"),
            os.path.join(os.getenv("LOCALAPPDATA", ""), "Google", "Chrome", "Application", "chrome.exe"),
        ]
        for path in possible_paths:
            if path and os.path.exists(path):
                return path
    
    elif system == "Darwin":  # macOS
        chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if os.path.exists(chrome_path):
            return chrome_path
    
    elif system == "Linux":
        # Linux: check common locations
        possible_paths = [
            "/usr/bin/google-chrome",
            "/usr/bin/google-chrome-stable",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
    
    return None


def main():
    # Get persistent profile directory from environment or use default
    PROFILE_DIR = os.getenv("CHROME_EXTENSION_PROFILE", "./chrome_profile_tiktok_sorter")
    
    # Extension Chrome Web Store URL
    EXT_URL = "https://chromewebstore.google.com/detail/tiktok-instagram-sorter/bmljpagafjlkebnopbdncpnifkknlobk"
    
    # Find Chrome executable
    chrome_path = find_chrome_executable()
    if not chrome_path:
        print("ERROR: Chrome executable not found.")
        print("\nPlease set CHROME_EXECUTABLE_PATH environment variable to your Chrome path,")
        print("or install Google Chrome in a standard location.")
        print("\nStandard locations checked:")
        if platform.system() == "Windows":
            print("  - %PROGRAMFILES%\\Google\\Chrome\\Application\\chrome.exe")
            print("  - %PROGRAMFILES(X86)%\\Google\\Chrome\\Application\\chrome.exe")
            print("  - %LOCALAPPDATA%\\Google\\Chrome\\Application\\chrome.exe")
        elif platform.system() == "Darwin":
            print("  - /Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        else:
            print("  - /usr/bin/google-chrome")
            print("  - /usr/bin/google-chrome-stable")
        sys.exit(1)
    
    print("="*70)
    print("Opening Chrome for Extension Installation")
    print("="*70)
    print(f"Profile directory: {os.path.abspath(PROFILE_DIR)}")
    print(f"Extension URL: {EXT_URL}")
    print("="*70)
    print("\nThis Chrome is NOT automated. Install the extension normally, then close Chrome.\n")
    
    # Build Chrome command
    chrome_args = [
        chrome_path,
        f"--user-data-dir={os.path.abspath(PROFILE_DIR)}",
        "--no-first-run",
        "--no-default-browser-check",
        "--start-maximized",
        EXT_URL,
    ]
    
    # Launch Chrome as unmanaged process
    try:
        process = subprocess.Popen(chrome_args)
        print("Chrome launched. Waiting for you to close it...\n")
        
        # Wait for Chrome process to exit
        process.wait()
        
        print("\n" + "="*70)
        print("Done. Extension is saved in PROFILE_DIR and will load in automation runs.")
        print("="*70)
        print(f"\nProfile location: {os.path.abspath(PROFILE_DIR)}")
        print("\nYou can now run launch_chrome_with_extension_automation.py")
        print("to use Playwright automation with the extension already installed.\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted. Chrome may still be running.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: Failed to launch Chrome: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
