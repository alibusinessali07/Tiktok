"""
Launch Chrome with Extension for Playwright Automation

Why this script exists:
The Chrome Web Store blocks extension installations when the browser is launched
by automation tools (Playwright, Chromedriver, etc.). The Web Store detects
automation flags and shows "installation is not enabled".

Workflow:
1. First run open_chrome_profile_for_extension_install.py to install the extension
   manually in normal Chrome (no automation)
2. Then run this script for Playwright automation - it reuses the same PROFILE_DIR
   so the extension is already installed and active

Profile directory:
- Default: ./chrome_profile_tiktok_sorter
- Override with CHROME_EXTENSION_PROFILE environment variable
- This must be the SAME profile directory used by the installation script
- The extension persists across runs because both scripts share the same profile
"""

import os
from pathlib import Path
from playwright.sync_api import sync_playwright


def main():
    # Get persistent profile directory from environment or use default
    # MUST be the same PROFILE_DIR used by open_chrome_profile_for_extension_install.py
    # Convert to absolute path to ensure we're using the exact same profile
    PROFILE_DIR = Path(os.getenv("CHROME_EXTENSION_PROFILE", "./chrome_profile_tiktok_sorter")).resolve()
    
    print("Using PROFILE_DIR:", PROFILE_DIR)
    print("If you still see fixed sizing, close all Chrome windows for this profile and rerun.\n")
    
    with sync_playwright() as p:
        # Launch Chrome with persistent context using the same profile
        # Extension should already be installed from the manual install script
        # ignore_default_args removes Playwright's extension-blocking and sandbox flags
        # viewport=None ensures Playwright does not emulate a fixed viewport
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            channel="chrome",
            headless=False,
            viewport=None,  # No viewport emulation - allows normal window resizing
            args=[
                "--window-position=0,0",
                "--window-size=1920,1080",
            ],
            ignore_default_args=[
                "--enable-automation",
                "--disable-extensions",  # Prevent Playwright from disabling extensions
                "--no-sandbox",  # Remove sandbox flag to avoid warnings
            ],
        )
        
        # Get existing page or create new one
        if context.pages:
            page = context.pages[0]
        else:
            page = context.new_page()
        
        # Set viewport size as fallback if window sizing is still stuck
        # This is only needed if viewport=None alone doesn't work
        page.set_viewport_size({"width": 1920, "height": 1080})
        
        print("\n" + "="*70)
        print("Chrome browser is now running with Playwright automation.")
        print("The extension should be loaded and active from the persistent profile.")
        print("="*70 + "\n")
        
        # Keep browser open until user presses Enter
        input("Press Enter to close the browser...\n")
        
        # Close context cleanly
        context.close()
        print("Browser closed.")


if __name__ == "__main__":
    main()
