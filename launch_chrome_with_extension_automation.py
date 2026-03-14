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

from playwright.sync_api import sync_playwright
from chrome_launcher import launch_persistent_chrome_context, resolve_profile_dir


def main():
    # Get persistent profile directory from environment or use default
    # MUST be the same PROFILE_DIR used by open_chrome_profile_for_extension_install.py
    # Convert to absolute path to ensure we're using the exact same profile
    PROFILE_DIR = resolve_profile_dir()
    
    print("Using PROFILE_DIR:", PROFILE_DIR)
    print("If you still see fixed sizing, close all Chrome windows for this profile and rerun.\n")
    
    with sync_playwright() as p:
        # Launch Chrome using shared launcher configuration.
        context, _ = launch_persistent_chrome_context(
            p,
            PROFILE_DIR,
            start_maximized=True,
            disable_viewport_emulation=True,
        )
        
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
