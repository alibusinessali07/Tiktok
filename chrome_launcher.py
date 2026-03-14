"""Shared Chrome launch helpers for Playwright scripts in this repo."""

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from playwright.sync_api import BrowserContext, Page, Playwright


_IGNORE_DEFAULT_ARGS = [
    "--enable-automation",
    "--disable-extensions",
    "--no-sandbox",
]


def resolve_profile_dir(
    profile_dir: Optional[Union[str, Path]] = None,
    *,
    env_var: str = "CHROME_EXTENSION_PROFILE",
    default: str = "./chrome_profile_tiktok_sorter",
) -> Path:
    """
    Resolve Chrome profile directory to an absolute path.
    Priority: explicit profile_dir -> env_var -> default.
    """
    if profile_dir is not None:
        return Path(profile_dir).resolve()
    return Path(os.getenv(env_var, default)).resolve()


def maximize_window(context: BrowserContext, page: Page) -> None:
    """Maximize Chromium window via CDP. Raises if the CDP call fails."""
    cdp = context.new_cdp_session(page)
    window_info = cdp.send("Browser.getWindowForTarget")
    cdp.send(
        "Browser.setWindowBounds",
        {"windowId": window_info["windowId"], "bounds": {"windowState": "maximized"}},
    )


def launch_persistent_chrome_context(
    playwright: Playwright,
    profile_dir: Union[str, Path],
    *,
    start_maximized: bool = False,
    disable_viewport_emulation: bool = False,
    additional_args: Optional[Sequence[str]] = None,
) -> Tuple[BrowserContext, Page]:
    """
    Launch a persistent Chrome context with extension-safe flags.
    Returns (context, first_page).
    """
    args = ["--window-position=0,0"]
    if start_maximized:
        args.append("--start-maximized")
    if additional_args:
        args.extend(additional_args)

    launch_kwargs = {
        "user_data_dir": str(Path(profile_dir).resolve()),
        "channel": "chrome",
        "headless": False,
        "args": args,
        "ignore_default_args": _IGNORE_DEFAULT_ARGS,
    }

    if disable_viewport_emulation:
        launch_kwargs["no_viewport"] = True
    else:
        launch_kwargs["viewport"] = None

    context = playwright.chromium.launch_persistent_context(**launch_kwargs)
    page = context.pages[0] if context.pages else context.new_page()

    if start_maximized:
        maximize_window(context, page)

    return context, page
