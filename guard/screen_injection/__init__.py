"""Screen-injection dataset: web pages with prompt injections at known positions.

Each page is fully determined by its seed. Injections use eight techniques: visible text, low
contrast, tiny text, fake dialogs, fake buttons, user comments, canvas-drawn text (pixels only)
and DOM-hidden text (DOM only). Decoys that look like instructions but are meant for people are
labelled too, so a guard learns the difference. See README.md for the annotation format.
"""

from .generate import TECHNIQUES, PageSpec, build_page, contrast

__all__ = ["TECHNIQUES", "PageSpec", "build_page", "contrast"]
