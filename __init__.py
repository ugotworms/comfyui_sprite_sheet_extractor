"""
__init__.py — ComfyUI custom node package for SpriteSheetExtractor.

Registers the node and adds two lightweight API routes that the web
extension uses for interactive tolerance adjustment after generation:

  POST /sprite_sheet/preview  → re-render at given tolerance, return base64 PNG
  POST /sprite_sheet/save     → re-render and overwrite the output file
"""

import base64
import io
import os

from .nodes import SpriteSheetExtractor, SpriteSheetPreview, _frame_cache, build_sprite_sheet

from PIL import Image

# ---------------------------------------------------------------------------
# API routes (registered on the live ComfyUI server instance)
# ---------------------------------------------------------------------------
try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.post("/sprite_sheet/preview")
    async def _sprite_sheet_preview(request):
        """
        Re-render the cached frames at a new tolerance and return a
        base64-encoded PNG so the JS panel can show a live preview.

        Expected JSON body:
          { node_id, tolerance, remove_background }
        """
        data           = await request.json()
        node_id        = str(data.get("node_id", ""))
        tolerance      = float(data.get("tolerance", 15.0))
        remove_bg      = bool(data.get("remove_background", True))

        if node_id not in _frame_cache:
            return web.Response(
                status=404,
                text="No cached frames for this node — run the workflow first.",
            )

        cache     = _frame_cache[node_id]
        sheet     = build_sprite_sheet(cache["frames"], tolerance, remove_bg, cache["size"])
        pil       = Image.fromarray(sheet, "RGBA")
        buf       = io.BytesIO()
        pil.save(buf, format="PNG")
        b64       = base64.b64encode(buf.getvalue()).decode("utf-8")

        return web.json_response({"image": f"data:image/png;base64,{b64}"})

    @PromptServer.instance.routes.post("/sprite_sheet/save")
    async def _sprite_sheet_save(request):
        """
        Re-render the cached frames at a new tolerance and overwrite the
        output PNG.  Called by the JS panel's Save button.

        Expected JSON body:
          { node_id, tolerance, remove_background, filename_prefix }
        """
        import folder_paths

        data            = await request.json()
        node_id         = str(data.get("node_id", ""))
        tolerance       = float(data.get("tolerance", 15.0))
        remove_bg       = bool(data.get("remove_background", True))
        prefix          = str(data.get("filename_prefix",
                               _frame_cache.get(node_id, {}).get("filename_prefix", "sprite_sheet")))

        if node_id not in _frame_cache:
            return web.Response(
                status=404,
                text="No cached frames for this node — run the workflow first.",
            )

        cache     = _frame_cache[node_id]
        sheet     = build_sprite_sheet(cache["frames"], tolerance, remove_bg, cache["size"])

        # Overwrite the exact file produced by the last workflow run rather
        # than incrementing the counter again.  Falls back to a plain prefix
        # name if the node hasn't been run yet in this session.
        if "saved_path" in cache:
            filepath = cache["saved_path"]
            filename = cache["filename"]
        else:
            prefix   = str(data.get("filename_prefix",
                           cache.get("filename_prefix", "sprite_sheet")))
            filename = f"{prefix}.png"
            filepath = os.path.join(folder_paths.get_output_directory(), filename)

        Image.fromarray(sheet, "RGBA").save(filepath)
        print(f"[SpriteSheetExtractor] Saved (interactive): {filepath}")

        return web.json_response({"saved": True, "filename": filename, "path": filepath})

except Exception as _e:
    print(f"[SpriteSheetExtractor] Warning — could not register API routes: {_e}")

# ---------------------------------------------------------------------------
# Node registration
# ---------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "SpriteSheetExtractor": SpriteSheetExtractor,
    "SpriteSheetPreview":   SpriteSheetPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SpriteSheetExtractor": "Sprite Sheet Extractor",
    "SpriteSheetPreview":   "Sprite Sheet Preview",
}

# Tells ComfyUI to serve ./web/* as /extensions/comfyui_sprite_sheet_extractor/*
WEB_DIRECTORY = "./web"
