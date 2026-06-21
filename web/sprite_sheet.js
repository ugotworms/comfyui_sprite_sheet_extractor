/**
 * sprite_sheet.js — ComfyUI web extensions for SpriteSheetExtractor and SpriteSheetPreview
 *
 * SpriteSheetExtractor: adds a "Tweak Tolerance" button that opens a floating
 *   panel for live background-removal adjustment + save without re-queuing.
 *
 * SpriteSheetPreview: adds a pixel-art animation canvas widget inside the node
 *   that loops through the sprite sheet frames at the configured FPS.
 */

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------

const PANEL_ID = "ss-tweak-panel";

function removePanel() {
  document.getElementById(PANEL_ID)?.remove();
}

function createTweakPanel({ nodeId, tolerance, removeBackground, filenamePrefix }) {
  removePanel();

  // ---- Outer shell -------------------------------------------------------
  const panel = document.createElement("div");
  panel.id = PANEL_ID;
  Object.assign(panel.style, {
    position:     "fixed",
    top:          "50%",
    left:         "50%",
    transform:    "translate(-50%, -50%)",
    width:        "640px",
    maxWidth:     "96vw",
    background:   "#1e1e1e",
    border:       "1px solid #3a3a3a",
    borderRadius: "12px",
    padding:      "20px",
    zIndex:       "99999",
    boxShadow:    "0 12px 48px rgba(0,0,0,0.85)",
    fontFamily:   '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    color:        "#ddd",
    userSelect:   "none",
  });

  // ---- Header ------------------------------------------------------------
  const header = el("div", { display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" });
  const title  = el("span", { fontSize: "15px", fontWeight: "600" });
  title.textContent = "Sprite Sheet — Tweak Tolerance";
  const closeBtn = el("button", { background: "none", border: "none", color: "#888", fontSize: "22px", cursor: "pointer", lineHeight: "1", padding: "0 2px" });
  closeBtn.textContent = "×";
  closeBtn.title = "Close";
  closeBtn.onclick = removePanel;
  header.append(title, closeBtn);

  // ---- Preview (checkerboard bg) -----------------------------------------
  const previewWrap = el("div", {
    background:   "repeating-conic-gradient(#2a2a2a 0% 25%, #1a1a1a 0% 50%) 0 0 / 16px 16px",
    borderRadius: "8px",
    display:      "flex",
    alignItems:   "center",
    justifyContent: "center",
    minHeight:    "100px",
    marginBottom: "12px",
    overflow:     "hidden",
  });
  const previewImg = document.createElement("img");
  Object.assign(previewImg.style, {
    imageRendering: "pixelated",
    maxWidth:       "100%",
    maxHeight:      "320px",
    display:        "block",
  });
  previewWrap.appendChild(previewImg);

  // ---- Status line -------------------------------------------------------
  const statusEl = el("p", { fontSize: "12px", color: "#777", margin: "0 0 12px", minHeight: "16px" });

  // ---- Tolerance row -----------------------------------------------------
  const tolRow   = el("div", { display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" });
  const tolLabel = el("label", { fontSize: "13px", color: "#aaa", width: "72px", flexShrink: "0" });
  tolLabel.textContent = "Tolerance";

  const tolSlider = document.createElement("input");
  tolSlider.type  = "range";
  tolSlider.min   = "0";
  tolSlider.max   = "100";
  tolSlider.step  = "0.5";
  tolSlider.value = String(tolerance);
  Object.assign(tolSlider.style, { flex: "1", accentColor: "#5a9de0" });

  const tolVal = el("span", { fontSize: "13px", fontWeight: "600", minWidth: "36px", textAlign: "right" });
  tolVal.textContent = parseFloat(tolerance).toFixed(1);

  tolRow.append(tolLabel, tolSlider, tolVal);

  // ---- Action buttons ----------------------------------------------------
  const btnRow  = el("div", { display: "flex", gap: "10px" });
  const applyBtn = makeBtn("Apply", "#2a2a2a", "#555");
  const saveBtn  = makeBtn("Save (overwrite)", "#152a44", "#5a9de0");
  btnRow.append(applyBtn, saveBtn);

  // ---- Assemble ----------------------------------------------------------
  panel.append(header, previewWrap, statusEl, tolRow, btnRow);
  document.body.appendChild(panel);

  // ---- Logic -------------------------------------------------------------
  let currentTol  = parseFloat(tolerance);
  let debouncer   = null;
  let busy        = false;

  async function fetchPreview(tol) {
    if (busy) return;
    busy = true;
    setStatus("Rendering…");
    setButtons(true);
    try {
      const res = await fetch("/sprite_sheet/preview", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ node_id: nodeId, tolerance: tol, remove_background: removeBackground }),
      });
      if (!res.ok) throw new Error(await res.text());
      const { image } = await res.json();
      previewImg.src = image;
      setStatus(`Tolerance: ${parseFloat(tol).toFixed(1)}`);
    } catch (e) {
      setStatus("Error: " + e.message);
    } finally {
      busy = false;
      setButtons(false);
    }
  }

  tolSlider.addEventListener("input", () => {
    currentTol = parseFloat(tolSlider.value);
    tolVal.textContent = currentTol.toFixed(1);
    clearTimeout(debouncer);
    debouncer = setTimeout(() => fetchPreview(currentTol), 280);
  });

  applyBtn.addEventListener("click", () => fetchPreview(currentTol));

  saveBtn.addEventListener("click", async () => {
    setStatus("Saving…");
    setButtons(true);
    try {
      const res = await fetch("/sprite_sheet/save", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({
          node_id:           nodeId,
          tolerance:         currentTol,
          remove_background: removeBackground,
          filename_prefix:   filenamePrefix,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      const { filename } = await res.json();
      setStatus(`Saved → ${filename}`);
    } catch (e) {
      setStatus("Error: " + e.message);
    } finally {
      setButtons(false);
    }
  });

  // Load initial preview
  fetchPreview(currentTol);

  // ---- Helpers -----------------------------------------------------------
  function setStatus(msg) { statusEl.textContent = msg; }

  function setButtons(disabled) {
    applyBtn.disabled = disabled;
    saveBtn.disabled  = disabled;
    applyBtn.style.opacity = disabled ? "0.5" : "1";
    saveBtn.style.opacity  = disabled ? "0.5" : "1";
  }
}

// ---------------------------------------------------------------------------
// ComfyUI extension registration
// ---------------------------------------------------------------------------

app.registerExtension({
  name: "SpriteSheet.Extractor",

  async nodeCreated(node) {
    if (node.comfyClass !== "SpriteSheetExtractor") return;

    // Add a button widget that opens the tweak panel
    node.addWidget("button", "✦ Tweak Tolerance", null, () => {
      createTweakPanel({
        nodeId:          String(node.id),
        tolerance:       widgetValue(node, "tolerance")      ?? 15,
        removeBackground: widgetValue(node, "remove_background") ?? true,
        filenamePrefix:  widgetValue(node, "filename_prefix") ?? "sprite_sheet",
      });
    }, { serialize: false });
  },
});

// ---------------------------------------------------------------------------
// Tiny helpers
// ---------------------------------------------------------------------------

/** Get the current value of a named widget on a node. */
function widgetValue(node, name) {
  return node.widgets?.find(w => w.name === name)?.value;
}

/** Create a styled div/span/etc element with inline style properties. */
function el(tag, styles = {}) {
  const e = document.createElement(tag);
  Object.assign(e.style, styles);
  return e;
}

/** Create a styled button. */
function makeBtn(label, bgColor, borderColor) {
  const btn = document.createElement("button");
  btn.textContent = label;
  Object.assign(btn.style, {
    flex:         "1",
    height:       "38px",
    borderRadius: "8px",
    border:       `1px solid ${borderColor}`,
    background:   bgColor,
    color:        "#ddd",
    fontSize:     "13px",
    fontWeight:   "500",
    cursor:       "pointer",
    transition:   "opacity 0.15s",
  });
  return btn;
}

// =============================================================================
// SpriteSheetPreview — animated canvas widget inside the node
// =============================================================================

app.registerExtension({
  name: "SpriteSheet.Preview",

  // ---- Per-node setup: add the animated canvas DOM widget ----------------
  async nodeCreated(node) {
    if (node.comfyClass !== "SpriteSheetPreview") return;

    // Initial display size from the widget's default value. Will be updated
    // once the first execution returns real frame data.
    const initialSize = widgetValue(node, "target_size") ?? 96;
    const [dispW, dispH] = sheetDisplaySize(initialSize, initialSize);

    // Checkerboard container so transparent frames look correct
    const container = document.createElement("div");
    Object.assign(container.style, {
      display:         "flex",
      alignItems:      "center",
      justifyContent:  "center",
      backgroundImage: "repeating-conic-gradient(#2c2c2c 0% 25%, #1a1a1a 0% 50%)",
      backgroundSize:  "12px 12px",
      borderRadius:    "6px",
      width:           dispW + "px",
      height:          dispH + "px",
      overflow:        "hidden",
    });

    const canvas = document.createElement("canvas");
    canvas.width  = initialSize;
    canvas.height = initialSize;
    Object.assign(canvas.style, {
      imageRendering: "pixelated",
      width:  dispW + "px",
      height: dispH + "px",
      display: "block",
    });

    container.appendChild(canvas);

    // addDOMWidget(name, type, element, options)
    const widget = node.addDOMWidget("animation_preview", "CANVAS", container, {
      serialize: false,
      getValue() { return null; },
      setValue() {},
    });

    // Give the widget an explicit height so LiteGraph sizes the node correctly
    widget.computeSize = () => [node.size?.[0] ?? dispW + 24, dispH + 12];

    node._ssCanvas    = canvas;
    node._ssContainer = container;
    node._ssTimer     = null;
    node._ssFrames    = [];
    node._ssIndex     = 0;

    // Clean up animation timer when node is deleted from the graph
    node.onRemoved = () => {
      if (node._ssTimer) {
        clearInterval(node._ssTimer);
        node._ssTimer = null;
      }
    };
  },

  // ---- Global setup: listen for execution events -------------------------
  async setup() {
    api.addEventListener("executed", (event) => {
      const { node: rawNodeId, output } = event.detail ?? {};
      if (!output?.sprite_meta?.length || !output?.images?.length) return;

      const node = app.graph?.getNodeById(
        typeof rawNodeId === "string" ? parseInt(rawNodeId) : rawNodeId
      );
      if (!node || node.comfyClass !== "SpriteSheetPreview") return;

      const { target_size, fps, frame_count } = output.sprite_meta[0];
      const { filename, subfolder, type }     = output.images[0];

      const imgUrl = api.apiURL(
        `/view?filename=${encodeURIComponent(filename)}`
        + `&type=${encodeURIComponent(type)}`
        + `&subfolder=${encodeURIComponent(subfolder ?? "")}`
        + `&t=${Date.now()}`
      );

      loadAndAnimate(node, imgUrl, target_size, fps, frame_count);
    });
  },
});

// ---------------------------------------------------------------------------
// Core animation logic
// ---------------------------------------------------------------------------

/**
 * Load the sprite sheet from imgUrl, chop it into frames, and start the
 * animation loop on the node's canvas widget.
 */
function loadAndAnimate(node, imgUrl, targetSize, fps, frameCount) {
  // Stop any previous animation
  if (node._ssTimer) {
    clearInterval(node._ssTimer);
    node._ssTimer = null;
  }

  const [dispW, dispH] = sheetDisplaySize(targetSize, targetSize);

  // Resize the canvas and its container to match new dimensions
  const canvas    = node._ssCanvas;
  const container = node._ssContainer;
  canvas.width  = targetSize;
  canvas.height = targetSize;
  Object.assign(canvas.style, { width: dispW + "px", height: dispH + "px" });
  Object.assign(container.style, { width: dispW + "px", height: dispH + "px" });

  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;

  const sheet = new Image();
  sheet.onload = () => {
    // Chop the horizontal sprite sheet into individual frames using an
    // offscreen canvas so we only pay the drawImage cost once per load.
    const off    = document.createElement("canvas");
    off.width    = sheet.naturalWidth;
    off.height   = sheet.naturalHeight;
    const offCtx = off.getContext("2d");
    offCtx.imageSmoothingEnabled = false;
    offCtx.drawImage(sheet, 0, 0);

    const frames = [];
    for (let i = 0; i < frameCount; i++) {
      const fc = document.createElement("canvas");
      fc.width  = targetSize;
      fc.height = targetSize;
      const fCtx = fc.getContext("2d");
      fCtx.imageSmoothingEnabled = false;
      fCtx.drawImage(off, i * targetSize, 0, targetSize, targetSize, 0, 0, targetSize, targetSize);
      frames.push(fc);
    }

    node._ssFrames = frames;
    node._ssIndex  = 0;

    // Draw immediately so the first frame appears before the first tick
    ctx.clearRect(0, 0, targetSize, targetSize);
    ctx.drawImage(frames[0], 0, 0);

    // Animation loop
    node._ssTimer = setInterval(() => {
      if (!node._ssFrames.length) return;
      ctx.clearRect(0, 0, targetSize, targetSize);
      ctx.drawImage(node._ssFrames[node._ssIndex], 0, 0);
      node._ssIndex = (node._ssIndex + 1) % node._ssFrames.length;
    }, 1000 / fps);

    // Tell LiteGraph the node size has changed
    node.setSize([
      Math.max(node.size?.[0] ?? 0, dispW + 24),
      node.computeSize()[1],
    ]);
    node.setDirtyCanvas(true, true);
  };

  sheet.onerror = () => {
    console.warn("[SpriteSheetPreview] Could not load sprite sheet:", imgUrl);
  };

  sheet.src = imgUrl;
}

/**
 * Compute a comfortable CSS display size for the animation canvas.
 * Scales the native pixel dimensions up so the art is clearly visible:
 *   64 px  → ×4 = 256 px
 *   96 px  → ×3 = 288 px
 *  128 px  → ×2 = 256 px
 * Capped at 4× so large target sizes don't overflow the node.
 */
function sheetDisplaySize(w, h) {
  const scale = Math.min(4, Math.max(2, Math.floor(288 / Math.max(w, h))));
  return [w * scale, h * scale];
}
