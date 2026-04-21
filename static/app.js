(() => {
  const EMBED_URL = "/api/embed";
  const KNN_URL = "/api/knn";
  const USE_MOCK = false;

  const MOCK_MODELS = [
    { id: "clip_vitb32_laion2b", name: "CLIP ViT-B/32 LAION-2B", patch_size: 32, image_size: 224, text_enabled: true },
    { id: "dinov2_small", name: "DINOv2 Small", patch_size: 14, image_size: 224, text_enabled: false },
  ];

  const state = {
    queryImage: null,
    queryImageBase64: null,
    models: [],
    selectedModel: null,
    gridSize: 14,
    searchMode: "image",
    hoveredPatch: null,
    selectedPatch: null,
    isDragging: false,
    dragStart: null,
    results: [],
    loading: false,
    canvasDisplay: { width: 0, height: 0 },
    lightboxIndex: -1,
  };

  const $ = (id) => document.getElementById(id);

  const canvas = $("queryCanvas");
  const ctx = canvas.getContext("2d");
  const uploadZone = $("uploadZone");
  const uploadPrompt = $("uploadPrompt");
  const clearBtn = $("clearBtn");
  const fileInput = $("fileInput");
  const modelSelect = $("modelSelect");
  const modeToggle = $("modeToggle");
  const selectionInfo = $("selectionInfo");
  const searchBtn = $("searchBtn");
  const resultsGrid = $("resultsGrid");
  const resultsCount = $("resultsCount");
  const loadingIndicator = $("loadingIndicator");
  const emptyState = $("emptyState");
  const resultCountSelect = $("resultCount");
  const lightbox = $("lightbox");
  const lightboxImg = $("lightboxImg");
  const textInputZone = $("textInputZone");
  const textInput = $("textInput");
  const textModeBtn = document.querySelector('.text-mode-btn');

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  // ── Models ──

  async function loadModels() {
    try {
      if (USE_MOCK) throw new Error("mock");
      const res = await fetch(`${EMBED_URL}/models`);
      if (!res.ok) throw new Error(res.statusText);
      const data = await res.json();
      state.models = data.models;
    } catch {
      state.models = MOCK_MODELS;
    }
    modelSelect.innerHTML = "";
    state.models.forEach((m, i) => {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.name;
      modelSelect.appendChild(opt);
    });
    if (state.models.length) {
      state.selectedModel = state.models[0];
      state.gridSize = Math.round(state.selectedModel.image_size / state.selectedModel.patch_size);
    }
    updateTextModeAvailability();
  }

  function updateTextModeAvailability() {
    const model = getSelectedModel();
    const supported = model && model.text_enabled;
    textModeBtn.disabled = !supported;
    if (!supported && state.searchMode === "text") {
      state.searchMode = "image";
      modeToggle.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
      modeToggle.querySelector('[data-mode="image"]').classList.add("active");
      updateModeVisibility();
    }
  }

  function updateModeVisibility() {
    const isText = state.searchMode === "text";
    textInputZone.classList.toggle("hidden", !isText);
    uploadZone.classList.toggle("hidden", isText);
    if (isText) {
      uploadZone.classList.remove("patch-mode");
      searchBtn.disabled = !textInput.value.trim();
    } else {
      searchBtn.disabled = !state.queryImage;
    }
  }

  function getSelectedModel() {
    const id = modelSelect.value;
    return state.models.find((m) => m.id === id) || state.models[0];
  }

  // ── Image upload ──

  function loadImage(file) {
    if (!file || !file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        state.queryImage = img;
        state.queryImageBase64 = e.target.result.split(",")[1];
        state.selectedPatch = null;
        state.hoveredPatch = null;
        selectionInfo.innerHTML = "";
        setupCanvas();
        drawCanvas();
        uploadZone.classList.add("has-image");
        searchBtn.disabled = false;
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }

  function clearImage() {
    state.queryImage = null;
    state.queryImageBase64 = null;
    state.selectedPatch = null;
    state.hoveredPatch = null;
    state.isDragging = false;
    state.dragStart = null;
    selectionInfo.innerHTML = "";
    uploadZone.classList.remove("has-image");
    searchBtn.disabled = true;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    fileInput.value = "";
  }

  // ── Canvas ──
function setupCanvas() {
    if (!state.queryImage) return;
    
    // Force a square canvas bounded by the container width or max height
    const containerW = uploadZone.clientWidth;
    const maxH = 460;
    const size = Math.min(containerW, maxH); 
    
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    canvas.style.width = size + "px";
    canvas.style.height = size + "px";
    
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    state.canvasDisplay = { width: size, height: size }; // Now strictly 1:1
  }

function drawCanvas() {
    if (!state.queryImage) return;
    const { width: size, height: _size } = state.canvasDisplay;
    ctx.clearRect(0, 0, size, size);

    // 1. Calculate center-crop coordinates for the source image
    const img = state.queryImage;
    const imgAspect = img.width / img.height;
    let sx = 0, sy = 0, sw = img.width, sh = img.height;

    if (imgAspect > 1) {
      // Landscape: crop the sides
      sw = img.height;
      sx = (img.width - sw) / 2;
    } else if (imgAspect < 1) {
      // Portrait: crop the top and bottom
      sh = img.width;
      sy = (img.height - sh) / 2;
    }

    // 2. Draw the cropped square onto the canvas
    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, size, size);

    // 3. Grid drawing logic (remains exactly as you wrote it)
    if (state.searchMode !== "patch") return;

    const gs = state.gridSize;
    const cw = size / gs;
    const ch = size / gs;

    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 0.5;
    for (let i = 1; i < gs; i++) {
      ctx.beginPath();
      ctx.moveTo(i * cw, 0);
      ctx.lineTo(i * cw, size);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i * ch);
      ctx.lineTo(size, i * ch);
      ctx.stroke();
    }

    if (state.hoveredPatch && !state.isDragging) {
      const { row, col } = state.hoveredPatch;
      ctx.fillStyle = "rgba(88,166,255,0.2)";
      ctx.fillRect(col * cw, row * ch, cw, ch);
      ctx.strokeStyle = "rgba(88,166,255,0.5)";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(col * cw + 0.5, row * ch + 0.5, cw - 1, ch - 1);
    }

    if (state.selectedPatch) {
      drawSelection();
    }
  }
  function drawSelection() {
    const { width: dw, height: dh } = state.canvasDisplay;
    const gs = state.gridSize;
    const cw = dw / gs;
    const ch = dh / gs;
    const sel = state.selectedPatch;

    let x, y, w, h;
    if (sel.type === "single") {
      x = sel.col * cw;
      y = sel.row * ch;
      w = cw;
      h = ch;
    } else {
      const r1 = Math.min(sel.startRow, sel.endRow);
      const r2 = Math.max(sel.startRow, sel.endRow);
      const c1 = Math.min(sel.startCol, sel.endCol);
      const c2 = Math.max(sel.startCol, sel.endCol);
      x = c1 * cw;
      y = r1 * ch;
      w = (c2 - c1 + 1) * cw;
      h = (r2 - r1 + 1) * ch;
    }

    ctx.fillStyle = "rgba(0,212,170,0.25)";
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = "rgba(0,212,170,0.8)";
    ctx.lineWidth = 2;
    ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
  }

  // ── Patch grid interaction ──

  function getGridCell(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const sx = state.canvasDisplay.width / rect.width;
    const sy = state.canvasDisplay.height / rect.height;
    const cx = mx * sx;
    const cy = my * sy;
    const cw = state.canvasDisplay.width / state.gridSize;
    const ch = state.canvasDisplay.height / state.gridSize;
    return {
      row: clamp(Math.floor(cy / ch), 0, state.gridSize - 1),
      col: clamp(Math.floor(cx / cw), 0, state.gridSize - 1),
    };
  }

  function onMouseMove(e) {
    if (!state.queryImage || state.searchMode !== "patch") return;
    const cell = getGridCell(e);
    state.hoveredPatch = cell;

    if (state.isDragging && state.dragStart) {
      state.selectedPatch = {
        type: "region",
        startRow: state.dragStart.row,
        startCol: state.dragStart.col,
        endRow: cell.row,
        endCol: cell.col,
      };
    }
    drawCanvas();
  }

  function onMouseDown(e) {
    if (!state.queryImage || state.searchMode !== "patch") return;
    e.preventDefault();
    const cell = getGridCell(e);
    state.isDragging = true;
    state.dragStart = cell;
    state.selectedPatch = { type: "single", row: cell.row, col: cell.col };
    drawCanvas();
  }

  function onMouseUp(e) {
    if (!state.isDragging) return;
    state.isDragging = false;
    const sel = state.selectedPatch;
    if (
      sel &&
      sel.type === "region" &&
      sel.startRow === sel.endRow &&
      sel.startCol === sel.endCol
    ) {
      state.selectedPatch = { type: "single", row: sel.startRow, col: sel.startCol };
    }
    updateSelectionInfo();
    drawCanvas();
  }

  function onMouseLeave() {
    state.hoveredPatch = null;
    if (state.isDragging) {
      state.isDragging = false;
      updateSelectionInfo();
    }
    drawCanvas();
  }

  function updateSelectionInfo() {
    const sel = state.selectedPatch;
    if (!sel || state.searchMode !== "patch") {
      selectionInfo.innerHTML = "";
      return;
    }
    if (sel.type === "single") {
      selectionInfo.innerHTML = `Patch (${sel.row}, ${sel.col}) <span class="clear-sel" id="clearSel">clear</span>`;
    } else {
      const r1 = Math.min(sel.startRow, sel.endRow);
      const r2 = Math.max(sel.startRow, sel.endRow);
      const c1 = Math.min(sel.startCol, sel.endCol);
      const c2 = Math.max(sel.startCol, sel.endCol);
      const count = (r2 - r1 + 1) * (c2 - c1 + 1);
      selectionInfo.innerHTML = `Region (${r1},${c1})&ndash;(${r2},${c2}) &middot; ${count} patches <span class="clear-sel" id="clearSel">clear</span>`;
    }
    const clearSel = $("clearSel");
    if (clearSel) clearSel.addEventListener("click", clearSelection);
  }

  function clearSelection() {
    state.selectedPatch = null;
    state.isDragging = false;
    state.dragStart = null;
    selectionInfo.innerHTML = "";
    drawCanvas();
  }

  function getSelectionBbox() {
    const sel = state.selectedPatch;
    if (!sel) return null;
    const gs = state.gridSize;
    if (sel.type === "single") {
      return {
        x: sel.col / gs,
        y: sel.row / gs,
        w: 1 / gs,
        h: 1 / gs,
      };
    }
    const r1 = Math.min(sel.startRow, sel.endRow);
    const r2 = Math.max(sel.startRow, sel.endRow);
    const c1 = Math.min(sel.startCol, sel.endCol);
    const c2 = Math.max(sel.startCol, sel.endCol);
    return {
      x: c1 / gs,
      y: r1 / gs,
      w: (c2 - c1 + 1) / gs,
      h: (r2 - r1 + 1) / gs,
    };
  }

  // ── Search ──

  async function search() {
    if (!state.selectedModel) return;

    const mode = state.searchMode;
    if (mode === "text") {
      if (!textInput.value.trim()) return;
    } else {
      if (!state.queryImage) return;
    }
    if (mode === "patch" && !state.selectedPatch) return;

    state.loading = true;
    state.results = [];
    resultsGrid.innerHTML = "";
    resultsCount.textContent = "";
    emptyState.classList.add("hidden");
    loadingIndicator.classList.add("visible");

    const resultCount = parseInt(resultCountSelect.value, 10) || 18;

    try {
      if (USE_MOCK) {
        state.results = mockSearch(mode, resultCount).results;
      } else {
        let embedData;

        if (mode === "text") {
          const embedRes = await fetch(`${EMBED_URL}/embed_text`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              text: textInput.value.trim(),
              model: state.selectedModel.id,
            }),
          });
          if (!embedRes.ok) throw new Error(`Embedder error: ${embedRes.status}`);
          embedData = await embedRes.json();
        } else {
          const embedBody = {
            image: state.queryImageBase64,
            model: state.selectedModel.id,
            mode: mode,
          };

          if (mode === "patch") {
            embedBody.bbox = getSelectionBbox();
            const sel = state.selectedPatch;
            embedBody.patch_grid = { grid_size: state.gridSize };
            if (sel.type === "single") {
              embedBody.patch_grid.row = sel.row;
              embedBody.patch_grid.col = sel.col;
            } else {
              embedBody.patch_grid.start_row = Math.min(sel.startRow, sel.endRow);
              embedBody.patch_grid.start_col = Math.min(sel.startCol, sel.endCol);
              embedBody.patch_grid.end_row = Math.max(sel.startRow, sel.endRow);
              embedBody.patch_grid.end_col = Math.max(sel.startCol, sel.endCol);
            }
          }

          const embedRes = await fetch(`${EMBED_URL}/embed`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(embedBody),
          });
          if (!embedRes.ok) throw new Error(`Embedder error: ${embedRes.status}`);
          embedData = await embedRes.json();
        }

        const knnRes = await fetch(`${KNN_URL}/${state.selectedModel.id}/neighbors?k=${resultCount}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ vector: embedData.vector }),
        });
        if (!knnRes.ok) throw new Error(`KNN error: ${knnRes.status}`);
        const knnData = await knnRes.json();

        state.results = knnData.neighbors.map((n) => {
          const id = Object.entries(n).find(([k]) => k !== "score")?.[1] ?? n.id;
          return {
            id: id,
            score: n.score,
            url: `/images/${id}`,
            thumbnail: null,
            patch_bbox: null,
          };
        });
      }
    } catch (err) {
      console.error("Search failed:", err);
        if (!USE_MOCK) state.results = mockSearch(mode, resultCount).results;
    }

    state.loading = false;
    loadingIndicator.classList.remove("visible");
    renderResults();
  }

  async function searchByImageSrc(src) {
    if (!state.selectedModel) return;

    state.loading = true;
    state.results = [];
    resultsGrid.innerHTML = "";
    resultsCount.textContent = "";
    emptyState.classList.add("hidden");
    loadingIndicator.classList.add("visible");

    const resultCount = parseInt(resultCountSelect.value, 10) || 18;

    try {
      const imgRes = await fetch(src);
      const blob = await imgRes.blob();
      const base64 = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result.split(",")[1]);
        reader.readAsDataURL(blob);
      });

      const queryImg = new Image();
      queryImg.crossOrigin = "anonymous";
      queryImg.onload = () => {
        state.queryImage = queryImg;
        state.queryImageBase64 = base64;
        state.selectedPatch = null;
        state.hoveredPatch = null;
        selectionInfo.innerHTML = "";
        uploadZone.classList.add("has-image");
        searchBtn.disabled = false;
        if (state.searchMode === "text") {
          state.searchMode = "image";
          modeToggle.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
          modeToggle.querySelector('[data-mode="image"]').classList.add("active");
          updateModeVisibility();
        }
        setupCanvas();
        drawCanvas();
      };
      queryImg.src = src;

      const embedRes = await fetch(`${EMBED_URL}/embed`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: base64,
          model: state.selectedModel.id,
          mode: "image",
        }),
      });
      if (!embedRes.ok) throw new Error(`Embedder error: ${embedRes.status}`);
      const embedData = await embedRes.json();

      const knnRes = await fetch(`${KNN_URL}/${state.selectedModel.id}/neighbors?k=${resultCount}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ vector: embedData.vector }),
      });
      if (!knnRes.ok) throw new Error(`KNN error: ${knnRes.status}`);
      const knnData = await knnRes.json();

      state.results = knnData.neighbors.map((n) => {
        const id = Object.entries(n).find(([k]) => k !== "score")?.[1] ?? n.id;
        return { id, score: n.score, url: `/images/${id}`, thumbnail: null, patch_bbox: null };
      });
    } catch (err) {
      console.error("More-like-this failed:", err);
    }

    state.loading = false;
    loadingIndicator.classList.remove("visible");
    renderResults();
  }

  function mockSearch(mode, count = 18) {
    const results = Array.from({ length: count }, (_, i) => {
      const score = +(Math.max(0.01, 1 - i * 0.045 + Math.random() * 0.02)).toFixed(4);
      const result = {
        id: `result_${i}`,
        score,
        url: null,
        thumbnail: null,
        patch_bbox: null,
      };
      if (mode === "patch") {
        result.patch_bbox = {
          x: +(0.05 + Math.random() * 0.55).toFixed(3),
          y: +(0.05 + Math.random() * 0.55).toFixed(3),
          w: +(0.1 + Math.random() * 0.25).toFixed(3),
          h: +(0.1 + Math.random() * 0.25).toFixed(3),
        };
      }
      return result;
    });
    return { results };
  }

  // ── Render results ──

  function renderResults() {
    resultsGrid.innerHTML = "";

    if (state.results.length === 0) {
      emptyState.classList.remove("hidden");
      resultsCount.textContent = "";
      return;
    }

    emptyState.classList.add("hidden");
    resultsCount.textContent = `${state.results.length} results`;

    state.results.forEach((result, i) => {
      const card = document.createElement("div");
      card.className = "result-card";

      const imgContainer = document.createElement("div");
      imgContainer.className = "result-image";

      if (result.thumbnail || result.url) {
        const img = document.createElement("img");
        img.src = result.thumbnail || result.url;
        img.loading = "lazy";
        img.alt = result.id;
        imgContainer.appendChild(img);
      } else {
        const hue = (i * 137 + 180) % 360;
        imgContainer.style.background = `linear-gradient(135deg, hsl(${hue},40%,20%), hsl(${(hue + 60) % 360},50%,30%))`;
        const label = document.createElement("span");
        label.className = "placeholder-label";
        label.textContent = result.id;
        imgContainer.appendChild(label);
      }

      if (result.patch_bbox) {
        const overlay = document.createElement("div");
        overlay.className = "patch-overlay";
        const bb = result.patch_bbox;
        overlay.style.left = `${bb.x * 100}%`;
        overlay.style.top = `${bb.y * 100}%`;
        overlay.style.width = `${bb.w * 100}%`;
        overlay.style.height = `${bb.h * 100}%`;
        imgContainer.appendChild(overlay);
      }

      const scoreEl = document.createElement("div");
      scoreEl.className = "result-score";
      scoreEl.textContent = result.score.toFixed(4);

      card.appendChild(imgContainer);
      card.appendChild(scoreEl);

      const src = result.thumbnail || result.url;
      if (src) {
        const mltBtn = document.createElement("button");
        mltBtn.className = "mlt-btn";
        mltBtn.title = "More like this";
        mltBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>`;
        mltBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          searchByImageSrc(src);
        });
        imgContainer.appendChild(mltBtn);

        const idx = i;
        card.addEventListener("click", () => {
          state.lightboxIndex = idx;
          lightboxImg.src = src;
          lightbox.classList.add("visible");
        });
      }

      resultsGrid.appendChild(card);
    });
  }

  // ── Event setup ──

  function init() {
    loadModels();

    uploadZone.addEventListener("click", (e) => {
      if (e.target === clearBtn || clearBtn.contains(e.target)) return;
      if (!state.queryImage) fileInput.click();
    });

    fileInput.addEventListener("change", (e) => {
      if (e.target.files[0]) loadImage(e.target.files[0]);
    });

    uploadZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      uploadZone.classList.add("drag-over");
    });

    uploadZone.addEventListener("dragleave", () => {
      uploadZone.classList.remove("drag-over");
    });

    uploadZone.addEventListener("drop", (e) => {
      e.preventDefault();
      uploadZone.classList.remove("drag-over");
      const file = e.dataTransfer.files[0];
      if (file) loadImage(file);
    });

    document.addEventListener("paste", (e) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (const item of items) {
        if (item.type.startsWith("image/")) {
          loadImage(item.getAsFile());
          break;
        }
      }
    });

    clearBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      clearImage();
    });

    modelSelect.addEventListener("change", () => {
      state.selectedModel = getSelectedModel();
      state.gridSize = Math.round(
        state.selectedModel.image_size / state.selectedModel.patch_size
      );
      state.selectedPatch = null;
      selectionInfo.innerHTML = "";
      updateTextModeAvailability();
      updateModeVisibility();
      drawCanvas();
    });

    modeToggle.addEventListener("click", (e) => {
      const btn = e.target.closest(".mode-btn");
      if (!btn || btn.disabled) return;
      modeToggle.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.searchMode = btn.dataset.mode;
      state.selectedPatch = null;
      state.hoveredPatch = null;
      selectionInfo.innerHTML = "";

      if (state.searchMode === "patch") {
        uploadZone.classList.add("patch-mode");
      } else {
        uploadZone.classList.remove("patch-mode");
      }
      updateModeVisibility();
      drawCanvas();
    });

    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("mouseleave", onMouseLeave);

    searchBtn.addEventListener("click", search);

    lightbox.addEventListener("click", () => {
      lightbox.classList.remove("visible");
      state.lightboxIndex = -1;
    });

    document.addEventListener("keydown", (e) => {
      if (state.lightboxIndex < 0 || !lightbox.classList.contains("visible")) return;
      if (e.key === "ArrowRight") {
        e.preventDefault();
        const next = state.lightboxIndex + 1;
        if (next < state.results.length) {
          const src = state.results[next].thumbnail || state.results[next].url;
          if (src) {
            state.lightboxIndex = next;
            lightboxImg.src = src;
          }
        }
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        const prev = state.lightboxIndex - 1;
        if (prev >= 0) {
          const src = state.results[prev].thumbnail || state.results[prev].url;
          if (src) {
            state.lightboxIndex = prev;
            lightboxImg.src = src;
          }
        }
      } else if (e.key === "Escape") {
        lightbox.classList.remove("visible");
        state.lightboxIndex = -1;
      }
    });

    textInput.addEventListener("input", () => {
      if (state.searchMode === "text") {
        searchBtn.disabled = !textInput.value.trim();
      }
    });

    textInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && state.searchMode === "text") {
        e.preventDefault();
        if (textInput.value.trim() && !searchBtn.disabled) search();
      }
    });

    let resizeTimer;
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (state.queryImage) {
          setupCanvas();
          drawCanvas();
        }
      }, 150);
    });
  }

  init();
})();
