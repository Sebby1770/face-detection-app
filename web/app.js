(() => {
  "use strict";

  const fileInput = document.getElementById("file");
  const cameraBtn = document.getElementById("camera");
  const snapBtn = document.getElementById("snap");
  const stopBtn = document.getElementById("stop");
  const canvas = document.getElementById("canvas");
  const video = document.getElementById("video");
  const ctx = canvas.getContext("2d");
  const status = document.getElementById("status");
  const hud = document.getElementById("hud");
  const paddingInput = document.getElementById("padding");
  const paddingLabel = document.getElementById("padding-label");
  const downloadBtn = document.getElementById("download");
  const stage = document.getElementById("stage");
  const help = document.getElementById("help");
  const featherInput = document.getElementById("feather");

  const state = {
    source: null,
    faces: [],
    extras: [],
    drawing: null,
    stream: null,
    raf: 0,
    tick: 0,
  };

  function mode() {
    const picked = document.querySelector("input[name='mode']:checked");
    return picked ? picked.value : "solid";
  }

  function shape() {
    const picked = document.querySelector("input[name='shape']:checked");
    return picked ? picked.value : "box";
  }

  function padding() {
    return Number(paddingInput.value) / 100;
  }

  function featherOn() {
    return Boolean(featherInput && featherInput.checked);
  }

  function setStatus(message) {
    status.textContent = message;
  }

  function setMode(value) {
    const input = document.querySelector(`input[name='mode'][value='${value}']`);
    if (input) input.checked = true;
  }

  function setShape(value) {
    const input = document.querySelector(`input[name='shape'][value='${value}']`);
    if (input) input.checked = true;
  }

  function expand(box, pad, width, height) {
    const extraX = box.width * pad;
    const extraY = box.height * pad;
    const x = Math.max(0, box.x - extraX);
    const y = Math.max(0, box.y - extraY);
    const x1 = Math.min(width, box.x + box.width + extraX);
    const y1 = Math.min(height, box.y + box.height + extraY);
    return { x, y, w: Math.max(1, x1 - x), h: Math.max(1, y1 - y) };
  }

  function iou(a, b) {
    const ax2 = a.x + a.width;
    const ay2 = a.y + a.height;
    const bx2 = b.x + b.width;
    const by2 = b.y + b.height;
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(ax2, bx2);
    const y2 = Math.min(ay2, by2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const union = a.width * a.height + b.width * b.height - inter;
    return union ? inter / union : 0;
  }

  function redactBox(x, y, w, h) {
    const kind = mode();
    const elliptical = shape() === "ellipse";
    ctx.save();
    if (featherOn()) ctx.filter = "blur(6px)";
    if (elliptical) {
      ctx.beginPath();
      ctx.ellipse(x + w / 2, y + h / 2, (w / 2) * 1.08, (h / 2) * 1.2, 0, 0, Math.PI * 2);
      ctx.clip();
    }
    if (kind === "solid") {
      ctx.fillStyle = "#000";
      ctx.fillRect(x, y, w, h);
      ctx.restore();
      return;
    }
    if (kind === "pixelate") {
      const size = Math.max(8, Math.round(Math.max(w, h) / 8));
      const tmp = document.createElement("canvas");
      tmp.width = Math.max(1, Math.ceil(w / size));
      tmp.height = Math.max(1, Math.ceil(h / size));
      const tctx = tmp.getContext("2d");
      tctx.imageSmoothingEnabled = false;
      tctx.drawImage(canvas, x, y, w, h, 0, 0, tmp.width, tmp.height);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(tmp, 0, 0, tmp.width, tmp.height, x, y, w, h);
      ctx.imageSmoothingEnabled = true;
      ctx.restore();
      return;
    }
    const slice = ctx.getImageData(x, y, w, h);
    const data = slice.data;
    const radius = 4;
    const copy = new Uint8ClampedArray(data);
    const width = slice.width;
    const height = slice.height;
    for (let py = 0; py < height; py += 1) {
      for (let px = 0; px < width; px += 1) {
        let r = 0;
        let g = 0;
        let b = 0;
        let n = 0;
        for (let dy = -radius; dy <= radius; dy += 1) {
          const yy = py + dy;
          if (yy < 0 || yy >= height) continue;
          for (let dx = -radius; dx <= radius; dx += 1) {
            const xx = px + dx;
            if (xx < 0 || xx >= width) continue;
            const i = (yy * width + xx) * 4;
            r += copy[i];
            g += copy[i + 1];
            b += copy[i + 2];
            n += 1;
          }
        }
        const o = (py * width + px) * 4;
        data[o] = r / n;
        data[o + 1] = g / n;
        data[o + 2] = b / n;
      }
    }
    ctx.putImageData(slice, x, y);
    ctx.restore();
  }

  async function detect(bitmap) {
    if (typeof FaceDetector !== "function") return [];
    try {
      const detector = new FaceDetector({ fastMode: true, maxDetectedFaces: 32 });
      const hits = await detector.detect(bitmap);
      return hits.map((hit) => ({
        x: hit.boundingBox.x,
        y: hit.boundingBox.y,
        width: hit.boundingBox.width,
        height: hit.boundingBox.height,
        kept: false,
      }));
    } catch (err) {
      return [];
    }
  }

  function carryKeeps(next) {
    const kept = state.faces.filter((face) => face.kept);
    return next.map((face, index) => {
      const hit = kept.find((prior) => iou(prior, face) > 0.35);
      return Object.assign(face, { kept: Boolean(hit && hit.kept), id: index + 1 });
    });
  }

  function paintEmpty() {
    canvas.width = 960;
    canvas.height = 540;
    ctx.fillStyle = "#070a0d";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#93a0ae";
    ctx.font = "18px Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Drop a photo here, or use Open image / Use camera", canvas.width / 2, canvas.height / 2);
  }

  function updateHud() {
    const live = state.faces.filter((face) => !face.kept);
    const kept = state.faces.filter((face) => face.kept);
    const ids = state.faces.map((face) => face.id || "?").join(",");
    if (!state.source || !state.faces.length) {
      hud.textContent = state.source ? "NO FACE this frame" : "Waiting for an image";
      return;
    }
    hud.textContent = `IDs: ${ids || "—"} · redacting ${live.length} · kept ${kept.length} · extras ${state.extras.length} · ${mode()} ${shape()}`;
  }

  function paint() {
    const source = state.source;
    if (!source) {
      paintEmpty();
      updateHud();
      return;
    }
    const width = source.videoWidth || source.width;
    const height = source.videoHeight || source.height;
    if (!width || !height) return;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    ctx.filter = "none";
    ctx.drawImage(source, 0, 0, width, height);
    state.faces.forEach((box, index) => {
      box.id = box.id || index + 1;
      const grown = expand(box, padding(), width, height);
      if (box.kept) {
        ctx.save();
        ctx.strokeStyle = "rgba(61, 214, 198, 0.95)";
        ctx.setLineDash([6, 4]);
        ctx.lineWidth = 2;
        ctx.strokeRect(grown.x, grown.y, grown.w, grown.h);
        ctx.restore();
        return;
      }
      redactBox(grown.x, grown.y, grown.w, grown.h);
    });
    state.extras.forEach((box) => {
      const grown = expand(box, padding(), width, height);
      redactBox(grown.x, grown.y, grown.w, grown.h);
    });
    if (state.drawing) {
      ctx.strokeStyle = "rgba(61, 214, 198, 0.9)";
      ctx.lineWidth = 2;
      ctx.strokeRect(state.drawing.x, state.drawing.y, state.drawing.width, state.drawing.height);
    }
    downloadBtn.disabled = false;
    updateHud();
  }

  async function ingest(source, label) {
    state.source = source;
    const bitmap = source instanceof HTMLVideoElement
      ? source
      : await createImageBitmap(source);
    state.faces = carryKeeps(await detect(bitmap));
    state.extras = [];
    paint();
    const count = state.faces.length;
    const detector = typeof FaceDetector === "function"
      ? "This browser can detect faces."
      : "This browser has no built-in face detector — click and drag to cover faces yourself.";
    setStatus(`${label}: ${count} face${count === 1 ? "" : "s"} found. ${detector} Click a face to keep it visible.`);
  }

  function loop() {
    if (!state.stream) return;
    state.tick += 1;
    if (state.tick % 8 === 1) {
      detect(video).then((faces) => {
        state.faces = carryKeeps(faces);
        paint();
      });
    } else {
      paint();
    }
    state.raf = requestAnimationFrame(loop);
  }

  function loadFile(file) {
    if (!file || !file.type.startsWith("image/")) return;
    stopCamera();
    state.faces = [];
    state.extras = [];
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = async () => {
      await ingest(image, file.name);
      URL.revokeObjectURL(url);
    };
    image.src = url;
  }

  fileInput.addEventListener("change", () => {
    const file = fileInput.files && fileInput.files[0];
    if (file) loadFile(file);
  });

  ["dragenter", "dragover"].forEach((name) => {
    stage.addEventListener(name, (event) => {
      event.preventDefault();
      stage.classList.add("drop-active");
    });
  });
  stage.addEventListener("dragleave", () => stage.classList.remove("drop-active"));
  stage.addEventListener("drop", (event) => {
    event.preventDefault();
    stage.classList.remove("drop-active");
    const file = event.dataTransfer.files && event.dataTransfer.files[0];
    if (file) loadFile(file);
  });

  cameraBtn.addEventListener("click", async () => {
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 1280 } },
        audio: false,
      });
      video.srcObject = state.stream;
      video.hidden = true;
      await video.play();
      snapBtn.hidden = false;
      stopBtn.hidden = false;
      setStatus("Camera live. Faces are redacted as they are found. Click a face to keep it visible.");
      state.source = video;
      state.tick = 0;
      loop();
    } catch (err) {
      setStatus("Camera permission was denied or no camera is available.");
    }
  });

  function stopCamera() {
    if (state.raf) cancelAnimationFrame(state.raf);
    state.raf = 0;
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
      state.stream = null;
    }
    snapBtn.hidden = true;
    stopBtn.hidden = true;
  }

  stopBtn.addEventListener("click", () => {
    stopCamera();
    setStatus("Camera stopped.");
  });

  snapBtn.addEventListener("click", async () => {
    const snap = document.createElement("canvas");
    snap.width = video.videoWidth;
    snap.height = video.videoHeight;
    snap.getContext("2d").drawImage(video, 0, 0);
    const kept = state.faces.filter((face) => face.kept);
    stopCamera();
    const image = new Image();
    image.onload = async () => {
      state.faces = kept;
      await ingest(image, "Captured frame");
    };
    image.src = snap.toDataURL("image/png");
  });

  paddingInput.addEventListener("input", () => {
    paddingLabel.textContent = `${paddingInput.value}%`;
    paint();
  });
  document.querySelectorAll("input[name='mode'], input[name='shape']").forEach((input) => {
    input.addEventListener("change", paint);
  });
  featherInput.addEventListener("change", paint);

  function canvasPoint(event) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (event.clientX - rect.left) * scaleX,
      y: (event.clientY - rect.top) * scaleY,
    };
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (!state.source) return;
    const point = canvasPoint(event);
    state.drawing = { x: point.x, y: point.y, width: 1, height: 1, originX: point.x, originY: point.y };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!state.drawing) return;
    const point = canvasPoint(event);
    const ox = state.drawing.originX;
    const oy = state.drawing.originY;
    state.drawing.x = Math.min(ox, point.x);
    state.drawing.y = Math.min(oy, point.y);
    state.drawing.width = Math.abs(point.x - ox);
    state.drawing.height = Math.abs(point.y - oy);
    paint();
  });
  canvas.addEventListener("pointerup", () => {
    if (!state.drawing) return;
    if (state.drawing.width > 6 && state.drawing.height > 6) {
      state.extras.push({
        x: state.drawing.x,
        y: state.drawing.y,
        width: state.drawing.width,
        height: state.drawing.height,
      });
    } else {
      const cx = state.drawing.originX;
      const cy = state.drawing.originY;
      const hit = state.faces.find((box) => (
        cx >= box.x && cy >= box.y && cx <= box.x + box.width && cy <= box.y + box.height
      ));
      if (hit) {
        hit.kept = !hit.kept;
        setStatus(hit.kept
          ? `ID ${hit.id} stays visible.`
          : `ID ${hit.id} is redacted again.`);
      }
    }
    state.drawing = null;
    paint();
  });

  document.getElementById("clear").addEventListener("click", () => {
    state.extras = [];
    paint();
  });
  document.getElementById("undo").addEventListener("click", () => {
    state.extras.pop();
    paint();
  });
  downloadBtn.addEventListener("click", () => {
    const link = document.createElement("a");
    link.download = "redacted.png";
    link.href = canvas.toDataURL("image/png");
    link.click();
  });

  function toggleHelp() {
    help.hidden = !help.hidden;
  }
  document.getElementById("help-open").addEventListener("click", toggleHelp);
  document.getElementById("help-close").addEventListener("click", () => {
    help.hidden = true;
  });

  document.addEventListener("keydown", (event) => {
    const tag = (event.target && event.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if (event.key === "?" || (event.key === "/" && event.shiftKey)) {
      event.preventDefault();
      toggleHelp();
      return;
    }
    if (event.key === "Escape") {
      help.hidden = true;
      return;
    }
    if (event.key === "s" || event.key === "S") setMode("solid");
    if (event.key === "b" || event.key === "B") setMode("blur");
    if (event.key === "p" || event.key === "P") setMode("pixelate");
    if (event.key === "e" || event.key === "E") setShape(shape() === "box" ? "ellipse" : "box");
    if (event.key === "f" || event.key === "F") featherInput.checked = !featherInput.checked;
    if (event.key === "z" || event.key === "Z") state.extras.pop();
    paint();
  });

  paintEmpty();
  if (typeof FaceDetector !== "function") {
    setStatus("This browser has no built-in face detector. Open a photo and drag boxes over faces yourself.");
  }
})();
