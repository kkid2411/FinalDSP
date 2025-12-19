let currentAudioData = null;
let currentFilename = null;
let lastProcessedUrl = null;
let eqGains = [0, 0, 0, 0, 0, 0, 0, 0, 0];
const EQ_FREQS = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];
const PLAYLIST_STORAGE_KEY = "ncn_playlist";
let playlist = [];
let currentPlaylistId = null;
let currentPosition = 0;
let isDragging = false;
let isDraggingEQ = false;
let draggingEQIndex = -1;
let isPlaying = false;
let autoEQEnabled = false;
let audioElement = null;
let seekbarUpdateInterval = null;
let fftUpdateInterval = null;
let loadingCount = 0;
let modeLabel = "None";
let eqResponseTimeout = null;
let lastEqResponse = null;
let eqUpdateTimer = null;
let eqRunId = 0;
let eqIsRunning = false;
let showOriginalOverlay = true; // Toggle hiển thị original overlay

async function refreshEQResponse() {
  if (!eqCurveCtx) return;

  try {
    const res = await fetch("/api/audio/eq-response", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        eq_gains: eqGains,
        sr: currentAudioData?.sample_rate || 44100,
        q: 1.0,
      }),
    });
    const data = await res.json();
    if (data.success && data.freqs_hz && data.mag_db) {
      lastEqResponse = data;
      drawEQResponse(eqCurveCtx, data.freqs_hz, data.mag_db);
    }
  } catch (error) {
    console.error("EQ response error:", error);
  }
}

function scheduleEQResponse() {
  clearTimeout(eqResponseTimeout);
  eqResponseTimeout = setTimeout(refreshEQResponse, 200);
}

function setLoading(isLoading, message = "Processing...") {
  const overlay = document.getElementById("loadingOverlay");
  const textEl = overlay ? overlay.querySelector(".loading-text") : null;
  if (!overlay) return;

  if (isLoading) {
    loadingCount += 1;
    if (textEl) textEl.textContent = message;
    overlay.classList.remove("hidden");
  } else {
    loadingCount = Math.max(0, loadingCount - 1);
    if (loadingCount === 0) {
      overlay.classList.add("hidden");
    }
  }
}

let audioContext = null;
let analyserNode = null;
let audioSource = null;
let connectedAudioElement = null; // Track audioElement đã kết nối
let spectrogramData = null;
let originalAudioBuffer = null;
let originalAudioSource = null;
let originalAnalyserNode = null;

let waveformCtx = null;
let fftCtx = null;
let spectrogramCtx = null;
let eqCurveCtx = null;
function initCanvas(id) {
  const canvas = document.getElementById(id);
  if (!canvas) return null;
  const parent = canvas.parentElement;

  function resize() {
    const { width, height } = parent.getBoundingClientRect();
    canvas.width = width * window.devicePixelRatio;
    canvas.height = height * window.devicePixelRatio;
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(
      window.devicePixelRatio,
      0,
      0,
      window.devicePixelRatio,
      0,
      0
    );
    return ctx;
  }

  window.addEventListener("resize", resize);
  return resize();
}

function setModeLabel(modeText) {
  modeLabel = modeText;
  const modeEl = document.querySelector(
    ".autoeq-panel .panel-row:nth-child(2) .value"
  );
  if (modeEl) modeEl.textContent = modeText;
}

function pickRandomMode() {
  const candidates = ["Music", "Vocal", "Podcast", "EDM", "Rock", "Classical"];
  return candidates[Math.floor(Math.random() * candidates.length)];
}

function cloneData(obj) {
  if (!obj) return null;
  try {
    return JSON.parse(JSON.stringify(obj));
  } catch (e) {
    return null;
  }
}

function randomEqGains() {
  const min = -8;
  const max = 8;
  return EQ_FREQS.map(() => {
    const r = Math.random() * (max - min) + min;
    return Math.round(r * 10) / 10;
  });
}

async function suggestEQFromModel() {
  if (!currentFilename) {
    console.warn("No audio file loaded, using random EQ");
    return randomEqGains();
  }

  try {
    const res = await fetch("/api/audio/suggest-eq", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: currentFilename,
      }),
    });

    const data = await res.json();
    if (data.success && data.eq_gains && data.eq_gains.length === 9) {
      console.log("EQ suggested by ML model:", data.eq_gains);
      return data.eq_gains;
    } else {
      console.warn("EQ suggestion failed, using random EQ:", data.error);
      return randomEqGains();
    }
  } catch (error) {
    console.error("Error suggesting EQ from model:", error);
    return randomEqGains();
  }
}

function applyProcessedCharts(source = null) {
  const data = source || currentAudioData;
  if (!data) return;
  const duration = data.duration || 1;
  if (waveformCtx && data.waveform) {
    drawWaveform(waveformCtx, data.waveform.data, data.waveform.time, duration);
  }
  if (fftCtx && data.fft) {
    drawFFT(fftCtx, data.fft.frequencies, data.fft.magnitude_db);
  }
  if (duration) {
    updateTimelineLabels(duration);
  }
}

function drawWaveform(
  ctx,
  data,
  time,
  duration,
  color = "#4fb4ff",
  isOverlay = false
) {
  if (!ctx || !data || data.length === 0) return;

  const canvas = ctx.canvas;
  const logicalWidth = canvas.width / window.devicePixelRatio;
  const logicalHeight = canvas.height / window.devicePixelRatio;

  if (!isOverlay) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  const padding = 12;
  const plotWidth = logicalWidth - padding * 2;
  const plotHeight = logicalHeight - padding * 2 - 12;

  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.4;

  const centerY = padding + plotHeight * 0.5;
  const maxAmp = Math.max(...data.map(Math.abs));

  const scale = (plotHeight * 0.45) / (maxAmp || 1);

  const firstX = Math.max(padding, Math.min(logicalWidth - padding, padding));
  ctx.moveTo(firstX, centerY);

  for (let i = 0; i < data.length; i++) {
    const x = padding + (time[i] / duration) * plotWidth;
    const y = centerY - data[i] * scale;

    const clampedX = Math.max(padding, Math.min(logicalWidth - padding, x));
    const clampedY = Math.max(padding, Math.min(padding + plotHeight, y));
    ctx.lineTo(clampedX, clampedY);
  }
  ctx.stroke();

  if (!isOverlay) {
    ctx.strokeStyle = "rgba(255,255,255,0.16)";
    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(padding, centerY);
    ctx.lineTo(padding + plotWidth, centerY);
    ctx.stroke();

    ctx.fillStyle = "rgba(255,255,255,0.6)";
    ctx.font = "10px system-ui";
    ctx.textAlign = "left";
    ctx.fillText("+1", padding, padding + 10);
    ctx.fillText("0", padding, centerY - 2);
    ctx.fillText("-1", padding, padding + plotHeight - 2);

    const ticks = 6;
    ctx.textAlign = "center";
    for (let i = 0; i < ticks; i++) {
      const ratio = i / (ticks - 1);
      const x = padding + ratio * plotWidth;
      ctx.strokeStyle = "rgba(255,255,255,0.12)";
      ctx.beginPath();
      ctx.moveTo(x, padding + plotHeight);
      ctx.lineTo(x, padding + plotHeight + 4);
      ctx.stroke();
      const t = ratio * duration;
      ctx.fillText(formatTime(t), x, padding + plotHeight + 12);
    }
  }
}

function drawFFT(
  ctx,
  frequencies,
  magnitude_db,
  isOverlay = false,
  overlayColor = null
) {
  if (!ctx || !frequencies || frequencies.length === 0) return;

  const canvas = ctx.canvas;
  const logicalWidth = canvas.width / window.devicePixelRatio;
  const logicalHeight = canvas.height / window.devicePixelRatio;

  if (!isOverlay) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  const padding = 12;
  const plotWidth = logicalWidth - padding * 2;
  const plotHeight = logicalHeight - padding * 2 - 12;

  // Adaptive dB range từ data thực tế
  const actualMinDb = Math.min(
    ...magnitude_db.filter((db) => isFinite(db) && db > -Infinity)
  );
  const actualMaxDb = Math.max(
    ...magnitude_db.filter((db) => isFinite(db) && db < Infinity)
  );
  const dbPadding = Math.max(5, (actualMaxDb - actualMinDb) * 0.1); // 10% padding
  const minDb = Math.max(-120, Math.floor(actualMinDb - dbPadding));
  const maxDb = Math.min(20, Math.ceil(actualMaxDb + dbPadding));
  const dbRange = maxDb - minDb;

  // Linear scale cho frequencies (từ 0 đến max)
  const minFreq = 0;
  const maxFreq = frequencies[frequencies.length - 1] || 22050;
  const freqRange = maxFreq - minFreq;

  const bars = frequencies.length;

  for (let i = 0; i < bars; i++) {
    const db = magnitude_db[i];
    const normalized = (db - minDb) / dbRange;
    const amp = Math.max(0, Math.min(1, normalized)) * plotHeight;

    // Linear scale cho trục X
    const freq = frequencies[i];
    const xNorm = (freq - minFreq) / freqRange;
    const x = padding + xNorm * plotWidth;
    const y = padding + plotHeight - amp;

    if (isOverlay && overlayColor) {
      ctx.fillStyle = overlayColor;
    } else {
      const grad = ctx.createLinearGradient(0, y, 0, padding + plotHeight);
      grad.addColorStop(0, "#4a90ff");
      grad.addColorStop(1, "#7f5cff");
      ctx.fillStyle = grad;
    }

    // Tính barWidth dựa trên khoảng cách linear scale
    const nextFreq = i < bars - 1 ? frequencies[i + 1] : maxFreq;
    const nextXNorm = (nextFreq - minFreq) / freqRange;
    const nextX = padding + nextXNorm * plotWidth;
    const barW = Math.max(
      1,
      Math.min((nextX - x) * 0.9, plotWidth - (x - padding))
    );
    const barH = Math.min(amp, padding + plotHeight - y);

    if (barW > 0 && barH > 0) {
      ctx.fillRect(x, y, barW, barH);
    }
  }

  if (!isOverlay) {
    // Vẽ axes chính (border)
    ctx.strokeStyle = "rgba(255,255,255,0.3)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    // Bottom axis
    ctx.moveTo(padding, padding + plotHeight);
    ctx.lineTo(padding + plotWidth, padding + plotHeight);
    // Left axis
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, padding + plotHeight);
    // Top axis
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding + plotWidth, padding);
    // Right axis
    ctx.moveTo(padding + plotWidth, padding);
    ctx.lineTo(padding + plotWidth, padding + plotHeight);
    ctx.stroke();

    // Vẽ grid lines và labels cho trục Y (adaptive)
    ctx.strokeStyle = "rgba(255,255,255,0.25)"; // Tăng độ đậm từ 0.16
    ctx.lineWidth = 1;
    ctx.fillStyle = "rgba(255,255,255,0.8)"; // Tăng độ đậm từ 0.6
    ctx.font = "10px system-ui";
    ctx.textAlign = "left";

    // Tính số grid lines phù hợp
    const dbStep = Math.pow(10, Math.floor(Math.log10(dbRange)) - 1) * 5; // Bước 5, 10, 20, 50...
    const gridDbValues = [];
    for (
      let db = Math.ceil(minDb / dbStep) * dbStep;
      db <= maxDb;
      db += dbStep
    ) {
      gridDbValues.push(db);
    }

    gridDbValues.forEach((db) => {
      const yNorm = (db - minDb) / dbRange;
      const y = padding + (1 - yNorm) * plotHeight;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(padding + plotWidth, y);
      ctx.stroke();
      // Điều chỉnh vị trí label để không bị che
      ctx.fillText(
        `${db >= 0 ? "+" : ""}${Math.round(db)} dB`,
        padding - 2,
        y + 4
      );
    });

    // Vẽ grid lines và labels cho trục X (linear scale)
    ctx.textAlign = "center";
    const tickFreqs = [0, 1000, 2000, 5000, 10000, 15000, 20000];
    tickFreqs.forEach((f) => {
      if (f < minFreq || f > maxFreq) return;
      const xNorm = (f - minFreq) / freqRange;
      const x = padding + xNorm * plotWidth;
      ctx.strokeStyle = "rgba(255,255,255,0.2)"; // Tăng độ đậm từ 0.12
      ctx.beginPath();
      ctx.moveTo(x, padding + plotHeight);
      ctx.lineTo(x, padding + plotHeight + 4);
      ctx.stroke();
      ctx.fillText(formatFrequency(f), x, padding + plotHeight + 12);
    });
  }
}

function drawFFTRealTime(ctx, analyser) {
  if (!ctx || !analyser) return;

  const canvas = ctx.canvas;
  const logicalWidth = canvas.width / window.devicePixelRatio;
  const logicalHeight = canvas.height / window.devicePixelRatio;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const padding = 12; // Tăng padding từ 2 lên 12 để có chỗ cho labels
  const plotWidth = logicalWidth - padding * 2;
  const plotHeight = logicalHeight - padding * 2 - 12; // Trừ thêm 12 cho labels

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  analyser.getByteFrequencyData(dataArray);

  // Tính adaptive dB range từ data real-time
  let actualMinDb = Infinity;
  let actualMaxDb = -Infinity;
  const sampleRate = audioContext?.sampleRate || 44100;
  const nyquist = sampleRate / 2;

  for (let i = 0; i < bufferLength; i++) {
    const value = dataArray[i];
    const normalized = value / 255;
    const db = -120 + normalized * 140; // -120 to +20 dB range từ analyser
    if (isFinite(db)) {
      actualMinDb = Math.min(actualMinDb, db);
      actualMaxDb = Math.max(actualMaxDb, db);
    }
  }

  // Nếu không có data, dùng default range
  if (!isFinite(actualMinDb) || !isFinite(actualMaxDb)) {
    actualMinDb = -120;
    actualMaxDb = 20;
  }

  const dbPadding = Math.max(5, (actualMaxDb - actualMinDb) * 0.1);
  const minDb = Math.max(-120, Math.floor(actualMinDb - dbPadding));
  const maxDb = Math.min(20, Math.ceil(actualMaxDb + dbPadding));
  const dbRange = maxDb - minDb;

  // Linear scale cho frequencies (từ 0 đến max)
  const minFreq = 0;
  const maxFreq = nyquist;
  const freqRange = maxFreq - minFreq;

  for (let i = 0; i < bufferLength; i++) {
    const value = dataArray[i];
    const normalized = value / 255;
    const db = -120 + normalized * 140;
    const normalizedAmp = (db - minDb) / dbRange;
    const amp = Math.max(0, Math.min(1, normalizedAmp)) * plotHeight;

    // Linear scale cho trục X
    const freq = (i / bufferLength) * nyquist;
    const xNorm = (freq - minFreq) / freqRange;
    const x = padding + xNorm * plotWidth;
    const y = padding + plotHeight - amp;

    const grad = ctx.createLinearGradient(0, y, 0, padding + plotHeight);
    grad.addColorStop(0, "#4a90ff");
    grad.addColorStop(1, "#7f5cff");
    ctx.fillStyle = grad;

    // Tính barWidth dựa trên linear scale
    const nextFreq = ((i + 1) / bufferLength) * nyquist;
    const nextXNorm = (nextFreq - minFreq) / freqRange;
    const nextX = padding + nextXNorm * plotWidth;
    const barW = Math.max(
      1,
      Math.min((nextX - x) * 0.9, plotWidth - (x - padding))
    );
    const barH = Math.min(amp, padding + plotHeight - y);

    if (barW > 0 && barH > 0) {
      ctx.fillRect(x, y, barW, barH);
    }
  }

  // Vẽ axes chính (border)
  ctx.strokeStyle = "rgba(255,255,255,0.3)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  // Bottom axis
  ctx.moveTo(padding, padding + plotHeight);
  ctx.lineTo(padding + plotWidth, padding + plotHeight);
  // Left axis
  ctx.moveTo(padding, padding);
  ctx.lineTo(padding, padding + plotHeight);
  // Top axis
  ctx.moveTo(padding, padding);
  ctx.lineTo(padding + plotWidth, padding);
  // Right axis
  ctx.moveTo(padding + plotWidth, padding);
  ctx.lineTo(padding + plotWidth, padding + plotHeight);
  ctx.stroke();

  // Vẽ grid lines và labels cho trục Y (adaptive)
  ctx.strokeStyle = "rgba(255,255,255,0.25)";
  ctx.lineWidth = 1;
  ctx.fillStyle = "rgba(255,255,255,0.8)";
  ctx.font = "10px system-ui";
  ctx.textAlign = "left";

  // Tính số grid lines phù hợp
  const dbStep = Math.pow(10, Math.floor(Math.log10(dbRange)) - 1) * 5;
  const gridDbValues = [];
  for (let db = Math.ceil(minDb / dbStep) * dbStep; db <= maxDb; db += dbStep) {
    gridDbValues.push(db);
  }

  gridDbValues.forEach((db) => {
    const yNorm = (db - minDb) / dbRange;
    const y = padding + (1 - yNorm) * plotHeight;
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(padding + plotWidth, y);
    ctx.stroke();
    // Điều chỉnh vị trí label để không bị che
    ctx.fillText(
      `${db >= 0 ? "+" : ""}${Math.round(db)} dB`,
      padding - 2,
      y + 4
    );
  });

  // Vẽ grid lines và labels cho trục X (linear scale)
  ctx.textAlign = "center";
  const tickFreqs = [0, 1000, 2000, 5000, 10000, 15000, 20000];
  tickFreqs.forEach((f) => {
    if (f < minFreq || f > maxFreq) return;
    const xNorm = (f - minFreq) / freqRange;
    const x = padding + xNorm * plotWidth;
    ctx.strokeStyle = "rgba(255,255,255,0.2)";
    ctx.beginPath();
    ctx.moveTo(x, padding + plotHeight);
    ctx.lineTo(x, padding + plotHeight + 4);
    ctx.stroke();
    ctx.fillText(formatFrequency(f), x, padding + plotHeight + 12);
  });
}

function drawFFTRealTimeOverlay(ctx, analyser) {
  if (!ctx || !analyser) return;

  const canvas = ctx.canvas;
  const logicalWidth = canvas.width / window.devicePixelRatio;
  const logicalHeight = canvas.height / window.devicePixelRatio;

  const padding = 12; // Đồng bộ với drawFFTRealTime()
  const plotWidth = logicalWidth - padding * 2;
  const plotHeight = logicalHeight - padding * 2 - 12; // Trừ 12px cho labels

  const bufferLength = analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  analyser.getByteFrequencyData(dataArray);

  // Tính adaptive dB range từ data real-time (dùng cùng range với processed)
  // Lấy range từ processed analyser nếu có
  let minDb = -120;
  let maxDb = 20;
  let dbRange = maxDb - minDb;

  if (analyserNode) {
    const processedData = new Uint8Array(analyserNode.frequencyBinCount);
    analyserNode.getByteFrequencyData(processedData);

    let actualMinDb = Infinity;
    let actualMaxDb = -Infinity;
    for (let i = 0; i < processedData.length; i++) {
      const value = processedData[i];
      const normalized = value / 255;
      const db = -120 + normalized * 140;
      if (isFinite(db)) {
        actualMinDb = Math.min(actualMinDb, db);
        actualMaxDb = Math.max(actualMaxDb, db);
      }
    }

    if (isFinite(actualMinDb) && isFinite(actualMaxDb)) {
      const dbPadding = Math.max(5, (actualMaxDb - actualMinDb) * 0.1);
      minDb = Math.max(-120, Math.floor(actualMinDb - dbPadding));
      maxDb = Math.min(20, Math.ceil(actualMaxDb + dbPadding));
      dbRange = maxDb - minDb;
    }
  }

  // Linear scale cho frequencies (từ 0 đến max)
  const sampleRate = audioContext?.sampleRate || 44100;
  const nyquist = sampleRate / 2;
  const minFreq = 0;
  const maxFreq = nyquist;
  const freqRange = maxFreq - minFreq;

  ctx.fillStyle = "rgba(255, 215, 0, 0.6)"; // Màu vàng mờ

  for (let i = 0; i < bufferLength; i++) {
    const value = dataArray[i];
    const normalized = value / 255;
    const db = -120 + normalized * 140;

    const normalizedAmp = (db - minDb) / dbRange;
    const amp = Math.max(0, Math.min(1, normalizedAmp)) * plotHeight;

    // Linear scale cho trục X
    const freq = (i / bufferLength) * nyquist;
    const xNorm = (freq - minFreq) / freqRange;
    const x = padding + xNorm * plotWidth;
    const y = padding + plotHeight - amp;

    // Tính barWidth dựa trên linear scale
    const nextFreq = ((i + 1) / bufferLength) * nyquist;
    const nextXNorm = (nextFreq - minFreq) / freqRange;
    const nextX = padding + nextXNorm * plotWidth;
    const barW = Math.max(
      1,
      Math.min((nextX - x) * 0.9, plotWidth - (x - padding))
    );
    const barH = Math.min(amp, padding + plotHeight - y);

    if (barW > 0 && barH > 0) {
      ctx.fillRect(x, y, barW, barH);
    }
  }
}

function formatFrequency(hz) {
  if (hz >= 1000) {
    return `${(hz / 1000).toFixed(1)}k`;
  }
  return `${Math.round(hz)}`;
}

function drawSpectrogram(ctx, data, frequencies, times, currentTime = null) {
  if (!ctx || !data || data.length === 0) return;

  const canvas = ctx.canvas;
  const logicalWidth = canvas.width / window.devicePixelRatio;
  const logicalHeight = canvas.height / window.devicePixelRatio;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const freqBins = data.length;
  const timeFrames = data[0].length;
  const cellWidth = logicalWidth / timeFrames;
  const cellHeight = logicalHeight / freqBins;

  const leftMargin = 0;
  const bottomMargin = 0;
  const plotWidth = logicalWidth;
  const plotHeight = logicalHeight;
  const plotCellWidth = plotWidth / timeFrames;
  const plotCellHeight = plotHeight / freqBins;

  const minDb = -80;
  const maxDb = 0;

  for (let f = 0; f < freqBins; f++) {
    for (let t = 0; t < timeFrames; t++) {
      const db = data[f][t];
      const normalized = Math.max(
        0,
        Math.min(1, (db - minDb) / (maxDb - minDb))
      );

      let r, g, b;
      if (normalized < 0.5) {
        r = 255;
        g = 159 + normalized * 96;
        b = 74;
      } else {
        r = 255 - (normalized - 0.5) * 255;
        g = 74;
        b = 255;
      }

      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      const x = t * plotCellWidth;
      const y = (freqBins - f - 1) * plotCellHeight;

      const cellW = Math.min(plotCellWidth, logicalWidth - x);
      const cellH = Math.min(plotCellHeight, logicalHeight - y);
      if (
        cellW > 0 &&
        cellH > 0 &&
        x >= 0 &&
        y >= 0 &&
        x + cellW <= logicalWidth &&
        y + cellH <= logicalHeight
      ) {
        ctx.fillRect(x, y, cellW, cellH);
      }
    }
  }

  if (currentTime !== null && times && times.length > 0) {
    const duration = times[times.length - 1];
    if (duration > 0) {
      const position = currentTime / duration;
      const xPos = position * plotWidth;

      const clampedX = Math.max(0, Math.min(logicalWidth, xPos));

      ctx.strokeStyle = "#ff4a88";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(clampedX, 0);
      ctx.lineTo(clampedX, logicalHeight);
      ctx.stroke();
    }
  }
}

function formatFrequency(freq) {
  if (freq >= 1000) {
    return freq / 1000 + "k";
  }
  return freq.toString();
}

function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function updateFFTLabels(frequencies) {
  const labelsContainer = document.querySelector(".fft-labels");
  if (!labelsContainer) return;

  labelsContainer.innerHTML = "";

  const span0 = document.createElement("span");
  span0.textContent = "0";
  labelsContainer.appendChild(span0);

  EQ_FREQS.forEach((freq) => {
    const span = document.createElement("span");
    span.textContent = formatFrequency(freq);
    labelsContainer.appendChild(span);
  });
}

function updateSpectrogramLabels(times) {
  const labelsContainer = document.querySelector(".spectrogram-labels");
  if (!labelsContainer || !times || times.length === 0) return;

  labelsContainer.innerHTML = "";

  const duration = times[times.length - 1];

  const numLabels = 6;
  for (let i = 0; i < numLabels; i++) {
    const span = document.createElement("span");

    const time = (i / (numLabels - 1)) * duration;
    span.textContent = formatTime(time);
    labelsContainer.appendChild(span);
  }
}

function updateEQCurveLabels() {
  const labelsContainer = document.querySelector(".eq-curve-labels");
  if (!labelsContainer) return;

  labelsContainer.innerHTML = "";

  EQ_FREQS.forEach((freq) => {
    const span = document.createElement("span");
    span.textContent = formatFrequency(freq);
    labelsContainer.appendChild(span);
  });
}

function drawEQResponse(ctx, freqs, magDb) {
  if (!ctx || !freqs || !freqs.length || !magDb || !magDb.length) return;

  const canvas = ctx.canvas;
  const logicalWidth = canvas.width / window.devicePixelRatio;
  const logicalHeight = canvas.height / window.devicePixelRatio;
  const padding = 12;
  const plotWidth = logicalWidth - padding * 2;
  const plotHeight = logicalHeight - padding * 2 - 12;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const minDb = -12;
  const maxDb = 12;
  const dbRange = maxDb - minDb;

  const minFreq = 20;
  const maxFreq = freqs[freqs.length - 1] || 20000;
  const logMin = Math.log10(minFreq);
  const logMax = Math.log10(maxFreq);

  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1;
  const gridDb = [12, 0, -12];
  ctx.fillStyle = "rgba(255,255,255,0.6)";
  ctx.font = "10px system-ui";
  ctx.textAlign = "left";
  gridDb.forEach((db) => {
    const yNorm = (db - minDb) / dbRange;
    const y = padding + (1 - yNorm) * plotHeight;
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(padding + plotWidth, y);
    ctx.stroke();
    ctx.fillText(`${db} dB`, padding, y - 2);
  });

  ctx.strokeStyle = "#4fb4ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < freqs.length; i++) {
    const f = Math.max(minFreq, freqs[i]);
    const xNorm = (Math.log10(f) - logMin) / (logMax - logMin);
    const x = padding + xNorm * plotWidth;

    const db = Math.max(minDb, Math.min(maxDb, magDb[i]));
    const yNorm = (db - minDb) / dbRange;
    const y = padding + (1 - yNorm) * plotHeight;

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,0.6)";
  ctx.textAlign = "center";
  const tickFreqs = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];
  tickFreqs.forEach((f) => {
    if (f < minFreq || f > maxFreq) return;
    const x =
      padding + ((Math.log10(f) - logMin) / (logMax - logMin)) * plotWidth;
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.beginPath();
    ctx.moveTo(x, padding + plotHeight);
    ctx.lineTo(x, padding + plotHeight + 4);
    ctx.stroke();
    ctx.fillText(formatFrequency(f), x, padding + plotHeight + 12);
  });
}

async function uploadAudio(file) {
  const formData = new FormData();
  formData.append("file", file);

  setLoading(true, "Uploading...");
  try {
    const res = await fetch("/api/audio/upload", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (data.success) {
      // Cleanup original audio của file cũ
      cleanupOriginalAudio();

      currentFilename = data.filename;
      currentAudioData = data;
      // Lưu original data để so sánh
      if (!currentAudioData.original) {
        currentAudioData.original = {
          waveform: data.waveform ? { ...data.waveform } : null,
          fft: null, // Sẽ được lưu sau khi analyze
        };
      }

      // Load original audio buffer nếu toggle bật
      if (showOriginalOverlay) {
        await loadOriginalAudioBuffer();
      }

      const detectedMode = data.detected_mode || pickRandomMode();
      setModeLabel(detectedMode);

      const trackName = document.querySelector(".track-name");
      if (trackName) trackName.textContent = data.filename;

      if (waveformCtx) {
        drawWaveform(
          waveformCtx,
          data.waveform.data,
          data.waveform.time,
          data.duration
        );
      }

      currentPosition = 0;
      updateSeekbar(0);

      if (audioElement) {
        audioElement.pause();
        audioElement.src = "";
      }
      isPlaying = false;
      togglePlayPauseIcon();

      if (seekbarUpdateInterval) {
        clearInterval(seekbarUpdateInterval);
        seekbarUpdateInterval = null;
      }

      if (fftUpdateInterval) {
        clearInterval(fftUpdateInterval);
        fftUpdateInterval = null;
      }

      if (audioSource) {
        audioSource.disconnect();
        audioSource = null;
      }
      connectedAudioElement = null;
      analyserNode = null;

      updateTimelineLabels(data.duration);

      eqGains = [0, 0, 0, 0, 0, 0, 0, 0, 0];
      updateEQBarsUI();
      scheduleEQResponse();

      await analyzeAudio();
    } else {
      alert("Error: " + data.error);
    }
  } catch (error) {
    alert("Upload error: " + error.message);
  } finally {
    setLoading(false);
  }
}

function applyEQGains(gains) {
  if (!Array.isArray(gains) || gains.length !== eqGains.length) return;
  eqGains = [...gains];
  updateEQBarsUI();
  scheduleEQResponse();
}

function loadPlaylistFromStorage() {
  try {
    const raw = localStorage.getItem(PLAYLIST_STORAGE_KEY);
    const parsed = raw ? JSON.parse(raw) : [];

    let changed = false;
    const normalized = parsed.map((item) => {
      if (item.id) return item;
      changed = true;
      return {
        ...item,
        id: generatePlaylistId(item.filename || "track"),
      };
    });

    if (changed) {
      localStorage.setItem(PLAYLIST_STORAGE_KEY, JSON.stringify(normalized));
    }

    return normalized;
  } catch (e) {
    console.error("Cannot load playlist from storage:", e);
    return [];
  }
}

function savePlaylistToStorage() {
  try {
    localStorage.setItem(PLAYLIST_STORAGE_KEY, JSON.stringify(playlist));
  } catch (e) {
    console.error("Cannot save playlist to storage:", e);
  }
}

function renderPlaylistUI(activeId = null) {
  const listEl = document.querySelector(".playlist-list");
  if (!listEl) return;

  listEl.innerHTML = "";

  if (!playlist.length) {
    const empty = document.createElement("div");
    empty.className = "playlist-empty";
    empty.textContent = "Chưa có bài hát nào";
    listEl.appendChild(empty);
    return;
  }

  playlist.forEach((item) => {
    const itemEl = document.createElement("div");
    itemEl.className = "playlist-item";

    if (
      (activeId && item.id && item.id === activeId) ||
      (!activeId && currentFilename && item.filename === currentFilename)
    ) {
      itemEl.classList.add("active");
    }

    const icon = document.createElement("div");
    icon.className = "icon play";

    const info = document.createElement("div");
    info.className = "info";

    const name = document.createElement("span");
    name.className = "name";
    name.textContent = item.name || item.filename || "Unknown";

    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = item.tag || "Custom";

    info.appendChild(name);
    info.appendChild(tag);
    itemEl.appendChild(icon);
    itemEl.appendChild(info);

    itemEl.addEventListener("click", async () => {
      await loadPlaylistItem(item);
      renderPlaylistUI(item.id || item.filename);
    });

    listEl.appendChild(itemEl);
  });
}

function generatePlaylistId(filename) {
  const suffix = Math.random().toString(16).slice(2, 8);
  return `${filename || "track"}-${Date.now()}-${suffix}`;
}

function addCurrentTrackToPlaylist() {
  if (!currentFilename) {
    alert("Bạn cần mở một file audio trước khi thêm vào playlist.");
    return;
  }

  const newItem = {
    id: generatePlaylistId(currentFilename),
    filename: currentFilename,
    name: currentAudioData?.filename || currentFilename,
    tag: currentAudioData?.tag || "Custom",
    eqGains: [...eqGains],
    detectedMode: modeLabel || "None",
  };
  playlist.push(newItem);
  currentPlaylistId = newItem.id;
  savePlaylistToStorage();

  renderPlaylistUI(currentPlaylistId);
}

async function loadPlaylistItem(item) {
  if (!item || !item.filename) return;

  currentPlaylistId = item.id || item.filename;
  currentFilename = item.filename;
  currentAudioData = currentAudioData || {};
  currentAudioData.filename = item.filename;
  currentAudioData.name = item.name || item.filename;
  currentAudioData.tag = item.tag || "Custom";
  setModeLabel(item.detectedMode || pickRandomMode());

  const trackName = document.querySelector(".track-name");
  if (trackName) trackName.textContent = currentAudioData.name;
  setModeLabel(item.detectedMode || pickRandomMode());

  if (audioElement) {
    audioElement.pause();
  }
  isPlaying = false;
  togglePlayPauseIcon();
  currentPosition = 0;
  updateSeekbar(0);
  if (seekbarUpdateInterval) {
    clearInterval(seekbarUpdateInterval);
    seekbarUpdateInterval = null;
  }
  if (fftUpdateInterval) {
    clearInterval(fftUpdateInterval);
    fftUpdateInterval = null;
  }

  if (item.eqGains && Array.isArray(item.eqGains)) {
    applyEQGains(item.eqGains);
  } else {
    applyEQGains(eqGains);
  }

  await analyzeAudio();
  await processEQ();
  await loadAudioWithEQ();
  scheduleEQResponse();
}

async function analyzeAudio() {
  if (!currentFilename) return;

  setLoading(true, "Analyzing audio...");
  try {
    const res = await fetch("/api/audio/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: currentFilename }),
    });

    const data = await res.json();
    if (data.success) {
      if (data.fft) {
        if (!currentAudioData) currentAudioData = {};
        currentAudioData.fft = data.fft;
        // Lưu original FFT để so sánh
        if (!currentAudioData.original) {
          currentAudioData.original = {};
        }
        if (!currentAudioData.original.fft) {
          currentAudioData.original.fft = { ...data.fft };
        }
      }

      if (fftCtx && data.fft && !isPlaying) {
        drawFFT(fftCtx, data.fft.frequencies, data.fft.magnitude_db);
        updateFFTLabels(data.fft.frequencies);
      }

      if (spectrogramCtx && data.spectrogram) {
        spectrogramData = data.spectrogram;
        drawSpectrogram(
          spectrogramCtx,
          data.spectrogram.data,
          data.spectrogram.frequencies,
          data.spectrogram.times,
          null
        );

        updateSpectrogramLabels(data.spectrogram.times);
      }
    }
  } catch (error) {
    console.error("Analyze error:", error);
  } finally {
    setLoading(false);
  }
}

async function processEQ(eqGainsToUse = null) {
  if (!currentFilename) return;

  // Dùng snapshot nếu có, không thì dùng eqGains hiện tại
  const gains = eqGainsToUse || eqGains;

  setLoading(true, "Processing EQ...");
  try {
    const res = await fetch("/api/audio/process", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: currentFilename,
        eq_gains: gains,
      }),
    });

    const data = await res.json();
    if (data.success) {
      let duration = currentAudioData?.duration || 0;
      if (
        data.waveform &&
        Array.isArray(data.waveform.time) &&
        data.waveform.time.length
      ) {
        const lastTime = data.waveform.time[data.waveform.time.length - 1] || 0;
        if (lastTime) {
          duration = Math.max(duration, lastTime);
          if (!currentAudioData) currentAudioData = {};
          currentAudioData.duration = duration;
          updateTimelineLabels(duration);
        }
      }

      if (waveformCtx && data.waveform) {
        const waveformDuration = duration || 1;
        // Vẽ processed waveform (màu xanh)
        drawWaveform(
          waveformCtx,
          data.waveform.data,
          data.waveform.time,
          waveformDuration
        );

        // Vẽ original waveform overlay (màu vàng mờ) nếu có và toggle bật
        if (showOriginalOverlay && currentAudioData?.original?.waveform) {
          drawWaveform(
            waveformCtx,
            currentAudioData.original.waveform.data,
            currentAudioData.original.waveform.time,
            waveformDuration,
            "rgba(255, 215, 0, 0.6)", // Màu vàng mờ
            true // isOverlay
          );
        }

        if (!currentAudioData) currentAudioData = {};
        currentAudioData.waveform = data.waveform;
      }

      if (fftCtx && data.fft && !isPlaying) {
        // Vẽ processed FFT (màu xanh)
        drawFFT(fftCtx, data.fft.frequencies, data.fft.magnitude_db);

        // Vẽ original FFT overlay (màu vàng mờ) nếu có và toggle bật
        if (showOriginalOverlay && currentAudioData?.original?.fft) {
          drawFFT(
            fftCtx,
            currentAudioData.original.fft.frequencies,
            currentAudioData.original.fft.magnitude_db,
            true, // isOverlay
            "rgba(255, 215, 0, 0.6)" // Màu vàng mờ
          );
        }

        if (!currentAudioData) currentAudioData = {};
        currentAudioData.fft = data.fft;

        updateFFTLabels(data.fft.frequencies);
      }

      if (spectrogramCtx && data.spectrogram) {
        spectrogramData = data.spectrogram;
        drawSpectrogram(
          spectrogramCtx,
          data.spectrogram.data,
          data.spectrogram.frequencies,
          data.spectrogram.times,
          null
        );

        updateSpectrogramLabels(data.spectrogram.times);
      }

      scheduleEQResponse();
      lastProcessedUrl = null;
    }
  } catch (error) {
    console.error("Process error:", error);
  } finally {
    setLoading(false);
  }
}

function scheduleEqApply(options = {}) {
  const { wasPlaying = false, savedPosition = 0 } = options;
  clearTimeout(eqUpdateTimer);
  const runId = ++eqRunId;
  // Snapshot eqGains để tránh race condition
  const eqGainsSnapshot = [...eqGains];
  eqUpdateTimer = setTimeout(
    () =>
      performEqUpdate(runId, {
        wasPlaying,
        savedPosition,
        eqGains: eqGainsSnapshot,
      }),
    300
  );
}

async function performEqUpdate(
  runId,
  { wasPlaying, savedPosition, eqGains: eqGainsSnapshot }
) {
  if (runId !== eqRunId) return;
  if (eqIsRunning) return;

  eqIsRunning = true;
  try {
    await processEQ(eqGainsSnapshot);
    if (runId !== eqRunId) return;

    if (wasPlaying) {
      // Thêm check eqRunId trước khi load audio
      if (runId !== eqRunId) return;
      const newAudio = await loadAudioWithEQ(eqGainsSnapshot);
      if (!newAudio) return;

      newAudio.currentTime = savedPosition;

      try {
        if (audioContext && audioContext.state === "suspended") {
          await audioContext.resume();
        }

        setupWebAudioAPI(newAudio);

        await newAudio.play();
        isPlaying = true;
        togglePlayPauseIcon();

        // Load và start original audio nếu toggle bật
        if (showOriginalOverlay) {
          if (!originalAudioBuffer && currentFilename) {
            await loadOriginalAudioBuffer();
          }
          if (originalAudioBuffer) {
            startOriginalAudio(savedPosition * (newAudio.duration || 0));
          }
        }
      } catch (error) {
        if (error.name !== "AbortError") {
          console.error("Error playing audio:", error);
        }
        return;
      }

      if (seekbarUpdateInterval) {
        clearInterval(seekbarUpdateInterval);
      }
      seekbarUpdateInterval = setInterval(() => {
        if (newAudio && newAudio.duration && !isDragging) {
          currentPosition = newAudio.currentTime / newAudio.duration;
          updateSeekbar(currentPosition);
        }
      }, 100);

      if (fftUpdateInterval) {
        clearInterval(fftUpdateInterval);
      }
      fftUpdateInterval = setInterval(() => {
        updateRealTimeVisualization();
      }, 50);
    }
  } catch (error) {
    console.error("EQ update error:", error);
  } finally {
    eqIsRunning = false;
  }
}

function updateEQGain(bandIndex, value) {
  const gainDb = (value / 100) * 24 - 12;
  eqGains[bandIndex] = gainDb;

  scheduleEQResponse();

  const wasPlaying = isPlaying && audioElement;
  let savedPosition = 0;
  if (wasPlaying) {
    savedPosition = audioElement.currentTime || 0;
    audioElement.pause();
    isPlaying = false;
    togglePlayPauseIcon();
    if (seekbarUpdateInterval) {
      clearInterval(seekbarUpdateInterval);
      seekbarUpdateInterval = null;
    }
    if (fftUpdateInterval) {
      clearInterval(fftUpdateInterval);
      fftUpdateInterval = null;
    }
  }

  scheduleEqApply({ wasPlaying, savedPosition });
}

function updateEQHandlePosition(bar, fill, handle) {
  const fillHeight = parseFloat(fill.style.height) || 0;
  const barHeight = bar.offsetHeight;
  const handleTop = barHeight * (1 - fillHeight / 100) - 2;
  handle.style.top = `${Math.max(-2, Math.min(barHeight - 2, handleTop))}px`;
}

function updateEQBarsUI() {
  const eqBars = document.querySelectorAll(".eq-bar");
  eqBars.forEach((bar, index) => {
    if (index < eqGains.length) {
      const fill = bar.querySelector(".fill");
      const handle = bar.querySelector(".eq-handle");
      if (fill) {
        const gainDb = eqGains[index];
        const percent = ((gainDb + 12) / 24) * 100;
        const height = Math.max(0, Math.min(100, percent));
        fill.style.height = height + "%";

        if (handle) {
          updateEQHandlePosition(bar, fill, handle);
        }
      }
    }
  });
}

function formatTime(seconds) {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

function updateTimelineLabels(duration) {
  const timelineLabels = document.querySelectorAll(".timeline-labels span");
  if (timelineLabels.length >= 6 && duration > 0) {
    const intervals = [0, 0.25, 0.5, 0.75, 1.0];
    timelineLabels.forEach((label, index) => {
      if (index < intervals.length) {
        const time = intervals[index] * duration;
        label.textContent = formatTime(time);
      } else if (index === intervals.length) {
        label.textContent = formatTime(duration);
      }
    });
  }
}

function updateSeekbar(position) {
  const seekbar = document.querySelector(".seekbar");
  if (!seekbar) return;

  const track = seekbar.querySelector(".track");
  const thumb = seekbar.querySelector(".thumb");
  const positionPercent = Math.max(0, Math.min(100, position * 100));

  if (track) {
    track.style.width = `${positionPercent}%`;
  }
  if (thumb) {
    thumb.style.left = `${positionPercent}%`;
  }

  const duration = currentAudioData?.duration || 0;
  const currentTime = position * duration;
  const timeLabels = document.querySelectorAll(".time-label");

  if (timeLabels[0]) {
    timeLabels[0].textContent = formatTime(currentTime);
  }
  if (timeLabels[1]) {
    timeLabels[1].textContent = formatTime(duration);
  }
}

function setupSeekbar() {
  const seekbar = document.querySelector(".seekbar");
  if (!seekbar) return;

  const handleSeek = (e, updateAudio = false) => {
    const rect = seekbar.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const position = Math.max(0, Math.min(1, x / rect.width));
    currentPosition = position;
    updateSeekbar(position);

    if (waveformCtx && currentAudioData) {
      const duration = currentAudioData.duration;
      drawWaveform(
        waveformCtx,
        currentAudioData.waveform.data,
        currentAudioData.waveform.time,
        duration
      );

      const ctx = waveformCtx;
      const { width, height } = ctx.canvas;
      const xPos = position * width;
      ctx.strokeStyle = "#ff4a88";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(xPos, 0);
      ctx.lineTo(xPos, height);
      ctx.stroke();
    }

    if (
      updateAudio &&
      audioElement &&
      audioElement.duration &&
      currentAudioData
    ) {
      const seekTime = position * audioElement.duration;
      audioElement.currentTime = seekTime;

      // Seek original audio nếu toggle bật và đang phát
      if (showOriginalOverlay && isPlaying) {
        if (originalAudioBuffer) {
          stopOriginalAudio();
          startOriginalAudio(seekTime);
        } else if (currentFilename) {
          // Load buffer async, không block seek
          loadOriginalAudioBuffer().then(() => {
            if (originalAudioBuffer && isPlaying) {
              startOriginalAudio(seekTime);
            }
          });
        }
      }
    }
  };

  seekbar.addEventListener("click", (e) => handleSeek(e, true));

  seekbar.addEventListener("mousedown", (e) => {
    isDragging = true;
    handleSeek(e, false);
  });

  document.addEventListener("mousemove", (e) => {
    if (isDragging) {
      handleSeek(e, false);
    }
  });

  document.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;

      if (audioElement && audioElement.duration && currentAudioData) {
        audioElement.currentTime = currentPosition * audioElement.duration;
      }
    }
  });

  seekbar.addEventListener("touchstart", (e) => {
    e.preventDefault();
    isDragging = true;
    const touch = e.touches[0];
    const rect = seekbar.getBoundingClientRect();
    const x = touch.clientX - rect.left;
    const position = Math.max(0, Math.min(1, x / rect.width));
    currentPosition = position;
    updateSeekbar(position);
  });

  document.addEventListener("touchmove", (e) => {
    if (isDragging) {
      e.preventDefault();
      const touch = e.touches[0];
      const rect = seekbar.getBoundingClientRect();
      const x = touch.clientX - rect.left;
      const position = Math.max(0, Math.min(1, x / rect.width));
      currentPosition = position;
      updateSeekbar(position);
    }
  });

  document.addEventListener("touchend", () => {
    if (isDragging) {
      isDragging = false;

      if (audioElement && audioElement.duration && currentAudioData) {
        audioElement.currentTime = currentPosition * audioElement.duration;
      }
    }
  });
}

window.addEventListener("DOMContentLoaded", () => {
  waveformCtx = initCanvas("waveform");
  fftCtx = initCanvas("fft");
  spectrogramCtx = initCanvas("spectrogram");
  eqCurveCtx = initCanvas("eq-curve");

  setModeLabel("None");

  updateEQCurveLabels();

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "audio/*";
  fileInput.style.display = "none";
  fileInput.addEventListener("change", (e) => {
    if (e.target.files[0]) {
      uploadAudio(e.target.files[0]);
    }
  });
  document.body.appendChild(fileInput);

  const openFileBtn = document.querySelector(".top-left .btn");
  if (openFileBtn) {
    openFileBtn.addEventListener("click", () => {
      fileInput.click();
    });
  }

  const topButtons = document.querySelectorAll(".top-left .btn");
  const addToListBtn = topButtons.length > 1 ? topButtons[1] : null;
  if (addToListBtn) {
    addToListBtn.addEventListener("click", () => {
      addCurrentTrackToPlaylist();
    });
  }

  scheduleEQResponse();

  // Toggle hiển thị original overlay
  const showOriginalToggle = document.getElementById("showOriginalToggle");
  if (showOriginalToggle) {
    showOriginalToggle.checked = showOriginalOverlay;
    showOriginalToggle.addEventListener("change", async (e) => {
      showOriginalOverlay = e.target.checked;

      if (showOriginalOverlay) {
        // Load original audio buffer nếu chưa có
        if (!originalAudioBuffer && currentFilename) {
          await loadOriginalAudioBuffer();
        }
        // Start original audio nếu đang phát
        if (isPlaying && originalAudioBuffer && audioElement) {
          startOriginalAudio(audioElement.currentTime || 0);
        }
      } else {
        // Cleanup original audio khi toggle tắt
        cleanupOriginalAudio();
      }

      // Vẽ lại visualization nếu có data
      if (currentAudioData) {
        if (currentAudioData.waveform && waveformCtx) {
          const duration = currentAudioData.duration || 1;
          drawWaveform(
            waveformCtx,
            currentAudioData.waveform.data,
            currentAudioData.waveform.time,
            duration
          );
          if (showOriginalOverlay && currentAudioData.original?.waveform) {
            drawWaveform(
              waveformCtx,
              currentAudioData.original.waveform.data,
              currentAudioData.original.waveform.time,
              duration,
              "rgba(255, 215, 0, 0.6)",
              true
            );
  }
        }
        if (currentAudioData.fft && fftCtx && !isPlaying) {
          drawFFT(
            fftCtx,
            currentAudioData.fft.frequencies,
            currentAudioData.fft.magnitude_db
          );
          if (showOriginalOverlay && currentAudioData.original?.fft) {
            drawFFT(
              fftCtx,
              currentAudioData.original.fft.frequencies,
              currentAudioData.original.fft.magnitude_db,
              true,
              "rgba(255, 215, 0, 0.6)"
            );
          }
          updateFFTLabels(currentAudioData.fft.frequencies);
        }
      }
    });
  }

  playlist = loadPlaylistFromStorage();
  renderPlaylistUI();

  const eqBars = document.querySelectorAll(".eq-bar");
  eqBars.forEach((bar, index) => {
    const fill = bar.querySelector(".fill");
    if (fill) {
      let handle = bar.querySelector(".eq-handle");
      if (!handle) {
        handle = document.createElement("div");
        handle.className = "eq-handle";
        handle.style.cssText = `
          position: absolute;
          left: -2px;
          right: -2px;
          width: calc(100% + 4px);
          height: 4px;
          background: #4fb4ff;
          border-radius: 2px;
          cursor: ns-resize;
          z-index: 10;
          pointer-events: auto;
          transition: background 0.15s ease;
        `;
        bar.appendChild(handle);

        handle.addEventListener("mouseenter", () => {
          if (!isDraggingEQ) {
            handle.style.background = "#7f5cff";
          }
        });
        handle.addEventListener("mouseleave", () => {
          if (!isDraggingEQ) {
            handle.style.background = "#4fb4ff";
          }
        });
      }

      updateEQHandlePosition(bar, fill, handle);

      const updateEQFromY = (clientY) => {
        const rect = bar.getBoundingClientRect();
        const percent = 1 - (clientY - rect.top) / rect.height;
        const height = Math.max(0, Math.min(100, percent * 100));
        fill.style.height = height + "%";
        updateEQHandlePosition(bar, fill, handle);
        updateEQGain(index, height);
      };

      const handleMouseDown = (e) => {
        e.preventDefault();
        e.stopPropagation();
        isDraggingEQ = true;
        draggingEQIndex = index;
        bar.style.cursor = "ns-resize";
        handle.style.background = "#7f5cff";
        updateEQFromY(e.clientY);
      };

      handle.addEventListener("mousedown", handleMouseDown);
      bar.addEventListener("mousedown", (e) => {
        if (e.target === bar || e.target === fill) {
          handleMouseDown(e);
        }
      });

      const handleTouchStart = (e) => {
        e.preventDefault();
        e.stopPropagation();
        isDraggingEQ = true;
        draggingEQIndex = index;
        const touch = e.touches[0];
        updateEQFromY(touch.clientY);
      };

      handle.addEventListener("touchstart", handleTouchStart);
      bar.addEventListener("touchstart", (e) => {
        if (e.target === bar || e.target === fill) {
          handleTouchStart(e);
        }
      });
    }
  });

  document.addEventListener("mousemove", (e) => {
    if (isDraggingEQ && draggingEQIndex >= 0) {
      const bar = eqBars[draggingEQIndex];
      const fill = bar.querySelector(".fill");
      const handle = bar.querySelector(".eq-handle");
      if (fill && handle) {
        const rect = bar.getBoundingClientRect();
        const percent = 1 - (e.clientY - rect.top) / rect.height;
        const height = Math.max(0, Math.min(100, percent * 100));
        fill.style.height = height + "%";
        updateEQHandlePosition(bar, fill, handle);
        updateEQGain(draggingEQIndex, height);
      }
    }
  });

  document.addEventListener("mouseup", () => {
    if (isDraggingEQ) {
      isDraggingEQ = false;
      if (draggingEQIndex >= 0) {
        const bar = eqBars[draggingEQIndex];
        const handle = bar.querySelector(".eq-handle");
        bar.style.cursor = "";
        if (handle) {
          handle.style.background = "#4fb4ff";
        }
        draggingEQIndex = -1;
      }
    }
  });

  document.addEventListener("touchmove", (e) => {
    if (isDraggingEQ && draggingEQIndex >= 0 && e.touches.length > 0) {
      e.preventDefault();
      const bar = eqBars[draggingEQIndex];
      const fill = bar.querySelector(".fill");
      const handle = bar.querySelector(".eq-handle");
      if (fill && handle) {
        const touch = e.touches[0];
        const rect = bar.getBoundingClientRect();
        const percent = 1 - (touch.clientY - rect.top) / rect.height;
        const height = Math.max(0, Math.min(100, percent * 100));
        fill.style.height = height + "%";
        updateEQHandlePosition(bar, fill, handle);
        updateEQGain(draggingEQIndex, height);
      }
    }
  });

  document.addEventListener("touchend", () => {
    if (isDraggingEQ) {
      isDraggingEQ = false;
      if (draggingEQIndex >= 0) {
        const bar = eqBars[draggingEQIndex];
        const handle = bar.querySelector(".eq-handle");
        if (handle) {
          handle.style.background = "#4fb4ff";
        }
        draggingEQIndex = -1;
      }
    }
  });

  scheduleEQResponse();

  updateEQBarsUI();

  setupAutoEQ();

  setupSeekbar();

  setupPlayPauseButton();

  setupStopButton();

  if (!currentAudioData) {
    if (waveformCtx) {
      drawWaveform(waveformCtx, [], [], 1);
    }
    if (fftCtx) {
      drawFFT(fftCtx, [], []);
    }

    updateSeekbar(0);
  }
});

function updateAutoEQUI(enabled) {
  const fftCheckbox = document.querySelector(
    ".fft-panel .switch input[type='checkbox']"
  );
  if (fftCheckbox) {
    fftCheckbox.checked = enabled;
  }

  const autoeqValue = document.querySelector(".autoeq-panel .panel-row .value");
  if (autoeqValue) {
    autoeqValue.textContent = enabled ? "ON" : "OFF";
  }
}

function setupAutoEQ() {
  const fftCheckbox = document.querySelector(
    ".fft-panel .switch input[type='checkbox']"
  );

  if (fftCheckbox) {
    fftCheckbox.checked = autoEQEnabled;
    updateAutoEQUI(autoEQEnabled);

    fftCheckbox.addEventListener("change", async (e) => {
      autoEQEnabled = e.target.checked;
      updateAutoEQUI(autoEQEnabled);

      if (autoEQEnabled && currentAudioData) {
        setLoading(true, "Suggesting EQ...");
        const newGains = await suggestEQFromModel();
        setLoading(false);

        eqGains = [...newGains];
        updateEQBarsUI();
        scheduleEQResponse();

        const wasPlaying = isPlaying && audioElement;
        let savedPosition = 0;
        if (wasPlaying) {
          savedPosition = audioElement.currentTime || 0;
          audioElement.pause();
          isPlaying = false;
          togglePlayPauseIcon();
          if (seekbarUpdateInterval) {
            clearInterval(seekbarUpdateInterval);
            seekbarUpdateInterval = null;
          }
          if (fftUpdateInterval) {
            clearInterval(fftUpdateInterval);
            fftUpdateInterval = null;
          }
        }

        scheduleEqApply({ wasPlaying, savedPosition });
      } else {
        console.log("Auto-EQ disabled");
      }
    });
  }

  const autoeqRow = document.querySelector(".autoeq-panel .panel-row");
  if (autoeqRow) {
    autoeqRow.style.cursor = "pointer";
    autoeqRow.addEventListener("click", () => {
      autoEQEnabled = !autoEQEnabled;
      updateAutoEQUI(autoEQEnabled);

      if (fftCheckbox) {
        fftCheckbox.checked = autoEQEnabled;
        fftCheckbox.dispatchEvent(new Event("change"));
      }
    });
  }
}

function togglePlayPauseIcon() {
  const playIcon = document.querySelector(".play-icon");
  const pauseIcon = document.querySelector(".pause-icon");

  if (isPlaying) {
    if (playIcon) playIcon.style.display = "none";
    if (pauseIcon) pauseIcon.style.display = "block";
  } else {
    if (playIcon) playIcon.style.display = "block";
    if (pauseIcon) pauseIcon.style.display = "none";
  }
}

function setupWebAudioAPI(audioElement) {
  try {
    if (!audioElement) {
      return false;
    }

    // Nếu audioSource đã tồn tại và đang kết nối với audioElement khác
    if (
      audioSource &&
      connectedAudioElement &&
      connectedAudioElement !== audioElement
    ) {
      // Disconnect audioSource cũ
      try {
        audioSource.disconnect();
      } catch (e) {
        // Ignore error if already disconnected
      }
      // Xóa flag của audioElement cũ
      if (connectedAudioElement._audioSourceConnected) {
        delete connectedAudioElement._audioSourceConnected;
      }
      audioSource = null;
      connectedAudioElement = null;
    }

    // Nếu audioSource đã tồn tại và đang kết nối với cùng audioElement
    if (
      audioSource &&
      connectedAudioElement === audioElement &&
      audioElement._audioSourceConnected
    ) {
      return true;
    }

    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    if (audioContext.state === "suspended") {
      audioContext.resume();
    }

    if (!analyserNode) {
      analyserNode = audioContext.createAnalyser();
      analyserNode.fftSize = 2048;
      analyserNode.smoothingTimeConstant = 0.8;
    }

    if (!audioSource) {
      try {
        audioSource = audioContext.createMediaElementSource(audioElement);
        audioSource.connect(analyserNode);
        analyserNode.connect(audioContext.destination);

        audioElement._audioSourceConnected = true;
        connectedAudioElement = audioElement; // Lưu reference
      } catch (error) {
        if (error.name === "InvalidStateError") {
          audioElement._audioSourceConnected = true;
          connectedAudioElement = audioElement;
          return false;
        }
        throw error;
      }
    }

    return true;
  } catch (error) {
    console.error("Error setting up Web Audio API:", error);
    return false;
  }
}

async function loadOriginalAudioBuffer() {
  if (!currentFilename || !showOriginalOverlay) return;

  try {
    const res = await fetch("/api/audio/play", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: currentFilename,
        original: true,
      }),
    });

    const data = await res.json();
    if (!data.success) {
      console.error("Error loading original audio:", data.error);
      return;
    }

    const audioUrl = data.audio_url;
    const response = await fetch(audioUrl);
    const arrayBuffer = await response.arrayBuffer();

    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

    originalAudioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    console.log("Original audio buffer loaded");
  } catch (error) {
    console.error("Error loading original audio buffer:", error);
  }
}

function setupOriginalAnalyser() {
  if (!originalAudioBuffer || !audioContext || !showOriginalOverlay) {
    cleanupOriginalAudio();
    return false;
  }

  try {
    if (!originalAnalyserNode) {
      originalAnalyserNode = audioContext.createAnalyser();
      originalAnalyserNode.fftSize = 2048;
      originalAnalyserNode.smoothingTimeConstant = 0.8;
    }

    if (originalAudioSource) {
      try {
        originalAudioSource.stop();
      } catch (e) {
        // Ignore error if already stopped
      }
      originalAudioSource.disconnect();
    }

    originalAudioSource = audioContext.createBufferSource();
    originalAudioSource.buffer = originalAudioBuffer;
    originalAudioSource.connect(originalAnalyserNode);
    // Không connect đến destination để không phát ra loa

    return true;
  } catch (error) {
    console.error("Error setting up original analyser:", error);
    return false;
  }
}

function startOriginalAudio(currentTime = 0) {
  if (!showOriginalOverlay || !originalAudioBuffer || !audioContext) return;

  try {
    const setupSuccess = setupOriginalAnalyser();
    if (setupSuccess && originalAudioSource) {
      originalAudioSource.start(0, currentTime);
    }
  } catch (error) {
    console.error("Error starting original audio:", error);
  }
}

function stopOriginalAudio() {
  if (originalAudioSource) {
    try {
      originalAudioSource.stop();
    } catch (e) {
      // Ignore error if already stopped
    }
    originalAudioSource.disconnect();
    originalAudioSource = null;
  }
}

function cleanupOriginalAudio() {
  stopOriginalAudio();
  originalAnalyserNode = null;
  originalAudioBuffer = null;
}

async function loadAudioWithEQ(eqGainsToUse = null) {
  if (!currentFilename) return null;

  // Dùng snapshot nếu có, không thì dùng eqGains hiện tại
  const gains = eqGainsToUse || eqGains;

  setLoading(true, "Loading audio...");
  try {
    const res = await fetch("/api/audio/play", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        filename: currentFilename,
        eq_gains: gains,
      }),
    });

    const data = await res.json();
    if (!data.success) {
      console.error("Error loading audio:", data.error);
      return null;
    }

    if (!audioElement) {
      audioElement = new Audio();

      audioElement.addEventListener(
        "loadedmetadata",
        () => {
          console.log("Audio loaded, duration:", audioElement.duration);
          const duration = audioElement.duration || 0;
          if (duration) {
            if (!currentAudioData) currentAudioData = {};
            currentAudioData.duration = duration;
            updateTimelineLabels(duration);
          }

          if (
            !audioSource &&
            !audioElement._audioSourceConnected &&
            audioElement.src
          ) {
            setupWebAudioAPI(audioElement);
          }
        },
        { once: false }
      );

      audioElement.addEventListener("timeupdate", () => {
        if (!isDragging && audioElement.duration) {
          currentPosition = audioElement.currentTime / audioElement.duration;
          updateSeekbar(currentPosition);
        }
      });

      audioElement.addEventListener("ended", () => {
        isPlaying = false;
        currentPosition = 0;
        updateSeekbar(0);
        togglePlayPauseIcon();
        if (seekbarUpdateInterval) {
          clearInterval(seekbarUpdateInterval);
          seekbarUpdateInterval = null;
        }
        if (fftUpdateInterval) {
          clearInterval(fftUpdateInterval);
          fftUpdateInterval = null;
        }

        if (fftCtx && currentAudioData && currentAudioData.fft) {
          drawFFT(
            fftCtx,
            currentAudioData.fft.frequencies,
            currentAudioData.fft.magnitude_db
          );
        }
        if (spectrogramCtx && spectrogramData) {
          drawSpectrogram(
            spectrogramCtx,
            spectrogramData.data,
            spectrogramData.frequencies,
            spectrogramData.times,
            null
          );
        }
      });

      audioElement.addEventListener("error", (e) => {
        console.error("Audio error:", e);
        alert("Lỗi khi phát audio. Vui lòng thử lại.");
        isPlaying = false;
        togglePlayPauseIcon();
      });
    }

    const newUrl = data.audio_url;

    let currentSrc = "";
    let newUrlPath = "";
    try {
      if (audioElement.src) {
        currentSrc = new URL(audioElement.src, window.location.origin).pathname;
      }
      if (newUrl) {
        newUrlPath = new URL(newUrl, window.location.origin).pathname;
      }
    } catch (e) {
      currentSrc = audioElement.src || "";
      newUrlPath = newUrl || "";
    }
    if (currentSrc !== newUrlPath || !audioElement.src) {
      if (isPlaying) {
        audioElement.pause();
        isPlaying = false;
      }

      if (audioSource) {
        try {
          audioSource.disconnect();
        } catch (e) {}
        audioSource = null;
        connectedAudioElement = null;

        if (audioElement._audioSourceConnected) {
          delete audioElement._audioSourceConnected;
        }
      }
      audioElement.src = newUrl;
      audioElement.load();

      await new Promise((resolve, reject) => {
        const onLoadedMetadata = () => {
          audioElement.removeEventListener("loadedmetadata", onLoadedMetadata);
          audioElement.removeEventListener("error", onError);
          resolve();
        };
        const onError = (e) => {
          audioElement.removeEventListener("loadedmetadata", onLoadedMetadata);
          audioElement.removeEventListener("error", onError);
          reject(e);
        };
        if (audioElement.readyState >= 1) {
          onLoadedMetadata();
        } else {
          audioElement.addEventListener("loadedmetadata", onLoadedMetadata, {
            once: true,
          });
          audioElement.addEventListener("error", onError, { once: true });
        }
      });
    } else {
      if (
        !audioSource &&
        !audioElement._audioSourceConnected &&
        audioElement.readyState >= 1
      ) {
        setupWebAudioAPI(audioElement);
      }
    }

    lastProcessedUrl = data.audio_url;
    audioElement._mode = "eq";
    return audioElement;
  } catch (error) {
    console.error("Error loading audio:", error);
    return null;
  } finally {
    setLoading(false);
  }
}

function updateRealTimeVisualization() {
  if (!isPlaying || !audioElement) return;

  if (fftCtx && analyserNode) {
    // Vẽ processed FFT (màu xanh)
    drawFFTRealTime(fftCtx, analyserNode);

    // Vẽ original FFT overlay (màu vàng mờ) nếu toggle bật
    if (showOriginalOverlay && originalAnalyserNode) {
      drawFFTRealTimeOverlay(fftCtx, originalAnalyserNode);
    }
  }

  if (spectrogramCtx && spectrogramData && audioElement.duration) {
    const currentTime = audioElement.currentTime;
    drawSpectrogram(
      spectrogramCtx,
      spectrogramData.data,
      spectrogramData.frequencies,
      spectrogramData.times,
      currentTime
    );
  }
}

async function setupPlayPauseButton() {
  const playPauseBtn = document.getElementById("playPauseBtn");
  if (!playPauseBtn) return;

  playPauseBtn.addEventListener("click", async () => {
    if (!currentAudioData) {
      alert("Vui lòng upload file audio trước!");
      return;
    }

    if (!isPlaying) {
      let audio = audioElement;

      if (!audio || !audio.src || audio._mode !== "eq") {
        audio = await loadAudioWithEQ();
        if (!audio) {
          alert("Không thể tải audio. Vui lòng thử lại.");
          return;
        }
      }

      if (currentAudioData.duration) {
        audio.currentTime = currentPosition * currentAudioData.duration;
      }

      if (
        !audioSource &&
        !audio._audioSourceConnected &&
        audio.readyState >= 1
      ) {
        setupWebAudioAPI(audio);
      }

      try {
        if (audioContext && audioContext.state === "suspended") {
          await audioContext.resume();
        }

        try {
          await audio.play();
          isPlaying = true;
          togglePlayPauseIcon();

          // Load và start original audio nếu toggle bật
          if (showOriginalOverlay) {
            if (!originalAudioBuffer && currentFilename) {
              await loadOriginalAudioBuffer();
            }
            if (originalAudioBuffer) {
              startOriginalAudio(audio.currentTime || 0);
            }
          }
        } catch (error) {
          if (error.name !== "AbortError") {
            console.error("Error playing audio:", error);
            alert("Không thể phát audio. Vui lòng thử lại.");
            isPlaying = false;
            togglePlayPauseIcon();
          }
        }

        if (!seekbarUpdateInterval) {
          seekbarUpdateInterval = setInterval(() => {
            if (audio && audio.duration && !isDragging) {
              currentPosition = audio.currentTime / audio.duration;
              updateSeekbar(currentPosition);
            }
          }, 100);
        }

        if (!fftUpdateInterval) {
          fftUpdateInterval = setInterval(() => {
            updateRealTimeVisualization();
          }, 50);
        }
      } catch (error) {
        console.error("Play error:", error);
        alert("Không thể phát audio. Vui lòng thử lại.");
        isPlaying = false;
        togglePlayPauseIcon();
      }
    } else {
      if (audioElement) {
        audioElement.pause();
      }
      // Pause original audio
      stopOriginalAudio();
      isPlaying = false;
      togglePlayPauseIcon();

      if (seekbarUpdateInterval) {
        clearInterval(seekbarUpdateInterval);
        seekbarUpdateInterval = null;
      }

      if (fftUpdateInterval) {
        clearInterval(fftUpdateInterval);
        fftUpdateInterval = null;
      }

      if (fftCtx && currentAudioData && currentAudioData.fft) {
        drawFFT(
          fftCtx,
          currentAudioData.fft.frequencies,
          currentAudioData.fft.magnitude_db
        );
      }
      if (spectrogramCtx && spectrogramData) {
        drawSpectrogram(
          spectrogramCtx,
          spectrogramData.data,
          spectrogramData.frequencies,
          spectrogramData.times,
          null
        );
      }
    }
  });
}

function setupStopButton() {
  const stopBtn = document.getElementById("stopBtn");
  if (!stopBtn) return;

  stopBtn.addEventListener("click", () => {
    if (!currentAudioData) return;

    if (audioElement) {
      audioElement.pause();
      audioElement.currentTime = 0;
    }

    isPlaying = false;
    currentPosition = 0;
    updateSeekbar(0);
    togglePlayPauseIcon();

    if (seekbarUpdateInterval) {
      clearInterval(seekbarUpdateInterval);
      seekbarUpdateInterval = null;
    }

    if (fftUpdateInterval) {
      clearInterval(fftUpdateInterval);
      fftUpdateInterval = null;
    }

    if (fftCtx && currentAudioData && currentAudioData.fft) {
      drawFFT(
        fftCtx,
        currentAudioData.fft.frequencies,
        currentAudioData.fft.magnitude_db
      );
    }
    if (spectrogramCtx && spectrogramData) {
      drawSpectrogram(
        spectrogramCtx,
        spectrogramData.data,
        spectrogramData.frequencies,
        spectrogramData.times,
        null
      );
    }
  });
}
