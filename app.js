/**
 * app.js
 * ---------------------------------------------------------------
 * Orchestrates the whole prototype. Loaded as an ES module so it
 * can `import` MediaPipe's Tasks Vision package directly from a
 * CDN — no build step, no npm install, just open index.html.
 * ---------------------------------------------------------------
 */
import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const { HAND_CONNECTIONS, getFingerStates, SIGN_VOCAB, matchSign, I18N } = window.SignSpeakVocab;

// ---------- DOM ----------
const modelStatus = document.getElementById("modelStatus");
const modelStatusText = document.getElementById("modelStatusText");
const langEnBtn = document.getElementById("langEnBtn");
const langTaBtn = document.getElementById("langTaBtn");

const video = document.getElementById("webcam");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");
const recBadge = document.getElementById("recBadge");
const cameraToggleBtn = document.getElementById("cameraToggleBtn");

const recognizedCard = document.getElementById("recognizedCard");
const recognizedWord = document.getElementById("recognizedWord");
const recognizedWordTa = document.getElementById("recognizedWordTa");
const confidenceFill = document.getElementById("confidenceFill");
const vocabStrip = document.getElementById("vocabStrip");

const micBtn = document.getElementById("micBtn");
const micHint = document.getElementById("micHint");
const transcriptFinal = document.getElementById("transcriptFinal");
const transcriptInterim = document.getElementById("transcriptInterim");
const transcriptEmpty = document.getElementById("transcriptEmpty");

const historyList = document.getElementById("historyList");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");

// ---------- state ----------
let currentLang = "en";
let handLandmarker = null;
let stream = null;
let running = false;
let rafId = null;

const BUFFER_SIZE = 10;
const STABILITY_RATIO = 0.7;
const NO_HAND_RESET_FRAMES = 15;
let signBuffer = [];
let noHandFrames = 0;
let lockedSignId = null;
let flashUntil = 0;

let recognition = null;
let recognizing = false;
let recogLang = "en-IN";

let history = [];
let cachedVoices = [];

// ---------- i18n ----------
function applyLanguage(lang) {
  currentLang = lang;
  document.body.dataset.lang = lang;
  langEnBtn.classList.toggle("active", lang === "en");
  langTaBtn.classList.toggle("active", lang === "ta");

  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (I18N[lang][key]) el.textContent = I18N[lang][key];
  });

  cameraToggleBtn.textContent = running ? I18N[lang].stopCamera : I18N[lang].startCamera;
  micHint.textContent = recognizing ? I18N[lang].micHintListening : I18N[lang].micHintIdle;
  renderHistory();
}
langEnBtn.addEventListener("click", () => applyLanguage("en"));
langTaBtn.addEventListener("click", () => applyLanguage("ta"));

// ---------- model status ----------
function setModelStatus(state, text) {
  modelStatus.classList.remove("ready", "error");
  if (state === "ready") modelStatus.classList.add("ready");
  if (state === "error") modelStatus.classList.add("error");
  modelStatusText.textContent = text;
}

// ---------- vocab strip ----------
function renderVocabStrip() {
  vocabStrip.innerHTML = "";
  SIGN_VOCAB.forEach((sign) => {
    const chip = document.createElement("span");
    chip.className = "vocab-chip";
    chip.dataset.signId = sign.id;
    chip.textContent = `${sign.emoji} ${sign.en}`;
    vocabStrip.appendChild(chip);
  });
}
function updateVocabHighlight(activeId) {
  document.querySelectorAll(".vocab-chip").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.signId === activeId);
  });
}

// ---------- MediaPipe HandLandmarker ----------
async function initModel() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    const commonOptions = {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
      },
      runningMode: "VIDEO",
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    };
    try {
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        ...commonOptions,
        baseOptions: { ...commonOptions.baseOptions, delegate: "GPU" },
      });
    } catch (gpuErr) {
      console.warn("GPU delegate unavailable, falling back to CPU:", gpuErr);
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        ...commonOptions,
        baseOptions: { ...commonOptions.baseOptions, delegate: "CPU" },
      });
    }
    setModelStatus("ready", I18N[currentLang].modelReady);
    cameraToggleBtn.disabled = false;
  } catch (err) {
    console.error(err);
    setModelStatus("error", I18N[currentLang].modelError);
  }
}

// ---------- camera ----------
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    const syncCanvasSize = () => {
      overlay.width = video.videoWidth;
      overlay.height = video.videoHeight;
    };
    if (video.videoWidth) syncCanvasSize();
    video.addEventListener("loadedmetadata", syncCanvasSize, { once: true });

    cameraPlaceholder.style.display = "none";
    recBadge.classList.remove("hidden");
    running = true;
    cameraToggleBtn.textContent = I18N[currentLang].stopCamera;
    cameraToggleBtn.classList.add("is-active");
    rafId = requestAnimationFrame(predictLoop);
  } catch (err) {
    console.error(err);
    setModelStatus("error", I18N[currentLang].cameraError);
  }
}

function stopCamera() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  video.srcObject = null;
  cameraPlaceholder.style.display = "flex";
  recBadge.classList.add("hidden");
  cameraToggleBtn.textContent = I18N[currentLang].startCamera;
  cameraToggleBtn.classList.remove("is-active");
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

  lockedSignId = null;
  signBuffer = [];
  noHandFrames = 0;
  setConfidence(0);
  updateVocabHighlight(null);
}

cameraToggleBtn.addEventListener("click", () => (running ? stopCamera() : startCamera()));

// ---------- prediction loop ----------
function predictLoop() {
  if (!running) return;
  const now = performance.now();
  if (handLandmarker && video.readyState >= 2) {
    const result = handLandmarker.detectForVideo(video, now);
    drawSkeleton(result, now);
    processResult(result, now);
  }
  rafId = requestAnimationFrame(predictLoop);
}

function drawSkeleton(result, now) {
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  if (!result.landmarks || result.landmarks.length === 0) return;

  const color = now < flashUntil ? "#F2A93B" : "#3FDBC4";
  result.landmarks.forEach((lm) => {
    overlayCtx.strokeStyle = color;
    overlayCtx.lineWidth = Math.max(2, overlay.width * 0.004);
    HAND_CONNECTIONS.forEach(([a, b]) => {
      const p1 = lm[a], p2 = lm[b];
      overlayCtx.beginPath();
      overlayCtx.moveTo(p1.x * overlay.width, p1.y * overlay.height);
      overlayCtx.lineTo(p2.x * overlay.width, p2.y * overlay.height);
      overlayCtx.stroke();
    });
    overlayCtx.fillStyle = color;
    lm.forEach((pt) => {
      overlayCtx.beginPath();
      overlayCtx.arc(pt.x * overlay.width, pt.y * overlay.height, Math.max(2.5, overlay.width * 0.006), 0, Math.PI * 2);
      overlayCtx.fill();
    });
  });
}

function processResult(result, now) {
  const hasHand = result.landmarks && result.landmarks.length > 0;
  if (!hasHand) {
    noHandFrames++;
    if (noHandFrames > NO_HAND_RESET_FRAMES) {
      lockedSignId = null;
      signBuffer = [];
      setConfidence(0);
      updateVocabHighlight(null);
    }
    return;
  }
  noHandFrames = 0;

  const states = getFingerStates(result.landmarks[0]);
  const match = matchSign(states);
  signBuffer.push(match ? match.id : null);
  if (signBuffer.length > BUFFER_SIZE) signBuffer.shift();

  const counts = {};
  signBuffer.forEach((id) => { if (id) counts[id] = (counts[id] || 0) + 1; });
  let bestId = null, bestCount = 0;
  Object.entries(counts).forEach(([id, c]) => { if (c > bestCount) { bestId = id; bestCount = c; } });

  const ratio = signBuffer.length ? bestCount / signBuffer.length : 0;
  setConfidence(bestId ? ratio : 0);

  if (bestId && ratio >= STABILITY_RATIO && bestId !== lockedSignId) {
    lockedSignId = bestId;
    const sign = SIGN_VOCAB.find((s) => s.id === bestId);
    onSignRecognized(sign, now);
  }
}

function setConfidence(ratio) {
  confidenceFill.style.width = Math.round(ratio * 100) + "%";
}

function onSignRecognized(sign, now) {
  recognizedWord.textContent = sign.en;
  recognizedWordTa.textContent = sign.ta;
  recognizedCard.classList.add("flash");
  flashUntil = now + 550;
  setTimeout(() => recognizedCard.classList.remove("flash"), 550);
  updateVocabHighlight(sign.id);

  const spoken = currentLang === "ta" ? sign.ta : sign.en;
  addHistory("sign", spoken);
  speak(spoken, currentLang);
}

// ---------- text-to-speech ----------
function refreshVoices() {
  cachedVoices = "speechSynthesis" in window ? window.speechSynthesis.getVoices() : [];
}
if ("speechSynthesis" in window) {
  refreshVoices();
  window.speechSynthesis.onvoiceschanged = refreshVoices;
}
function speak(text, langPrefix) {
  if (!("speechSynthesis" in window)) return;
  const utter = new SpeechSynthesisUtterance(text);
  const prefix = langPrefix === "ta" ? "ta" : "en";
  const voice =
    cachedVoices.find((v) => v.lang.toLowerCase().startsWith(prefix)) ||
    cachedVoices.find((v) => v.lang.toLowerCase().startsWith("en"));
  if (voice) utter.voice = voice;
  utter.lang = voice ? voice.lang : prefix === "ta" ? "ta-IN" : "en-IN";
  utter.rate = 0.95;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utter);
}

// ---------- speech-to-text ----------
function buildRecognition() {
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) return null;
  const r = new SR();
  r.continuous = true;
  r.interimResults = true;
  r.lang = recogLang;

  r.onresult = (e) => {
    let interim = "", final = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const transcript = e.results[i][0].transcript;
      if (e.results[i].isFinal) final += transcript;
      else interim += transcript;
    }
    if (interim || final) transcriptEmpty.style.display = "none";
    transcriptInterim.textContent = interim;
    if (final.trim()) {
      transcriptFinal.textContent = final.trim();
      addHistory("speech", final.trim());
    }
  };
  r.onerror = (e) => {
    console.warn("Speech recognition error:", e.error);
  };
  r.onend = () => {
    if (recognizing) {
      try { r.start(); } catch (e) { /* already running */ }
    }
  };
  return r;
}

micBtn.addEventListener("click", () => {
  if (!recognizing) {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      micHint.textContent = I18N[currentLang].speechUnsupported;
      return;
    }
    recognition = buildRecognition();
    recognition.start();
    recognizing = true;
    micBtn.classList.add("listening");
    micBtn.setAttribute("aria-pressed", "true");
    micHint.textContent = I18N[currentLang].micHintListening;
  } else {
    recognizing = false;
    if (recognition) recognition.stop();
    micBtn.classList.remove("listening");
    micBtn.setAttribute("aria-pressed", "false");
    micHint.textContent = I18N[currentLang].micHintIdle;
  }
});

document.querySelectorAll(".recog-lang-toggle .chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    document.querySelectorAll(".recog-lang-toggle .chip").forEach((c) => c.classList.remove("active"));
    chip.classList.add("active");
    recogLang = chip.dataset.reclang;
    if (recognizing && recognition) {
      recognizing = false;
      recognition.stop();
      setTimeout(() => {
        recognition = buildRecognition();
        recognition.start();
        recognizing = true;
      }, 150);
    }
  });
});

// ---------- history ----------
function loadHistory() {
  try {
    history = JSON.parse(localStorage.getItem("signspeak_history") || "[]");
  } catch (e) {
    history = [];
  }
  renderHistory();
}
function saveHistory() {
  try {
    localStorage.setItem("signspeak_history", JSON.stringify(history.slice(-50)));
  } catch (e) { /* storage unavailable, skip persistence */ }
}
function addHistory(direction, text) {
  history.push({ direction, text, time: new Date().toISOString() });
  saveHistory();
  renderHistory();
}
function renderHistory() {
  historyList.innerHTML = "";
  if (history.length === 0) {
    const li = document.createElement("li");
    li.className = "history-empty";
    li.textContent = I18N[currentLang].historyEmpty;
    historyList.appendChild(li);
    return;
  }
  history.slice().reverse().forEach((item) => {
    const li = document.createElement("li");
    li.className = "history-item " + item.direction;

    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = item.direction === "sign" ? "SIGN" : "SPEECH";

    const text = document.createElement("span");
    text.textContent = item.text;

    const time = document.createElement("span");
    time.className = "time";
    time.textContent = new Date(item.time).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    li.append(tag, text, time);
    historyList.appendChild(li);
  });
}
clearHistoryBtn.addEventListener("click", () => {
  history = [];
  saveHistory();
  renderHistory();
});

// ---------- boot ----------
renderVocabStrip();
loadHistory();
applyLanguage("en");
initModel();
