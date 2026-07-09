/**
 * signs.js
 * ---------------------------------------------------------------
 * Everything vocabulary-related lives in this one file on purpose:
 * to add a new sign, you only ever need to touch SIGN_VOCAB below.
 *
 * HOW RECOGNITION WORKS (read this before editing):
 * MediaPipe's HandLandmarker gives us 21 (x,y,z) points per hand,
 * every frame. getFingerStates() turns those 21 points into a
 * simple 5-value pattern: is each finger extended (true) or
 * curled (false)? SIGN_VOCAB then matches that 5-value pattern
 * against known signs.
 *
 * This is a *geometric* classifier, not a trained model — it is
 * fast, needs zero training data, and is honest about being a
 * starting point. It only recognizes static, one-handed shapes
 * held toward the camera. Real Indian Sign Language includes
 * motion and facial grammar that this simple version can't see.
 * See README.md -> "Roadmap to a trained model" for how to evolve
 * this into the CNN/LSTM pipeline described in the project brief.
 * ---------------------------------------------------------------
 */

// Standard 21-point MediaPipe hand connections, used to draw the skeleton.
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],          // thumb
  [0,5],[5,6],[6,7],[7,8],          // index
  [5,9],[9,10],[10,11],[11,12],    // middle
  [9,13],[13,14],[14,15],[15,16],  // ring
  [13,17],[17,18],[18,19],[19,20], // pinky
  [0,17]                            // palm base
];

/**
 * Reads the 21 landmarks for one hand and returns which fingers are
 * extended. Works best when the palm faces the camera (the natural
 * position for showing a sign).
 */
function getFingerStates(lm) {
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const wrist = lm[0];

  // Non-thumb fingers: a finger counts as "extended" when its tip sits
  // noticeably farther from the wrist than its own middle knuckle (PIP).
  // That ratio test still works when the hand is tilted or rotated,
  // unlike a plain "is the tip above the knuckle" check.
  const isExtended = (tipIdx, pipIdx) =>
    dist(lm[tipIdx], wrist) > dist(lm[pipIdx], wrist) * 1.1;

  // Thumb moves in its own plane, so it's measured against the pinky's
  // knuckle instead of the wrist: a thumb tucked into the palm sits
  // close to that point, an extended thumb sits clearly further out.
  const pinkyMcp = lm[17];
  const thumbExtended = dist(lm[4], pinkyMcp) > dist(lm[2], pinkyMcp) * 1.05;

  return {
    thumb: thumbExtended,
    index: isExtended(8, 6),
    middle: isExtended(12, 10),
    ring: isExtended(16, 14),
    pinky: isExtended(20, 18),
  };
}

/**
 * The starter vocabulary. Each pattern is [thumb, index, middle, ring, pinky].
 * Patterns must stay unique — matchSign() does an exact match.
 *
 * To add a sign: hold the handshape in front of the camera, note which
 * fingers are up, add a row below. Keep patterns distinct from existing
 * ones or recognition will be ambiguous.
 */
const SIGN_VOCAB = [
  { id: "hello",    pattern: [true,  true,  true,  true,  true ], emoji: "🖐️", en: "Hello",      ta: "வணக்கம்" },
  { id: "no",       pattern: [false, false, false, false, false], emoji: "✊", en: "No",         ta: "இல்லை" },
  { id: "thankyou", pattern: [true,  false, false, false, false], emoji: "👍", en: "Thank you",  ta: "நன்றி" },
  { id: "wait",     pattern: [false, true,  false, false, false], emoji: "☝️", en: "Wait",       ta: "காத்திருங்கள்" },
  { id: "yes",      pattern: [false, true,  true,  false, false], emoji: "✌️", en: "Yes",        ta: "ஆம்" },
  { id: "iloveyou", pattern: [true,  true,  false, false, true ], emoji: "🤟", en: "I love you", ta: "நான் உன்னை காதலிக்கிறேன்" },
  { id: "call",     pattern: [true,  false, false, false, true ], emoji: "🤙", en: "Call",       ta: "அழை" },
  { id: "water",    pattern: [false, false, false, false, true ], emoji: "🤏", en: "Water",      ta: "தண்ணீர்" },
];

/** Exact-match the finger pattern against SIGN_VOCAB. Returns null if no rule fits. */
function matchSign(states) {
  const key = [states.thumb, states.index, states.middle, states.ring, states.pinky];
  return SIGN_VOCAB.find(s => s.pattern.every((v, i) => v === key[i])) || null;
}

// UI strings — every piece of interface chrome, in both languages.
const I18N = {
  en: {
    tagline: "Two-way sign language translation, live",
    signToSpeechTitle: "Sign → Speech",
    signToSpeechSub: "Show a sign to the camera",
    speechToTextTitle: "Speech → Text",
    speechToTextSub: "For the other person to speak into",
    cameraOffLabel: "Camera is off",
    startCamera: "Start Camera",
    stopCamera: "Stop Camera",
    recognizedLabel: "Recognized sign",
    micHintIdle: "Tap to start listening",
    micHintListening: "Listening…",
    transcriptEmpty: "Spoken words will appear here as large text",
    historyTitle: "Conversation history",
    clearHistory: "Clear",
    historyEmpty: "Your conversation will appear here",
    footerNote: "Prototype vocabulary: 8 signs today — built to expand toward 50–100+ common Indian Sign Language words.",
    modelLoading: "Loading AI model…",
    modelReady: "Model ready",
    modelError: "Model failed to load",
    cameraError: "Camera access was blocked or unavailable",
    speechUnsupported: "Speech recognition isn't supported in this browser — try Chrome or Edge",
  },
  ta: {
    tagline: "இரு-வழி சைகை மொழி மொழிபெயர்ப்பு, நேரடியாக",
    signToSpeechTitle: "சைகை → பேச்சு",
    signToSpeechSub: "கேமராவிற்கு ஒரு சைகையைக் காட்டுங்கள்",
    speechToTextTitle: "பேச்சு → எழுத்து",
    speechToTextSub: "மற்ற நபர் பேச இதைப் பயன்படுத்தவும்",
    cameraOffLabel: "கேமரா ஆஃப் செய்யப்பட்டுள்ளது",
    startCamera: "கேமராவைத் தொடங்கு",
    stopCamera: "கேமராவை நிறுத்து",
    recognizedLabel: "கண்டறியப்பட்ட சைகை",
    micHintIdle: "கேட்கத் தொடங்க தட்டவும்",
    micHintListening: "கேட்கிறது…",
    transcriptEmpty: "பேசப்படும் வார்த்தைகள் இங்கே பெரிய எழுத்தில் தோன்றும்",
    historyTitle: "உரையாடல் வரலாறு",
    clearHistory: "அழி",
    historyEmpty: "உங்கள் உரையாடல் இங்கே தோன்றும்",
    footerNote: "தற்போதைய சொற்களஞ்சியம்: 8 சைகைகள் — 50–100+ இந்திய சைகை மொழி சொற்களாக விரிவடையும் வகையில் கட்டமைக்கப்பட்டுள்ளது.",
    modelLoading: "AI மாடல் ஏற்றப்படுகிறது…",
    modelReady: "மாடல் தயார்",
    modelError: "மாடலை ஏற்ற முடியவில்லை",
    cameraError: "கேமரா அணுகல் தடுக்கப்பட்டது அல்லது கிடைக்கவில்லை",
    speechUnsupported: "இந்த உலாவியில் பேச்சு அங்கீகாரம் ஆதரிக்கப்படவில்லை — Chrome அல்லது Edge பயன்படுத்தவும்",
  },
};

// Expose everything on one namespace since this file loads as a plain
// (non-module) script, while app.js loads as an ES module.
window.SignSpeakVocab = { HAND_CONNECTIONS, getFingerStates, SIGN_VOCAB, matchSign, I18N };
