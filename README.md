# SignSpeak AI — Working Prototype

A browser-based, two-way sign-language translator. No install, no build step —
open `index.html` in Chrome or Edge and it runs.

This README explains exactly what's real, what's simplified, and how to grow
this into the full CNN/LSTM system from the original pitch.

---

## 1. What's actually working here

| Feature from the pitch                  | Status in this prototype                                            |
|------------------------------------------|-----------------------------------------------------------------------|
| Camera captures hand gestures            | ✅ Real — `getUserMedia`                                              |
| AI detects hand landmarks                | ✅ Real — Google's MediaPipe HandLandmarker (21-point model), running live in the browser |
| Model recognizes the sign                | ⚠️ Simplified — a rule-based geometric classifier (see §2), not a trained CNN/LSTM yet |
| App speaks the translated word           | ✅ Real — Web Speech API (`speechSynthesis`)                          |
| Speech from another person → text        | ✅ Real — Web Speech API (`SpeechRecognition`)                        |
| Tamil + English display                  | ✅ Real — full bilingual UI + bilingual vocabulary                    |
| Conversation history                     | ✅ Real — saved locally in the browser (`localStorage`)               |
| Offline mode                             | 🚧 Not built yet — see §5 roadmap                                     |

Nothing here is a mockup or fake data — every panel is wired to a working
browser API. The one deliberate simplification is *how* a sign is
classified, explained below.

## 2. How sign recognition works right now

MediaPipe returns 21 (x, y, z) points per hand, 30+ times a second.
`signs.js` turns that into a 5-value pattern — is each finger extended or
curled? — and matches it against a small table of known handshapes
(`SIGN_VOCAB`). A 10-frame smoothing buffer requires ~70% agreement before
a sign is accepted, which is what stops flickering/false triggers.

This is intentionally **not** a trained model. It needs zero training data,
runs instantly, and is honest about being a starting point — which mirrors
the "Challenges" slide in the original pitch: start with a small vocabulary
(here, 8 signs) and expand over time. It recognizes static, one-handed
shapes only. Real Indian Sign Language uses motion, two-handed signs, and
facial grammar (non-manual markers) that this version can't see — that gap
is exactly what a CNN/LSTM/Transformer model would close.

**Current vocabulary (8 signs):** Hello, No, Thank you, Wait, Yes,
I love you, Call, Water — shown live as chips under the camera.

## 3. Running it

Camera and microphone access require a **secure context** — `file://` will
usually get silently blocked by the browser. Two easy options:

**Option A — quick local test:**
```bash
cd signspeak-ai
python3 -m http.server 8000
```
Then open `http://localhost:8000` in Chrome.

**Option B — real deployment (recommended for a live demo):**
Push this folder to a GitHub repo and enable **GitHub Pages** (Settings →
Pages → deploy from branch). You'll get a free `https://` URL — ideal for
showing judges, since it works on any laptop with no setup.

No npm install, no bundler. `app.js` imports MediaPipe straight from a CDN
(jsDelivr) as an ES module.

## 4. File map

```
index.html   structure + all UI text (English shown by default)
style.css    visual design — dark "stage" theme, teal = vision, amber = voice
signs.js     ← edit this to add signs. Vocabulary table + the finger-pattern classifier
app.js       camera loop, MediaPipe wiring, speech I/O, history, language toggle
```

### Adding a new sign
Open `signs.js`, hold the handshape up, note which fingers are extended,
add a row to `SIGN_VOCAB`:
```js
{ id: "food", pattern: [true, false, false, false, false], emoji: "🤌", en: "Food", ta: "உணவு" }
```
Patterns must stay unique (checked by the exact-match logic in `matchSign`).

## 5. Roadmap to the full pitch

1. **Trained classifier.** Record short landmark sequences per sign
   (MediaPipe already gives you the numbers — just log them to CSV while
   someone signs). Train an LSTM or small Transformer in Python
   (TensorFlow/PyTorch), export to TensorFlow.js, and swap `matchSign()`
   for a model prediction. This is what turns "8 rule-based handshapes"
   into "50–100+ real ISL signs with motion."
2. **Two-handed and dynamic signs.** `numHands: 2` is already enabled in
   `app.js` — the classifier just isn't using the second hand or motion
   over time yet. A sequence model naturally fixes this.
3. **True offline mode.** Add a service worker to cache the app shell and
   the MediaPipe model file after first load, so the vocabulary keeps
   working with no connection.
4. **AI avatar (speech/text → sign).** The hardest and most novel stretch
   goal from the brief — animating a hand/avatar to sign back. Worth
   scoping separately once the core loop above is solid.

## 6. Known limitations (good to say out loud to judges)

- The classifier assumes the palm faces the camera; sideways or rotated
  hands reduce accuracy.
- Tamil text-to-speech quality depends on the OS/browser's installed
  voices — not every machine has a Tamil voice installed.
- `SpeechRecognition` is a Chrome/Edge API; it isn't supported in Firefox.
- This models a *starter* vocabulary, not linguistically complete ISL —
  intentionally, per the brief's own "start with 50–100 signs" guidance.

## 7. Suggested live-demo flow

1. Open the deployed link, show the model loading, then "Model ready."
2. Sign → Speech: hold up "Hello," let it recognize + speak it out loud.
3. Speech → Text: have a judge speak into the mic, watch the live
   transcript appear in large text.
4. Toggle EN / தமிழ் and repeat a sign — show the spoken output switch
   language.
5. Point at the vocabulary chips and the Roadmap section to address the
   "how do you scale to real ISL" question before it's even asked.
