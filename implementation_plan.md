# AI Interview Proctoring System with EyeGazer.js

This project is a modern, premium AI mock interview and proctoring application. It integrates Brown HCI's **WebGazer.js** to calibrate the candidate's eyes using a 9-point calibration system and subsequently track their gaze. If the candidate looks outside the laptop screen (browser window bounds) or switches tabs, the application flags the event with a high-resolution timestamp and logs it in a proctoring audit trail.

---

## User Review Required

> [!IMPORTANT]
> **Webcam Permissions**: The application requires webcam access. On launch, the browser will request permission to capture video, which runs entirely in the client-side browser to protect user privacy.
>
> **Fullscreen Mode Recommended**: To track if the user looks outside the *physical laptop screen*, we will prompt the user to enter Fullscreen mode upon starting the interview. This aligns the browser viewport bounds with the screen boundaries.

---

## Proposed Changes

We will create a new self-contained project folder under `C:/Users/Manav/Downloads/eyeGazer/ai-interview`.

### Directory Structure
```
ai-interview/
├── index.html          # Main HTML5 document (Layout, Calibration UI, Dashboard)
├── css/
│   └── style.css       # Premium, dark-themed styling (glassmorphism, alerts, animations)
├── js/
│   ├── webgazer.js     # WebGazer library (copied from WebGazer submodule for self-containment)
│   ├── interview.js    # AI Interview logic (questions, text-to-speech, recording simulator)
│   └── proctor.js      # Gaze tracking, calibration, out-of-bounds flagging, and tab-focus hooks
```

---

### [Component: UI and Layout]

#### [NEW] [index.html](file:///C:/Users/Manav/Downloads/eyeGazer/ai-interview/index.html)
The primary entry point. It will contain:
- A glassmorphic calibration overlay (hidden once calibrated) displaying the 9-point calibration system.
- An interactive AI interview simulation panel displaying questions (with Text-to-Speech support) and an input textarea with Speech-to-Text capabilities.
- A live webcam container styling WebGazer's video preview.
- A proctoring dashboard consisting of:
  - Real-time status cards (current focus state, total flags count, tab switch count).
  - A real-time incident timeline showing proctoring violations.
  - A "Download Report" button to export logs as JSON/CSV.

#### [NEW] [style.css](file:///C:/Users/Manav/Downloads/eyeGazer/ai-interview/css/style.css)
- Implement a premium dark dashboard aesthetic with neon teal/indigo/rose highlights.
- Add styles for the `#webgazerVideoContainer` to cleanly clip and place the webcam inside the dashboard's camera feed card rather than floating arbitrarily.
- Define animations for alert states (e.g., flashing red borders when looking away).
- Configure responsive layout grids using CSS Flexbox and Grid.

---

### [Component: Proctoring and Calibration Logic]

#### [NEW] [proctor.js](file:///C:/Users/Manav/Downloads/eyeGazer/ai-interview/js/proctor.js)
Handles the core integration with WebGazer:
1. **Calibration Controller**:
   - Manages the state of the 9 calibration points (requiring 5 clicks per point).
   - Displays a 5-second precision test in the center to calculate calibration accuracy.
2. **Out of Bounds Detection**:
   - Hooks into `webgazer.setGazeListener()`.
   - Compares the gaze coordinates `(x, y)` against the viewport bounds `[0, window.innerWidth]` and `[0, window.innerHeight]`.
   - **Noise-filtering**: Employs a time-window threshold. A brief glance outside (e.g., <500ms) will show a visual warning, but will not write a permanent flag. If they look away for >= 1.0 second, a flag is added to the log.
3. **Face-Lost Tracking**:
   - Detects when `data === null` (WebGazer loses face registration). If lost for >= 1.5 seconds, logs a "Face Lost" proctoring incident.
4. **Tab Switch Tracking**:
   - Monitors `visibilitychange` and window `blur` events. Instantly logs an incident if the user navigates away or blurs the browser window.
5. **Auditing & Export**:
   - Stores logs in memory with structures: `{ timestamp: "18:45:30", type: "Gaze Out of Bounds", duration: "2.5s", details: "Looked to the left of the screen" }`.
   - Implements JSON/CSV file download functionality.

#### [NEW] [interview.js](file:///C:/Users/Manav/Downloads/eyeGazer/ai-interview/js/interview.js)
Simulates a real-world AI interview:
- Pre-populated set of technical/behavioral interview questions.
- **Text-to-Speech (TTS)**: Reads the question out loud using the browser's `SpeechSynthesis` API.
- **Speech-to-Text (STT)**: Integrates the browser's `webkitSpeechRecognition` to dictate the candidate's answers in real-time when clicking the "Record Answer" button.
- Manages interview progression (Next/Prev Question, Finish Interview).

---

### [Dependency Management]

We will copy `webgazer.js` (and its `.map` file) from `WebGazer/www/` to `ai-interview/js/` to make our application fully local and self-contained, avoiding external CDN dependency issues.

---

## Verification Plan

### Automated Verification
- We can serve the project locally using Python's built-in HTTP server or Node `http-server` and verify code linting.
- Check browser console logs for any TensorFlow or WebGazer initialization warnings.

### Manual Verification
1. **Calibration Flow**: Click all 9 points 5 times and check if the opacity increases and color turns yellow, followed by the 5-second accuracy check.
2. **Gaze Tracking Bounds**: Look outside the screen (left, right, up, down) and verify:
   - Visual red warning border flashes around the camera feed.
   - An incident is appended to the proctoring log after 1 second of looking away.
3. **Face-Lost Behavior**: Cover the webcam or look completely away, verifying that "Face Lost" is logged after 1.5 seconds.
4. **Tab-Switching Behavior**: Open a new tab or click outside the browser window, verifying that a "Focus Lost / Tab Switch" incident is immediately logged.
5. **Interview Flow**: Test question cycling, Text-to-Speech (reading questions), and Speech-to-Text dictation.
6. **Report Export**: Click "Download Report" and inspect the downloaded CSV/JSON file structure.
