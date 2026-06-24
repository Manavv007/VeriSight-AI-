/* =================================================================
   interview.js — AI interview simulation
   - Pre-populated question bank (technical + behavioral)
   - Text-to-Speech (SpeechSynthesis) to read questions aloud
   - Speech-to-Text (webkitSpeechRecognition) to dictate answers
   - Question navigation (Prev / Next / Finish)
   Loose coupling: dispatches `interview:complete` on finish.
   Exposes window.Interview for inspection / report export.
   ================================================================= */
(function () {
  "use strict";

  /** @typedef {{ category: string, text: string }} Question */

  /** @type {Question[]} */
  const QUESTIONS = [
    { category: "Warm-up", text: "Tell me about yourself and walk me through your background." },
    { category: "Behavioral", text: "Describe a challenging project you worked on. What made it difficult and how did you handle it?" },
    { category: "Technical — JavaScript", text: "Explain the difference between the call stack and the event loop in JavaScript. How are asynchronous callbacks scheduled?" },
    { category: "Technical — Web", text: "What happens, step by step, when a user types a URL into the browser and presses Enter?" },
    { category: "System Design", text: "How would you design a URL shortening service like bit.ly? Talk through the data model and how you would scale reads." },
    { category: "Behavioral", text: "Tell me about a time you disagreed with a teammate. How did you reach a resolution?" },
    { category: "Technical — Data", text: "When would you choose a SQL database over a NoSQL database, and what trade-offs are you accepting?" },
    { category: "Reflection", text: "Where do you see the biggest area for your own technical growth, and what are you doing about it?" },
  ];

  const state = {
    index: 0,
    responses: QUESTIONS.map((q) => ({ category: q.category, question: q.text, answer: "" })),
    recognizing: false,
    finished: false,
  };

  // ---- DOM refs ----
  const el = {};
  function cacheDom() {
    el.counter = document.getElementById("question-counter");
    el.category = document.getElementById("question-category");
    el.text = document.getElementById("question-text");
    el.answer = document.getElementById("answer-input");
    el.recordStatus = document.getElementById("record-status");
    el.btnSpeak = document.getElementById("btn-speak");
    el.btnRecord = document.getElementById("btn-record");
    el.btnPrev = document.getElementById("btn-prev");
    el.btnNext = document.getElementById("btn-next");
  }

  // =====================================================
  // Text-to-Speech
  // =====================================================
  const tts = {
    supported: typeof window.speechSynthesis !== "undefined",
    voice: null,
    pickVoice() {
      if (!this.supported) return;
      const voices = window.speechSynthesis.getVoices();
      // Prefer a natural English voice when available.
      this.voice =
        voices.find((v) => /en(-|_)?US/i.test(v.lang) && /Google|Natural|Samantha|Zira/i.test(v.name)) ||
        voices.find((v) => /^en/i.test(v.lang)) ||
        voices[0] ||
        null;
    },
    speak(textToSpeak) {
      if (!this.supported) return;
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(textToSpeak);
      if (this.voice) u.voice = this.voice;
      u.rate = 1.0;
      u.pitch = 1.0;
      u.lang = (this.voice && this.voice.lang) || "en-US";
      window.speechSynthesis.speak(u);
    },
    stop() {
      if (this.supported) window.speechSynthesis.cancel();
    },
  };
  if (tts.supported) {
    tts.pickVoice();
    // Voices load asynchronously in most browsers.
    window.speechSynthesis.onvoiceschanged = () => tts.pickVoice();
  }

  // =====================================================
  // Speech-to-Text
  // =====================================================
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const stt = {
    supported: !!SpeechRecognition,
    recognition: null,
    baseText: "",
    init() {
      if (!this.supported) return;
      const r = new SpeechRecognition();
      r.continuous = true;
      r.interimResults = true;
      r.lang = "en-US";

      r.onresult = (event) => {
        let finalText = "";
        let interim = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) finalText += transcript;
          else interim += transcript;
        }
        if (finalText) {
          this.baseText = (this.baseText + " " + finalText).trim();
        }
        const composed = (this.baseText + (interim ? " " + interim : "")).trim();
        el.answer.value = composed;
        saveCurrentAnswer();
      };

      r.onerror = (event) => {
        if (event.error === "not-allowed" || event.error === "service-not-allowed") {
          setRecordStatus("Mic blocked", false);
          stopRecording();
        } else if (event.error === "no-speech") {
          setRecordStatus("No speech…", true);
        }
      };

      r.onend = () => {
        // Browser may auto-stop; restart if the user is still in recording mode.
        if (state.recognizing) {
          try { r.start(); } catch (_) { /* already started */ }
        }
      };

      this.recognition = r;
    },
    start() {
      if (!this.supported) return;
      if (!this.recognition) this.init();
      this.baseText = el.answer.value.trim();
      try { this.recognition.start(); } catch (_) { /* ignore double start */ }
    },
    stop() {
      if (this.supported && this.recognition) {
        try { this.recognition.stop(); } catch (_) {}
      }
    },
  };

  // =====================================================
  // Rendering & navigation
  // =====================================================
  function render() {
    const q = QUESTIONS[state.index];
    el.category.textContent = q.category;
    el.text.textContent = q.text;
    el.counter.textContent = `Question ${state.index + 1} / ${QUESTIONS.length}`;
    el.answer.value = state.responses[state.index].answer || "";
    el.btnPrev.disabled = state.index === 0;
    el.btnNext.textContent = state.index === QUESTIONS.length - 1 ? "Finish ✓" : "Next ›";
  }

  function saveCurrentAnswer() {
    state.responses[state.index].answer = el.answer.value;
  }

  function goTo(index) {
    saveCurrentAnswer();
    if (state.recognizing) stopRecording();
    tts.stop();
    state.index = Math.max(0, Math.min(QUESTIONS.length - 1, index));
    render();
  }

  function next() {
    if (state.index === QUESTIONS.length - 1) {
      finish();
    } else {
      goTo(state.index + 1);
    }
  }

  function finish() {
    if (state.finished) return;
    saveCurrentAnswer();
    if (state.recognizing) stopRecording();
    tts.stop();
    state.finished = true;
    document.dispatchEvent(
      new CustomEvent("interview:complete", {
        detail: { responses: state.responses.slice() },
      })
    );
  }

  // =====================================================
  // Recording UI helpers
  // =====================================================
  function setRecordStatus(label, live) {
    el.recordStatus.textContent = label;
    el.recordStatus.classList.toggle("is-live", !!live);
  }

  function startRecording() {
    if (!stt.supported) {
      setRecordStatus("STT unavailable", false);
      return;
    }
    state.recognizing = true;
    stt.start();
    el.btnRecord.classList.add("is-recording");
    el.btnRecord.textContent = "⏹ Stop Recording";
    setRecordStatus("● Listening…", true);
  }

  function stopRecording() {
    state.recognizing = false;
    stt.stop();
    el.btnRecord.classList.remove("is-recording");
    el.btnRecord.textContent = "🎙️ Record Answer";
    setRecordStatus("Idle", false);
    saveCurrentAnswer();
  }

  function toggleRecording() {
    if (state.recognizing) stopRecording();
    else startRecording();
  }

  // =====================================================
  // Wiring
  // =====================================================
  function bind() {
    el.btnSpeak.addEventListener("click", () => {
      if (!tts.supported) {
        el.btnSpeak.textContent = "🔇 TTS unavailable";
        el.btnSpeak.disabled = true;
        return;
      }
      tts.speak(QUESTIONS[state.index].text);
    });

    el.btnRecord.addEventListener("click", toggleRecording);
    el.btnPrev.addEventListener("click", () => goTo(state.index - 1));
    el.btnNext.addEventListener("click", next);
    el.answer.addEventListener("input", saveCurrentAnswer);

    // Graceful degradation labels.
    if (!stt.supported) {
      el.btnRecord.title = "Speech-to-Text needs Chrome, Edge, or Safari.";
      el.btnRecord.textContent = "🎙️ Record (unsupported)";
    }
    if (!tts.supported) {
      el.btnSpeak.title = "Text-to-Speech is not supported in this browser.";
    }

    // When proctoring calibration completes, read the first question aloud.
    document.addEventListener("gazeproctor:ready", () => {
      if (tts.supported) setTimeout(() => tts.speak(QUESTIONS[state.index].text), 400);
    });
  }

  function init() {
    cacheDom();
    if (!el.text) return; // DOM not present
    stt.init();
    bind();
    render();
  }

  // Public API for the report exporter and debugging.
  window.Interview = {
    getResponses: () => state.responses.slice(),
    getProgress: () => ({
      index: state.index,
      total: QUESTIONS.length,
      answered: state.responses.filter((r) => r.answer && r.answer.trim().length > 0).length,
      finished: state.finished,
    }),
    finish,
    speakCurrent: () => tts.speak(QUESTIONS[state.index].text),
    isTtsSupported: () => tts.supported,
    isSttSupported: () => stt.supported,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
