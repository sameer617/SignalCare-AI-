// session_player.js
// ------------------
// Replays the precomputed per-turn analysis (analysis.json) in sync with
// <video> playback time. Each turn has a start_time/end_time (from Whisper);
// as the video's currentTime passes a turn's start_time, that turn's
// transcript line, sentiment, and distress signals are revealed.

(function () {
    const layout = document.querySelector(".session-layout");
    const sessionId = layout.dataset.sessionId;
    const taxonomy = JSON.parse(layout.dataset.taxonomy);

    const player = document.getElementById("player");
    const transcriptLines = document.getElementById("transcript-lines");
    const sentimentPill = document.getElementById("sentiment-pill");
    const sentimentConfidence = document.getElementById("sentiment-confidence");
    const sentimentTrend = document.getElementById("sentiment-trend");

    const MAX_TREND_BARS = 40;

    let turns = [];
    let revealedCount = 0;

    function severityClass(code) {
        const entry = taxonomy[code];
        const severity = entry ? entry.severity : null;
        return severity ? "severity-" + severity.toLowerCase() : "severity-medium";
    }

    function signalLabel(code) {
        const entry = taxonomy[code];
        return entry ? entry.name : code;
    }

    function signalTooltip(code, confidence) {
        const entry = taxonomy[code];
        const definition = entry ? entry.definition : "";
        return `${code} (confidence ${(confidence * 100).toFixed(0)}%): ${definition}`;
    }

    function renderTurn(turn) {
        const line = document.createElement("div");
        line.className = "transcript-line sentiment-" + turn.sentiment;

        const text = document.createElement("span");
        text.className = "transcript-text";
        text.textContent = turn.text;
        line.appendChild(text);

        if (turn.distress_signals.length > 0) {
            const tags = document.createElement("div");
            tags.className = "distress-tags";
            turn.distress_signals.forEach((code) => {
                const tag = document.createElement("span");
                const shortCode = code.split("_")[0];
                const confidence = turn.distress_signal_confidence[code] || 0;
                tag.className = "distress-tag " + severityClass(shortCode);
                tag.textContent = `${shortCode} — ${signalLabel(shortCode)} (${(confidence * 100).toFixed(0)}%)`;
                tag.title = signalTooltip(shortCode, confidence);
                tags.appendChild(tag);
            });
            line.appendChild(tags);
        }

        transcriptLines.appendChild(line);
        line.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    function updateBadge(code, fired, confidence, contributingCodes) {
        const badge = document.getElementById("badge-" + code);
        if (!badge) return;

        const stateEl = badge.querySelector(".badge-state");
        badge.classList.toggle("fired", !!fired);

        if (fired) {
            const codes = (contributingCodes || []).map((c) => c.split("_")[0]).join(", ");
            const confPct = (confidence * 100).toFixed(0);
            stateEl.textContent = `Detected (${confPct}%)`;
            badge.title = `${signalLabel(code)} - confidence ${confPct}%` +
                (codes ? ` - contributing: ${codes}` : "");
        } else {
            stateEl.textContent = "Not detected";
            badge.title = signalLabel(code);
        }
    }

    function updateSentimentTrend(sentiment) {
        const bar = document.createElement("div");
        bar.className = "sentiment-trend-bar sentiment-" + sentiment;
        const heights = { positive: "100%", neutral: "55%", negative: "30%" };
        bar.style.height = heights[sentiment] || "55%";
        sentimentTrend.appendChild(bar);
        while (sentimentTrend.children.length > MAX_TREND_BARS) {
            sentimentTrend.removeChild(sentimentTrend.firstChild);
        }
    }

    function applyTurn(turn) {
        sentimentPill.textContent = turn.sentiment;
        sentimentPill.className = "sentiment-pill sentiment-" + turn.sentiment;
        sentimentConfidence.textContent =
            `confidence ${(turn.sentiment_confidence * 100).toFixed(0)}%`;
        updateSentimentTrend(turn.sentiment);

        updateBadge("S05", turn.volatility_fired, turn.volatility_confidence);
        updateBadge("S08", turn.drift_fired, turn.drift_confidence);
        updateBadge("S09", turn.rumination_fired, turn.rumination_confidence, turn.rumination_codes);
        updateBadge("S12", turn.escalation_fired, turn.escalation_confidence, turn.escalation_codes);
    }

    function resetSignals() {
        sentimentPill.textContent = "–";
        sentimentPill.className = "sentiment-pill";
        sentimentConfidence.textContent = "";
        sentimentTrend.innerHTML = "";
        ["S05", "S08", "S09", "S12"].forEach((code) => updateBadge(code, false, 0));
    }

    function onTimeUpdate() {
        const currentTime = player.currentTime;
        while (revealedCount < turns.length && turns[revealedCount].start_time <= currentTime) {
            const turn = turns[revealedCount];
            renderTurn(turn);
            applyTurn(turn);
            revealedCount += 1;
        }
    }

    function onSeeked() {
        // Reset and re-reveal everything up to the new position, so seeking
        // backward/forward keeps the transcript + signals consistent.
        transcriptLines.innerHTML = "";
        resetSignals();
        revealedCount = 0;
        onTimeUpdate();
    }

    fetch(`/sessions/${sessionId}/analysis.json`)
        .then((res) => res.json())
        .then((data) => {
            turns = data;
            player.addEventListener("timeupdate", onTimeUpdate);
            player.addEventListener("seeked", onSeeked);
        });
})();
