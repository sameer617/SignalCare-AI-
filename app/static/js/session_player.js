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
    const sentimentValue = document.getElementById("sentiment-value");

    let turns = [];
    let revealedCount = 0;

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
                tag.className = "distress-tag";
                const shortCode = code.split("_")[0];
                const confidence = turn.distress_signal_confidence[code] || 0;
                tag.textContent = `${shortCode} (${(confidence * 100).toFixed(0)}%)`;
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

        badge.classList.toggle("fired", !!fired);

        if (fired) {
            const codes = (contributingCodes || []).map((c) => c.split("_")[0]).join(", ");
            const confPct = (confidence * 100).toFixed(0);
            badge.title = `${signalLabel(code)} - confidence ${confPct}%` +
                (codes ? ` - contributing: ${codes}` : "");
        } else {
            badge.title = signalLabel(code);
        }
    }

    function applyTurn(turn) {
        sentimentValue.textContent =
            `${turn.sentiment} (confidence ${(turn.sentiment_confidence * 100).toFixed(0)}%)`;

        updateBadge("S05", turn.volatility_fired, turn.volatility_confidence);
        updateBadge("S08", turn.drift_fired, turn.drift_confidence);
        updateBadge("S09", turn.rumination_fired, turn.rumination_confidence, turn.rumination_codes);
        updateBadge("S12", turn.escalation_fired, turn.escalation_confidence, turn.escalation_codes);
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
