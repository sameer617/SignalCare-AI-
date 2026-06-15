// session_chat.js
// ----------------
// "Ask about this session" chatbot widget. Loads prior chat history for the
// session, lets the counselor ask follow-up questions about the transcript
// and detected signals, and persists the conversation via
// /sessions/{id}/chat.

(function () {
    const layout = document.querySelector(".session-layout");
    const sessionId = layout.dataset.sessionId;

    const toggleBtn = document.getElementById("chat-toggle");
    const closeBtn = document.getElementById("chat-close");
    const panel = document.getElementById("chat-panel");
    const messagesEl = document.getElementById("chat-messages");
    const form = document.getElementById("chat-form");
    const input = document.getElementById("chat-input");

    function appendMessage(role, content) {
        const bubble = document.createElement("div");
        bubble.className = "chat-message chat-" + role;
        bubble.textContent = content;
        messagesEl.appendChild(bubble);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return bubble;
    }

    function setFormDisabled(disabled) {
        input.disabled = disabled;
        form.querySelector("button").disabled = disabled;
    }

    toggleBtn.addEventListener("click", () => {
        panel.hidden = !panel.hidden;
        if (!panel.hidden) {
            input.focus();
        }
    });

    closeBtn.addEventListener("click", () => {
        panel.hidden = true;
    });

    form.addEventListener("submit", (event) => {
        event.preventDefault();
        const question = input.value.trim();
        if (!question) return;

        appendMessage("user", question);
        input.value = "";
        setFormDisabled(true);
        const thinking = appendMessage("assistant", "Thinking...");

        fetch(`/sessions/${sessionId}/chat`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: question }),
        })
            .then((res) => {
                if (!res.ok) throw new Error("Request failed");
                return res.json();
            })
            .then((data) => {
                thinking.textContent = data.reply;
            })
            .catch(() => {
                thinking.textContent = "Sorry, something went wrong. Please try again.";
            })
            .finally(() => {
                setFormDisabled(false);
                messagesEl.scrollTop = messagesEl.scrollHeight;
                input.focus();
            });
    });

    fetch(`/sessions/${sessionId}/chat`)
        .then((res) => res.json())
        .then((data) => {
            (data.messages || []).forEach((m) => appendMessage(m.role, m.content));
        });
})();
