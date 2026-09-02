(() => {
  const redactBtn = document.getElementById("nav-redact");
  const attendBtn = document.getElementById("nav-attendance");
  const redact = document.getElementById("view-redact");
  const attend = document.getElementById("view-attendance");
  if (!redactBtn || !attendBtn || !redact || !attend) return;

  function show(mode) {
    const attendance = mode === "attendance";
    redact.hidden = attendance;
    attend.hidden = !attendance;
    redactBtn.classList.toggle("active", !attendance);
    attendBtn.classList.toggle("active", attendance);
    try {
      history.replaceState(null, "", attendance ? "#attendance" : "#redact");
    } catch {
      /* ignore */
    }
  }

  redactBtn.addEventListener("click", () => show("redact"));
  attendBtn.addEventListener("click", () => show("attendance"));
  show(location.hash.replace("#", "") === "attendance" ? "attendance" : "redact");
})();
