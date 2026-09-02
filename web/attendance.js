const KEY = "fra.web.v1";
const DEMO_PINS = { R1: "1234", R2: "2468", R3: "13579" };

function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function today() { return new Date().toISOString().slice(0, 10); }
function monday(dateStr) {
  const d = new Date(dateStr + "T00:00:00");
  d.setDate(d.getDate() - ((d.getDay() + 6) % 7));
  return d.toISOString().slice(0, 10);
}
function addDays(dateStr, n) {
  const d = new Date(dateStr + "T00:00:00");
  d.setDate(d.getDate() + n);
  return d.toISOString().slice(0, 10);
}
function isoWeekday(dateStr) {
  const day = new Date(dateStr + "T00:00:00").getDay();
  return day === 0 ? 7 : day;
}
function isSchoolDay(db, dateStr) {
  const holidays = new Set((db.holidays || []).map((h) => h.date));
  if (holidays.has(dateStr)) return false;
  const weekends = Array.isArray(db.weekendDays) && db.weekendDays.length
    ? db.weekendDays
    : [6, 7];
  return !weekends.includes(isoWeekday(dateStr));
}
function initials(name) {
  return (name || "?")
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => part[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();
}

function load() {
  const raw = localStorage.getItem(KEY);
  if (raw) {
    const db = JSON.parse(raw);
    migrate(db);
    return db;
  }
  return seed();
}
function save(db) { localStorage.setItem(KEY, JSON.stringify(db)); }

function migrate(db) {
  (db.students || []).forEach((s) => {
    if (s.pin === undefined || s.pin === "") {
      if (DEMO_PINS[s.roll]) s.pin = DEMO_PINS[s.roll];
      else if (s.pin === undefined) s.pin = "";
    }
  });
  if (!Array.isArray(db.weekendDays) || !db.weekendDays.length) {
    db.weekendDays = [6, 7];
  }
}

function seed() {
  const db = {
    students: [
      { id: 1, roll: "R1", name: "Ada Lovelace", section: "A", email: "ada@school.edu", active: true, pin: "1234" },
      { id: 2, roll: "R2", name: "Grace Hopper", section: "A", email: "grace@school.edu", active: true, pin: "2468" },
      { id: 3, roll: "R3", name: "Alan Turing", section: "B", email: "alan@school.edu", active: true, pin: "13579" },
    ],
    attendance: [],
    holidays: [{ date: "2026-08-19", name: "Founders Day" }],
    weekendDays: [6, 7],
    nextId: 4,
    theme: "dark",
  };
  const start = monday(today());
  db.students.forEach((s, i) => {
    for (let d = 0; d < 5; d += 1) {
      if (s.roll === "R3" && d > 1) continue;
      db.attendance.push({
        id: db.attendance.length + 1,
        student_id: s.id,
        date: addDays(start, d),
        time: "09:0" + d + ":00",
        status: i === 1 && d === 1 ? "Late" : "Present",
        period: "Morning",
        source: "cli",
      });
    }
  });
  const asOf = today();
  function markIfNeeded(student, date) {
    if (db.attendance.some((a) => a.student_id === student.id && a.date === date)) return;
    db.attendance.push({
      id: db.attendance.length + 1,
      student_id: student.id,
      date,
      time: "09:00:00",
      status: "Present",
      period: "Morning",
      source: "cli",
    });
  }
  function fillStreak(student, n) {
    let d = asOf;
    let count = 0;
    while (count < n) {
      if (isSchoolDay(db, d)) {
        markIfNeeded(student, d);
        count += 1;
      }
      d = addDays(d, -1);
    }
  }
  fillStreak(db.students[0], 3);
  fillStreak(db.students[1], 2);
  save(db);
  return db;
}

const $ = (id) => document.getElementById(id);

function table(headers, rows, empty = "None") {
  if (!rows.length) return `<div class="card"><p class="hint">${empty}</p></div>`;
  return `<table><thead><tr>${headers.map((h) => `<th>${h}</th>`).join("")}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
}

function activeStudents(db) { return db.students.filter((s) => s.active); }

function classify(status, time, period) {
  if (status) return status;
  const start = period === "Afternoon" ? 13 : 9;
  const [hh, mm] = (time || "09:00").split(":").map(Number);
  const minutes = hh * 60 + mm;
  return minutes <= start * 60 + 10 ? "Present" : "Late";
}

function digest(db, date) {
  const students = activeStudents(db);
  const marks = db.attendance.filter((a) => a.date === date);
  const first = new Map();
  marks.slice().sort((a, b) => a.time.localeCompare(b.time)).forEach((a) => {
    if (!first.has(a.student_id)) first.set(a.student_id, a);
  });
  let present = 0, late = 0, excused = 0;
  first.forEach((a) => {
    if (a.status === "Late") { present += 1; late += 1; }
    else if (a.status === "Excused") excused += 1;
    else present += 1;
  });
  const absent = students.length - present - excused;
  return { present, late, excused, absent, total: students.length, rate: students.length ? (present / students.length) * 100 : 0 };
}

function weekly(db, start) {
  const days = [];
  for (let i = 0; i < 7; i += 1) {
    const day = addDays(start, i);
    if (isSchoolDay(db, day)) days.push(day);
  }
  return activeStudents(db).map((s) => {
    let present = 0, late = 0, excused = 0;
    days.forEach((day) => {
      const mark = db.attendance.filter((a) => a.student_id === s.id && a.date === day).sort((a, b) => a.time.localeCompare(b.time))[0];
      if (!mark) return;
      if (mark.status === "Excused") excused += 1;
      else if (mark.status === "Late") late += 1;
      else present += 1;
    });
    const absent = days.length - present - late - excused;
    const rate = days.length ? ((present + late) / days.length) * 100 : 0;
    return { ...s, present, late, excused, absent, days: days.length, rate };
  });
}

function firstMark(db, studentId, date) {
  return db.attendance
    .filter((a) => a.student_id === studentId && a.date === date)
    .sort((a, b) => a.time.localeCompare(b.time))[0];
}

function presentStreak(db, student, asOf) {
  let streak = 0;
  let d = asOf;
  for (let i = 0; i < 800; i += 1) {
    if (!isSchoolDay(db, d)) {
      d = addDays(d, -1);
      continue;
    }
    const mark = firstMark(db, student.id, d);
    if (mark && (mark.status === "Excused" || mark.status === "Sick")) {
      d = addDays(d, -1);
      continue;
    }
    if (mark && (mark.status === "Present" || mark.status === "Late")) {
      streak += 1;
      d = addDays(d, -1);
      continue;
    }
    break;
  }
  return streak;
}

function icsEscape(value) {
  return String(value || "")
    .replace(/\\/g, "\\\\")
    .replace(/;/g, "\\;")
    .replace(/,/g, "\\,")
    .replace(/\r\n/g, "\\n")
    .replace(/\n/g, "\\n");
}

function buildAttendanceIcs(db, start, end) {
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}/, "");
  const lines = [
    "BEGIN:VCALENDAR",
    "VERSION:2.0",
    "PRODID:-//Sebby1770//Face Recognition Attendance//EN",
    "CALSCALE:GREGORIAN",
    "METHOD:PUBLISH",
  ];
  activeStudents(db).forEach((s) => {
    for (let d = start; d <= end; d = addDays(d, 1)) {
      const mark = firstMark(db, s.id, d);
      if (!mark || !["Present", "Late", "Excused"].includes(mark.status)) continue;
      const compact = d.replace(/-/g, "");
      lines.push("BEGIN:VEVENT");
      lines.push(`UID:${s.roll}-${d}@face-recognition-attendance`);
      lines.push(`DTSTAMP:${stamp}`);
      lines.push(`DTSTART;VALUE=DATE:${compact}`);
      lines.push(`SUMMARY:${icsEscape(`${s.name} (${s.roll}) ${mark.status}`)}`);
      if (mark.note) lines.push(`DESCRIPTION:${icsEscape(mark.note)}`);
      lines.push("END:VEVENT");
    }
  });
  lines.push("END:VCALENDAR");
  return lines.join("\r\n") + "\r\n";
}

function renderDash() {
  const db = load();
  const d = digest(db, today());
  const streaks = activeStudents(db).map((s) => presentStreak(db, s, today()));
  const longest = streaks.length ? Math.max(...streaks) : 0;
  $("kpis").innerHTML = [
    ["Students", activeStudents(db).length],
    ["Present today", d.present],
    ["Late", d.late],
    ["Excused", d.excused],
    ["Absent", d.absent],
    ["Rate", d.rate.toFixed(0) + "%"],
    ["Longest streak", longest],
  ].map(([k, v]) => `<div class="kpi"><strong>${v}</strong><span>${k}</span></div>`).join("");
  const rows = db.attendance.filter((a) => a.date === today()).map((a) => {
    const s = db.students.find((x) => x.id === a.student_id);
    return `<tr><td>${escapeHtml(s ? s.roll : "?")}</td><td>${escapeHtml(s ? s.name : "?")}</td><td>${escapeHtml(a.time)}</td><td><span class="pill ${escapeHtml(a.status)}">${escapeHtml(a.status)}</span></td><td>${escapeHtml(a.period || "")}</td></tr>`;
  });
  $("today-table").innerHTML = table(["Roll", "Name", "Time", "Status", "Period"], rows, "No marks yet today.");
}

function renderStudents() {
  const db = load();
  const stats = weekly(db, monday(today()));
  const rateById = new Map(stats.map((s) => [s.id, s.rate]));
  $("student-cards").innerHTML = db.students.map((s) => {
    const rate = rateById.has(s.id) ? `${rateById.get(s.id).toFixed(0)}%` : (s.active ? "0%" : "archived");
    return `<article class="student-card">
      <div class="avatar" aria-hidden="true">${escapeHtml(initials(s.name))}</div>
      <div>
        <strong>${escapeHtml(s.name)}</strong>
        <div class="meta">${escapeHtml(s.roll)} · ${escapeHtml(s.section || "—")} · ${escapeHtml(rate)}</div>
      </div>
    </article>`;
  }).join("");
  $("student-table").innerHTML = table(
    ["Roll", "Name", "Section", "Streak", "Email", "PIN", "Active", ""],
    db.students.map((s) => `<tr>
      <td>${escapeHtml(s.roll)}</td><td>${escapeHtml(s.name)}</td><td>${escapeHtml(s.section || "—")}</td>
      <td>${s.active ? presentStreak(db, s, today()) : "—"}</td>
      <td>${escapeHtml(s.email || "—")}</td>
      <td>${s.pin ? "set (demo)" : "—"}</td>
      <td>${s.active ? "yes" : "archived"}</td>
      <td>
        <button class="btn ghost" data-archive="${escapeHtml(s.roll)}">${s.active ? "Archive" : "Restore"}</button>
        <button class="btn ghost" data-set-pin="${escapeHtml(s.roll)}">Set PIN</button>
      </td>
    </tr>`),
  );
  $("mark-roll").innerHTML = activeStudents(db).map((s) => `<option value="${escapeHtml(s.roll)}">${escapeHtml(s.roll)} — ${escapeHtml(s.name)}</option>`).join("");
}

function renderRecords() {
  const db = load();
  const date = $("rec-date").value || today();
  $("rec-date").value = date;
  const rows = db.attendance.filter((a) => a.date === date).map((a) => {
    const s = db.students.find((x) => x.id === a.student_id);
    return `<tr><td>${escapeHtml(s ? s.roll : "?")}</td><td>${escapeHtml(s ? s.name : "?")}</td><td>${escapeHtml(a.time)}</td><td><span class="pill ${escapeHtml(a.status)}">${escapeHtml(a.status)}</span></td><td>${escapeHtml(a.period || "")}</td></tr>`;
  });
  $("rec-table").innerHTML = table(["Roll", "Name", "Time", "Status", "Period"], rows);
}

function renderWeekly() {
  const db = load();
  const start = $("week-start").value || monday(today());
  $("week-start").value = start;
  const rows = weekly(db, start).map((s) =>
    `<tr><td>${escapeHtml(s.roll)}</td><td>${escapeHtml(s.name)}</td><td>${s.present}</td><td>${s.late}</td><td>${s.excused}</td><td>${s.absent}</td><td>${s.rate.toFixed(0)}%</td></tr>`
  );
  $("week-table").innerHTML = table(["Roll", "Name", "Present", "Late", "Excused", "Absent", "Rate"], rows);
}

function renderHolidays() {
  const db = load();
  $("holiday-table").innerHTML = table(
    ["Date", "Name", ""],
    db.holidays.map((h) => `<tr><td>${escapeHtml(h.date)}</td><td>${escapeHtml(h.name)}</td><td><button class="btn ghost" data-del-h="${escapeHtml(h.date)}">Remove</button></td></tr>`),
    "No holidays.",
  );
}

function renderAlerts() {
  const db = load();
  const start = monday(today());
  const stats = weekly(db, start);
  const risk = stats.filter((s) => s.rate < 75);
  $("alert-box").innerHTML = table(
    ["Roll", "Name", "Rate", "Absent"],
    risk.map((s) => `<tr><td>${escapeHtml(s.roll)}</td><td>${escapeHtml(s.name)}</td><td>${s.rate.toFixed(0)}%</td><td>${s.absent}</td></tr>`),
    "Nobody is under 75% this week.",
  );
}

function renderKiosk() {
  $("kiosk-pin-display").textContent = kioskPin || "••••";
}

function renderCalendar() {
  const db = load();
  const start = $("cal-from").value || monday(today());
  const end = $("cal-to").value || addDays(monday(today()), 6);
  $("cal-from").value = start;
  $("cal-to").value = end;
  if (end < start) {
    $("cal-grid").innerHTML = `<div class="card"><p class="hint">End date must be on or after start date.</p></div>`;
    return;
  }
  const dates = [];
  for (let d = start; d <= end; d = addDays(d, 1)) dates.push(d);
  if (dates.length > 93) {
    $("cal-grid").innerHTML = `<div class="card"><p class="hint">Range cannot exceed 93 days.</p></div>`;
    return;
  }
  const holidayMap = new Map(db.holidays.map((h) => [h.date, h.name]));
  const weekendSet = new Set(
    Array.isArray(db.weekends) && db.weekends.length
      ? db.weekends
      : dates.filter((day) => {
          const weekends = Array.isArray(db.weekendDays) && db.weekendDays.length
            ? db.weekendDays
            : [6, 7];
          return weekends.includes(isoWeekday(day));
        }),
  );
  const dateRow = `<div class="heat-dates"><div class="heat-label"></div><div class="heat-cells">${
    dates.map((d) => `<span title="${escapeHtml(d)}">${escapeHtml(d.slice(8))}</span>`).join("")
  }</div></div>`;
  const rows = activeStudents(db).map((s) => {
    const cells = dates.map((day) => {
      if (holidayMap.has(day)) {
        return `<div class="heat Holiday" title="${escapeHtml(day)} ${escapeHtml(holidayMap.get(day))}"></div>`;
      }
      if (weekendSet.has(day)) {
        return `<div class="heat Weekend" title="${escapeHtml(day)} Weekend"></div>`;
      }
      const mark = firstMark(db, s.id, day);
      const status = mark ? mark.status : "Absent";
      const cls = mark ? mark.status : "";
      return `<div class="heat ${escapeHtml(cls)}" title="${escapeHtml(day)} ${escapeHtml(status)}"></div>`;
    }).join("");
    return `<div class="heat-row"><div class="heat-label">${escapeHtml(s.name)}<div class="meta">${escapeHtml(s.roll)}</div></div><div class="heat-cells">${cells}</div></div>`;
  }).join("");
  $("cal-grid").innerHTML = `<div class="card heatmap">${dateRow}${rows || "<p class='hint'>No students.</p>"}</div>`;
}

function show(view) {
  document.querySelectorAll(".view").forEach((el) => { el.hidden = el.id !== `view-${view}`; });
  document.querySelectorAll(".sidebar nav button").forEach((btn) => btn.classList.toggle("active", btn.dataset.view === view));
  const titles = {
    dash: "Dashboard",
    students: "Students",
    kiosk: "Kiosk",
    calendar: "Calendar",
    mark: "Mark attendance",
    records: "Records",
    weekly: "Weekly stats",
    holidays: "Holidays",
    alerts: "Alerts",
  };
  $("title").textContent = titles[view];
  ({
    dash: renderDash,
    students: renderStudents,
    kiosk: renderKiosk,
    calendar: renderCalendar,
    mark: renderStudents,
    records: renderRecords,
    weekly: renderWeekly,
    holidays: renderHolidays,
    alerts: renderAlerts,
  })[view]();
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  $("theme").value = theme;
}

let kioskPin = "";

document.querySelectorAll(".sidebar nav button").forEach((btn) => btn.addEventListener("click", () => show(btn.dataset.view)));
$("theme").addEventListener("change", (event) => {
  const db = load();
  db.theme = event.target.value;
  save(db);
  applyTheme(db.theme);
});
$("form-student").addEventListener("submit", (event) => {
  event.preventDefault();
  const db = load();
  const data = Object.fromEntries(new FormData(event.target));
  if (db.students.some((s) => s.roll === data.roll.trim())) return alert("Roll already exists.");
  db.students.push({
    id: db.nextId++,
    roll: data.roll.trim(),
    name: data.name.trim(),
    section: data.section.trim(),
    email: data.email.trim(),
    active: true,
    pin: "",
  });
  save(db);
  event.target.reset();
  renderStudents();
});
$("student-table").addEventListener("click", (event) => {
  const archiveRoll = event.target.dataset.archive;
  if (archiveRoll) {
    const db = load();
    const s = db.students.find((x) => x.roll === archiveRoll);
    if (s) s.active = !s.active;
    save(db);
    renderStudents();
    return;
  }
  const pinRoll = event.target.dataset.setPin;
  if (!pinRoll) return;
  const entered = window.prompt(
    `Set a 4–8 digit PIN for ${pinRoll}.\nDemo only: stored in localStorage, not hashed.`,
  );
  if (entered == null) return;
  const pin = entered.trim();
  if (!/^\d{4,8}$/.test(pin)) {
    alert("PIN must be 4–8 digits.");
    return;
  }
  const db = load();
  const s = db.students.find((x) => x.roll === pinRoll);
  if (s) s.pin = pin;
  save(db);
  renderStudents();
});
$("form-mark").addEventListener("submit", (event) => {
  event.preventDefault();
  const db = load();
  const data = Object.fromEntries(new FormData(event.target));
  const student = db.students.find((s) => s.roll === data.roll && s.active);
  if (!student) return;
  const date = data.date || today();
  const now = new Date();
  const time = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:00`;
  db.attendance.push({
    id: db.attendance.length + 1,
    student_id: student.id,
    date,
    time,
    status: classify(data.status, time, data.period),
    period: data.period,
    source: "gui",
    note: (data.note || "").trim(),
  });
  save(db);
  $("mark-msg").textContent = `Marked ${student.name}.`;
  renderDash();
});
$("form-holiday").addEventListener("submit", (event) => {
  event.preventDefault();
  const db = load();
  const data = Object.fromEntries(new FormData(event.target));
  db.holidays = db.holidays.filter((h) => h.date !== data.date);
  db.holidays.push({ date: data.date, name: data.name || "Holiday" });
  save(db);
  renderHolidays();
});
$("holiday-table").addEventListener("click", (event) => {
  const date = event.target.dataset.delH;
  if (!date) return;
  const db = load();
  db.holidays = db.holidays.filter((h) => h.date !== date);
  save(db);
  renderHolidays();
});
$("rec-date").addEventListener("change", renderRecords);
$("week-start").addEventListener("change", renderWeekly);
$("cal-from").addEventListener("change", renderCalendar);
$("cal-to").addEventListener("change", renderCalendar);
$("btn-ics").addEventListener("click", () => {
  const db = load();
  const start = $("cal-from").value || monday(today());
  const end = $("cal-to").value || addDays(monday(today()), 6);
  if (end < start) {
    alert("End date must be on or after start date.");
    return;
  }
  const ics = buildAttendanceIcs(db, start, end);
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([ics], { type: "text/calendar" }));
  a.download = `attendance-${start}-${end}.ics`;
  a.click();
});
$("btn-export").addEventListener("click", () => {
  const db = load();
  const blob = new Blob([JSON.stringify(db, null, 2)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "attendance.json";
  a.click();
});
$("btn-cam").addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    $("cam").srcObject = stream;
  } catch (err) {
    alert("Camera unavailable: " + err.message);
  }
});
$("btn-cam-mark").addEventListener("click", () => $("form-mark").requestSubmit());

$("pin-pad").addEventListener("click", (event) => {
  const key = event.target.dataset.key;
  if (key === undefined) return;
  if (key === "del") kioskPin = kioskPin.slice(0, -1);
  else if (key === "clr") kioskPin = "";
  else if (kioskPin.length < 8) kioskPin += key;
  renderKiosk();
});
$("kiosk-mark").addEventListener("click", () => {
  const roll = $("kiosk-roll").value.trim();
  const db = load();
  const student = db.students.find((s) => s.roll === roll);
  if (!student) {
    $("kiosk-msg").textContent = "Unknown roll.";
    return;
  }
  if (!student.active) {
    $("kiosk-msg").textContent = "Student is archived.";
    return;
  }
  if (!/^\d{4,8}$/.test(kioskPin)) {
    $("kiosk-msg").textContent = "PIN must be 4–8 digits.";
    return;
  }
  if (!student.pin || student.pin !== kioskPin) {
    $("kiosk-msg").textContent = "Invalid PIN.";
    return;
  }
  const now = new Date();
  const time = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:00`;
  const note = ($("kiosk-note").value || "").trim();
  db.attendance.push({
    id: db.attendance.length + 1,
    student_id: student.id,
    date: today(),
    time,
    status: "Present",
    period: "Morning",
    source: "kiosk",
    note,
  });
  save(db);
  kioskPin = "";
  if ($("kiosk-note")) $("kiosk-note").value = "";
  renderKiosk();
  $("kiosk-msg").textContent = `Marked ${student.name} Present.`;
  renderDash();
});

const initial = load();
applyTheme(initial.theme || "dark");
$("rec-date").value = today();
$("week-start").value = monday(today());
$("cal-from").value = monday(today());
$("cal-to").value = addDays(monday(today()), 6);
document.querySelector('#form-mark [name="date"]').value = today();
show("dash");

function apiOrigin() {
  try {
    const raw = new URLSearchParams(window.location.search).get("api");
    if (!raw) return "";
    return String(raw).replace(/\/$/, "");
  } catch (err) {
    return "";
  }
}

async function hydrateFromApi() {
  const origin = apiOrigin();
  if (!origin) return;
  try {
    const [stRes, attRes, holRes] = await Promise.all([
      fetch(`${origin}/students`),
      fetch(`${origin}/attendance`),
      fetch(`${origin}/holidays`).catch(() => null),
    ]);
    if (!stRes.ok || !attRes.ok) return;
    const st = await stRes.json();
    const att = await attRes.json();
    const db = load();
    const pinByRoll = new Map((db.students || []).map((s) => [s.roll, s.pin || ""]));
    db.students = (st.students || []).map((s) => ({
      id: s.id,
      roll: s.roll_number,
      name: s.name,
      section: s.section || "",
      email: s.email || "",
      active: s.active !== 0,
      pin: pinByRoll.get(s.roll_number) || "",
    }));
    db.attendance = (att.records || []).map((a, i) => ({
      id: a.id || i + 1,
      student_id: a.student_id,
      date: a.date,
      time: a.time,
      status: a.status,
      period: a.period || "",
      source: a.source || "api",
      note: a.note || "",
    }));
    if (holRes && holRes.ok) {
      const hol = await holRes.json();
      if (Array.isArray(hol.holidays)) {
        db.holidays = hol.holidays.map((h) => ({
          date: h.date,
          name: h.name || "Holiday",
        }));
      }
    }
    save(db);
    const active = document.querySelector(".sidebar nav button.active");
    show(active ? active.dataset.view : "dash");
  } catch (err) {
    // Keep the localStorage demo if the live API is unreachable.
  }
}
hydrateFromApi();
