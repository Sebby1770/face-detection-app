"""Tkinter application: home dashboard, register, train, take attendance,
kiosk, calendar, records, stats, settings, and student management screens."""
from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import (
    BOTH,
    BOTTOM,
    END,
    LEFT,
    RIGHT,
    X,
    Y,
    BooleanVar,
    Entry,
    Frame,
    Label,
    Listbox,
    StringVar,
    Tk,
    filedialog,
    messagebox,
    simpledialog,
    ttk,
)
from typing import Callable

import cv2
from PIL import Image, ImageTk

from . import config
from .camera import Camera
from .database import Database, calendar_cell_letter, week_dates, week_start
from .face_engine import (
    FaceEngine,
    FaceEngineError,
    invalidate_model_if_needed,
    is_low_quality_sample,
    mark_model_stale,
    model_is_stale,
    sample_hash,
)


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
def _apply_style(root: Tk) -> ttk.Style:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:  # pragma: no cover - depends on platform
        pass

    base_font = (config.FONT_FAMILY, 11)
    bold_font = (config.FONT_FAMILY, 11, "bold")
    h1_font = (config.FONT_FAMILY, 22, "bold")

    root.configure(bg=config.COLOR_BG)
    style.configure("TFrame", background=config.COLOR_BG)
    style.configure("Surface.TFrame", background=config.COLOR_SURFACE)
    style.configure("Sidebar.TFrame", background=config.COLOR_SIDEBAR)

    style.configure(
        "TLabel",
        background=config.COLOR_BG,
        foreground=config.COLOR_TEXT,
        font=base_font,
    )
    style.configure(
        "Surface.TLabel",
        background=config.COLOR_SURFACE,
        foreground=config.COLOR_TEXT,
        font=base_font,
    )
    style.configure(
        "Muted.TLabel",
        background=config.COLOR_SURFACE,
        foreground=config.COLOR_MUTED,
        font=base_font,
    )
    style.configure(
        "H1.TLabel",
        background=config.COLOR_BG,
        foreground=config.COLOR_TEXT,
        font=h1_font,
    )
    style.configure(
        "Sidebar.TLabel",
        background=config.COLOR_SIDEBAR,
        foreground=config.COLOR_SIDEBAR_TEXT,
        font=bold_font,
    )
    style.configure(
        "SidebarTitle.TLabel",
        background=config.COLOR_SIDEBAR,
        foreground=config.COLOR_SIDEBAR_TEXT,
        font=(config.FONT_FAMILY, 14, "bold"),
    )
    style.configure(
        "SidebarMuted.TLabel",
        background=config.COLOR_SIDEBAR,
        foreground=config.COLOR_MUTED,
        font=(config.FONT_FAMILY, 10),
    )

    style.configure(
        "Primary.TButton",
        background=config.COLOR_PRIMARY,
        foreground="white",
        font=bold_font,
        padding=(16, 10),
        borderwidth=0,
    )
    style.map(
        "Primary.TButton",
        background=[("active", config.COLOR_PRIMARY_DARK)],
        foreground=[("active", "white")],
    )
    style.configure(
        "Accent.TButton",
        background=config.COLOR_ACCENT,
        foreground="white",
        font=bold_font,
        padding=(16, 10),
        borderwidth=0,
    )
    style.map(
        "Accent.TButton",
        background=[("active", config.COLOR_ACCENT_DARK)],
        foreground=[("active", "white")],
    )
    style.configure(
        "Danger.TButton",
        background=config.COLOR_DANGER,
        foreground="white",
        font=bold_font,
        padding=(12, 8),
        borderwidth=0,
    )
    style.map(
        "Danger.TButton",
        background=[("active", "#dc2626")],
        foreground=[("active", "white")],
    )
    style.configure(
        "Nav.TButton",
        background=config.COLOR_SIDEBAR,
        foreground=config.COLOR_SIDEBAR_TEXT,
        font=base_font,
        padding=(20, 14),
        anchor="w",
        borderwidth=0,
    )
    style.map(
        "Nav.TButton",
        background=[("active", config.COLOR_PRIMARY)],
        foreground=[("active", "white")],
    )
    style.configure(
        "NavActive.TButton",
        background=config.COLOR_PRIMARY,
        foreground="white",
        font=bold_font,
        padding=(20, 14),
        anchor="w",
        borderwidth=0,
    )
    style.configure(
        "Ghost.TButton",
        background=config.COLOR_SIDEBAR,
        foreground=config.COLOR_SIDEBAR_TEXT,
        font=(config.FONT_FAMILY, 10),
        padding=(10, 6),
        borderwidth=0,
    )
    style.map(
        "Ghost.TButton",
        background=[("active", config.COLOR_PRIMARY)],
        foreground=[("active", "white")],
    )

    style.configure(
        "TEntry",
        fieldbackground=config.COLOR_SURFACE,
        foreground=config.COLOR_TEXT,
        padding=8,
    )
    style.configure(
        "TCombobox",
        fieldbackground=config.COLOR_SURFACE,
        background=config.COLOR_SURFACE,
        foreground=config.COLOR_TEXT,
        padding=6,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", config.COLOR_SURFACE)],
        foreground=[("readonly", config.COLOR_TEXT)],
    )
    style.configure(
        "TCheckbutton",
        background=config.COLOR_SURFACE,
        foreground=config.COLOR_TEXT,
        font=base_font,
    )
    style.configure(
        "TRadiobutton",
        background=config.COLOR_SURFACE,
        foreground=config.COLOR_TEXT,
        font=base_font,
    )
    style.configure(
        "TProgressbar",
        background=config.COLOR_PRIMARY,
        troughcolor=config.COLOR_BORDER,
    )
    style.configure(
        "Treeview",
        background=config.COLOR_SURFACE,
        foreground=config.COLOR_TEXT,
        rowheight=28,
        fieldbackground=config.COLOR_SURFACE,
        font=base_font,
    )
    style.configure(
        "Treeview.Heading",
        background=config.COLOR_PRIMARY,
        foreground="white",
        font=bold_font,
        padding=8,
    )
    style.map(
        "Treeview",
        background=[("selected", config.COLOR_PRIMARY)],
        foreground=[("selected", "white")],
    )
    style.map("Treeview.Heading", background=[("active", config.COLOR_PRIMARY_DARK)])
    style.configure("TNotebook", background=config.COLOR_BG, borderwidth=0)
    style.configure(
        "TNotebook.Tab",
        padding=(14, 8),
        font=bold_font,
        background=config.COLOR_SURFACE,
        foreground=config.COLOR_TEXT,
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", config.COLOR_PRIMARY)],
        foreground=[("selected", "white")],
    )

    return style


def _card(parent: Frame, **kwargs) -> Frame:
    return Frame(
        parent,
        bg=config.COLOR_SURFACE,
        highlightthickness=1,
        highlightbackground=config.COLOR_BORDER,
        **kwargs,
    )


def face_box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def push_motion_sample(
    history: dict[int, list[tuple[float, float]]],
    student_id: int,
    center: tuple[float, float],
    cap: int | None = None,
) -> list[tuple[float, float]]:
    limit = int(cap if cap is not None else config.LIVENESS_HISTORY)
    samples = history.setdefault(student_id, [])
    samples.append(center)
    overflow = len(samples) - max(1, limit)
    if overflow > 0:
        del samples[:overflow]
    return samples


def motion_liveness_ok(
    samples: list[tuple[float, float]],
    min_px: float | None = None,
) -> bool:
    """True when max-min on x or y meets the still-photo motion threshold."""
    if len(samples) < 2:
        return False
    threshold = float(
        min_px if min_px is not None else config.LIVENESS_MIN_MOTION_PX
    )
    xs = [point[0] for point in samples]
    ys = [point[1] for point in samples]
    return (max(xs) - min(xs) >= threshold) or (max(ys) - min(ys) >= threshold)


def _draw_face_label(frame, box, color, label: str) -> None:
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(frame, (x, y - 26), (x + w, y), color, -1)
    cv2.putText(
        frame,
        label,
        (x + 6, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def save_unknown_crop(face, confidence: float) -> str | None:
    try:
        now = datetime.now()
        dest = config.UNKNOWNS_DIR / now.strftime("%Y-%m-%d")
        dest.mkdir(parents=True, exist_ok=True)
        stamp = now.strftime("%H%M%S_%f")
        path = dest / f"{stamp}_{float(confidence):.0f}.png"
        if not cv2.imwrite(str(path), face):
            return None
        return str(path) if path.is_file() else None
    except OSError:
        return None
    except Exception:  # noqa: BLE001
        return None


def process_live_faces(
    app: "AttendanceApp",
    frame,
    *,
    period: str | None,
    source: str,
    marked_today: dict[int, str],
    motion: dict[int, list[tuple[float, float]]],
    on_unknown: Callable | None = None,
    on_insert: Callable | None = None,
) -> None:
    """Detect faces, enforce motion liveness on first mark, then dwell-touch."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = app.face_engine.detect_faces(gray)
    yellow = (0, 255, 255)
    green = (16, 185, 129)
    red = (239, 68, 68)

    for box in faces:
        x, y, w, h = box
        face = app.face_engine.crop_and_resize(gray, box)
        student_id, conf = app.face_engine.predict(face)

        if student_id is None:
            _draw_face_label(frame, box, red, f"Unknown ({conf:.0f})")
            if on_unknown is not None:
                on_unknown(face, conf)
            continue

        row = app.db.get_student(student_id)
        if row is None:
            _draw_face_label(frame, box, red, "Unknown")
            if on_unknown is not None:
                on_unknown(face, conf)
            continue

        name = str(row["name"])
        if student_id in marked_today:
            app.db.mark_attendance(
                student_id=student_id,
                confidence=conf,
                period=period,
                source=source,
                touch=True,
            )
            _draw_face_label(frame, box, green, marked_today[student_id])
            continue

        samples = push_motion_sample(
            motion, student_id, face_box_center((x, y, w, h))
        )
        if not motion_liveness_ok(samples):
            _draw_face_label(frame, box, yellow, "Move slightly…")
            continue

        inserted = app.db.mark_attendance(
            student_id=student_id,
            confidence=conf,
            period=period,
            source=source,
        )
        marked_today[student_id] = name
        _draw_face_label(frame, box, green, f"{name}  ({conf:.0f})")
        if inserted and on_insert is not None:
            on_insert(row, student_id, conf)


def _show_preview(label: Label, frame) -> None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    target_w = max(label.winfo_width(), 600)
    target_h = max(label.winfo_height(), 420)
    img.thumbnail((target_w, target_h))
    photo = ImageTk.PhotoImage(img)
    label.configure(image=photo)
    label.image = photo


# ---------------------------------------------------------------------------
# Base page
# ---------------------------------------------------------------------------
class Page(Frame):
    """Common base for all routable pages."""

    title: str = ""

    def __init__(self, master: Frame, app: "AttendanceApp") -> None:
        super().__init__(master, bg=config.COLOR_BG)
        self.app = app

    def on_show(self) -> None:
        """Called every time the page is displayed."""

    def on_hide(self) -> None:
        """Called when navigating away. Use to release resources."""


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
class DashboardPage(Page):
    title = "Dashboard"

    def __init__(self, master: Frame, app: "AttendanceApp") -> None:
        super().__init__(master, app)

        ttk.Label(self, text="Welcome back", style="H1.TLabel").pack(
            anchor="w", padx=30, pady=(24, 0)
        )
        ttk.Label(
            self,
            text="A modern face-recognition attendance system.",
        ).pack(anchor="w", padx=30, pady=(0, 8))
        self.samples_warn = Label(
            self,
            text="",
            bg=config.COLOR_BG,
            fg=config.COLOR_MUTED,
            font=(config.FONT_FAMILY, 11),
            wraplength=900,
            justify="left",
        )
        self.samples_warn.pack(anchor="w", padx=30, pady=(0, 18))

        cards = Frame(self, bg=config.COLOR_BG)
        cards.pack(fill=X, padx=24)

        self.card_students = self._kpi(cards, "Total Students", "0", config.COLOR_PRIMARY)
        self.card_today = self._kpi(cards, "Present Today", "0", config.COLOR_ACCENT)
        self.card_late = self._kpi(cards, "Late Today", "0", config.COLOR_WARNING)

        for c in (self.card_students, self.card_today, self.card_late):
            c["frame"].pack(side=LEFT, padx=8, pady=8, fill=X, expand=True)

        cards2 = Frame(self, bg=config.COLOR_BG)
        cards2.pack(fill=X, padx=24)

        self.card_unknown = self._kpi(
            cards2, "Unknown Faces Today", "0", config.COLOR_DANGER
        )
        self.card_week = self._kpi(cards2, "This Week's Rate", "0%", config.COLOR_PRIMARY)
        self.card_streak = self._kpi(cards2, "Longest streak", "0", config.COLOR_ACCENT)
        self.card_records = self._kpi(cards2, "Total Records", "0", config.COLOR_MUTED)

        for c in (self.card_unknown, self.card_week, self.card_streak, self.card_records):
            c["frame"].pack(side=LEFT, padx=8, pady=8, fill=X, expand=True)

        actions = Frame(self, bg=config.COLOR_BG)
        actions.pack(fill=X, padx=30, pady=(24, 12))
        ttk.Label(actions, text="Quick actions", style="H1.TLabel").pack(anchor="w")

        btns = Frame(self, bg=config.COLOR_BG)
        btns.pack(fill=X, padx=24, pady=(8, 24))
        ttk.Button(
            btns,
            text="Take Attendance",
            style="Accent.TButton",
            command=lambda: self.app.show("attendance"),
        ).pack(side=LEFT, padx=6)
        ttk.Button(
            btns,
            text="Register New Student",
            style="Primary.TButton",
            command=lambda: self.app.show("register"),
        ).pack(side=LEFT, padx=6)
        ttk.Button(
            btns,
            text="Weekly Stats",
            style="Primary.TButton",
            command=lambda: self.app.show("stats"),
        ).pack(side=LEFT, padx=6)
        ttk.Button(
            btns,
            text="View Records",
            style="Primary.TButton",
            command=lambda: self.app.show("records"),
        ).pack(side=LEFT, padx=6)

    def _kpi(self, parent: Frame, label: str, value: str, accent: str) -> dict:
        frame = _card(parent)
        bar = Frame(frame, bg=accent, height=4)
        bar.pack(fill=X)
        caption = Label(
            frame,
            text=label,
            bg=config.COLOR_SURFACE,
            fg=config.COLOR_MUTED,
            font=(config.FONT_FAMILY, 11),
        )
        caption.pack(anchor="w", padx=18, pady=(16, 4))
        value_lbl = Label(
            frame,
            text=value,
            bg=config.COLOR_SURFACE,
            fg=config.COLOR_TEXT,
            font=(config.FONT_FAMILY, 30, "bold"),
        )
        value_lbl.pack(anchor="w", padx=18, pady=(0, 16))
        return {"frame": frame, "value": value_lbl, "label": caption}

    def on_show(self) -> None:
        s = self.app.db.stats()
        self.card_students["value"].config(text=str(s["total_students"]))
        if s.get("holiday"):
            name = str(s.get("holiday_name") or "Holiday")
            self.card_today["label"].config(text=f"Holiday — {name}")
        elif s.get("weekend") or s.get("school_day") is False:
            self.card_today["label"].config(text="Weekend — no class")
        else:
            self.card_today["label"].config(text="Present Today")
        self.card_today["value"].config(text=str(s["present_today"]))
        self.card_late["value"].config(text=str(s["late_today"]))
        self.card_unknown["value"].config(text=str(s["unknown_today"]))
        self.card_week["value"].config(text=f"{s['week_rate']:.0f}%")
        longest = 0
        try:
            streaks = self.app.db.streaks_report()
            if streaks:
                longest = max(int(row.get("streak") or 0) for row in streaks)
        except Exception:  # noqa: BLE001
            longest = 0
        self.card_streak["value"].config(text=str(longest))
        self.card_records["value"].config(text=str(s["total_records"]))
        missing = [
            row
            for row in self.app.db.enrollment_roster(min_samples=1)
            if int(row.get("samples") or 0) == 0
        ]
        if missing:
            names = ", ".join(str(row["name"]) for row in missing[:5])
            extra = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
            self.samples_warn.config(
                text=(
                    f"{len(missing)} student(s) have no face samples: "
                    f"{names}{extra}"
                )
            )
        else:
            self.samples_warn.config(text="")


# ---------------------------------------------------------------------------
# Register page (capture face samples for a new student)
# ---------------------------------------------------------------------------
class RegisterPage(Page):
    title = "Register Student"

    def __init__(self, master: Frame, app: "AttendanceApp") -> None:
        super().__init__(master, app)
        self._camera: Camera | None = None
        self._capturing: bool = False
        self._sample_count: int = 0
        self._student_id: int | None = None
        self._after_id: str | None = None
        self._seen_hashes: set[str] = set()

        ttk.Label(self, text="Register New Student", style="H1.TLabel").pack(
            anchor="w", padx=30, pady=(24, 12)
        )

        body = Frame(self, bg=config.COLOR_BG)
        body.pack(fill=BOTH, expand=True, padx=24, pady=(0, 24))

        form = _card(body)
        form.pack(side=LEFT, fill=Y, padx=(0, 12), ipadx=12, ipady=12)

        Label(form, text="Student details", bg=config.COLOR_SURFACE,
              fg=config.COLOR_TEXT, font=(config.FONT_FAMILY, 14, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(8, 18)
        )

        self.var_roll = StringVar()
        self.var_name = StringVar()
        self.var_email = StringVar()
        self.var_dept = StringVar()
        self.var_section = StringVar()

        for i, (label, var) in enumerate(
            [
                ("Roll number*", self.var_roll),
                ("Full name*", self.var_name),
                ("Email", self.var_email),
                ("Department", self.var_dept),
                ("Section", self.var_section),
            ],
            start=1,
        ):
            Label(form, text=label, bg=config.COLOR_SURFACE,
                  fg=config.COLOR_TEXT, font=(config.FONT_FAMILY, 10)).grid(
                row=i, column=0, sticky="w", padx=12, pady=(6, 2)
            )
            ttk.Entry(form, textvariable=var, width=28).grid(
                row=i, column=1, sticky="we", padx=12, pady=(6, 2)
            )

        self.lbl_status = Label(
            form,
            text="Fill in details, then capture from the webcam or import photos.",
            bg=config.COLOR_SURFACE,
            fg=config.COLOR_MUTED,
            font=(config.FONT_FAMILY, 10),
            wraplength=260,
            justify="left",
        )
        self.lbl_status.grid(row=10, column=0, columnspan=2, sticky="we", padx=12, pady=(16, 8))

        self.progress = ttk.Progressbar(
            form, orient="horizontal", mode="determinate",
            maximum=config.SAMPLES_PER_STUDENT,
        )
        self.progress.grid(row=11, column=0, columnspan=2, sticky="we", padx=12, pady=(2, 12))

        self.btn_import = ttk.Button(
            form, text="Import photos…", style="Accent.TButton",
            command=self._import_photos,
        )
        self.btn_import.grid(row=12, column=0, columnspan=2, sticky="we", padx=12, pady=(2, 6))

        self.btn_start = ttk.Button(
            form, text="Start Capture", style="Primary.TButton",
            command=self._start_capture,
        )
        self.btn_start.grid(row=13, column=0, columnspan=2, sticky="we", padx=12, pady=(2, 6))

        self.btn_stop = ttk.Button(
            form, text="Cancel", style="Danger.TButton",
            command=self._stop_capture, state="disabled",
        )
        self.btn_stop.grid(row=14, column=0, columnspan=2, sticky="we", padx=12, pady=(0, 12))

        preview = _card(body)
        preview.pack(side=RIGHT, fill=BOTH, expand=True)

        Label(preview, text="Camera preview", bg=config.COLOR_SURFACE,
              fg=config.COLOR_TEXT, font=(config.FONT_FAMILY, 14, "bold")).pack(
            anchor="w", padx=14, pady=(10, 4)
        )
        self.preview_label = Label(preview, bg=config.COLOR_PREVIEW, width=80, height=24)
        self.preview_label.pack(fill=BOTH, expand=True, padx=14, pady=14)

    def _details_or_error(self) -> tuple[str, str] | None:
        roll = self.var_roll.get().strip()
        name = self.var_name.get().strip()
        if not roll or not name:
            messagebox.showerror("Missing info", "Roll number and name are required.")
            return None
        if self.app.db.get_student_by_roll(roll):
            messagebox.showerror(
                "Duplicate",
                f"A student with roll number '{roll}' already exists.",
            )
            return None
        return roll, name

    def _create_student(self, roll: str, name: str) -> int | None:
        try:
            return self.app.db.add_student(
                roll_number=roll,
                name=name,
                email=self.var_email.get().strip(),
                department=self.var_dept.get().strip(),
                section=self.var_section.get().strip(),
            )
        except sqlite3.IntegrityError as exc:
            messagebox.showerror("Database error", str(exc))
            return None

    def _clear_form(self) -> None:
        self.var_roll.set("")
        self.var_name.set("")
        self.var_email.set("")
        self.var_dept.set("")
        self.var_section.set("")

    def _import_photos(self) -> None:
        parsed = self._details_or_error()
        if parsed is None:
            return
        roll, name = parsed
        directory = filedialog.askdirectory(
            title="Select a folder of student photos",
        )
        if not directory:
            return

        sid = self._create_student(roll, name)
        if sid is None:
            return

        try:
            count = self.app.face_engine.enroll_from_folder(Path(directory), sid)
        except FaceEngineError as exc:
            self.app.db.delete_student(sid)
            messagebox.showerror("Import failed", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            self.app.db.delete_student(sid)
            messagebox.showerror("Import failed", str(exc))
            return

        self._sample_count = count
        self.progress["value"] = min(count, config.SAMPLES_PER_STUDENT)
        mark_model_stale(f"Imported folder samples for id={sid}")
        self.lbl_status.config(
            text=(
                f"Imported {count} face sample(s) for '{name}'. "
                "Visit Train Model to rebuild the recognizer."
            ),
            fg=config.COLOR_ACCENT,
        )
        self._clear_form()
        messagebox.showinfo(
            "Imported",
            f"Wrote {count} samples for {name} ({roll}).\nTrain the model next.",
        )

    def _start_capture(self) -> None:
        parsed = self._details_or_error()
        if parsed is None:
            return
        roll, name = parsed
        sid = self._create_student(roll, name)
        if sid is None:
            return
        self._student_id = sid

        self._sample_count = 0
        self._seen_hashes = set()
        self.progress["value"] = 0
        self._capturing = True
        self.btn_start["state"] = "disabled"
        self.btn_import["state"] = "disabled"
        self.btn_stop["state"] = "normal"
        self.lbl_status.config(
            text="Look at the camera. Move your head slightly for varied angles.",
            fg=config.COLOR_TEXT,
        )

        self._camera = Camera()
        if not self._camera.open():
            messagebox.showerror("Camera", "Could not open the webcam.")
            self._stop_capture()
            return

        student_dir = config.FACES_DIR / str(self._student_id)
        student_dir.mkdir(parents=True, exist_ok=True)

        self._loop_capture(student_dir)

    def _loop_capture(self, student_dir: Path) -> None:
        if not self._capturing or self._camera is None:
            return
        ok, frame = self._camera.read()
        if not ok or frame is None:
            self._after_id = self.after(30, lambda: self._loop_capture(student_dir))
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.app.face_engine.detect_faces(gray)

        if faces:
            faces.sort(key=lambda b: b[2] * b[3], reverse=True)
            box = faces[0]
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (37, 99, 235), 2)

            face_img = self.app.face_engine.extract_face(
                gray, detect=True, require_face=True
            )
            if face_img is not None:
                blur = float(cv2.Laplacian(face_img, cv2.CV_64F).var())
                if (
                    not is_low_quality_sample(face_img)
                    and blur >= float(config.ENROLL_MIN_LAPLACIAN)
                ):
                    digest = sample_hash(face_img)
                    if digest not in self._seen_hashes:
                        self._seen_hashes.add(digest)
                        self._sample_count += 1
                        sample_path = student_dir / f"{self._sample_count:03d}.png"
                        cv2.imwrite(str(sample_path), face_img)
                        self.progress["value"] = self._sample_count
                        self.lbl_status.config(
                            text=(
                                f"Captured {self._sample_count}/"
                                f"{config.SAMPLES_PER_STUDENT} unique samples..."
                            ),
                            fg=config.COLOR_TEXT,
                        )

        self._render(frame)

        if self._sample_count >= config.SAMPLES_PER_STUDENT:
            self._finish_capture()
            return
        self._after_id = self.after(50, lambda: self._loop_capture(student_dir))

    def _render(self, frame_bgr) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        target_w = max(self.preview_label.winfo_width(), 480)
        target_h = max(self.preview_label.winfo_height(), 360)
        img.thumbnail((target_w, target_h))
        photo = ImageTk.PhotoImage(img)
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo  # prevent GC

    def _finish_capture(self) -> None:
        self._capturing = False
        self.btn_start["state"] = "normal"
        self.btn_import["state"] = "normal"
        self.btn_stop["state"] = "disabled"
        if self._camera:
            self._camera.release()
            self._camera = None
        mark_model_stale("New student face samples captured")
        self.lbl_status.config(
            text=(
                f"Done! Captured {self._sample_count} samples for "
                f"'{self.var_name.get()}'. Now visit 'Train Model' to "
                "rebuild the recognizer."
            ),
            fg=config.COLOR_ACCENT,
        )
        self._clear_form()

    def _stop_capture(self) -> None:
        self._capturing = False
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self._camera:
            self._camera.release()
            self._camera = None
        if self._student_id and self._sample_count == 0:
            self.app.db.delete_student(self._student_id)
        self._student_id = None
        self.btn_start["state"] = "normal"
        self.btn_import["state"] = "normal"
        self.btn_stop["state"] = "disabled"
        self.lbl_status.config(
            text="Capture canceled.", fg=config.COLOR_DANGER
        )

    def on_hide(self) -> None:
        self._stop_capture()


# ---------------------------------------------------------------------------
# Train page
# ---------------------------------------------------------------------------
class TrainPage(Page):
    title = "Train Model"

    def __init__(self, master: Frame, app: "AttendanceApp") -> None:
        super().__init__(master, app)

        ttk.Label(self, text="Train Recognition Model", style="H1.TLabel").pack(
            anchor="w", padx=30, pady=(24, 6)
        )
        ttk.Label(
            self,
            text="Builds the LBPH model from every face captured during registration.",
        ).pack(anchor="w", padx=30, pady=(0, 18))

        card = _card(self)
        card.pack(fill=X, padx=24, pady=(0, 24))

        Label(
            card,
            text="When to retrain",
            bg=config.COLOR_SURFACE,
            fg=config.COLOR_TEXT,
            font=(config.FONT_FAMILY, 14, "bold"),
        ).pack(anchor="w", padx=18, pady=(14, 4))
        Label(
            card,
            text=(
                "Retrain after registering a new student or after deleting "
                "a student. Training takes a few seconds."
            ),
            bg=config.COLOR_SURFACE,
            fg=config.COLOR_MUTED,
            font=(config.FONT_FAMILY, 10),
            wraplength=720,
            justify="left",
        ).pack(anchor="w", padx=18, pady=(0, 16))

        self.lbl_status = Label(
            card,
            text="Model status: not loaded.",
            bg=config.COLOR_SURFACE,
            fg=config.COLOR_TEXT,
            font=(config.FONT_FAMILY, 11, "bold"),
        )
        self.lbl_status.pack(anchor="w", padx=18, pady=(0, 8))

        self.lbl_threshold = Label(
            card,
            text="",
            bg=config.COLOR_SURFACE,
            fg=config.COLOR_MUTED,
            font=(config.FONT_FAMILY, 10),
        )
        self.lbl_threshold.pack(anchor="w", padx=18, pady=(0, 12))

        self.progress = ttk.Progressbar(card, orient="horizontal", mode="indeterminate")
        self.progress.pack(fill=X, padx=18, pady=(0, 12))

        ttk.Button(
            card, text="Train Now", style="Primary.TButton",
            command=self._train_async,
        ).pack(anchor="w", padx=18, pady=(0, 18))

    def _train_async(self) -> None:
        self.progress.start(12)
        self.lbl_status.config(text="Training...", fg=config.COLOR_TEXT)

        def _worker() -> None:
            try:
                count = self.app.face_engine.train_from_dataset()
                msg = f"Trained on {count} face samples. Model is ready."
                color = config.COLOR_ACCENT
            except Exception as exc:  # noqa: BLE001
                msg = f"Training failed: {exc}"
                color = config.COLOR_DANGER

            def _done() -> None:
                self.progress.stop()
                self.lbl_status.config(text=msg, fg=color)

            self.after(0, _done)

        threading.Thread(target=_worker, daemon=True).start()

    def on_show(self) -> None:
        self.lbl_threshold.config(
            text=(
                f"Unknown when LBPH distance > "
                f"{config.get_confidence_threshold():.1f}  "
                "(change this on the Settings page or with --threshold)."
            )
        )
        stale = invalidate_model_if_needed() or model_is_stale()
        if stale:
            self.lbl_status.config(
                text="Model status: STALE — retrain recommended after student changes.",
                fg=config.COLOR_WARNING,
            )
        elif self.app.face_engine.is_loaded or config.MODEL_PATH.exists():
            self.lbl_status.config(text="Model status: trained model on disk.",
                                   fg=config.COLOR_ACCENT)
        else:
            self.lbl_status.config(text="Model status: no model trained yet.",
                                   fg=config.COLOR_WARNING)


# ---------------------------------------------------------------------------
# Take Attendance page
# ---------------------------------------------------------------------------
class AttendancePage(Page):
    title = "Take Attendance"

    def __init__(self, master: Frame, app: "AttendanceApp") -> None:
        super().__init__(master, app)
        self._camera: Camera | None = None
        self._running = False
        self._after_id: str | None = None
        self._marked_today: dict[int, str] = {}
        self._motion: dict[int, list[tuple[float, float]]] = {}
        self._last_unknown_at: datetime | None = None

        ttk.Label(self, text="Take Attendance", style="H1.TLabel").pack(
            anchor="w", padx=30, pady=(24, 6)
        )
        ttk.Label(
            self,
            text="Move slightly to pass liveness, then stay in frame — time_out tracks dwell.",
        ).pack(anchor="w", padx=30, pady=(0, 14))

        body = Frame(self, bg=config.COLOR_BG)
        body.pack(fill=BOTH, expand=True, padx=24, pady=(0, 24))

        cam_card = _card(body)
        cam_card.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 12))
        self.preview_label = Label(cam_card, bg=config.COLOR_PREVIEW)
        self.preview_label.pack(fill=BOTH, expand=True, padx=10, pady=10)

        controls = Frame(cam_card, bg=config.COLOR_SURFACE)
        controls.pack(fill=X, padx=10, pady=(0, 10))

        Label(
            controls, text="Period",
            bg=config.COLOR_SURFACE, fg=config.COLOR_TEXT,
            font=(config.FONT_FAMILY, 10),
        ).pack(side=LEFT, padx=(4, 4))
        self.var_period = StringVar(value=self.app.db.get_default_period_name())
        self.cmb_period = ttk.Combobox(
            controls, textvariable=self.var_period, width=14, state="readonly"
        )
        self.cmb_period.pack(side=LEFT, padx=4)

        self.btn_start = ttk.Button(
            controls, text="Start Camera", style="Accent.TButton",
            command=self._start,
        )
        self.btn_start.pack(side=LEFT, padx=4)
        self.btn_stop = ttk.Button(
            controls, text="Stop", style="Danger.TButton",
            command=self._stop, state="disabled",
        )
        self.btn_stop.pack(side=LEFT, padx=4)

        self.lbl_unknown = Label(
            controls,
            text="Unknown this session: 0",
            bg=config.COLOR_SURFACE,
            fg=config.COLOR_MUTED,
            font=(config.FONT_FAMILY, 10),
        )
        self.lbl_unknown.pack(side=RIGHT, padx=8)

        side = _card(body)
        side.pack(side=RIGHT, fill=Y, ipadx=8, ipady=8)
        Label(side, text="Marked this session", bg=config.COLOR_SURFACE,
              fg=config.COLOR_TEXT, font=(config.FONT_FAMILY, 14, "bold")).pack(
            anchor="w", padx=10, pady=(8, 8)
        )
        self.tree = ttk.Treeview(
            side, columns=("name", "roll", "time", "status"), show="headings", height=18
        )
        self.tree.heading("name", text="Name")
        self.tree.heading("roll", text="Roll #")
        self.tree.heading("time", text="Time")
        self.tree.heading("status", text="Status")
        self.tree.column("name", width=150)
        self.tree.column("roll", width=80)
        self.tree.column("time", width=70)
        self.tree.column("status", width=70)
        self.tree.pack(fill=BOTH, expand=True, padx=10, pady=(0, 10))

    def _refresh_periods(self) -> None:
        names = [str(r["name"]) for r in self.app.db.list_periods()]
        if not names:
            names = [config.DEFAULT_PERIOD_NAME]
        self.cmb_period["values"] = names
        current = self.var_period.get()
        if current not in names:
            default = self.app.db.get_default_period_name()
            self.var_period.set(default if default in names else names[0])

    def _start(self) -> None:
        if not self.app.face_engine.is_loaded:
            try:
                loaded = self.app.face_engine.load()
            except Exception as exc:  # noqa: BLE001
                messagebox.showerror(
                    "Model error",
                    f"Could not load the trained model:\n{exc}",
                )
                return
            if not loaded:
                messagebox.showerror(
                    "No model",
                    "No trained model found. Register at least one student "
                    "and click 'Train Model' first.",
                )
                return

        if invalidate_model_if_needed() or model_is_stale():
            proceed = messagebox.askyesno(
                "Model may be out of date",
                "Students were added or removed since the last train.\n"
                "Recognition may be inaccurate until you retrain.\n\n"
                "Continue taking attendance anyway?",
            )
            if not proceed:
                return

        self._camera = Camera()
        if not self._camera.open():
            messagebox.showerror("Camera", "Could not open the webcam.")
            return
        self._running = True
        self._marked_today.clear()
        self._motion.clear()
        self.btn_start["state"] = "disabled"
        self.btn_stop["state"] = "normal"
        self.cmb_period["state"] = "disabled"
        self._loop()

    def _loop(self) -> None:
        if not self._running or self._camera is None:
            return
        ok, frame = self._camera.read()
        if not ok or frame is None:
            self._after_id = self.after(30, self._loop)
            return

        period = self.var_period.get().strip() or None

        def _on_insert(row, student_id: int, _conf: float) -> None:
            time_str = datetime.now().strftime("%H:%M:%S")
            latest = self.app.db.get_attendance(
                date=datetime.now().strftime("%Y-%m-%d"),
                student_id=student_id,
            )
            status = "Present"
            if not latest.empty:
                status = str(latest.iloc[0]["status"])
            self.tree.insert(
                "", 0,
                values=(row["name"], row["roll_number"], time_str, status),
            )

        process_live_faces(
            self.app,
            frame,
            period=period,
            source="camera",
            marked_today=self._marked_today,
            motion=self._motion,
            on_unknown=self._log_unknown,
            on_insert=_on_insert,
        )
        _show_preview(self.preview_label, frame)

        self._after_id = self.after(40, self._loop)

    def _log_unknown(self, face, conf: float) -> None:
        now = datetime.now()
        last = self._last_unknown_at
        if (
            last is not None
            and 0 <= (now - last).total_seconds() < config.UNKNOWN_LOG_COOLDOWN_SECONDS
        ):
            return
        crop_path = save_unknown_crop(face, conf)
        if self.app.db.log_unknown_face(confidence=conf, path=crop_path):
            self._last_unknown_at = now
            self.app.unknown_sightings += 1
            self.lbl_unknown.config(
                text=f"Unknown this session: {self.app.unknown_sightings}"
            )
        elif crop_path:
            try:
                Path(crop_path).unlink(missing_ok=True)
            except OSError:
                pass

    def _stop(self) -> None:
        self._running = False
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self._camera:
            self._camera.release()
            self._camera = None
        self.btn_start["state"] = "normal"
        self.btn_stop["state"] = "disabled"
        self.cmb_period["state"] = "readonly"

    def on_show(self) -> None:
        self._refresh_periods()
        self.lbl_unknown.config(
            text=f"Unknown this session: {self.app.unknown_sightings}"
        )

    def on_hide(self) -> None:
        self._stop()


# ---------------------------------------------------------------------------
# Calendar heatmap page
# ---------------------------------------------------------------------------
class CalendarPage(Page):
    title = "Calendar"

    def __init__(self, master: Frame, app: "AttendanceApp") -> None:
        super().__init__(master, app)
        start = week_start()
        end = week_dates(start, days=7)[-1]

        ttk.Label(self, text="Calendar", style="H1.TLabel").pack(
            anchor="w", padx=30, pady=(24, 6)
        )
        ttk.Label(
            self,
            text="P=Present  L=Late  E=Excused  .=unmarked  W=weekend  H=holiday",
        ).pack(anchor="w", padx=30, pady=(0, 12))

        controls = Frame(self, bg=config.COLOR_BG)
        controls.pack(fill=X, padx=24)
        ttk.Label(controls, text="From:").pack(side=LEFT, padx=4)
        self.var_from = StringVar(value=start)
        ttk.Entry(controls, textvariable=self.var_from, width=14).pack(
            side=LEFT, padx=4
        )
        ttk.Label(controls, text="To:").pack(side=LEFT, padx=(12, 4))
        self.var_to = StringVar(value=end)
        ttk.Entry(controls, textvariable=self.var_to, width=14).pack(
            side=LEFT, padx=4
        )
        ttk.Label(controls, text="Section:").pack(side=LEFT, padx=(12, 4))
        self.var_section = StringVar(value="")
        self.cmb_section = ttk.Combobox(
            controls, textvariable=self.var_section, width=12, state="normal"
        )
        self.cmb_section.pack(side=LEFT, padx=4)
        ttk.Button(
            controls, text="Refresh", style="Primary.TButton", command=self._reload
        ).pack(side=LEFT, padx=4)

        self.lbl_summary = ttk.Label(self, text="")
        self.lbl_summary.pack(anchor="w", padx=30, pady=(10, 0))

        card = _card(self)
        card.pack(fill=BOTH, expand=True, padx=24, pady=14)
        self._tree_host = Frame(card, bg=config.COLOR_SURFACE)
        self._tree_host.pack(fill=BOTH, expand=True, padx=10, pady=10)
        self.tree: ttk.Treeview | None = None
        self._scroll_x: ttk.Scrollbar | None = None
        self._scroll_y: ttk.Scrollbar | None = None

    def _reload(self) -> None:
        start = self.var_from.get().strip() or week_start()
        end = self.var_to.get().strip() or week_dates(week_start(), days=7)[-1]
        try:
            datetime.strptime(start, "%Y-%m-%d")
            datetime.strptime(end, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Bad date", "Dates must be YYYY-MM-DD.")
            return
        section = self.var_section.get().strip() or None
        try:
            grid = self.app.db.calendar_grid(start, end, section=section)
        except ValueError as exc:
            messagebox.showerror("Calendar", str(exc))
            return

        dates = list(grid["dates"])
        holidays = grid.get("holidays") or {}
        weekends = set(grid.get("weekends") or [])
        self.lbl_summary.config(
            text=(
                f"{grid['from']} → {grid['to']}  ·  "
                f"{len(grid['students'])} student(s)  ·  {len(dates)} day(s)"
            )
        )
        self._rebuild_tree(dates)
        assert self.tree is not None
        for student in grid["students"]:
            cells = [student["name"]]
            for day in dates:
                cells.append(
                    calendar_cell_letter(
                        day, student["days"].get(day), holidays, weekends
                    )
                )
            self.tree.insert("", "end", values=cells)

    def _rebuild_tree(self, dates: list[str]) -> None:
        if self.tree is not None:
            self.tree.destroy()
        if self._scroll_x is not None:
            self._scroll_x.destroy()
        if self._scroll_y is not None:
            self._scroll_y.destroy()
        cols = ("name", *dates)
        self.tree = ttk.Treeview(
            self._tree_host, columns=cols, show="headings"
        )
        self.tree.heading("name", text="Name")
        self.tree.column("name", width=180, anchor="w", stretch=False)
        for day in dates:
            heading = day[5:] if len(day) >= 10 else day
            self.tree.heading(day, text=heading)
            self.tree.column(day, width=46, anchor="center", stretch=False)
        self._scroll_y = ttk.Scrollbar(
            self._tree_host, orient="vertical", command=self.tree.yview
        )
        self._scroll_x = ttk.Scrollbar(
            self._tree_host, orient="horizontal", command=self.tree.xview
        )
        self.tree.configure(
            yscrollcommand=self._scroll_y.set,
            xscrollcommand=self._scroll_x.set,
        )
        self.tree.grid(row=0, column=0, sticky="nsew")
        self._scroll_y.grid(row=0, column=1, sticky="ns")
        self._scroll_x.grid(row=1, column=0, sticky="ew")
        self._tree_host.grid_rowconfigure(0, weight=1)
        self._tree_host.grid_columnconfigure(0, weight=1)

    def on_show(self) -> None:
        self.cmb_section["values"] = [""] + self.app.db.list_sections()
        if not self.var_from.get().strip():
            start = week_start()
            self.var_from.set(start)
            self.var_to.set(week_dates(start, days=7)[-1])
        self._reload()


# ---------------------------------------------------------------------------
# PIN / camera kiosk
# ---------------------------------------------------------------------------
class KioskPage(Page):
    title = "Kiosk"

    def __init__(self, master: Frame, app: "AttendanceApp") -> None:
        super().__init__(master, app)
        self._camera: Camera | None = None
        self._running = False
        self._after_id: str | None = None
        self._marked_today: dict[int, str] = {}
        self._motion: dict[int, list[tuple[float, float]]] = {}
        self._last_unknown_at: datetime | None = None

        ttk.Label(self, text="Kiosk", style="H1.TLabel").pack(
            anchor="w", padx=30, pady=(24, 6)
        )
        ttk.Label(
            self,
            text="Mark with a roll + PIN, or start the camera for face recognition.",
        ).pack(anchor="w", padx=30, pady=(0, 12))

        body = Frame(self, bg=config.COLOR_BG)
        body.pack(fill=BOTH, expand=True, padx=24, pady=(0, 24))

        form_card = _card(body)
        form_card.pack(side=LEFT, fill=Y, padx=(0, 12), ipadx=8, ipady=8)

        Label(
            form_card,
            text="PIN kiosk",
            bg=config.COLOR_SURFACE,
            fg=config.COLOR_TEXT,
            font=(config.FONT_FAMILY, 16, "bold"),
        ).pack(anchor="w", padx=18, pady=(16, 12))

        Label(
            form_card,
            text="Roll number",
            bg=config.COLOR_SURFACE,
            fg=config.COLOR_MUTED,
            font=(config.FONT_FAMILY, 11),
        ).pack(anchor="w", padx=18)
        self.var_roll = StringVar()
        self.ent_roll = Entry(
            form_card,
            textvariable=self.var_roll,
            font=(config.FONT_FAMILY, 28),
            width=12,
            bg=config.COLOR_SURFACE,
            fg=config.COLOR_TEXT,
            insertbackground=config.COLOR_TEXT,
            relief="solid",
            bd=1,
        )
        self.ent_roll.pack(anchor="w", padx=18, pady=(4, 14))

        Label(
            form_card,
            text="PIN",
            bg=config.COLOR_SURFACE,
            fg=config.COLOR_MUTED,
            font=(config.FONT_FAMILY, 11),
        ).pack(anchor="w", padx=18)
        self.var_pin = StringVar()
        self.ent_pin = Entry(
            form_card,
            textvariable=self.var_pin,
            show="•",
            font=(config.FONT_FAMILY, 28),
            width=12,
            bg=config.COLOR_SURFACE,
            fg=config.COLOR_TEXT,
            insertbackground=config.COLOR_TEXT,
            relief="solid",
            bd=1,
        )
        self.ent_pin.pack(anchor="w", padx=18, pady=(4, 14))

        Label(
            form_card,
            text="Period",
            bg=config.COLOR_SURFACE,
            fg=config.COLOR_MUTED,
            font=(config.FONT_FAMILY, 11),
        ).pack(anchor="w", padx=18)
        self.var_period = StringVar(value=self.app.db.get_default_period_name())
        self.cmb_period = ttk.Combobox(
            form_card, textvariable=self.var_period, width=16, state="readonly"
        )
        self.cmb_period.pack(anchor="w", padx=18, pady=(4, 16))

        ttk.Button(
            form_card,
            text="Mark",
            style="Accent.TButton",
            command=self._mark_pin,
        ).pack(anchor="w", padx=18, pady=(0, 8))

        self.lbl_status = Label(
            form_card,
            text="",
            bg=config.COLOR_SURFACE,
            fg=config.COLOR_MUTED,
            font=(config.FONT_FAMILY, 12, "bold"),
            wraplength=280,
            justify="left",
        )
        self.lbl_status.pack(anchor="w", padx=18, pady=(8, 16))

        cam_card = _card(body)
        cam_card.pack(side=RIGHT, fill=BOTH, expand=True)
        self.preview_label = Label(cam_card, bg=config.COLOR_PREVIEW)
        self.preview_label.pack(fill=BOTH, expand=True, padx=10, pady=10)
        cam_controls = Frame(cam_card, bg=config.COLOR_SURFACE)
        cam_controls.pack(fill=X, padx=10, pady=(0, 10))
        self.btn_start = ttk.Button(
            cam_controls,
            text="Start camera",
            style="Primary.TButton",
            command=self._start,
        )
        self.btn_start.pack(side=LEFT, padx=4)
        self.btn_stop = ttk.Button(
            cam_controls,
            text="Stop",
            style="Danger.TButton",
            command=self._stop,
            state="disabled",
        )
        self.btn_stop.pack(side=LEFT, padx=4)
        self.ent_roll.bind("<Return>", lambda _e: self._mark_pin())
        self.ent_pin.bind("<Return>", lambda _e: self._mark_pin())

    def _refresh_periods(self) -> None:
        names = [str(r["name"]) for r in self.app.db.list_periods()]
        if not names:
            names = [config.DEFAULT_PERIOD_NAME]
        self.cmb_period["values"] = names
        current = self.var_period.get()
        if current not in names:
            default = self.app.db.get_default_period_name()
            self.var_period.set(default if default in names else names[0])

    def _set_status(self, text: str, ok: bool | None = None) -> None:
        if ok is True:
            color = config.COLOR_ACCENT
        elif ok is False:
            color = config.COLOR_DANGER
        else:
            color = config.COLOR_MUTED
        self.lbl_status.config(text=text, fg=color)

    def _mark_pin(self) -> None:
        roll = self.var_roll.get().strip()
        pin = self.var_pin.get().strip()
        period = self.var_period.get().strip() or None
        if not roll:
            self._set_status("Enter a roll number.", ok=False)
            return
        if not pin:
            self._set_status("Enter a PIN.", ok=False)
            return
        try:
            inserted = self.app.db.mark_with_pin(
                roll, pin, period=period, source="kiosk"
            )
        except ValueError as exc:
            self._set_status(str(exc), ok=False)
            return
        student = self.app.db.get_student_by_roll(roll)
        name = student["name"] if student is not None else roll
        if inserted:
            self._set_status(f"Marked {name} ({roll}).", ok=True)
            self.var_pin.set("")
        else:
            self._set_status(
                f"{name} already marked for this period.", ok=False
            )

    def _start(self) -> None:
        if self._running:
            return
        if not self.app.face_engine.is_loaded:
            try:
                loaded = self.app.face_engine.load()
            except Exception as exc:  # noqa: BLE001
                self._set_status(f"Model error: {exc}", ok=False)
                return
            if not loaded:
                self._set_status("No trained model. Train first.", ok=False)
                return
        self._camera = Camera()
        if not self._camera.open():
            self._set_status("Could not open the webcam.", ok=False)
            return
        self._running = True
        self._marked_today.clear()
        self._motion.clear()
        self.btn_start["state"] = "disabled"
        self.btn_stop["state"] = "normal"
        self._set_status("Camera running — move slightly to mark.", ok=None)
        self._loop()

    def _loop(self) -> None:
        if not self._running or self._camera is None:
            return
        ok, frame = self._camera.read()
        if not ok or frame is None:
            self._after_id = self.after(30, self._loop)
            return
        period = self.var_period.get().strip() or None
        process_live_faces(
            self.app,
            frame,
            period=period,
            source="kiosk",
            marked_today=self._marked_today,
            motion=self._motion,
            on_unknown=self._log_unknown,
            on_insert=lambda row, _sid, _c: self._set_status(
                f"Camera marked {row['name']} ({row['roll_number']}).", ok=True
            ),
        )
        _show_preview(self.preview_label, frame)
        self._after_id = self.after(40, self._loop)

    def _log_unknown(self, face, conf: float) -> None:
        now = datetime.now()
        last = self._last_unknown_at
        if (
            last is not None
            and 0 <= (now - last).total_seconds() < config.UNKNOWN_LOG_COOLDOWN_SECONDS
        ):
            return
        crop_path = save_unknown_crop(face, conf)
        if self.app.db.log_unknown_face(confidence=conf, path=crop_path):
            self._last_unknown_at = now
            self.app.unknown_sightings += 1
        elif crop_path:
            try:
                Path(crop_path).unlink(missing_ok=True)
            except OSError:
                pass

    def _stop(self) -> None:
        self._running = False
        if self._after_id:
            try:
                self.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None
        if self._camera:
            self._camera.release()
            self._camera = None
        self.btn_start["state"] = "normal"
        self.btn_stop["state"] = "disabled"

    def on_show(self) -> None:
        self._refresh_periods()

    def on_hide(self) -> None:
        self._stop()


# ---------------------------------------------------------------------------
# Records / attendance log page
# ---------------------------------------------------------------------------
class RecordsPage(Page):
    title = "Attendance Records"

    def __init__(self, master: Frame, app: "AttendanceApp") -> None:
        super().__init__(master, app)
        self._mode = "present"  # or "absentees"

        ttk.Label(self, text="Attendance Records", style="H1.TLabel").pack(
            anchor="w", padx=30, pady=(24, 12)
        )

        controls = Frame(self, bg=config.COLOR_BG)
        controls.pack(fill=X, padx=24)
        ttk.Label(controls, text="Date (YYYY-MM-DD):").pack(side=LEFT, padx=4)
        self.var_date = StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        ttk.Entry(controls, textvariable=self.var_date, width=14).pack(side=LEFT, padx=4)
        ttk.Label(controls, text="Section:").pack(side=LEFT, padx=(12, 4))
        self.var_section = StringVar(value="")
        self.cmb_section = ttk.Combobox(
            controls, textvariable=self.var_section, width=12, state="normal"
        )
        self.cmb_section.pack(side=LEFT, padx=4)
        ttk.Label(controls, text="Period:").pack(side=LEFT, padx=(12, 4))
        self.var_period = StringVar(value="")
        self.cmb_period = ttk.Combobox(
            controls, textvariable=self.var_period, width=12, state="normal"
        )
        self.cmb_period.pack(side=LEFT, padx=4)
        ttk.Button(controls, text="Filter", style="Primary.TButton",
                   command=self._reload).pack(side=LEFT, padx=4)
        ttk.Button(controls, text="Show All", style="Primary.TButton",
                   command=self._show_all).pack(side=LEFT, padx=4)
        ttk.Button(controls, text="Absentees", style="Primary.TButton",
                   command=self._show_absentees).pack(side=LEFT, padx=4)
        ttk.Button(controls, text="Export unknowns", style="Accent.TButton",
                   command=self._export_unknowns).pack(side=RIGHT, padx=4)
        ttk.Button(controls, text="Export JSON", style="Accent.TButton",
                   command=lambda: self._export(as_json=True)).pack(side=RIGHT, padx=4)
        ttk.Button(controls, text="Export CSV", style="Accent.TButton",
                   command=lambda: self._export(as_json=False)).pack(side=RIGHT, padx=4)

        card = _card(self)
        card.pack(fill=BOTH, expand=True, padx=24, pady=14)

        cols = (
            "date", "time", "time_out", "duration", "name", "roll",
            "department", "section", "period", "confidence", "status",
        )
        self.tree = ttk.Treeview(card, columns=cols, show="headings")
        for col, width in zip(
            cols, (100, 80, 80, 80, 150, 90, 100, 70, 80, 80, 80), strict=False
        ):
            heading = col.replace("_", " ").capitalize()
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill=BOTH, expand=True, padx=10, pady=10)

    def _section_filter(self) -> str | None:
        value = self.var_section.get().strip()
        return value or None

    def _period_filter(self) -> str | None:
        value = self.var_period.get().strip()
        return value or None

    def _refresh_section_choices(self) -> None:
        sections = self.app.db.list_sections()
        self.cmb_section["values"] = [""] + sections
        periods = [str(r["name"]) for r in self.app.db.list_periods()]
        self.cmb_period["values"] = [""] + periods

    def _populate(self, df) -> None:
        self.tree.delete(*self.tree.get_children())
        for _, r in df.iterrows():
            duration = r.get("duration_seconds")
            if duration is None or str(duration) in {"", "nan", "None"}:
                duration_s = "—"
            else:
                try:
                    duration_s = f"{int(duration)}s"
                except (TypeError, ValueError):
                    duration_s = "—"
            time_out = r.get("time_out")
            if time_out is None or str(time_out) in {"", "nan", "None"}:
                time_out_s = "—"
            else:
                time_out_s = str(time_out)
            self.tree.insert(
                "",
                "end",
                values=(
                    r.get("date", ""),
                    r.get("time", ""),
                    time_out_s,
                    duration_s,
                    r.get("name", ""),
                    r.get("roll_number", ""),
                    r.get("department", "") or "—",
                    r.get("section", "") or "—",
                    r.get("period", "") or "—",
                    f"{r['confidence']:.1f}" if r.get("confidence") is not None else "—",
                    r.get("status", ""),
                ),
            )

    def _populate_absentees(self, df, date: str) -> None:
        self.tree.delete(*self.tree.get_children())
        for _, r in df.iterrows():
            self.tree.insert(
                "",
                "end",
                values=(
                    date,
                    "—",
                    "—",
                    "—",
                    r.get("name", ""),
                    r.get("roll_number", ""),
                    r.get("department", "") or "—",
                    r.get("section", "") or "—",
                    "—",
                    "—",
                    "Absent",
                ),
            )

    def _reload(self) -> None:
        self._mode = "present"
        date = self.var_date.get().strip() or None
        try:
            if date:
                datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Bad date", "Date must be YYYY-MM-DD.")
            return
        df = self.app.db.get_attendance(
            date=date, section=self._section_filter(), period=self._period_filter()
        )
        self._populate(df)

    def _show_all(self) -> None:
        self._mode = "present"
        self.var_date.set("")
        df = self.app.db.get_attendance(
            section=self._section_filter(), period=self._period_filter()
        )
        self._populate(df)

    def _show_absentees(self) -> None:
        self._mode = "absentees"
        date = self.var_date.get().strip() or datetime.now().strftime("%Y-%m-%d")
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Bad date", "Date must be YYYY-MM-DD.")
            return
        self.var_date.set(date)
        df = self.app.db.get_absentees(date=date, section=self._section_filter())
        self._populate_absentees(df, date)

    def _export(self, as_json: bool = False) -> None:
        date = self.var_date.get().strip() or None
        section = self._section_filter()
        period = self._period_filter()
        ext = ".json" if as_json else ".csv"
        kind = "JSON" if as_json else "CSV"
        filetypes = (
            [("JSON files", "*.json"), ("All files", "*.*")]
            if as_json
            else [("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if self._mode == "absentees":
            if not date:
                date = datetime.now().strftime("%Y-%m-%d")
            default_name = f"absentees_{date}{ext}"
            path = filedialog.asksaveasfilename(
                defaultextension=ext,
                initialdir=str(config.EXPORTS_DIR),
                initialfile=default_name,
                filetypes=filetypes,
            )
            if not path:
                return
            if as_json:
                out = self.app.db.export_absentees_json(Path(path), date=date, section=section)
            else:
                out = self.app.db.export_absentees_csv(Path(path), date=date, section=section)
            messagebox.showinfo("Exported", f"Saved absentees {kind} to:\n{out}")
            return

        default_name = f"attendance_{date or 'all'}{ext}"
        path = filedialog.asksaveasfilename(
            defaultextension=ext,
            initialdir=str(config.EXPORTS_DIR),
            initialfile=default_name,
            filetypes=filetypes,
        )
        if not path:
            return
        if as_json:
            out = self.app.db.export_attendance_json(
                Path(path), date=date, section=section, period=period
            )
        else:
            out = self.app.db.export_attendance_csv(
                Path(path), date=date, section=section, period=period
            )
        messagebox.showinfo("Exported", f"Saved {kind} to:\n{out}")

    def _export_unknowns(self) -> None:
        date = self.var_date.get().strip() or None
        try:
            if date:
                datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Bad date", "Date must be YYYY-MM-DD.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            initialdir=str(config.EXPORTS_DIR),
            initialfile=f"unknowns_{date or 'all'}.csv",
            filetypes=[
                ("CSV files", "*.csv"),
                ("JSON files", "*.json"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        if Path(path).suffix.lower() == ".json":
            out = self.app.db.export_unknown_faces_json(Path(path), date=date)
            kind = "JSON"
        else:
            out = self.app.db.export_unknown_faces_csv(Path(path), date=date)
            kind = "CSV"
        messagebox.showinfo("Exported", f"Saved unknown-face {kind} to:\n{out}")

    def on_show(self) -> None:
        self._refresh_section_choices()
        self._reload()


# ---------------------------------------------------------------------------
# Weekly stats page
# ---------------------------------------------------------------------------
class StatsPage(Page):
    title = "Weekly Stats"

    def __init__(self, master: Frame, app: "AttendanceApp") -> None:
        super().__init__(master, app)

        ttk.Label(self, text="Weekly attendance", style="H1.TLabel").pack(
            anchor="w", padx=30, pady=(24, 6)
        )
        ttk.Label(
            self,
            text="Present / late / excused / absent per student over a 5- or 7-day window.",
        ).pack(anchor="w", padx=30, pady=(0, 12))

        controls = Frame(self, bg=config.COLOR_BG)
        controls.pack(fill=X, padx=24)
        ttk.Label(controls, text="Start date:").pack(side=LEFT, padx=4)
        self.var_start = StringVar(value=week_start())
        ttk.Entry(controls, textvariable=self.var_start, width=14).pack(side=LEFT, padx=4)
        ttk.Label(controls, text="Days:").pack(side=LEFT, padx=(12, 4))
        self.var_days = StringVar(value=str(config.WEEK_DAYS_DEFAULT))
        self.cmb_days = ttk.Combobox(
            controls,
            textvariable=self.var_days,
            width=6,
            values=("5", "7"),
            state="readonly",
        )
        self.cmb_days.pack(side=LEFT, padx=4)
        ttk.Label(controls, text="Section:").pack(side=LEFT, padx=(12, 4))
        self.var_section = StringVar(value="")
        self.cmb_section = ttk.Combobox(
            controls, textvariable=self.var_section, width=12, state="normal"
        )
        self.cmb_section.pack(side=LEFT, padx=4)
        ttk.Button(
            controls, text="Refresh", style="Primary.TButton", command=self._reload
        ).pack(side=LEFT, padx=4)
        ttk.Button(
            controls, text="Export HTML", style="Accent.TButton",
            command=self._export_html,
        ).pack(side=RIGHT, padx=4)
        ttk.Button(
            controls, text="Export JSON", style="Accent.TButton",
            command=lambda: self._export(as_json=True),
        ).pack(side=RIGHT, padx=4)
        ttk.Button(
            controls, text="Export CSV", style="Accent.TButton",
            command=lambda: self._export(as_json=False),
        ).pack(side=RIGHT, padx=4)

        self.lbl_summary = ttk.Label(self, text="")
        self.lbl_summary.pack(anchor="w", padx=30, pady=(10, 0))

        card = _card(self)
        card.pack(fill=BOTH, expand=True, padx=24, pady=14)

        cols = ("roll", "name", "section", "present", "late", "excused", "absent", "rate")
        self.tree = ttk.Treeview(card, columns=cols, show="headings")
        for col, width in zip(cols, (100, 180, 80, 70, 60, 70, 70, 80), strict=False):
            self.tree.heading(col, text=col.capitalize())
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill=BOTH, expand=True, padx=10, pady=10)

    def _reload(self) -> None:
        start = self.var_start.get().strip() or week_start()
        try:
            datetime.strptime(start, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Bad date", "Start date must be YYYY-MM-DD.")
            return
        try:
            days = int(self.var_days.get() or config.WEEK_DAYS_DEFAULT)
        except ValueError:
            days = config.WEEK_DAYS_DEFAULT
        section = self.var_section.get().strip() or None
        df = self.app.db.weekly_summary(start, days=days, section=section)
        rate = self.app.db.weekly_rate(start, days=days, section=section)
        self.lbl_summary.config(
            text=f"Window {start} · {days} day(s) · overall rate {rate:.1f}%"
        )
        self.tree.delete(*self.tree.get_children())
        for _, r in df.iterrows():
            self.tree.insert(
                "",
                "end",
                values=(
                    r["roll_number"],
                    r["name"],
                    r.get("section") or "—",
                    int(r["present"]),
                    int(r["late"]),
                    int(r.get("excused", 0)),
                    int(r["absent"]),
                    f"{r['attendance_rate']:.1f}%",
                ),
            )

    def _window(self) -> tuple[str, str, int, str | None] | None:
        start = self.var_start.get().strip() or week_start()
        try:
            datetime.strptime(start, "%Y-%m-%d")
        except ValueError:
            messagebox.showerror("Bad date", "Start date must be YYYY-MM-DD.")
            return None
        try:
            days = int(self.var_days.get() or config.WEEK_DAYS_DEFAULT)
        except ValueError:
            days = config.WEEK_DAYS_DEFAULT
        days = max(1, days)
        end = (
            datetime.strptime(start, "%Y-%m-%d") + timedelta(days=days - 1)
        ).strftime("%Y-%m-%d")
        section = self.var_section.get().strip() or None
        return start, end, days, section

    def _export(self, as_json: bool = False) -> None:
        window = self._window()
        if window is None:
            return
        start, end, _days, section = window
        ext = ".json" if as_json else ".csv"
        kind = "JSON" if as_json else "CSV"
        filetypes = (
            [("JSON files", "*.json"), ("All files", "*.*")]
            if as_json
            else [("CSV files", "*.csv"), ("All files", "*.*")]
        )
        path = filedialog.asksaveasfilename(
            defaultextension=ext,
            initialdir=str(config.EXPORTS_DIR),
            initialfile=f"range_{start}_{end}{ext}",
            filetypes=filetypes,
        )
        if not path:
            return
        if as_json:
            out = self.app.db.export_range_report_json(
                Path(path), start, end, section=section
            )
        else:
            out = self.app.db.export_range_report_csv(
                Path(path), start, end, section=section
            )
        messagebox.showinfo("Exported", f"Saved range report {kind} to:\n{out}")

    def _export_html(self) -> None:
        window = self._window()
        if window is None:
            return
        start, end, _days, section = window
        path = filedialog.asksaveasfilename(
            defaultextension=".html",
            initialdir=str(config.EXPORTS_DIR),
            initialfile=f"range_{start}_{end}.html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
        )
        if not path:
            return
        out = self.app.db.export_range_report_html(
            Path(path), start, end, section=section
        )
        messagebox.showinfo("Exported", f"Saved range report HTML to:\n{out}")

    def on_show(self) -> None:
        self.cmb_section["values"] = [""] + self.app.db.list_sections()
        if not self.var_start.get().strip():
            self.var_start.set(week_start())
        self._reload()


# ---------------------------------------------------------------------------
# Settings page
# ---------------------------------------------------------------------------
class SettingsPage(Page):
    title = "Settings"

    def __init__(self, master: Frame, app: "AttendanceApp") -> None:
        super().__init__(master, app)

        ttk.Label(self, text="Settings", style="H1.TLabel").pack(
            anchor="w", padx=30, pady=(24, 6)
        )
        ttk.Label(
            self,
            text="Theme, recognition threshold, and late-arrival grace. Stored in the database.",
        ).pack(anchor="w", padx=30, pady=(0, 18))

        card = _card(self)
        card.pack(fill=X, padx=24, pady=(0, 24))

        self.var_theme = StringVar(value=config.get_theme())
        self.var_threshold = StringVar(value=f"{config.get_confidence_threshold():.1f}")
        self.var_grace = StringVar(value=str(self.app.db.get_grace_minutes()))
        self.var_period = StringVar(value=self.app.db.get_default_period_name())

        Label(
            card, text="Appearance",
            bg=config.COLOR_SURFACE, fg=config.COLOR_TEXT,
            font=(config.FONT_FAMILY, 14, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 8))

        Label(
            card, text="Theme",
            bg=config.COLOR_SURFACE, fg=config.COLOR_TEXT,
        ).grid(row=1, column=0, sticky="w", padx=18, pady=6)
        theme_row = Frame(card, bg=config.COLOR_SURFACE)
        theme_row.grid(row=1, column=1, sticky="w", padx=18, pady=6)
        ttk.Radiobutton(
            theme_row, text="Light", value="light", variable=self.var_theme
        ).pack(side=LEFT, padx=(0, 12))
        ttk.Radiobutton(
            theme_row, text="Dark", value="dark", variable=self.var_theme
        ).pack(side=LEFT)

        Label(
            card, text="Recognition",
            bg=config.COLOR_SURFACE, fg=config.COLOR_TEXT,
            font=(config.FONT_FAMILY, 14, "bold"),
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 8))

        Label(
            card, text="LBPH threshold",
            bg=config.COLOR_SURFACE, fg=config.COLOR_TEXT,
        ).grid(row=3, column=0, sticky="w", padx=18, pady=6)
        ttk.Entry(card, textvariable=self.var_threshold, width=12).grid(
            row=3, column=1, sticky="w", padx=18, pady=6
        )
        Label(
            card,
            text="Lower distance = stricter. Faces worse than this are Unknown.",
            bg=config.COLOR_SURFACE, fg=config.COLOR_MUTED,
            font=(config.FONT_FAMILY, 10),
        ).grid(row=4, column=0, columnspan=2, sticky="w", padx=18, pady=(0, 8))

        Label(
            card, text="Periods",
            bg=config.COLOR_SURFACE, fg=config.COLOR_TEXT,
            font=(config.FONT_FAMILY, 14, "bold"),
        ).grid(row=5, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 8))

        Label(
            card, text="Default period",
            bg=config.COLOR_SURFACE, fg=config.COLOR_TEXT,
        ).grid(row=6, column=0, sticky="w", padx=18, pady=6)
        self.cmb_period = ttk.Combobox(
            card, textvariable=self.var_period, width=16, state="readonly"
        )
        self.cmb_period.grid(row=6, column=1, sticky="w", padx=18, pady=6)

        Label(
            card, text="Late grace (minutes)",
            bg=config.COLOR_SURFACE, fg=config.COLOR_TEXT,
        ).grid(row=7, column=0, sticky="w", padx=18, pady=6)
        ttk.Entry(card, textvariable=self.var_grace, width=12).grid(
            row=7, column=1, sticky="w", padx=18, pady=6
        )
        Label(
            card,
            text="Arrival after period start + grace is marked Late.",
            bg=config.COLOR_SURFACE, fg=config.COLOR_MUTED,
            font=(config.FONT_FAMILY, 10),
        ).grid(row=8, column=0, columnspan=2, sticky="w", padx=18, pady=(0, 8))

        Label(
            card, text="School week",
            bg=config.COLOR_SURFACE, fg=config.COLOR_TEXT,
            font=(config.FONT_FAMILY, 14, "bold"),
        ).grid(row=9, column=0, columnspan=2, sticky="w", padx=18, pady=(16, 8))

        Label(
            card, text="Weekend",
            bg=config.COLOR_SURFACE, fg=config.COLOR_TEXT,
        ).grid(row=10, column=0, sticky="w", padx=18, pady=6)
        weekend_row = Frame(card, bg=config.COLOR_SURFACE)
        weekend_row.grid(row=10, column=1, sticky="w", padx=18, pady=6)
        self.var_sat = BooleanVar(value=True)
        self.var_sun = BooleanVar(value=True)
        ttk.Checkbutton(
            weekend_row, text="Saturday", variable=self.var_sat
        ).pack(side=LEFT, padx=(0, 12))
        ttk.Checkbutton(
            weekend_row, text="Sunday", variable=self.var_sun
        ).pack(side=LEFT)

        ttk.Button(
            card, text="Save settings", style="Primary.TButton",
            command=self._save,
        ).grid(row=11, column=0, columnspan=2, sticky="w", padx=18, pady=(12, 18))

        holidays = _card(self)
        holidays.pack(fill=BOTH, expand=True, padx=24, pady=(0, 24))
        Label(
            holidays, text="Holidays",
            bg=config.COLOR_SURFACE, fg=config.COLOR_TEXT,
            font=(config.FONT_FAMILY, 14, "bold"),
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=18, pady=(16, 8))

        self.var_holiday_date = StringVar()
        self.var_holiday_name = StringVar()
        Label(
            holidays, text="Date (YYYY-MM-DD)",
            bg=config.COLOR_SURFACE, fg=config.COLOR_TEXT,
        ).grid(row=1, column=0, sticky="w", padx=18, pady=4)
        ttk.Entry(holidays, textvariable=self.var_holiday_date, width=16).grid(
            row=1, column=1, sticky="w", padx=8, pady=4
        )
        Label(
            holidays, text="Name",
            bg=config.COLOR_SURFACE, fg=config.COLOR_TEXT,
        ).grid(row=1, column=2, sticky="w", padx=8, pady=4)
        ttk.Entry(holidays, textvariable=self.var_holiday_name, width=22).grid(
            row=1, column=3, sticky="we", padx=18, pady=4
        )
        ttk.Button(
            holidays, text="Add", style="Primary.TButton",
            command=self._add_holiday,
        ).grid(row=2, column=0, sticky="w", padx=18, pady=(4, 8))
        ttk.Button(
            holidays, text="Remove selected", style="Danger.TButton",
            command=self._remove_holiday,
        ).grid(row=2, column=1, columnspan=2, sticky="w", padx=8, pady=(4, 8))

        list_frame = Frame(holidays, bg=config.COLOR_SURFACE)
        list_frame.grid(
            row=3, column=0, columnspan=4, sticky="nsew", padx=18, pady=(4, 18)
        )
        scroll = ttk.Scrollbar(list_frame, orient="vertical")
        self.holiday_list = Listbox(
            list_frame,
            height=6,
            yscrollcommand=scroll.set,
            exportselection=False,
        )
        scroll.config(command=self.holiday_list.yview)
        self.holiday_list.pack(side=LEFT, fill=BOTH, expand=True)
        scroll.pack(side=RIGHT, fill=Y)
        holidays.columnconfigure(3, weight=1)
        holidays.rowconfigure(3, weight=1)
        self._holiday_dates: list[str] = []

    def _refresh_periods(self) -> None:
        names = [str(r["name"]) for r in self.app.db.list_periods()]
        self.cmb_period["values"] = names
        if names and self.var_period.get() not in names:
            self.var_period.set(names[0])

    def _save(self) -> None:
        try:
            threshold = float(self.var_threshold.get().strip())
            config.set_confidence_threshold(threshold)
        except ValueError:
            messagebox.showerror(
                "Threshold",
                "Confidence threshold must be a positive number (e.g. 70).",
            )
            return
        try:
            grace = int(self.var_grace.get().strip())
            if grace < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Grace", "Grace minutes must be an integer ≥ 0.")
            return

        theme = self.var_theme.get().strip().lower() or "light"
        period = self.var_period.get().strip()
        weekend_parts: list[str] = []
        if self.var_sat.get():
            weekend_parts.append("6")
        if self.var_sun.get():
            weekend_parts.append("7")
        self.app.db.set_weekend_days(
            ",".join(weekend_parts) if weekend_parts else "none"
        )
        self.app.db.set_setting("confidence_threshold", str(threshold))
        self.app.db.set_setting("grace_minutes", str(grace))
        if period:
            self.app.db.set_setting("default_period", period)
        self.app.db.set_setting("theme", theme)

        if theme != config.get_theme():
            self.app.set_theme(theme)
            return
        messagebox.showinfo("Saved", "Settings updated.")

    def _reload_holidays(self) -> None:
        self.holiday_list.delete(0, END)
        self._holiday_dates = []
        for row in self.app.db.list_holidays():
            day = str(row["date"])
            name = str(row["name"] or "Holiday")
            self._holiday_dates.append(day)
            self.holiday_list.insert(END, f"{day} — {name}")

    def _add_holiday(self) -> None:
        date = self.var_holiday_date.get().strip()
        name = self.var_holiday_name.get().strip() or "Holiday"
        try:
            self.app.db.add_holiday(date, name)
        except ValueError:
            messagebox.showerror("Holiday", "Date must be YYYY-MM-DD.")
            return
        self.var_holiday_date.set("")
        self.var_holiday_name.set("")
        self._reload_holidays()

    def _remove_holiday(self) -> None:
        sel = self.holiday_list.curselection()
        if not sel:
            messagebox.showinfo("Holidays", "Select a holiday to remove.")
            return
        date = self._holiday_dates[int(sel[0])]
        self.app.db.delete_holiday(date)
        self._reload_holidays()

    def on_show(self) -> None:
        self.var_theme.set(config.get_theme())
        self.var_threshold.set(f"{config.get_confidence_threshold():.1f}")
        self.var_grace.set(str(self.app.db.get_grace_minutes()))
        self.var_period.set(self.app.db.get_default_period_name())
        days = self.app.db.get_weekend_days()
        self.var_sat.set(6 in days)
        self.var_sun.set(7 in days)
        self._refresh_periods()
        self._reload_holidays()


# ---------------------------------------------------------------------------
# Students page (manage list, delete)
# ---------------------------------------------------------------------------
class StudentsPage(Page):
    title = "Students"

    def __init__(self, master: Frame, app: "AttendanceApp") -> None:
        super().__init__(master, app)

        header = Frame(self, bg=config.COLOR_BG)
        header.pack(fill=X, padx=30, pady=(24, 12))
        ttk.Label(header, text="Registered Students", style="H1.TLabel").pack(side=LEFT)
        ttk.Button(
            header, text="Refresh", style="Primary.TButton",
            command=self._reload,
        ).pack(side=RIGHT, padx=4)
        ttk.Button(
            header, text="Delete Selected", style="Danger.TButton",
            command=self._delete,
        ).pack(side=RIGHT, padx=4)
        ttk.Button(
            header, text="Mark Excused", style="Primary.TButton",
            command=self._mark_excused,
        ).pack(side=RIGHT, padx=4)
        ttk.Button(
            header, text="Import CSV…", style="Accent.TButton",
            command=self._import_csv,
        ).pack(side=RIGHT, padx=4)
        ttk.Button(
            header, text="Set PIN…", style="Primary.TButton",
            command=self._set_pin,
        ).pack(side=RIGHT, padx=4)

        filters = Frame(self, bg=config.COLOR_BG)
        filters.pack(fill=X, padx=30, pady=(0, 8))
        ttk.Label(filters, text="Section:").pack(side=LEFT, padx=4)
        self.var_section = StringVar(value="")
        self.cmb_section = ttk.Combobox(
            filters, textvariable=self.var_section, width=14, state="normal"
        )
        self.cmb_section.pack(side=LEFT, padx=4)
        ttk.Button(
            filters, text="Filter", style="Primary.TButton", command=self._reload
        ).pack(side=LEFT, padx=4)

        card = _card(self)
        card.pack(fill=BOTH, expand=True, padx=24, pady=14)

        cols = (
            "id", "roll", "name", "email", "department", "section",
            "samples", "registered_on",
        )
        self.tree = ttk.Treeview(card, columns=cols, show="headings")
        for col, width in zip(
            cols, (50, 100, 180, 160, 110, 80, 80, 150), strict=False
        ):
            self.tree.heading(col, text=col.replace("_", " ").capitalize())
            self.tree.column(col, width=width, anchor="w")
        self.tree.pack(fill=BOTH, expand=True, padx=10, pady=10)

    def _section_filter(self) -> str | None:
        value = self.var_section.get().strip()
        return value or None

    def _reload(self) -> None:
        sections = self.app.db.list_sections()
        self.cmb_section["values"] = [""] + sections
        self.tree.delete(*self.tree.get_children())
        samples_by_id = {
            int(item["id"]): int(item.get("samples") or 0)
            for item in self.app.db.enrollment_roster()
        }
        for r in self.app.db.list_students(section=self._section_filter()):
            self.tree.insert(
                "", "end",
                values=(
                    r["id"], r["roll_number"], r["name"],
                    r["email"] or "—", r["department"] or "—",
                    r["section"] or "—",
                    samples_by_id.get(int(r["id"]), 0),
                    r["registered_on"],
                ),
            )

    def _set_pin(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select a student to set a PIN.")
            return
        values = self.tree.item(sel[0])["values"]
        roll = str(values[1])
        name = values[2]
        pin = simpledialog.askstring(
            "Set PIN",
            f"4–8 digit PIN for {name} ({roll}):",
            show="*",
            parent=self.winfo_toplevel(),
        )
        if pin is None:
            return
        try:
            self.app.db.set_pin(roll, pin)
        except ValueError as exc:
            messagebox.showerror("PIN", str(exc))
            return
        messagebox.showinfo("PIN", f"PIN saved for {roll}.")

    def _import_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Import students CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            result = self.app.db.import_students_csv(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Import failed", str(exc))
            return
        self._reload()
        messagebox.showinfo(
            "Imported",
            f"Added {result['added']} student(s).\n"
            f"Skipped {result['skipped']} existing or invalid roll(s).",
        )

    def _mark_excused(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select a student to mark Excused.")
            return
        values = self.tree.item(sel[0])["values"]
        student_id = int(values[0])
        name = values[2]
        today = datetime.now().strftime("%Y-%m-%d")
        if not messagebox.askyesno(
            "Mark Excused",
            f"Mark '{name}' as Excused for {today}?",
        ):
            return
        inserted = self.app.db.mark_attendance(
            student_id, status="Excused", source="gui"
        )
        if not inserted:
            messagebox.showinfo(
                "Skipped",
                "Cooldown is still active for this student. Try again shortly.",
            )
            return
        messagebox.showinfo("Excused", f"{name} marked Excused for {today}.")

    def _delete(self) -> None:
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("No selection", "Select a student to delete.")
            return
        values = self.tree.item(sel[0])["values"]
        student_id = int(values[0])
        name = values[2]
        if not messagebox.askyesno(
            "Confirm delete",
            f"Delete '{name}' and all of their attendance records?\n"
            "Their face samples on disk will also be removed.",
        ):
            return

        student_dir = config.FACES_DIR / str(student_id)
        if student_dir.exists():
            for p in student_dir.glob("*"):
                try:
                    p.unlink()
                except OSError:
                    pass
            try:
                student_dir.rmdir()
            except OSError:
                pass

        self.app.db.delete_student(student_id)
        mark_model_stale(f"Student id={student_id} deleted")
        self._reload()
        messagebox.showinfo(
            "Deleted",
            "Student removed. Remember to retrain the model from the "
            "'Train Model' tab.",
        )

    def on_show(self) -> None:
        self._reload()


# ---------------------------------------------------------------------------
# Main app shell
# ---------------------------------------------------------------------------
class AttendanceApp:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title(f"{config.APP_NAME} v{config.APP_VERSION}")
        self.root.geometry(config.WINDOW_SIZE)
        self.root.minsize(960, 600)

        self.db = Database()
        self._load_settings()
        _apply_style(self.root)

        self.face_engine = FaceEngine()
        try:
            self.face_engine.load()
        except Exception:
            pass

        self.unknown_sightings = 0
        self.var_dark = BooleanVar(value=config.get_theme() == "dark")

        self._build_layout()
        self._build_pages()
        self.show("dashboard")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _load_settings(self) -> None:
        theme = self.db.get_setting("theme") or "light"
        config.apply_theme(theme)
        raw = self.db.get_setting("confidence_threshold")
        if raw:
            try:
                config.set_confidence_threshold(float(raw))
            except ValueError:
                pass

    def _build_layout(self) -> None:
        self.sidebar = Frame(self.root, bg=config.COLOR_SIDEBAR, width=220)
        self.sidebar.pack(side=LEFT, fill=Y)
        self.sidebar.pack_propagate(False)

        ttk.Label(
            self.sidebar,
            text=" Attendance",
            style="SidebarTitle.TLabel",
        ).pack(fill=X, padx=18, pady=(22, 4))
        ttk.Label(
            self.sidebar,
            text="Face Recognition System",
            style="Sidebar.TLabel",
        ).pack(fill=X, padx=18, pady=(0, 22))

        self._nav_buttons: dict[str, ttk.Button] = {}
        for key, label in [
            ("dashboard", "Dashboard"),
            ("register", "Register Student"),
            ("train", "Train Model"),
            ("attendance", "Take Attendance"),
            ("kiosk", "Kiosk"),
            ("calendar", "Calendar"),
            ("records", "Records"),
            ("stats", "Weekly Stats"),
            ("students", "Students"),
            ("settings", "Settings"),
        ]:
            b = ttk.Button(
                self.sidebar, text=f"  {label}",
                style="Nav.TButton",
                command=lambda k=key: self.show(k),
            )
            b.pack(fill=X, padx=0, pady=1)
            self._nav_buttons[key] = b

        footer = Frame(self.sidebar, bg=config.COLOR_SIDEBAR)
        footer.pack(side=BOTTOM, fill=X, padx=12, pady=12)
        self.var_dark = BooleanVar(value=config.get_theme() == "dark")
        ttk.Checkbutton(
            footer,
            text="Dark theme",
            variable=self.var_dark,
            command=self._on_theme_toggle,
            style="TCheckbutton",
        ).pack(anchor="w", pady=(0, 8))
        # Checkbutton style uses surface colors; paint the footer widget bg.
        ttk.Label(
            footer,
            text=f"v{config.APP_VERSION}",
            style="SidebarMuted.TLabel",
        ).pack(anchor="w")

        self.content = Frame(self.root, bg=config.COLOR_BG)
        self.content.pack(side=RIGHT, fill=BOTH, expand=True)

    def _on_theme_toggle(self) -> None:
        desired = "dark" if self.var_dark.get() else "light"
        if desired != config.get_theme():
            self.set_theme(desired)

    def set_theme(self, name: str) -> None:
        if self._current_key is not None:
            try:
                self.pages[self._current_key].on_hide()
            except Exception:
                pass
        applied = config.apply_theme(name)
        self.db.set_setting("theme", applied)
        current = self._current_key
        self._current_key = None
        for widget in self.root.winfo_children():
            widget.destroy()
        self.root.configure(bg=config.COLOR_BG)
        _apply_style(self.root)
        self.var_dark = BooleanVar(value=applied == "dark")
        self._build_layout()
        self._build_pages()
        self.show(current or "dashboard")

    def _build_pages(self) -> None:
        self.pages: dict[str, Page] = {
            "dashboard":  DashboardPage(self.content, self),
            "register":   RegisterPage(self.content, self),
            "train":      TrainPage(self.content, self),
            "attendance": AttendancePage(self.content, self),
            "kiosk":      KioskPage(self.content, self),
            "calendar":   CalendarPage(self.content, self),
            "records":    RecordsPage(self.content, self),
            "stats":      StatsPage(self.content, self),
            "students":   StudentsPage(self.content, self),
            "settings":   SettingsPage(self.content, self),
        }
        self._current_key: str | None = None

    def show(self, key: str) -> None:
        if key == self._current_key:
            return
        if self._current_key is not None:
            self.pages[self._current_key].on_hide()
            self.pages[self._current_key].pack_forget()

        for k, btn in self._nav_buttons.items():
            btn.configure(style="NavActive.TButton" if k == key else "Nav.TButton")

        page = self.pages[key]
        page.pack(fill=BOTH, expand=True)
        page.on_show()
        self._current_key = key

    def _on_close(self) -> None:
        if self._current_key is not None:
            try:
                self.pages[self._current_key].on_hide()
            except Exception:
                pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
