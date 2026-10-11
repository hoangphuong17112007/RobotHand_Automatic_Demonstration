"""Cau noi RTC <-> AI: Lock-free Ring Buffer dang SPSC (single-producer/single-consumer).

Nguyen tac O(1):
- Moi kenh (snapshot, proposal) la mot ring buffer co kich thuoc co dinh.
- Producer ghi payload vao slot roi moi tang trunk tu (seq) len 1 (commit).
- Consumer doc seq (nguyen tu nho GIL), doc slot o seq-1, va kiem tra lai seq
  (double-check) de chan doc "gom" khi slot bi ghi de trong luc doc.
- Khong dung khoa mutex nao cho duong dan doc/ghi nong; read-toi-luc nao la
  truy cap duy nhat, khong bao gio bi block.
- `threading.Event` chi duoc dung ngoai le: cho tinh hieu (blocking wait) boi
  thread tieu dung, khong tham gia vao duong dan dong bo du lieu.

Packet (Snapshot/Proposal) deu mang Timestamp (ts = thoi gian sim luc chup/infer,
wall = dong ho tuong) de khau tru do tre (Latency Compensation).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Snapshot:
    ts: float                    # sim time (s) luc chup anh
    wall: float                  # wall clock (s) luc chup anh
    candidates: list = field(default_factory=list)
    patches: list = field(default_factory=list)   # list[np.ndarray 24x24]
    clouds: list = field(default_factory=list)    # list[np.ndarray (n,3)] per candidate
    depth: np.ndarray | None = None
    seg: np.ndarray | None = None


@dataclass
class Proposal:
    ts: float                    # sim time (s) luc chup anh goc
    wall: float                  # wall clock (s) luc node AI xuat proposal
    x: float
    y: float
    z_top: float
    yaw: float = 0.0             # huong truc kep (rad, quanh z) -> RTC dat close axis
    clearance: float = 1.0       # khe gap nho nhat (m) quanh vi tri gap
    grasp_ready: bool = True     # AI da xac nhan: gắp duoc, khong bi can tro
    blocked: set = field(default_factory=set)   # cac linh kien dang la vat can
    body: int | None = None
    name: str = ""
    score: float = 0.0
    # "khieu" (poke) de linh kien ra khoi goc/vach truoc khi gap; thay vi gap ngay.
    action: str = "grasp"        # "grasp" | "poke"
    poke_xy: tuple = None        # (x, y) tam linh kien can khieu
    poke_dir: tuple = None       # (dx, dy) don vi huong day (song song vach, ra khoi goc)


class Bridge:
    """Ring buffer khong khoa cho moi kenh; co thanh ghi seq de commit."""

    def __init__(self, history_size: int = 3, poke_budget: int = 4):
        self._snap_n = max(1, history_size)
        self._prop_n = max(1, history_size)
        self._snap_ring: list[Snapshot | None] = [None] * self._snap_n
        self._prop_ring: list[Proposal | None] = [None] * self._prop_n
        self._snap_seq = 0                       # chi producer (RtcNode) ghi
        self._prop_seq = 0                       # chi producer (AiNode) ghi
        self._blocked = frozenset()
        self._poke_budget = max(0, poke_budget)
        self._snap_ev = threading.Event()
        self._prop_ev = threading.Event()
        self._stop = threading.Event()

    # -------------------------------------------------- kenh blocked (swap ref)

    def set_blocked(self, bodies: set[int]) -> None:
        self._blocked = frozenset(bodies)        # swap nguyen tu (GIL)

    def blocked_ids(self) -> set[int]:
        return set(self._blocked)

    # --------------------------------------------------- kenh poke budget (swap ref)

    def set_poke_budget(self, n: int) -> None:
        self._poke_budget = max(0, int(n))

    def poke_budget(self) -> int:
        return self._poke_budget

    # --------------------------------------------- publish (producer, O(1))

    def publish_snapshot(self, snap: Snapshot) -> None:
        i = self._snap_seq % self._snap_n
        self._snap_ring[i] = snap                # 1. ghi payload
        self._snap_seq += 1                      # 2. commit (visibility)
        self._snap_ev.set()

    def publish_proposal(self, p: Proposal) -> None:
        i = self._prop_seq % self._prop_n
        self._prop_ring[i] = p
        self._prop_seq += 1
        self._prop_ev.set()

    # ------------------------------------------- read (consumer, khong block)

    def latest_snapshot(self) -> Snapshot | None:
        seq = self._snap_seq
        if seq == 0:
            return None
        obj = self._snap_ring[(seq - 1) % self._snap_n]
        if self._snap_seq == seq:                # double-check chua bi ghi de
            return obj
        return self._snap_ring[(self._snap_seq - 1) % self._snap_n]

    def snapshot_after(self, since_ts: float) -> Snapshot | None:
        """Snapshot moi nhat co ts > since_ts (quet nguoc toi da bang kich thuoc ring)."""
        seq = self._snap_seq
        for k in range(min(seq, self._snap_n)):
            obj = self._snap_ring[(seq - 1 - k) % self._snap_n]
            if obj is not None and obj.ts > since_ts:
                return obj
        return None

    def latest_proposal(self) -> Proposal | None:
        seq = self._prop_seq
        if seq == 0:
            return None
        obj = self._prop_ring[(seq - 1) % self._prop_n]
        if self._prop_seq == seq:
            return obj
        return self._prop_ring[(self._prop_seq - 1) % self._prop_n]

    def proposal_after(self, since_ts: float) -> Proposal | None:
        seq = self._prop_seq
        for k in range(min(seq, self._prop_n)):
            obj = self._prop_ring[(seq - 1 - k) % self._prop_n]
            if obj is not None and obj.ts > since_ts:
                return obj
        return None

    # ---------------------------------- cho tin hieu (chi ngoai le, khong dong bo du lieu)

    def wait_snapshot(self, timeout: float = 60.0) -> Snapshot | None:
        if self._snap_ev.wait(timeout):
            self._snap_ev.clear()
            return self.latest_snapshot()
        return None

    def wait_proposal(self, since_ts: float, timeout: float = 20.0) -> Proposal | None:
        deadline = time.monotonic() + timeout
        while True:
            p = self.proposal_after(since_ts)
            if p is not None:
                return p
            if self._stop.is_set() or time.monotonic() > deadline:
                return None
            self._prop_ev.wait(0.05)

    def request_stop(self) -> None:
        self._stop.set()
        self._snap_ev.set()
        self._prop_ev.set()

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()