"""Tham so he thong cellet lap rap tu dong."""
from __future__ import annotations

import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "main.xml"
OUT_DIR = ROOT / "out"
TELEMETRY_DIR = OUT_DIR / "telemetry"

ARM_JOINTS = [
    "joint_hip",
    "joint_hindarm",
    "joint_forearm",
    "joint_wrist",
    "joint_hand",
    "joint_roll",
]
ARM_ACTUATORS = [
    "hip_ctrl",
    "hindarm_ctrl",
    "forearm_ctrl",
    "wrist_ctrl",
    "hand_ctrl",
    "roll_ctrl",
]
GRIPPER_ACTUATOR = "gripper_ctrl"
EE_SITE = "ee_site"
TCP_SITE = "ee_tcp"
FT_SITE = "ft_site"
TOUCH_SITE = "tactile_pad"
CAMERA = "wrist_cam"

PART_BODIES_PREFIX = "part_"
# chi chon/gap nhung vi co material nay (linh kien do); "" = tat ca
TARGET_MATERIAL = ""  # sử dụng tất cả linh kiện (theo bài toán demo)

SOURCE_BIN = (0.0, 0.365)
TARGET_BIN = (0.0, -0.365)
# vi tri dat duoc danh dau tren khay dich (khop voi marker trong main.xml)
TARGET_MARK = (0.0, -0.365)
# Khoảng chấp nhận đặt vào khay vàng (m)
PLACE_TOL = 0.035  # chặt hơn để tăng yêu cầu độ chính xác
TARGET_DROP_RADIUS = 0.06
# Vi tri tren khay dich duoc chon lai sau MOI chu ky gắp: tranh linh kien
# da tha truoc do chiem cho (tha cham len nhau -> chu ky sau that bai).
DROP_CLEARANCE = 0.055        # khoang cach toi thieu giua tam 2 linh kien tren khay dich
DROP_RINGS = (0.028, 0.045, 0.060)  # ban kinh cac vong ung vien quanh TARGET_MARK
DROP_RING_SAMPLES = 8         # so ung vien tren moi vong
TRAY_X_MAX = 0.17             # bien trong khay dich theo x (tuong tai +-0.203)
TRAY_Y_MIN = -0.415           # bien trong khay dich theo y (tuong tai -0.43 / -0.30)
TRAY_Y_MAX = -0.315
TRAY_OCCUPIED_Y = 0.10        # part co y < nguong nay duoc coi la dang nam o khay dich  # bán kính khay vàng/zone thả
SOURCE_BIN_RADIUS = 0.18  # bán kính khay xanh nguồn
# Kiểm tra linh kiện rơi ra ngoài khay xanh (nguồn) trong quá trình không mong muốn
SOURCE_CONTAIN_Y_MIN = 0.29  # biên khay xanh (tương ứng bin src)
SOURCE_CONTAIN_Y_MAX = 0.44
SOURCE_CONTAIN_X_MAX = 0.21
BIN_HALF = (0.20, 0.065)
BIN_FLOOR_Z = 0.006
BIN_WALL_TOP_Z = 0.046

APPROACH_DOWN = (0.0, 0.0, -1.0)
APPROACH_UP = (0.0, 0.0, 1.0)
CLOSE_X = (1.0, 0.0, 0.0)

TIP_Z_HOVER = 0.22
TIP_Z_GRASP = 0.011
TIP_Z_DROP = 0.025

KP_POS = 600.0
KD_POS = 45.0
KP_ROT = 12.0
KD_ROT = 2.0
TASK_DAMP_EXTRA = 0.0

GRIP_OPEN = 0.0
# Mo hoan toan cang kep truoc MOI chu ky gap: tranh truong hop cang kep
# van con hep/ket sau chu ky truoc (luc preclose 3 Nm khong du de mo lai).
GRIP_OPEN_GAP = 0.060        # gap (m) duoc coi la cang kep da mo hoan toan
GRIP_OPEN_TIMEOUT = 0.60     # thoi gian toi da (s) danh cho lenh mo hoan toan
# q_des kep (PD): neu vuot qua q_grip thi momen se keo mo -> mat linh kien
GRIP_CLOSE_Q = 1.25
GRIP_KP = 100.0
GRIP_KD = 3.0
GRIP_TORQUE_MAX = 12.0
# Kep CHAM de giam truot: 2 giai doan - (1) kep cham voi momen nho trong
# GRIP_CLOSE_DURATION, (2) nap luc giu GRIP_SQUEEZE_DURATION roi moi nang.
# Truoc day kep nhanh (momen 3 Nm, 0.3s) -> mieng kep bo/van vao linh kien
# lam part lat/truot khi nang. Giam momen + keo dai thoi gian cham.
GRIP_CLOSE_DURATION = 0.55   # thoi gian kep cham (momen thap, cham)
GRIP_CLOSE_TMAX = 1.2        # momen toi da pha kep cham (giam toc do kep)
GRIP_SQUEEZE_DURATION = 0.35 # thoi gian nap luc giu sau khi cham
GRIP_SQUEEZE_TORQUE = 5.0    # momen giu khi da cham du (6.0 -> 5.0)
GRIP_HOLD_TORQUE = 6.0
# mo cham tai cho: PD mo manh (tau -12) + lech gear (idol/dich su) keo 2 ngon
# mo lech -> mieng day linh kien sang 1 ben roi rot -> dat lech khoang danh dau.
GRIP_RELEASE_TORQUE = -1.5
GRIP_STALL_MARGIN = 0.20

# ====== Phat hien truot (slip) va goi lai Deep Grasping ======
# Khi phat hien NGUY CO truot (chua truot han): dung tay, goi Deep Grasp tinh
# lai toa do kep (re-snapshot + re-aim). Khi linh kien da truot han: that bai.
SLIP_TILT_DEG = 12.0          # part nghieng nhieu hon muc nay khi kep -> nguy co
SLIP_LIFT_PROGRESS = 0.012    # part phai len duoc ~1.2cm trong SLIP_LIFT_WAIT
SLIP_LIFT_WAIT = 0.45         # (s) doi toi da cho part thuc su dang duoc nang
SLIP_FALLEN = 0.010           # part rot xuong duoi muc bat dau nang -> truot het
TRANSFER_SLIP_DROP = 0.12     # part thap hon tip nhieu khi van chuyen -> rot khoi kep
GRASP_REGASPS_MAX = 2         # so lan "Deep Grasp tinh lai toa do kep" toi da/chu ky

# ====== Kep banh ranh tranh vach khay (wall) + "gal" part o goc ======
# Transporter truoc day chi tinh clearance giua cac linh kien voi nhau, BO QUA
# vach khay xanh -> part sat vach bi coi la "sach, gap duoc" -> pads dap vach/
# lat part -> truot. Them tinh nang: (1) vach la vat can ao khi band cham tuong,
# (2) khi part o goc kha/phia vach -> chay "gal" (poke): day part song song vach
# ra khoi goc roi moi gap, vi tri/day do Transporter de xuat (AI quyet dinh).
TRANSPORTER_WALL_BLOCK = True
POKE_WALL_MARGIN = 0.024      # (m) part cach vach duoi muc nay -> coi la "ke goc"
POKE_GAP_MARGIN = 0.022       # them vao be rong part khi mo kep trong luc gal
POKE_PAD_ABOVE = 0.004        # day kep cham LEN tren dinh part mot it (tranh can)
POKE_SWEEP_DIST = 0.022       # (m) do quet cua moi lan gal (nho - tranh day part ra khoi khay)
POKE_SPEED = 0.08             # (m/s) van toc quet
POKE_DWELL = 0.20             # (s) dung ngat giua moi pha gal
POKE_MAX = 4                  # so lan gal toi da trong 1 chu ky truoc khi gap bo sung
# vi tri bin nguon (khay xanh) - dung de xac dinh vach trong-trong.
# Mat trong vach tu main.xml: vach x o +-0.20, vach y o 0.30/0.43 - day vach 0.003.
SOURCE_BIN_X_MAX = 0.200
SOURCE_BIN_Y_LO = 0.303
SOURCE_BIN_Y_HI = 0.427
# kep "doi hinh" truoc khi xuong: mo kep = rong linh kien + margin -> han che
# quac dong khi kep (chan day linh kien ra khoi giua hai mieng)
GAP_PRE_MARGIN = 0.014
GAP_PRE_KP = 400.0
GAP_PRE_TMAX = 3.0
GAP_GEOM0 = 75          # mieng kep (geom1) - neu khong tim thay se tu tim
GAP_GEOM1 = 90
# kep nhe (momend thap) trong luc xuong phan duoi de mieng kep tu can chinh
# linh kien lech/nghieng truoc khi kep day o do sau (GRIP_CLOSE_TMAX).
# Momend thap de tranh day/ban linh kien ra khi mieng cham phia tren.
GRIP_SOFT_TMAX = 0.5
SQUEEZE_ENGAGE = 99.0     # cham it nhat chieu cao nay cua mat ben linh kien
SQUEEZE_LATE = 0.010       # khong kep som hon vi tri nay (gan grasp_z)
GRASP_PROBES = [(0.0, 0.0), (0.018, -0.010), (-0.018, 0.010), (0.000, 0.016)]
GRASP_MAX_PROBES = 4
GRIP_FINGER0 = "gripper_gripper_left_site"
GRIP_FINGER1 = "gripper_gripper_right_site"
GRIPPER_JOINT = "gripper_servo_gear"

#chan DESCEND: load phai vuot nguong LIEN TUC va tip gan grasp_z moi cho la blocked
FT_CONTACT_THRESHOLD = 25.0
FT_SAFETY_THRESHOLD = 40.0
FT_PERSIST_TIME = 0.06
FT_NEAR_Z = 0.05

#do hoi tu truoc khi xuong (giam lech xy -> ngon kep cham linh kien)
HOVER_TOL = 0.006
LIFT_TOL = 0.012
WAYPT_TOL = 0.003
HOVER_DWELL = 0.20
RELEASE_DWELL = 0.40
# do sau kep: dap kep nam [tip+0.021, tip+0.070]. Dat tip = day linh kien
# - GRASP_PAD_ABOVE de duoi dap kep tham 2mm vao duoi day linh kien (bao trum
# het chieu cao) va khong cham sau hon nua -> ngan tay kep ram vao dong phan
# ben duoi (lam lac linh kien truoc khi kep). Khong tinh duoc -> GRASP_DEPTH.
GRASP_PAD_ABOVE = 0.023
GRASP_DEPTH = 0.045
# chinh sua chieu cao kep khi 2 mieng kep lech z (do gear lech/gia -> slide_x
# 2 ngon khac nhau): ha them toi da bang gia tri nay de mieng CAO van che
# duoc linh kien; vuot qua thi deu chan (ram dong/hoac mat kep cham san).
PAD_Z_MAX_DROP = 0.030
# mieng kep nam ben tren TCP: mieng kep cham duoi = tip + PAD_BELOW_TIP.
# Khi mieng kep da cham duoc it nhat SQUEEZE_ENGAGE vao mat ben linh kien
# (va khong som hon SQUEEZE_LATE tren grasp_z) -> kep nhe ngay trong luc
# van xuong: kep tu can chinh linh kien lech/nghieng, han cham goc mieng
# vao goc linh kien lam no lat/sang (dau) truoc khi kep day o grasp_z.
PAD_BELOW_TIP = 0.021

IK_TOL_POS = 2e-3
IK_TOL_ROT = 6e-2
IK_MAX_ITER = 300
IK_DAMPING = 0.05

CONTROL_DT = 0.001
WAYPOINT_SPEED = 0.25
WAYPOINT_ACCEL = 0.6

# ---------------------------------------------------------------- Transporter
# Danh gia kha nang gap trong cum (khe gap min cho 2 ngon kep xuong 2 ben).
GRASP_CLEAR_REQ = 0.006      # khe gap toi thieu (m) de coi la "gap duoc"
GRASP_CLEAR_GOOD = 0.018     # khe gap "tot" -> diem toi da
GRASP_BAND_EXTRA = 0.004     # mo rong them vung kep moi ben (m) khi do khe
GRASP_PAD_WINDOW = 0.008     # nua do day cua "cua so" ngon kep theo truc kep (m)
GRASP_ZNECK = 0.045          # linh kien lan can thap hon ZNECK so voi dinh thi bo qua
# linh kien dang la vat can cua lan gap hien tai -> them vao blocked (doi gat):
BLOCK_OBSTACLES = True
# Ngan ngoi khi zero linh kien nao "sach" - van de xuat ung vien it can nhat:
TRANSPORTER_FALLBACK = True
# ICP kiem chung pose: nguong sai so (RMSE) va ty le diem ben trong box
GRASP_ICP_RMSE_MAX = 0.006
GRASP_ICP_INLIER_MIN = 0.25

# ---------------------------------------------------------------- RTC <-> AI
# Ngan ngoi diem proposal toi thieu de RTC tien hanh gap.
PROP_SCORE_MIN = 0.20
# Lock-free ring buffer: do dai lich su moi kenh (O(1) doc/ghi).
BRIDGE_HISTORY = 4
# Lich su vi tri canh tay/camera ngan han de bu tru do tre (Latency Compensation).
POSE_HIST_LEN = 240
# Toi da tuoi (s) cua du lieu anh duoc phep ngoai suy vi tri linh kien.
LATCOMP_MAX_AGE = 0.15
# Refine bang Vision-Force Servoing trong khi ha xuong (micro-adjustments).
VFS_ENABLE = False
VFS_EVERY = 80               # chu ky refresh lien tuc (so buoc sim)
VFS_MAX_OFF = 0.008          # gioi han bu tru ngang moi lan tinh chinh (m)
VFS_EMA = 0.5                # he so loc mu (Exponential Moving Average)
