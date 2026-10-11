# mujoco — Mô phỏng cánh tay robot 6 bậc tự do (PoC gắp & đặt linh kiện)

Dự án **Proof-of-Concept** mô phỏng một cell lắp ráp tự động trong MuJoCo:
cánh tay robot 6-DOF gắp linh kiện từ khay nguồn (xanh) thả vào khay đích (vàng),
kèm pipeline điều khiển (Impedance + IK) và AI thị giác (nhận dạng → đánh giá
khả năng gắp → chấm điểm CNN).

## 1. Cấu trúc thư mục

```
mujoco/
├── main.xml                 # Model chính: tay 6-DOF + 2 khay + 30 linh kiện + camera + cảm biến F/T
├── models/                  # 8 XML: Base, Hip, Hindarm, Forearm, Wrist, Hand, Gripper, materials
├── meshes/<bộ phận>/        # Mesh STL của từng bộ phận (Base, Hip, Hindarm, Forearm, Wrist, Hand, Gripper)
├── linhkien/lk1..lk5.stl     # Mesh linh kiện (mm) — được khai báo trong main.xml, scale 0.001
├── automation/              # GĐ2-GĐ5: điều khiển & AI
│   ├── config.py            # Tham số hệ thống (IK, F/T, grasp, vùng bin, poke...)
│   ├── kinematics.py        # FK + IK giải nghịch (DLS) + read_frame
│   ├── controller.py        # Impedance controller + Gripper (điều khiển momen)
│   ├── sensors.py           # SensorBus: F/T 6 trục, touch, opening (mm), q/qd
│   ├── mission.py           # FSM pick-and-place + quỹ đạo quintic/slerp + "gal" (poke)
│   ├── runner.py            # Vòng chạy chính + telemetry + ghi CSV
│   ├── perception.py        # Camera RGB-D-Seg + unproject pixel→world + candidates
│   ├── graspai.py           # CNN scorer 24x24 patch + profile Mask R-CNN/Deep Grasp & DexRepNet
│   ├── transporter.py       # Transporter Network: clearance, blocker, kẹt vách/góc → poke
│   ├── icp.py               # BoxIcp: khớp pose 6D (x,y,yaw) với point cloud
│   ├── bridge.py            # Cầu nối lock-free (ring buffer Snapshot/Proposal)
│   ├── ai_node.py           # Thread AI: snapshot → pipeline 3 tầng → proposal
│   ├── rtc_node.py          # Thread RTC (Real Time Controller): mô phỏng + mission + snapshot ↔ proposal
│   └── analytics.py         # KPI, Force profile, Cycle time, Trajectory, Dashboard
├── run_benchmark.py         # Baseline GĐ2: chạy N chu kỳ → out/results + telemetry + dashboard
├── run_ai_pick.py           # GĐ3/GĐ4: pipeline RTC (Real Time Controller) + AI chạy song song (không viewer)
├── run_viewer.py            # GĐ5: mở Viewer 3D và TỰ ĐỘNG chạy toàn bộ pipeline
├── test_move.py             # Điều khiển khớp, đọc cảm biến F/T + touch
├── test_camera.py           # Xuất ảnh từ camera gắn cổ tay → out/wrist_cam.png
├── test_pickplace.py        # Chạy 1 chu kỳ gắp-đặt đầy đủ
├── analysis/evaluation.ipynb # Đánh giá: tỉ lệ thành công, cycle time, force profile
├── out/                     # Telemetry + kết quả + dashboard (bị .gitignore)
├── requirements.txt         # Toàn bộ gói phụ thuộc (mujoco==3.15.0, numpy, pandas, torch...)
├── MuJoCo/                  # SDK MuJoCo 3.15.0 — bản Windows (bị .gitignore)
├── third_party/glfw-3.4.bin.WIN64/   # GLFW 3.4 cho các tutorial C
├── MuJoCo_tutorials-main/   # Tutorial C tham khảo (1 conveyor, 2 cylinder, 3 gripper, 4 belt)
└── .vscode/                 # Cấu hình build/debug cho VS Code
```

Nguồn `models/` + `meshes/`: https://github.com/felixokolo/MuJoCo_tutorials (thư mục `5`).

## 2. Cài đặt môi trường

```powershell
pip install -r requirements.txt      # mujoco==3.15.0, numpy, matplotlib, pandas, torch (CPU), ...
```

Bộ compiler C (chỉ cần nếu build tutorial trong `MuJoCo_tutorials-main/`):

- MinGW-w64 gcc 16.2 + gdb 17.2 + mingw32-make (WinLibs, cài qua winget)
- GLFW 3.4 đã có sẵn trong `third_party/` (không cần cài thêm)

Mở terminal **mới** sau khi cài để PATH có hiệu lực, hoặc dùng terminal tích hợp
trong VS Code (`.vscode/settings.json` đã khai báo PATH).

## 3. Chạy mô phỏng

Kiểm tra cơ bản:

```powershell
python test_move.py        # điều khiển các khớp, in ra cảm biến F/T + touch
python test_camera.py      # xuất ảnh từ camera gắn cổ tay -> out/wrist_cam.png
python test_pickplace.py   # chạy 1 chu kỳ gắp-đặt, in telemetry và kết quả
```

Demo tự động (GĐ5) — mở **Viewer 3D và tự động chạy pipeline RTC + AI**:

```powershell
python run_viewer.py --cycles 8 --task 1
python run_viewer.py --cycles 6 --task 2
```

Pipeline đầu cuối không cần viewer:

```powershell
python run_ai_pick.py --cycles 6 --infer-sleep 0.02   # GĐ3/GĐ4: RTC + AI
python run_benchmark.py --cycles 6                    # baseline GĐ2 (chọn mục tiêu ground-truth)
```

Phân tích kết quả: mở `analysis/evaluation.ipynb` hoặc

```powershell
python -m automation.analytics --out out             # dashboard từ out/telemetry/pick.csv (baseline)
```

Cách thủ công: mở `MuJoCo\bin\simulate.exe` rồi kéo `main.xml` vào cửa sổ,
điều khiển các trục ở mục **control** bên phải.

Kiểm tra model không cần giao diện:

```powershell
MuJoCo\bin\compile.exe main.xml out.mjb
```

## 4. Model `main.xml`

- **Thời gian**: `timestep=0.001` (1 ms/bước), `integrator=implicitfast`,
  `solver=Newton`, `cone=pyramidal`.
- **Sàn** ở `z=-0.75`; **mặt bàn** (box `0.70×0.55×0.375`) có mặt trên `z=0`.
- **Khay nguồn** (xanh) quanh `y=+0.365`, **khay đích** (vàng) quanh `y=-0.365`;
  mỗi khay lòng rộng ~`0.40×0.13` m, tường cao 24 mm.
- **30 linh kiện** `part_00..part_29` là box với ma sát cao + va chạm mềm, được
  thả từ 3 lớp `z = 0.13 / 0.23 / 0.33` m rồi rơi tự do vào khay nguồn. Kích
  thước ~20–60 × 10–40 × 10–30 mm, mỗi part mang 1 trong 5 màu
  (`lk_orange/teal/purple/green/gray`) để dễ phân đoạn.
- **Cánh tay 6-DOF** gắn tại `(0,0,0.080)`; các khớp
  `joint_hip, joint_hindarm, joint_forearm, joint_wrist, joint_hand, joint_roll`
  (actuator `*_ctrl`, điều khiển bằng **momen**).
- **Camera** `wrist_cam` kiểu *eye-in-hand* (FOV 60°, khai báo 1280×1024) hướng
  theo trục tiếp cận của kẹp.
- **Site** `ft_site` (gốc cổ tay–kẹp), `ee_site` (mũi kẹp), `ee_tcp` (giữa hai
  ngón), `tactile_pad` (vùng chạm).
- **Cảm biến**: `ft_force`/`ft_torque` (6 trục tại `ft_site`), `gripper_touch`
  (touch tại `tactile_pad`), cùng `q_*`/`qd_*` cho 6 khớp.
- **Mesh** `linhkien/lk1..lk5.stl` được khai báo trong `<asset>` làm mẫu CAD;
  các vật thể mô phỏng hiện dùng box (mesh thật thay bằng cách sửa geom tương ứng).

## 5. Kiến trúc GĐ3/GĐ4 (RTC + AI đa luồng)

```
[Thread RTC]  Runner (mô phỏng 1ms + Impedance + mission FSM)
   │  snapshot RGB-D-Seg (wrist_cam) ──► Bridge (ring buffer lock-free)
   │                                          ▲
   │  proposal (x,y,yaw,score,action) ◄────────│
   ▼                                          │
[Thread AI]  ① Mask R-CNN (seg) → ② Transporter + ICP → ③ Deep Grasping (CNN patch 24x24)
```

- **`bridge.py`**: mỗi kênh (snapshot, proposal) là ring buffer kích thước cố
  định, ghi/đọc O(1) không khóa mutex; mọi packet mang `ts` (thời gian sim) +
  `wall` (đồng hồ tường) cho khâu bù trễ.
- **`rtc_node.py`**: sở hữu *độc quyền* `mjModel`/`mjData`, mọi thao tác MuJoCo
  (step, camera, sync) chỉ xảy ra trong luồng này → an toàn với viewer passive.
  Có bù trễ (Latency Compensation) ngoại suy vị trí part từ lúc chụp tới hiện
  tại, và Visual-Force Servoing khi hạ kẹp (mặc định tắt: `VFS_ENABLE=False`).
- **`mission.py`**: chọn mục tiêu **theo đề xuất của AI**; trước DESCEND có pha
  **refine bằng camera** (tương đương local-crop của Transporter/DeepGrasp) để
  triệt tiêu sai số nhìn xa.
- **Transporter-style**: part đã `loose`/bị chặn bị loại khỏi các chu kỳ sau;
  part kẹt vách/góc được đánh dấu để AI đề xuất **poke** ("gal") đẩy ra trước
  khi gắp (giới hạn `POKE_MAX=4` lần/chu kỳ).
- **Phát hiện trượt (slip)**: nếu part nghiêng > `SLIP_TILT_DEG` khi kẹp hoặc
  không nâng lên được trong `SLIP_LIFT_WAIT`, dừng tay và **gọi lại Deep Grasp**
  tính lại tọa độ kẹp (tối đa `GRASP_REGASPS_MAX=2` lần/chu kỳ).

## 6. Pipeline AI (3 tầng, trong `ai_node.py`)

1. **Mask R-CNN** — nhận dạng linh kiện từ kênh segmentation; sinh ứng viên
   (`PartCandidate` với vị trí world, `z_top`, yaw, kích thước box).
2. **Transporter Network + ICP** (`transporter.py`, `icp.py`) — khảo sát khả
   năng gắp trong cụm: tính *clearance* quanh vị trí gắp, phát hiện **blocker**
   và **kẹt vách (corner)**, rồi `BoxIcp` khớp/lọc pose 6D từ point cloud.
3. **Deep Grasping** (`graspai.py`) — CNN 5 lớp chấm điểm patch depth 24×24
   (`CnnScorer`, tự pretrain trên dữ liệu tổng hợp), hợp nhất với điểm
   clearance/ICP để xếp hạng.

Đầu ra **chỉ là toạ độ/mốc (Waypoint/Pose) + cờ `grasp_ready`** — AI không bao
giờ truy xuất trực tiếp vào `d.ctrl`/momen động cơ (phân cấp quyền hạn).

Các profile bài toán nằm trong `graspai.py` (`MaskRcnnDeepGraspProfile`,
`DexRepProfile`) và có thể chọn qua `choose_aim(task=...)`. Cờ `--task` hiện
được chuyển vào cả hai node và ghi log; node AI đang dùng `GraspProfile` chung
nên hai task cho kết quả tương đương, phần profile chuyên biệt để mở rộng.

## 7. Tham số điều khiển chính (`automation/config.py`)

- **Impedance**: `Kp=600`, `Kd=45` (vị trí), `Kr=12`, `Kdr=2` (hướng).
- **IK (DLS)**: `tol_pos=2e-3`, `tol_rot=6e-2`, `max_iter=300`, `damping=0.05`.
- **F/T**: ngưỡng va chạm `25 N`, an toàn `40 N`, phải vượt ngưỡng liên tục
  `0.06 s` mới coi là bị chặn.
- **Kẹp**: PD đóng → siết 2 giai đoạn (momen thấp `1.2 N·m` rồi nạp `5 N·m`),
  khi mang giữ hằng `6 N·m`; nhả chậm `-1.5 N·m`.
- **Vị trí thao tác**: hover `z=0.22`, gắp `z=0.011`, thả `z=0.025` m.

## 8. Kết quả & phân tích

`run_benchmark.py` / `run_ai_pick.py` / `run_viewer.py` ghi ra:

- `out/telemetry/*.csv` — telemetry mỗi 10 bước sim (tip, F/T, touch, opening, gap, warning)
- `out/results/*.csv` — kết quả từng chu kỳ (`ok`, `off_mark`, `loose`, `lost`, `no_target`, `timeout`)
- `out/dashboard/` — `force_profile*.png`, `cycle_times*.png`, `trajectory*.png`, `kpi*.csv`

`automation/analytics.py` tính KPI (tỉ lệ thành công, cycle time TB/min/max) và
`analysis/evaluation.ipynb` so sánh baseline GĐ2 với pipeline RTC+AI.

## 9. Build tutorial C (tham khảo)

```powershell
cd MuJoCo_tutorials-main\1
mingw32-make          # build belt.exe (tự copy mujoco.dll, glfw3.dll cạnh exe)
mingw32-make run
```

Tutorial 4 phải chạy với thư mục hiện hành là `bin` (exe load `../main.xml`),
`mingw32-make run` đã làm sẵn việc này.

Trong VS Code: `Ctrl+Shift+B` để build, hoặc chọn cấu hình debug
`C: belt.exe (tutorial 1)` / `C: belt.exe (tutorial 4)`.


## 10. Tác giả
System 3122
