"""Test chuyen dong: dieu khien 6 khop canh tay + gap va doc cam bien luc/cham.

Chay: python test_move.py
"""

import pathlib

import mujoco

MODEL_PATH = pathlib.Path(__file__).with_name("main.xml")

ACTUATOR_TARGETS = {
    "hip_ctrl": 0.0,
    "hindarm_ctrl": -0.6,
    "forearm_ctrl": 0.8,
    "wrist_ctrl": -0.4,
    "hand_ctrl": 0.0,
    "roll_ctrl": 0.0,
    "gripper_ctrl": -0.2,
}

SIM_TIME = 3.0  # giay


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    for act_name, target in ACTUATOR_TARGETS.items():
        data.ctrl[model.actuator(act_name).id] = target

    steps = int(SIM_TIME / model.opt.timestep)
    ft = model.sensor("ft_force").id
    touch = model.sensor("gripper_touch").id
    part_body = model.body("part_00").id
    arm_joints = ["joint_hip", "joint_hindarm", "joint_forearm",
                  "joint_wrist", "joint_hand", "joint_roll"]
    qadr = {n: int(model.jnt_qposadr[model.joint(n).id]) for n in arm_joints}

    for i in range(steps):
        mujoco.mj_step(model, data)
        if i % max(1, steps // 10) == 0:
            qpos = [round(float(data.qpos[qadr[name]]), 3)
                    for name in arm_joints]
            force = data.sensordata[ft:ft + 3]
            print(
                f"t={data.time:4.2f}s  cac khop={qpos}  "
                f"F/T=({force[0]:+6.2f},{force[1]:+6.2f},{force[2]:+6.2f})N  "
                f"touch={float(data.sensordata[touch]):.4f}  "
                f"part_00 z={data.xpos[part_body][2]:6.3f} m"
            )

    print("\nVi tri khop cuoi (rad):",
          [round(float(data.qpos[qadr[n]]), 3) for n in arm_joints])
    force = data.sensordata[ft:ft + 3]
    print(f"Luc F/T cuoi: ({force[0]:+.3f}, {force[1]:+.3f}, {force[2]:+.3f}) N")
    print(f"Cam bien cham: {float(data.sensordata[touch]):.4f}")


if __name__ == "__main__":
    main()
