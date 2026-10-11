"""Test camera: render anh tu camera gan tren co tay (wrist_cam).

Chay: python test_camera.py
Anh duoc luu vao out/wrist_cam.png
"""

import argparse
import pathlib

import mujoco
from PIL import Image

MODEL_PATH = pathlib.Path(__file__).with_name("main.xml")
CAMERA_NAME = "wrist_cam"
OUT_PATH = pathlib.Path(__file__).with_name("out") / "wrist_cam.png"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200, help="so buoc mo phong truoc khi chup")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--out", type=pathlib.Path, default=OUT_PATH)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    # cho mo phong chay mot lat de linh kien on dinh trong khay
    for _ in range(args.steps):
        mujoco.mj_step(model, data)

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    renderer.update_scene(data, camera=CAMERA_NAME)
    image = renderer.render()
    renderer.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.out)

    print(f"Camera '{CAMERA_NAME}': {args.width}x{args.height}")
    print(f"Vi tri camera (world): {data.cam_xpos[model.cam(CAMERA_NAME).id].round(4)}")
    print(f"Da luu anh: {args.out}")


if __name__ == "__main__":
    main()
