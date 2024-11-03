import mujoco
import matplotlib.pylab as plt
import numpy as np

def generate_world(x, y, rotx, roty):

def get_positions(frames):
    phi = np.linspace(-np.pi/2, 0, frames) 
    print(phi)

def generate_frames():
    frames = 120
    pos = get_positions(frames)
    for i in range(frames):
        xml = generate_world(**pos[i])
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, 1280, 1280)

        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        plt.imsave(f"frame_{i:03d}.png", renderer.render())
        renderer.close()
