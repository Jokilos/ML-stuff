import mujoco
from mujoco import viewer
import numpy as np
# from mujoco import MjModel, MjData, MjViewer
from xml.etree.ElementTree import ElementTree, fromstring, tostring
import time

def rot_M(axis, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    
    if axis == 0:
        a = [1, 0, 0, 0, c, -s, 0, s, c]
    elif axis == 1:
        a = [c, 0, -s, 0, 1, 0, s, 0, c]
    elif axis == 2:
        a = [c, -s, 0, s, c, 0, 0, 0, 1]
    else:
        return None
    return np.array(a).reshape(3,3)

def get_objpoints(edge):
    blk = 1/10
    objpoints = np.array([
        [1 - blk, 1 - blk, 0],
        [blk, 1 - blk, 0],
        [blk, blk, 0],
        [1 - blk, blk, 0],

        [0, blk, blk],
        [0, 1 - blk, blk],
        [0, 1 - blk, 1 - blk],
        [0, blk, 1 - blk],
    ]) 
    
    return objpoints * edge 

objPoints = get_objpoints(0.1)
objPoints = rot_M(2, -np.pi/2) @ rot_M(0, np.pi/2) @ objPoints.T
objPoints = objPoints.T + np.array([-0.3, -0.3, 0]) * 0.83
print(objPoints)

with open("car_3.xml", "r") as f:
    base_xml = f.read()

tree = ElementTree(fromstring(base_xml))
root = tree.getroot()

worldbody = root.find("worldbody")
for i, point in enumerate(objPoints):
    gstr = f"""
        <geom name="mypoint{i}" type="sphere" size="0.002" pos="{point[0]:.3f} {point[1]:.3f} {point[2]:.3f}" rgba="1 0 0 1"/>

    """

    geom = fromstring(gstr)
    print(gstr)

    worldbody.append(geom)

# box = fromstring("""
#     <geom name="mybox" type="box" material="cube_markers_1" size=".05 .05 .05"/>
# """)

modified_xml = tostring(root, encoding="unicode")

model = mujoco.MjModel.from_xml_string(modified_xml)
renderer = mujoco.Renderer(model, height=480, width=640)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
viewer = viewer.launch_passive(model, data)

def sim_step(n_steps: int, /, view=True, rendering_speed = 10, **controls: float):
    for control_name, value in controls.items():
        data.actuator(control_name).ctrl = value

    for _ in range(n_steps):
        step_start = time.time()
        mujoco.mj_step(model, data)
        if view:
            viewer.sync()
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step / rendering_speed)

    renderer.update_scene(data=data, camera="dash cam")
    img = renderer.render()
    return img

for _ in range(200):
    sim_step(200, "")

time.sleep(100)
