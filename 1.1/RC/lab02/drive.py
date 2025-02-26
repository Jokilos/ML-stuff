#!/usr/bin/env python
# coding: utf-8

# In[287]:


import mujoco
import matplotlib.pylab as plt
import numpy as np
import cv2


# In[288]:


def cv2_imshow(img):
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


# In[289]:


dim = 1280

prolog = """
<?xml version="1.0" ?>
<mujoco>
    <worldbody>

        <body name="floor" pos="0 0 -0.1">
            <geom size="2.0 2.0 0.02" rgba="0.2 0.2 0.2 1" type="box"/>
        </body>
        <body name="x_arrow" pos="0.5 0 0">
            <geom size="0.5 0.01 0.01" rgba="1 0 0 0.5" type="box"/>
        </body>
        <body name="y_arrow" pos="0 0.5 0">
            <geom size="0.01 0.5 0.01" rgba="0 1 0 0.5" type="box"/>
        </body>
        <body name="z_arrow" pos="0 0 0.5">
            <geom size="0.01 0.01 0.5" rgba="0 0 1 0.5" type="box"/>
        </body>

"""

epilog = f"""
    </worldbody>

    <visual>
        <global offwidth="{dim}" offheight="{dim}"/>
    </visual>
</mujoco>
"""

def generate_car(car_pos, wheel_pos, car_rot, wheel_aa):
    xml = f"""
        <body name="car" pos="{car_pos}" axisangle="0 0 1 {car_rot:.3f}">
            <geom size="0.2 0.1 0.02" rgba="1 1 1 0.9" type="box"/>
        </body>"""
    
    for i in range(4):
        xml += f"""
        <body name="wheel_{i+1}" pos="{wheel_pos[i]}" axisangle="{wheel_aa[i]}">
            <geom size="0.07 0.01" rgba="1 1 1 0.9" type="cylinder"/>
        </body>"""

    return xml

def generate_radar(radar_pos, radar_aa):
    xml = f"""
        <body name="radar_1" pos="{radar_pos[0]}" axisangle="{radar_aa[0]}">
        <geom size="0.01 0.01 0.1" rgba="1 1 1 1" type="box"/>
        </body>
        <body name="radar_2" pos="{radar_pos[1]}" axisangle="{radar_aa[1]}">
            <geom size="0.03 0.01" rgba="1 0 0 1" type="cylinder"/>
        </body>
    """

    return xml


# In[290]:


def rot_mtx(axis, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    
    if axis == 'x':
        a = [1, 0, 0, 0, c, -s, 0, s, c]
    elif axis == 'y':
        a = [c, 0, -s, 0, 1, 0, s, 0, c]
    elif axis == 'z':
        a = [c, -s, 0, s, c, 0, 0, 0, 1]
    else:
        a = [0] * 9
    
    return np.array(a).reshape(3,3)

def rot_to_aa(rot):
    axis_angle = cv2.Rodrigues(rot)[0]

    angle = np.linalg.norm(axis_angle)

    if angle > 0:
        axis = axis_angle / angle  # Normalize the axis
    else:
        axis = np.zeros(3)

    return (axis, angle)  
    


# In[291]:


# wheel_add = np.array([1,1,-1,1,-1,-1,1,-1]).reshape(4,2) * 0.1
# wheel_vecs = np.hstack([wheel_add, np.zeros(4).reshape(-1,1)]).T
# z_rot = (rot_mtx('z', np.pi))

# radar_add = np.array([0, -0.1, 0.2, 0, -0.15, 0.29]).reshape(2,3).T
# rot_radar_add = (z_rot @ radar_add).T

# radar_mat = np.tile([-1, -1, 0], 2).reshape(2,3) + rot_radar_add

#radar_pos = [f"{x:.3f} {y} 0.0" for [x, y, _] in radar_mat]


# In[292]:


to_degrees = 180/np.pi

def get_positions(frames):
    phi = np.linspace(-np.pi/2, 0, frames)
    yy, xx = (np.sin(phi), np.cos(phi) - 1)
    car_rot = phi + np.pi/2 
    rad_rot = np.pi - phi

    args = np.array([xx, yy, car_rot, rad_rot]).T
    args = [{'x': x, 'y': y, 'rotc': cr, 'rotr': rr} for [x, y, cr, rr] in args]

    return(args)

def get_wheel_pos(x, y, z_rot):
    #generate 4 wheel positions
    #by multiplying displacement vectors by z_rot matrix

    wheel_add = np.array([1,1,-1,1,-1,-1,1,-1]).reshape(4,2) * 0.1
    wheel_vecs = np.hstack([wheel_add, np.zeros(4).reshape(-1,1)]).T
    rot_wheel_add = (z_rot @ wheel_vecs).T

    wheel_mat = np.tile([x, y, 0], 4).reshape(4,3) + rot_wheel_add
    wheel_pos = [f"{x:.3f} {y:.3f} 0.0" for [x, y, _] in wheel_mat]
    return wheel_pos

def get_radar_pos(x, y, z_rot):
    radar_add = np.array([0, -0.1, 0.1, 0, -0.15, 0.19]).reshape(2,3).T
    rot_radar_add = (z_rot @ radar_add).T

    radar_mat = np.tile([x, y, 0], 2).reshape(2,3) + rot_radar_add
    radar_pos = [f"{x:.3f} {y:.3f} {z:.3f}" for [x, y, z] in radar_mat]
    return radar_pos

def generate_world(x, y, rotc, rotr):
    #z rotation matrix
    z_rot = rot_mtx('z', rotc) 

    #generate car position
    car_pos = f"{x:.3f} {y:.3f} 0.0"

    wheel_pos = get_wheel_pos(x, y, z_rot)
    radar_pos = get_radar_pos(x, y, z_rot)

    wheel_aa = rot_to_aa(z_rot @ rot_mtx('x', np.pi/2))
    aa_cords = [f"{e[0]:.3f}" for e in wheel_aa[0]]
    wheel_aas = [f"{' '.join(aa_cords)} {wheel_aa[1] * to_degrees:.3f}"] * 4

    radar_aa = rot_to_aa(z_rot @ rot_mtx('x', np.pi/6))
    radar_aa_cords = [f"{e[0]:.3f}" for e in radar_aa[0]]
    radar_aas = [f"{' '.join(radar_aa_cords)} {radar_aa[1] * to_degrees:.3f}"] * 2

    car = generate_car(car_pos, wheel_pos, rotc * to_degrees, wheel_aas)
    radar = generate_radar(radar_pos, radar_aas)

    return prolog + car + radar + epilog

def generate_frames():
    frames = 30
    pos = get_positions(frames)
    for i in range(frames):
        xml = generate_world(**(pos[i]))
        #print(xml)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, dim, dim)

        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        img = renderer.render()
        plt.imsave(f"frames/frame_{i:03d}.png", img) 
        if i == 0:
             with open('half.xml', 'w') as file:
                file.write(xml)
        renderer.close()

generate_frames()


