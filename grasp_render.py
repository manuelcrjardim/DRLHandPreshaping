import os
import mujoco
import pickle
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFont
import pandas as pd

# --- Configuration ---
MODEL_XML_PATH = r".\mjcf\open_ai_assets\hand\shadow_hand_visualize_photorealistic.xml" #Must be changed accordingly
DATA_PKL_PATH = "final_trajectories_randomized.pkl"
OUTPUT_DIR = "rendered_frames"

IMAGE_WIDTH = 1080   
IMAGE_HEIGHT = 1080  

LABEL_HEIGHT = 80  
LABEL_FONT_SIZE = 48  
LABEL_BG_COLOR = (0, 0, 0, 180)  
LABEL_TEXT_COLOR = (255, 255, 255, 255)  

os.makedirs(OUTPUT_DIR, exist_ok=True)

# used to get object type
def get_object_type(full_object_name):

    parts = str(full_object_name).split('/') 
    if len(parts) > 1:
        type_part = parts[1].split('-')
        if type_part:
            return type_part[0]

def add_label_to_image(image, object_name, frame_index):
    
    object_type = get_object_type(object_name)
    labeled_image = image.copy()
    img_width, img_height = labeled_image.size
    overlay = Image.new('RGBA', (img_width, LABEL_HEIGHT), LABEL_BG_COLOR)    
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(overlay)
    label_text = f"Object Type: {object_type} | Frame: {frame_index}"

    bbox = draw.textbbox((0, 0), label_text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_x = (img_width - text_width) // 2
    text_y = (LABEL_HEIGHT - text_height) // 2
    draw.text((text_x, text_y), label_text, fill=LABEL_TEXT_COLOR, font=font)
    if labeled_image.mode != 'RGBA':
        labeled_image = labeled_image.convert('RGBA')
    
    final_image = Image.new('RGBA', (img_width, img_height + LABEL_HEIGHT), (0, 0, 0, 255))
    final_image.paste(labeled_image, (0, 0))
    final_image.paste(overlay, (0, img_height), overlay)
    final_image = final_image.convert('RGB')
    
    return final_image

def set_hand_color(model):
    
    matt_reddish_white = [0.9, 0.4, 0.4, 1.0]  
    
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and 'robot0:' in geom_name:
            model.geom_rgba[i] = matt_reddish_white

# load model and data 
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data = mujoco.MjData(model)

set_hand_color(model)
print(f"Model loaded successfully: {model.nq} positions, {model.nu} actuators, {model.njnt} joints")


# load trajectory data
with open(DATA_PKL_PATH, 'rb') as f:
    all_trajectories = pickle.load(f)

print("=== AVAILABLE OBJECTS ===")
for i, object_name in enumerate(all_trajectories.keys()):
    object_type = get_object_type(object_name)
    print(f"{i}: {object_name} (Type: {object_type}) - {len(all_trajectories[object_name])} trajectories")

OBJECT_TO_RENDER = 'sem/Watch-e743856943ce4fa49a9248bc70f7492'  # Trajectory will be picked for this object
TRAJECTORY_INDEX = 0  # Which trajectory to render (0-199)

if OBJECT_TO_RENDER not in all_trajectories:
    print(f"Error: Object '{OBJECT_TO_RENDER}' not found in data!")
    print("Available objects:", list(all_trajectories.keys()))
    exit()

# Get the specific trajectory
trajectory_dict = all_trajectories[OBJECT_TO_RENDER][TRAJECTORY_INDEX]
trajectory = [dof for timestep, dof in sorted(trajectory_dict.items())]

print(f"Selected object: {OBJECT_TO_RENDER}")
print(f"Object type: {get_object_type(OBJECT_TO_RENDER)}")
print(f"Trajectory index: {TRAJECTORY_INDEX}")
print(f"Trajectory length: {len(trajectory)} timesteps")

# Initialize renderer 
renderer = mujoco.Renderer(model, height=IMAGE_HEIGHT, width=IMAGE_WIDTH)

option = mujoco.MjvOption()
mujoco.mjv_defaultOption(option)
option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = 0
option.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = 0

camera = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(camera)
camera.azimuth = 135
camera.elevation = -25
camera.distance = 0.4
camera.lookat = [0.0, 0.0, 0.05]

print(f"Starting high-resolution rendering for {OBJECT_TO_RENDER}...")


object_output_dir = os.path.join(OUTPUT_DIR, OBJECT_TO_RENDER.replace('/', '_'))
os.makedirs(object_output_dir, exist_ok=True)

frames_to_render = [0, 25, 50, 100, 130, 199]  

for frame_index in frames_to_render:
    dof_positions = trajectory[frame_index]
    mujoco.mj_resetData(model, data)
    
    data.qpos[:] = np.array(dof_positions)    
    mujoco.mj_forward(model, data)
    
    renderer.update_scene(data, camera=camera, scene_option=option)
    pixels = renderer.render()
    
    img = Image.fromarray(pixels)

    contrast_enhancer = ImageEnhance.Contrast(img)
    img = contrast_enhancer.enhance(1.15)    
    color_enhancer = ImageEnhance.Color(img)
    img = color_enhancer.enhance(1.1)
    sharpness_enhancer = ImageEnhance.Sharpness(img)
    img = sharpness_enhancer.enhance(1.05)
    
    labeled_img = add_label_to_image(img, OBJECT_TO_RENDER, frame_index)
    
    output_path = os.path.join(object_output_dir, f"Watch_traj{TRAJECTORY_INDEX}_frame_{frame_index:04d}_labeled.png") # Name must be changed to match object
    labeled_img.save(output_path, quality=98, optimize=True)
    
    print(f"Rendered labeled frame {frame_index} -> {output_path}")

print(f"High-resolution rendering complete!")
print(f"Labeled frames saved to: {object_output_dir}")
print(f"Resolution: {IMAGE_WIDTH}x{IMAGE_HEIGHT + LABEL_HEIGHT}")