from truckscenes import TruckScenes
import os
from truckscenes.utils.geometry_utils import box_in_image, view_points
from truckscenes.utils.geometry_utils import  BoxVisibility
import numpy as np
import fiftyone as fo
from PIL import Image

BASE_PATH = path = os.environ["TRUCKSCN_PATH"]
trucksc = TruckScenes('v1.0-mini', BASE_PATH, True)

dataset = fo.Dataset("Truckscenes", overwrite=True)
dataset.add_group_field("group", default="CAMERA_LEFT_FRONT")


def camera_sample(group, filepath, sensor, token, scene):
    sample = fo.Sample(filepath=filepath, group=group.element(sensor))
    data_path, boxes, camera_intrinsic = trucksc.get_sample_data(token, box_vis_level=BoxVisibility.NONE,)
    image = Image.open(data_path)
    width, height = image.size
    shape = (height,width)
    polylines = []
    ego = trucksc.get('ego_pose', data["ego_pose_token"])
    ego_list = [ego]

    for box in boxes:
        if box_in_image(box,camera_intrinsic,shape,vis_level=BoxVisibility.ALL):
            c = np.array(trucksc.colormap[box.name]) / 255.0
            #print(box.name)
            corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :]
            front = [(corners[0][0]/width,corners[1][0]/height),
                    (corners[0][1]/width,corners[1][1]/height),
                    (corners[0][2]/width,corners[1][2]/height),
                    (corners[0][3]/width,corners[1][3]/height),]
            back =  [(corners[0][4]/width,corners[1][4]/height),
                    (corners[0][5]/width,corners[1][5]/height),
                    (corners[0][6]/width,corners[1][6]/height),
                    (corners[0][7]/width,corners[1][7]/height),]
            #print(corners.shape)
            polylines.append(fo.Polyline.from_cuboid(front + back, label=box.name))
    sample["cuboids"] = fo.Polylines(polylines=polylines)
    return sample


groups = ("CAMERA_LEFT_FRONT", "CAMERA_LEFT_BACK", "CAMERA_RIGHT_FRONT", "CAMERA_RIGHT_BACK")
samples = []
# Iterate over each scene
for scene in trucksc.scene:
    my_scene = scene
    token = my_scene['first_sample_token']
    my_sample = trucksc.get('sample', token)
    last_sample_token = my_scene['last_sample_token']
    
    # Iterate over each sample in the scene
    while not my_sample["next"] == "":
        scene_token = my_sample["scene_token"]
        group = fo.Group()
        # Iterate over each sensor in the sample
        for sensor in groups:
            data = trucksc.get('sample_data', my_sample['data'][sensor])
            filepath = trucksc.dataroot + data["filename"]

            # Check if the sensor is a camera
            if data["sensor_modality"] == "camera":
                sample = camera_sample(group, filepath, sensor, my_sample['data'][sensor],scene)

            # Add metadata to the sample
            sample["token"] = data["token"]
            sample["ego_pose_token"] = data["ego_pose_token"]
            sample["calibrated_sensor_token"] = data["calibrated_sensor_token"]
            sample["timestamp"] = data["timestamp"]
            sample["is_key_frame"] = data["is_key_frame"]
            sample["prev"] = data["prev"]
            sample["next"] = data["next"]
            sample["scene_token"] = scene_token

            
            samples.append(sample)

        token = my_sample["next"]

        my_sample = trucksc.get('sample', token)

dataset.add_samples(samples)
view = dataset.group_by("scene_token", order_by="timestamp")
dataset.persistent = True

session = fo.launch_app(dataset)