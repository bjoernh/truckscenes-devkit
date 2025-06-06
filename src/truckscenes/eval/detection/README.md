# MAN TruckScenes detection task

## Overview
- [Introduction](#introduction)
- [Participation](#participation)
- [Challenges](#challenges)
- [Submission rules](#submission-rules)
- [Results format](#results-format)
- [Classes and attributes](#classes-attributes-and-detection-ranges)
- [Evaluation metrics](#evaluation-metrics)
- [Leaderboard](#leaderboard)

## Introduction
Here we define the 3D object detection task on MAN TruckScenes.
The goal of this task is to place a 3D bounding box around 12 different object categories,
as well as estimating a set of attributes and the current velocity vector.

## Participation
The TruckScenes detection [evaluation server](https://huggingface.co/spaces/TruckScenes/MANTruckScenesDetectionChallenge) is open all year round for submission.
To participate in the challenge, please create an account at [Hugging Face](https://huggingface.co/).
Then upload your json result file including all of the required [meta data](#results-format).
After each challenge, the results will be exported to the TruckScenes [leaderboard](https://www.man.eu/truckscenes/).
This is the only way to benchmark your method against the test dataset.

## Challenges
To allow users to benchmark the performance of their method against the community, we host a single [leaderboard](https://www.man.eu/truckscenes/) all-year round.
Additionally we organize a number of challenges at leading Computer Vision conference workshops.
Users that submit their results during the challenge period are eligible for awards.
Any user that cannot attend the workshop (direct or via a representative) will be excluded from the challenge, but will still be listed on the leaderboard.

Click [here](https://huggingface.co/spaces/TruckScenes/MANTruckScenesDetectionChallenge) for the **Hugging Face detection evaluation server**.

## Submission rules
### Detection-specific rules
* The maximum time window of past sensor data and ego poses that may be used at inference time is approximately 0.5s (at most 6 *past* camera images, 11 *past* radar sweeps and 6 *past* lidar sweeps). At training time there are no restrictions.

### General rules
* We release annotations for the train and val set, but not for the test set.
* We release sensor data for train, val and test set.
* Users make predictions on the test set and submit the results to our evaluation server, which returns the metrics listed below.
* We do not use strata. Instead, we filter annotations and predictions beyond class specific distances.
* Users must limit the number of submitted boxes per sample to 500.
* Users must limit the number of lines in the submitted json file to less than 1Mio.
* Every submission provides method information. We encourage publishing code, but do not make it a requirement.
* Top leaderboard entries and their papers will be manually reviewed.
* Each user or team can submit at most five results *per day*. These results must come from different models, rather than submitting results from the same model at different training epochs or with slightly different parameters.
* Each user or team can only select 1 result to be ranked on the public leaderboard.
* Any attempt to circumvent these rules will result in a permanent ban of the team or company from all TruckScenes challenges. 

## Results format
We define a standardized detection result format that serves as an input to the evaluation code.
Results are evaluated for each 2Hz keyframe, also known as `sample`.
The detection results for a particular evaluation set (train/val/test) are stored in a single JSON file. 
For the train and val sets the evaluation can be performed by the user on their local machine.
For the test set the user needs to submit the single JSON result file to the official evaluation server.
The JSON file includes meta data `meta` on the type of inputs used for this method.
Furthermore it includes a dictionary `results` that maps each sample_token to a list of `sample_result` entries.
Each `sample_token` from the current evaluation set must be included in `results`, although the list of predictions may be empty if no object is detected.
```
submission {
    "meta": {
        "use_camera":        <bool>    -- True, if camera data was used as an input. Else false.
        "use_lidar":         <bool>    -- True, if lidar data was used as an input. Else false.
        "use_radar":         <bool>    -- True, if radar data was used as an input. Else false.
        "use_map":           <bool>    -- True, if map data was used as an input. Else false.
        "use_external":      <bool>    -- True, if external data was used as an input. Else false.
        "use_future_frames": <bool>    -- True, if future frames were used as an input during test. Else false.
        "use_tta":           <bool>    -- True, if test time augmentation was applied during test. Else false.
        "method_name":       <str>     -- Name of the used approach.
        "authors":           <str>     -- Authors of the method/paper. Empty string if not available.
        "affiliation":       <str>     -- Company, university etc. Empty string if not available.
        "description":       <str>     -- Short info about method, remarks. Empty string if not available.
        "code_url":          <str>     -- Link to open source code of the method. Empty string if not available.
        "paper_url":         <str>     -- Link to method's paper. Empty string if not available.
    },
    "results": {
        sample_token         <str>: List[sample_result] -- Maps each sample_token to a list of sample_results.
    }
}
```
For the predictions we create a new database table called `sample_result`.
The `sample_result` table is designed to mirror the `sample_annotation` table.
This allows for processing of results and annotations using the same tools.
A `sample_result` is a dictionary defined as follows:
```
sample_result {
    "sample_token":       <str>         -- Foreign key. Identifies the sample/keyframe for which objects are detected.
    "translation":        <float> [3]   -- Estimated bounding box location in m in the global frame: center_x, center_y, center_z.
    "size":               <float> [3]   -- Estimated bounding box size in m: width, length, height.
    "rotation":           <float> [4]   -- Estimated bounding box orientation as quaternion in the global frame: w, x, y, z.
    "velocity":           <float> [2]   -- Estimated bounding box velocity in m/s in the global frame: vx, vy.
    "detection_name":     <str>         -- The predicted class for this sample_result, e.g. car, pedestrian.
    "detection_score":    <float>       -- Object prediction score between 0 and 1 for the class identified by detection_name.
    "attribute_name":     <str>         -- Name of the predicted attribute or empty string for classes without attributes.
                                           See table below for valid attributes for each class, e.g. cycle.with_rider.
}
```
Note that the detection classes may differ from the general classes, as detailed below.

## Classes, attributes, and detection ranges
The MAN Truckscenes dataset comes with annotations for 27 classes.
Some of these only have a handful of samples.
Hence we merge similar classes and remove rare classes.
This results in 12 classes for the detection challenge.
Below we show the table of detection classes and their counterparts in the TruckScenes dataset.
For more information on the classes and their frequencies, see [this page](https://www.man.eu/truckscenes/).

| TruckScenes detection class | TruckScenes general class            |
|-----------------------------|--------------------------------------|
| void / ignore               | human.pedestrian.personal_mobility   |
| void / ignore               | human.pedestrian.stroller            |
| void / ignore               | human.pedestrian.wheelchair          |
| void / ignore               | movable_object.debris                |
| void / ignore               | movable_object.pushable_pullable     |
| void / ignore               | static_object.bicycle_rack           |
| void / ignore               | vehicle.emergency.ambulance          |
| void / ignore               | vehicle.emergency.police             |
| void / ignore               | vehicle.train                        |
| animal                      | animal                               |
| barrier                     | movable_object.barrier               |
| bicycle                     | vehicle.bicycle                      |
| bus                         | vehicle.bus.bendy                    |
| bus                         | vehicle.bus.rigid                    |
| car                         | vehicle.car                          |
| motorcycle                  | vehicle.motorcycle                   |
| other_vehicle               | vehicle.construction                 |
| other_vehicle               | vehicle.other                        |
| pedestrian                  | human.pedestrian.adult               |
| pedestrian                  | human.pedestrian.child               |
| pedestrian                  | human.pedestrian.construction_worker |
| pedestrian                  | human.pedestrian.police_officer      |
| traffic_cone                | movable_object.trafficcone           |
| traffic_sign                | static_object.traffic_sign           |
| trailer                     | vehicle.ego_trailer                  |
| trailer                     | vehicle.trailer                      |
| truck                       | vehicle.truck                        |

Below we list which TruckScenes classes can have which attributes.

For each TruckScenes detection class, the number of annotations decreases with increasing range from the ego vehicle, 
but the number of annotations per range varies by class. Therefore, each class has its own upper bound on evaluated
detection range, as shown below:

| TruckScenes detection class | Attributes                                            | Detection range (meters) |
|-----------------------------|-------------------------------------------------------|--------------------------|
| animal                      | void                                                  | 75                       |
| barrier                     | void                                                  | 75                       |
| traffic_cone                | void                                                  | 75                       |
| bicycle                     | cycle.{with_rider,   without_rider}                   | 75                       |
| motorcycle                  | cycle.{with_rider,   without_rider}                   | 75                       |
| pedestrian                  | pedestrian.{moving, standing,   sitting_lying_down}   | 75                       |
| traffic_sign                | traffic_sign.{pole_mounted,   overhanging, temporary} | 75                       |
| bus                         | vehicle.{moving, parked,   stopped}                   | 150                      |
| car                         | vehicle.{moving, parked,   stopped}                   | 150                      |
| other_vehicle               | vehicle.{moving, parked,   stopped}                   | 150                      |
| trailer                     | vehicle.{moving, parked,   stopped}                   | 150                      |
| truck                       | vehicle.{moving, parked,   stopped}                   | 150                      |

## Evaluation metrics
Below we define the metrics for the TruckScenes detection task.
Our final score is a weighted sum of mean Average Precision (mAP) and several True Positive (TP) metrics.

### Preprocessing
Before running the evaluation code the following pre-processing is done on the data
* All boxes (GT and prediction) are removed if they exceed the class-specific detection range. 
* All bikes and motorcycle boxes (GT and prediction) that fall inside a bike-rack are removed. The reason is that we do not annotate bikes inside bike-racks.  
* All boxes (GT) without lidar or radar points in them are removed. The reason is that we can not guarantee that they are actually visible in the frame. We do not filter the predicted boxes based on number of points.

### Average Precision metric
* **mean Average Precision (mAP)**:
We use the well-known Average Precision metric,
but define a match by considering the 2D center distance on the ground plane rather than intersection over union based affinities. 
Specifically, we match predictions with the ground truth objects that have the smallest center-distance up to a certain threshold.
For a given match threshold we calculate average precision (AP) by integrating the recall vs precision curve for recalls and precisions > 0.1.
We finally average over match thresholds of {0.5, 1, 2, 4} meters and compute the mean across classes.

### True Positive metrics
Here we define metrics for a set of true positives (TP) that measure translation / scale / orientation / velocity and attribute errors. 
All TP metrics are calculated using a threshold of 2m center distance during matching, and they are all designed to be positive scalars.

Matching and scoring happen independently per class and each metric is the average of the cumulative mean at each achieved recall level above 10%.
If 10% recall is not achieved for a particular class, all TP errors for that class are set to 1.
We define the following TP errors:
* **Average Translation Error (ATE)**: Euclidean center distance in 2D in meters.
* **Average Scale Error (ASE)**: Calculated as *1 - IOU* after aligning centers and orientation.
* **Average Orientation Error (AOE)**: Smallest yaw angle difference between prediction and ground-truth in radians. Orientation error is evaluated at 360 degree for all classes except barriers where it is only evaluated at 180 degrees. Orientation errors for cones are ignored.
* **Average Velocity Error (AVE)**: Absolute velocity error in m/s. Velocity error for barriers and cones are ignored.
* **Average Attribute Error (AAE)**: Calculated as *1 - acc*, where acc is the attribute classification accuracy. Attribute error for barriers and cones are ignored.

All errors are >= 0, but note that for translation and velocity errors the errors are unbounded, and can be any positive value.

The TP metrics are defined per class, and we then take a mean over classes to calculate mATE, mASE, mAOE, mAVE and mAAE.

Below we list which error terms are excluded for which TruckScenes detection classes. This is because some error terms
are ambiguous for specific classes that are rotation invariant, static, or without attributes.

| TruckScenes   detection class | Excluded error terms |
|-------------------------------|----------------------|
| animal                        | attribute error      |
| barrier                       | attribute error      |
|                               | velocity error       |
| traffic_cone                  | attribute error      |
|                               | velocity error       |
|                               | orientation error    |
| bicycle                       |                      |
| motorcycle                    |                      |
| pedestrian                    |                      |
| traffic_sign                  | velocity error       |
| bus                           |                      |
| car                           |                      |
| other_vehicle                 |                      |
| trailer                       |                      |
| truck                         |                      |

### Detection score
We are using the same error definitions and metrics as [nuScenes](https://www.nuscenes.org/object-detection). Thus,
* **nuScenes detection score (NDS)**:
We consolidate the above metrics by computing a weighted sum: mAP, mATE, mASE, mAOE, mAVE and mAAE.
As a first step we convert the TP errors to TP scores as *TP_score = max(1 - TP_error, 0.0)*.
We then assign a weight of *5* to mAP and *1* to each of the 5 TP scores and calculate the normalized sum.

We deviate at 2 points from the original NDS (nuScenes Detection Score) 
* we have additional 2 additional classes: animal + traffic sign
* we take into account higher evaluation ranges

Despite this similarity, due to the differences in sensor setup, vehicle and scene settings, the NDS calculated on the TruckScenes dataset can't be compared with NDS values achieved using the nuScenes dataset.

### Configuration
The default evaluation metrics configurations can be found in `truckscenes\eval\detection\configs\detection_cvpr_2024.json`. 

## Leaderboard
MAN TruckScenes will maintain a single leaderboard for the detection task.
For each submission the leaderboard will list method aspects and evaluation metrics.
Method aspects include input modalities (lidar, radar, vision), use of map data, external data, future frames (ff), and test time augmentation (tta).
To enable a fair comparison between methods, the user will be able to filter the methods by method aspects.


_Copied and adapted from [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit/tree/1.0.0/python-sdk/nuscenes/eval/detection)_