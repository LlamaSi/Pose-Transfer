OpenPose Demo - Output
====================================



## Contents
1. [Output Format](#output-format)
    1. [Keypoint Ordering](#keypoint-ordering)
    2. [Heatmap Ordering](#heatmap-ordering)
    3. [Face and Hands](#face-and-hands)
    4. [Pose Output Format](#pose-output-format)
    5. [Face Output Format](#face-output-format)
    6. [Hand Output Format](#hand-output-format)
3. [Reading Saved Results](#reading-saved-results)
4. [Keypoint Format in the C++ API](#keypoint-format-in-the-c-api)



## Output Format
There are 2 alternatives to save the OpenPose output.

1. The `write_json` flag saves the people pose data using a custom JSON writer. Each JSON file has a `people` array of objects, where each object has:
    1. An array `pose_keypoints_2d` containing the body part locations and detection confidence formatted as `x1,y1,c1,x2,y2,c2,...`. The coordinates `x` and `y` can be normalized to the range [0,1], [-1,1], [0, source size], [0, output size], etc., depending on the flag `keypoint_scale` (see flag for more information), while `c` is the confidence score in the range [0,1].
    2. The arrays `face_keypoints_2d`, `hand_left_keypoints_2d`, and `hand_right_keypoints_2d`, analogous to `pose_keypoints_2d`.
    3. The analogous 3-D arrays `body_keypoints_3d`, `face_keypoints_3d`, `hand_left_keypoints_2d`, and `hand_right_keypoints_2d` (if `--3d` is enabled, otherwise they will be empty). Instead of `x1,y1,c1,x2,y2,c2,...`, their format is `x1,y1,z1,c1,x2,y2,z2,c2,...`, where `c` is simply 1 or 0 depending on whether the 3-D reconstruction was successful or not.
    4. The body part candidates before being assembled into people (if `--part_candidates` is enabled).
```
{
    "version":1.1,
    "people":[
        {
            "pose_keypoints_2d":[582.349,507.866,0.845918,746.975,631.307,0.587007,...],
            "face_keypoints_2d":[468.725,715.636,0.189116,554.963,652.863,0.665039,...],
            "hand_left_keypoints_2d":[746.975,631.307,0.587007,615.659,617.567,0.377899,...],
            "hand_right_keypoints_2d":[617.581,472.65,0.797508,0,0,0,723.431,462.783,0.88765,...]
            "pose_keypoints_3d":[582.349,507.866,507.866,0.845918,507.866,746.975,631.307,0.587007,...],
            "face_keypoints_3d":[468.725,715.636,715.636,0.189116,715.636,554.963,652.863,0.665039,...],
            "hand_left_keypoints_3d":[746.975,631.307,631.307,0.587007,631.307,615.659,617.567,0.377899,...],
            "hand_right_keypoints_3d":[617.581,472.65,472.65,0.797508,472.65,0,0,0,723.431,462.783,0.88765,...]
        }
    ],
    // If `--part_candidates` enabled
    "part_candidates":[
        {
            "0":[296.994,258.976,0.845918,238.996,365.027,0.189116],
            "1":[381.024,321.984,0.587007],
            "2":[313.996,314.97,0.377899],
            "3":[238.996,365.027,0.189116],
            "4":[283.015,332.986,0.665039],
            "5":[457.987,324.003,0.430488,283.015,332.986,0.665039],
            "6":[],
            "7":[],
            "8":[],
            "9":[],
            "10":[],
            "11":[],
            "12":[],
            "13":[],
            "14":[293.001,242.991,0.674305],
            "15":[314.978,241,0.797508],
            "16":[],
            "17":[369.007,235.964,0.88765]
        }
    ]
}
```

2. (Deprecated) The `write_keypoint` flag uses the OpenCV cv::FileStorage default formats, i.e., JSON (available after OpenCV 3.0), XML, and YML. Note that it does not include any other information othern than keypoints.

Both of them follow the keypoint ordering described in the [Keypoint Ordering](#keypoint-ordering) section.



### Keypoint Ordering
The body part mapping order of any body model (e.g., COCO, MPI) can be extracted from the C++ API by using the `getPoseBodyPartMapping(const PoseModel poseModel)` function available in [poseParameters.hpp](../include/openpose/pose/poseParameters.hpp):
```
// C++ API call
#include <openpose/pose/poseParameters.hpp>
const auto& poseBodyPartMappingBody25 = getPoseBodyPartMapping(PoseModel::BODY_25);
const auto& poseBodyPartMappingCoco = getPoseBodyPartMapping(PoseModel::COCO_18);
const auto& poseBodyPartMappingMpi = getPoseBodyPartMapping(PoseModel::MPI_15);

// Result for BODY_25 (25 body parts consisting of COCO + foot)
// const std::map<unsigned int, std::string> POSE_BODY_25_BODY_PARTS {
//     {0,  "Nose"},
//     {1,  "Neck"},
//     {2,  "RShoulder"},
//     {3,  "RElbow"},
//     {4,  "RWrist"},
//     {5,  "LShoulder"},
//     {6,  "LElbow"},
//     {7,  "LWrist"},
//     {8,  "MidHip"},
//     {9,  "RHip"},
//     {10, "RKnee"},
//     {11, "RAnkle"},
//     {12, "LHip"},
//     {13, "LKnee"},
//     {14, "LAnkle"},
//     {15, "REye"},
//     {16, "LEye"},
//     {17, "REar"},
//     {18, "LEar"},
//     {19, "LBigToe"},
//     {20, "LSmallToe"},
//     {21, "LHeel"},
//     {22, "RBigToe"},
//     {23, "RSmallToe"},
//     {24, "RHeel"},
//     {25, "Background"}
// };
```