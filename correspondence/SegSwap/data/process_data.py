import os
import json
from lzstring import LZString
from pycocotools import mask as mask_utils
import numpy as np
from PIL import Image
from decord import VideoReader
from decord import cpu
import argparse
import cv2
from time import time
from tqdm import tqdm

def save_frames(frames, frame_idxes, output_folder, is_aria=False):
    # resize and save frames
    scale = 4
    if is_aria:
        scale = 2

    for img, fidx in zip(frames, frame_idxes):
        H, W, C = img.shape
        if H < 1408:
            break
        img2 = cv2.resize(img, (W//scale, H//scale))
        cv2.imwrite(os.path.join(output_folder, f'{fidx}.jpg'), img2)

def processVideo(takepath, take_name, ego_cam, exo_cams, outputpath, take_id):

    if not os.path.exists(f"{takepath}/{take_name}/frame_aligned_videos/{ego_cam}.mp4"):
        return -1

    # Subsample the ego video
    vr = VideoReader(
        f"{takepath}/{take_name}/frame_aligned_videos/{ego_cam}.mp4", ctx=cpu(0)
    )
    len_video = len(vr)
    # subsampling at 1fps -- none of the videos are annotated at more than 1 fps
    subsample_idx = np.arange(0, len_video, 30)
    
    if not os.path.exists(f"{outputpath}/{take_id}/{ego_cam}"):
        os.makedirs(f"{outputpath}/{take_id}/{ego_cam}")
        frames = vr.get_batch(subsample_idx).asnumpy()[...,::-1]
        save_frames(frames=frames, frame_idxes=subsample_idx, output_folder=f"{outputpath}/{take_id}/{ego_cam}", is_aria=True)

    # Subsample the exo videos
    for exo_cam in exo_cams:
        if not os.path.isfile(f"{outputpath}/{take_id}/{exo_cam}.mp4"):
            try:
                vr = VideoReader(
                    f"{takepath}/{take_name}/frame_aligned_videos/{exo_cam}.mp4", ctx=cpu(0)
                )
            except:
                print(f"{exo_cam} not available")
                continue
            os.makedirs(f"{outputpath}/{take_id}/{exo_cam}")
            frames = vr.get_batch(subsample_idx).asnumpy()[...,::-1]

            save_frames(frames=frames, frame_idxes=subsample_idx, output_folder=f"{outputpath}/{take_id}/{exo_cam}", is_aria=False)

    return subsample_idx.tolist()

def decode_mask(width, height, encoded_mask):
    try: 
        decomp_string = LZString.decompressFromEncodedURIComponent(encoded_mask)
    except:
        return None
    decomp_encoded = decomp_string.encode()
    rle_obj = {
        "size": [height, width],
        "counts": decomp_encoded,
    }       
    rle_obj['counts'] = rle_obj['counts'].decode('ascii')
    return rle_obj

def processMask(anno, new_anno):
    for object_id in anno.keys():
        new_anno[object_id] = {}
        for cam_id in anno[object_id].keys():
            new_anno[object_id][cam_id] = {}
            for frame_id in anno[object_id][cam_id]["annotation"].keys():
                width = anno[object_id][cam_id]["annotation"][frame_id]["width"]
                height = anno[object_id][cam_id]["annotation"][frame_id]["height"]
                encoded_mask = anno[object_id][cam_id]["annotation"][frame_id]["encodedMask"]
                coco_mask = decode_mask(width, height, encoded_mask)
                new_anno[object_id][cam_id][frame_id] = coco_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--takepath",
        help="EgoExo take data root",
        required=True
    )
    parser.add_argument(
        "--annotationpath",
        help="Annotations json file path",
        required=True
    )
    parser.add_argument(
        "--split_path",
        help="path to split.json",
        required=True
    )
    parser.add_argument(
        "--split",
        help="train/val/test split to process",
        required=True
    )
    parser.add_argument(
        "--outputpath",
        help="Output data root",
        required=True
        )
    args = parser.parse_args()

    with open(args.split_path, "r") as fp:
        data_split = json.load(fp)
    take_list = data_split[args.split]

    os.makedirs(args.outputpath, exist_ok=True)
    # Read the annotation file
    with open(args.annotationpath, "r") as f:
        annos = json.load(f)
    annos = annos['annotations']

    start = time()
    
    # Set to track incomplete takes
    incomplete_takes = set()
    video_not_found = set()

    for take_id in tqdm(take_list):
        if os.path.exists(f"{args.outputpath}/{take_id}"):
            # Get the original annotation for this take
            anno = annos[take_id]
            
            # Validate if all expected frames and masks exist
            annotation_path = f"{args.outputpath}/{take_id}/annotation.json"
            if os.path.exists(annotation_path):
                try:
                    # Load saved annotation to validate
                    with open(annotation_path, "r") as f:
                        saved_anno = json.load(f)
                    
                    # Check if required fields exist
                    if "masks" not in saved_anno or "subsample_idx" not in saved_anno:
                        raise ValueError("Missing masks or subsample_idx in saved annotation")
                    
                    # Collect all frames that should exist from saved masks
                    cam_ids = set()
                    for object_id, cameras in saved_anno["masks"].items():
                        for cam_id, frames in cameras.items():
                            cam_ids.add(cam_id)

                    for cam_id in cam_ids:
                        cam_folder = f"{args.outputpath}/{take_id}/{cam_id}"
                        if not os.path.exists(cam_folder):
                            raise ValueError("Missing cam folder")
                        else:
                            for frame_id in saved_anno['subsample_idx']:
                                frame_path = os.path.join(cam_folder, f"{frame_id}.jpg")
                                if not os.path.exists(frame_path):
                                    raise ValueError(f"Frame does not exist: {frame_path}")
                                # img = cv2.imread(frame_path)
                                # if img is None:
                                #     raise ValueError(f"Cannot load frame: {frame_path}")
                                # # Check if image has valid shape
                                # if len(img.shape) != 3 or img.shape[2] != 3:
                                #     raise ValueError(f"Invalid image shape: {img.shape}")
                except Exception as e:
                    print(f"  ✗ Error validating {take_id}: {e}")
                    incomplete_takes.add(take_id)
            else:
                print(f"  ✗ No annotation.json found for {take_id}")
                incomplete_takes.add(take_id)
        else:
            print(f"  ✗ No folder found for {take_id}")
            incomplete_takes.add(take_id)

    if len(incomplete_takes) > 0:
        print(f"Processing {len(incomplete_takes)} incomplete takes")
        for take_id in tqdm(incomplete_takes):
            # Create the output folder
            os.makedirs(f"{args.outputpath}/{take_id}", exist_ok=True)
            new_anno = {}
            # Get the corresponding take name
            anno = annos[take_id]
            take_name = anno["take_name"]

            valid_cams = set()
            for x in anno['object_masks'].keys(): 
                valid_cams.update(set(anno['object_masks'][x].keys()))
            
            ego_cams = []
            exo_cams = []
            for vc in valid_cams:
                if 'aria' in vc:
                    ego_cams.append(vc)
                else:
                    exo_cams.append(vc)

            if len(ego_cams) > 1:
                print(take_id, 'HAS MORE THAN ONE EGO')
                breakpoint()
            print(f"Processing take {take_id} {take_name}")
            
            # Process the masks
            print("Start processing masks")
            new_anno["masks"] = {}
            processMask(anno['object_masks'], new_anno["masks"])

            # # Process the videos
            print("Start processing Videos")
            subsample_idx = processVideo(args.takepath, take_name, ego_cam=ego_cams[0], exo_cams=exo_cams, outputpath=args.outputpath, take_id=take_id)
            if subsample_idx == -1:
                print(f"{args.takepath}/{take_name}/frame_aligned_videos/{ego_cams[0]}.mp4 does not exist")
                video_not_found.add(f"{args.takepath}/{take_name}/frame_aligned_videos/{ego_cams[0]}.mp4")
                continue
            new_anno["subsample_idx"] = subsample_idx

            # Save the annotation
            with open(f"{args.outputpath}/{take_id}/annotation.json", "w") as f:
                json.dump(new_anno, f)

    if len(incomplete_takes) >= 0:
        print(f"{len(incomplete_takes)} incomplete takes")
    if len(video_not_found) > 0:
        print(f"{len(video_not_found)} missing videos")
    
    if len(incomplete_takes) == len(video_not_found):
        print("All takes are already done!")

    else:
        print(f"Processing {len(incomplete_takes) - len(video_not_found)} incomplete takes")
        for take_id in tqdm(incomplete_takes):
            print(f"  ✗ {take_id} is incomplete")

    end = time()
    print(f"Total time: {end-start} seconds")