import json
from collections.abc import Generator
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
from jaxtyping import Float32, UInt8
from numpy import ndarray
from serde import field as serde_field
from serde import from_dict, serde
from tqdm import tqdm

from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.data.exoego.base_exo_ego import BaseExoEgoSequence, ExoBatchData, ExoData
from simplecv.data.exoego.skeleton.assembly_hands import (
    HAND_ID2NAME,
    HAND_IDS,
    HAND_LINKS,
)
from simplecv.video_io import MultiVideoReader


@serde
class ExoExtriCameras:
    # Use the "rename" parameter to indicate that the JSON key "0" should map to serde_field "left"
    C10404: Float32[ndarray, "4 4"] = serde_field(rename="C10404:rgb")
    C10118: Float32[ndarray, "4 4"] = serde_field(rename="C10118:rgb")
    C10119: Float32[ndarray, "4 4"] = serde_field(rename="C10119:rgb")
    C10095: Float32[ndarray, "4 4"] = serde_field(rename="C10095:rgb")
    C10379: Float32[ndarray, "4 4"] = serde_field(rename="C10379:rgb")
    C10395: Float32[ndarray, "4 4"] = serde_field(rename="C10395:rgb")
    C10115: Float32[ndarray, "4 4"] = serde_field(rename="C10115:rgb")
    C10390: Float32[ndarray, "4 4"] = serde_field(rename="C10390:rgb")


@serde
class EgoExtriCameras:
    # Use the "rename" parameter to indicate that the JSON key "0" should map to serde_field "left"
    C21176875: Float32[ndarray, "4 4"] = serde_field(rename="21176875:mono10bit")
    C21179183: Float32[ndarray, "4 4"] = serde_field(rename="21179183:mono10bit")
    C21110305: Float32[ndarray, "4 4"] = serde_field(rename="21110305:mono10bit")
    C21176623: Float32[ndarray, "4 4"] = serde_field(rename="21176623:mono10bit")


@serde
class Hand3DKeypoints:
    # Use the "rename" parameter to indicate that the JSON key "0" should map to serde_field "left"
    left: Float32[ndarray, "21 3"] = serde_field(rename="0")
    # And similarly for "1" -> "right"
    right: Float32[ndarray, "21 3"] = serde_field(rename="1")


@serde
class Hand2DKeypoints:
    # Use the "rename" parameter to indicate that the JSON key "0" should map to serde_field "left"
    left: Float32[ndarray, "21 2"] = serde_field(rename="0")
    # And similarly for "1" -> "right"
    right: Float32[ndarray, "21 2"] = serde_field(rename="1")


@serde
class Exo2DKeypoints:
    C10395: Hand2DKeypoints = serde_field(rename="C10395:rgb")
    C10379: Hand2DKeypoints = serde_field(rename="C10379:rgb")
    C10404: Hand2DKeypoints = serde_field(rename="C10404:rgb")
    C10119: Hand2DKeypoints = serde_field(rename="C10119:rgb")
    C10115: Hand2DKeypoints = serde_field(rename="C10115:rgb")
    C10390: Hand2DKeypoints = serde_field(rename="C10390:rgb")
    C10118: Hand2DKeypoints = serde_field(rename="C10118:rgb")
    C10095: Hand2DKeypoints = serde_field(rename="C10095:rgb")


@serde
class Ego2DKeypoints:
    C21176875: Hand2DKeypoints = serde_field(rename="21176875:mono10bit")
    C21179183: Hand2DKeypoints = serde_field(rename="21179183:mono10bit")
    C21110305: Hand2DKeypoints = serde_field(rename="21110305:mono10bit")
    C21176623: Hand2DKeypoints = serde_field(rename="21176623:mono10bit")


def load_ego_cameras(
    extrinsics_ego_path: Path,
    train_assembly_hands_json: Path,
    height: int,
    width: int,
) -> list[PinholeParameters]:
    with open(extrinsics_ego_path) as f:
        extrinsics_ego = json.load(f)

    exo_raw_extri: EgoExtriCameras = from_dict(EgoExtriCameras, extrinsics_ego)
    # assembly101 does not have camera intrinsics, so need to get them from assemblyhands
    with open(train_assembly_hands_json) as f:
        train_assembly_hands_dict: dict = json.load(f)

    all_calib_dict: dict = train_assembly_hands_dict["calibration"]
    # assume that all cameras have the same intrinsics for each capture, so only get a single one
    instrinsics_dict: dict[str, list[list[float]]] = next(
        iter(all_calib_dict.values())
    )["intrinsics"]

    pinhole_list: list[PinholeParameters] = []

    cam_name: str
    exo_camera: Float32[ndarray, "4 4"]
    for cam_name, exo_camera in asdict(exo_raw_extri).items():
        intri: Float32[ndarray, "3 3"] = np.array(
            instrinsics_dict[f"{cam_name}_rgb"], dtype=np.float32
        )
        intri = Intrinsics(
            camera_conventions="RDF",
            fl_x=float(intri[0, 0]),
            fl_y=float(intri[1, 1]),
            cx=float(intri[0, 2]),
            cy=float(intri[1, 2]),
            height=height,
            width=width,
        )
        extri = Extrinsics(
            world_R_cam=exo_camera[:3, :3],
            world_t_cam=exo_camera[:3, 3],
        )
        pinhole_param = PinholeParameters(
            name=cam_name,
            intrinsics=intri,
            extrinsics=extri,
        )
        pinhole_list.append(pinhole_param)

    return pinhole_list


class Assembely101Sequence(BaseExoEgoSequence):
    def __init__(
        self,
        data_path: Path,
        sequence_name: str,
        subject_id: str | None = None,
        load_labels: bool = False,
    ) -> None:
        self.encoding: Literal["av1", "h264"] = "av1"
        super().__init__(data_path, sequence_name, subject_id, load_labels)

    def __len__(self) -> int:
        assert len(self.video_path_list) > 0, "No videos found."
        # Make sure all cameras have the same number of images
        return len(self.exo_video_readers)

    def __iter__(self) -> Generator[ExoData, None, None]:
        for idx in range(len(self)):
            bgr_list: list[UInt8[ndarray, "H W 3"]] = self.exo_video_readers[idx]
            xyz: Float32[ndarray, "2 21 3"] = self.exo_batch_data.xyz_stack[idx]
            uv_dict: dict[str, Float32[ndarray, "2 21 2"]] = {
                cam_name: uv_stack[idx]
                for cam_name, uv_stack in self.exo_batch_data.uv_stack_dict.items()
            }
            yield ExoData(
                cam_params_list=self.exo_cam_list,
                bgr_list=bgr_list,
                xyz=xyz,
                uv_dict=uv_dict,
            )

    def load_video_paths(
        self, data_path: Path, sequence_name: str, subject_id: str | None = None
    ) -> list[Path]:
        """Load the paths to the video files."""
        video_dir: Path = data_path / "videos" / self.encoding / sequence_name
        assert video_dir.exists(), f"Directory {video_dir} does not exist"
        exo_video_files: list[Path] = sorted(
            [
                file
                for file in video_dir.iterdir()
                if file.is_file() and not file.name.startswith("HMC")
            ]
        )

        return exo_video_files

    def load_exo_cameras(
        self, data_path: Path, sequence_name: str, subject_id: str | None = None
    ) -> list[PinholeParameters]:
        extrinsics_exo_path: Path = (
            data_path
            / "assembly101_camera_and_hand_poses"
            / "camera_extrinsics_fixed"
            / f"{sequence_name}.json"
        )
        assert extrinsics_exo_path.exists(), (
            f"File {extrinsics_exo_path} does not exist"
        )
        assembly_hands_annotation_path: Path = data_path / "assembly-hands"
        train_assembly_hands_json: Path = (
            assembly_hands_annotation_path
            / "annotations"
            / "train"
            / "assemblyhands_train_exo_calib_v1-1.json"
        )
        assert train_assembly_hands_json.exists(), (
            f"File {train_assembly_hands_json} does not exist"
        )
        with open(extrinsics_exo_path) as f:
            extrinsics_fixed = json.load(f)

        # load videos to get height and width
        video_paths: list[Path] = self.load_video_paths(
            data_path, sequence_name, subject_id
        )
        # sort extrinsics_fixed by camera name
        extrinsics_fixed = dict(sorted(extrinsics_fixed.items()))
        exo_mv_video_reader = MultiVideoReader(video_paths)
        height, width = exo_mv_video_reader.height, exo_mv_video_reader.width
        exo_raw_extri: ExoExtriCameras = from_dict(ExoExtriCameras, extrinsics_fixed)
        # assembly101 does not have camera intrinsics, so need to get them from assemblyhands
        with open(train_assembly_hands_json) as f:
            train_assembly_hands_dict: dict = json.load(f)

        all_calib_dict: dict = train_assembly_hands_dict["calibration"]
        # assume that all cameras have the same intrinsics for each capture, so only get a single one
        instrinsics_dict: dict[str, list[list[float]]] = next(
            iter(all_calib_dict.values())
        )["intrinsics"]

        pinhole_list: list[PinholeParameters] = []

        cam_name: str
        exo_camera: Float32[ndarray, "4 4"]
        for cam_name, exo_camera in asdict(exo_raw_extri).items():
            intri: Float32[ndarray, "3 3"] = np.array(
                instrinsics_dict[f"{cam_name}_rgb"], dtype=np.float32
            )
            intri = Intrinsics(
                camera_conventions="RDF",
                fl_x=float(intri[0, 0]),
                fl_y=float(intri[1, 1]),
                cx=float(intri[0, 2]),
                cy=float(intri[1, 2]),
                height=height,
                width=width,
            )
            extri = Extrinsics(
                world_R_cam=exo_camera[:3, :3],
                world_t_cam=exo_camera[:3, 3],
            )
            pinhole_param = PinholeParameters(
                name=cam_name,
                intrinsics=intri,
                extrinsics=extri,
            )
            pinhole_list.append(pinhole_param)

        pinhole_list = list(sorted(pinhole_list, key=lambda x: x.name))
        return pinhole_list

    def load_exo_batch_data(
        self, data_path: Path, sequence_name: str, subject_id: str | None = None
    ) -> ExoBatchData:
        """Load the exocentric data for a sequence."""
        uv_json_path: Path = (
            data_path
            / "assembly101_camera_and_hand_poses"
            / "landmarks2D"
            / f"{sequence_name}.json"
        )
        assert data_path.exists(), f"File {data_path} does not exist"

        with open(uv_json_path) as f:
            all_uv_raw_dict: dict[str, dict[str, dict[str, list[list[float]]]]] = (
                json.loads(f.read())
            )

        uv_stack_dict: dict[str, Float32[ndarray, "num_frames 2 21 2"]]
        # sort all_2d_landmarks by frame number
        all_uv_raw_dict = dict(
            sorted(all_uv_raw_dict.items(), key=lambda item: int(item[0]))
        )
        uv_stack_dict = {}  # Initialize the dictionary first

        for cam_name in tqdm(
            [exo_cam.name for exo_cam in self.exo_cam_list], desc="Processing cameras"
        ):
            uv_list: list[Float32[ndarray, "num_frames 2 21 2"]] = []
            for uv_dict in tqdm(
                all_uv_raw_dict.values(),
                desc=f"Processing frames for {cam_name}",
                leave=False,
            ):
                left_right_uv_dict = uv_dict[f"{cam_name}:rgb"]
                uv_list.append(
                    np.stack(
                        (left_right_uv_dict["0"], left_right_uv_dict["1"]),
                        axis=0,
                        dtype=np.float32,
                    )
                )

            uv_final_stack: Float32[ndarray, "num_frames 2 21 2"] = np.stack(
                uv_list, axis=0
            )
            uv_stack_dict[cam_name] = uv_final_stack

        ### Load 3D keypoints ###
        xyz_json_path: Path = (
            data_path
            / "assembly101_camera_and_hand_poses"
            / "landmarks3D"
            / f"{sequence_name}.json"
        )
        assert xyz_json_path.exists(), f"File {xyz_json_path} does not exist"
        with open(xyz_json_path) as f:
            all_xyz_dict: dict[str, dict[str, list[list[float]]]] = json.loads(f.read())

        # sort all_3d_landmarks by frame number
        all_xyz_dict = dict(sorted(all_xyz_dict.items(), key=lambda item: int(item[0])))

        all_xyz_dict: dict[int, Hand3DKeypoints] = {
            int(k): from_dict(Hand3DKeypoints, v) for k, v in all_xyz_dict.items()
        }

        xyz_stack_list: list[Float32[ndarray, "2 21 3"]] = []
        for frame_number, _ in enumerate(tqdm(all_xyz_dict)):
            keypoints: Hand3DKeypoints = all_xyz_dict[frame_number]
            xyz_stack_list.append(
                np.stack((keypoints.left, keypoints.right), axis=0, dtype=np.float32)
            )

        # Concatenate keypoints from all frames vertically to get a (num_frames 21, 3) array.
        xyz_stack: Float32[ndarray, "num_frames 2 21 3"] = np.stack(
            xyz_stack_list, axis=0
        )
        return ExoBatchData(uv_stack_dict=uv_stack_dict, xyz_stack=xyz_stack)

    @property
    def hand_links(self) -> tuple[tuple[int, int], ...]:
        """Get the links between hand joints."""
        return HAND_LINKS

    @property
    def hand_ids(self) -> list[int]:
        """Get the IDs of hand joints."""
        return HAND_IDS

    @property
    def hand_id2name(self) -> dict[int, str]:
        """Get mapping from joint ID to joint name."""
        return HAND_ID2NAME

    @property
    def world_coordinate_system(self):
        return rr.ViewCoordinates.BUL

    @property
    def depth_paths(self) -> None:
        """Get mapping from joint ID to joint name."""
        return None
