from argparse import ArgumentParser
from pathlib import Path

import rerun as rr

from simplecv.apis.view_umetrack_data import main

if __name__ == "__main__":
    parser = ArgumentParser("Visualize data")
    parser.add_argument(
        "--data-path",
        type=Path,
        # default="/hdd/data/UmeTrack_data/raw_data/real/hand_hand/testing/user_12/",
        default="/mnt/12tbdrive/data/UmeTrack_data/raw_data/real/separate_hand/testing/user_19/",
        help="Path to data, should be a directory that looks like\
              'UmeTrack_data/raw_data/x/x/x/user_xx/",
    )
    rr.script_add_args(parser)
    args = parser.parse_args()
    rr.script_setup(args, "quest2_hand_tracking")
    main(args.data_path)
    rr.script_teardown(args)
