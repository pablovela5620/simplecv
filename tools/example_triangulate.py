from argparse import ArgumentParser
from pathlib import Path

import rerun as rr

from simplecv.data.easymocap import load_cameras


def log_camera():
    pass


def main():
    new_cameras = load_cameras(data_path=Path("data"))


# Example usage
if __name__ == "__main__":
    parser = ArgumentParser()
    rr.script_add_args(parser)

    args = parser.parse_args()
    rr.script_setup(args=args, application_id="triangulate")
    main()
    rr.script_teardown(args)
