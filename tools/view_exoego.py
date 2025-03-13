import tyro

from simplecv.apis.view_exoego_data import VisualzeConfig, visualize_exo_ego

if __name__ == "__main__":
    visualize_exo_ego(tyro.cli(VisualzeConfig))
