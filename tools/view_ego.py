import tyro

from simplecv.apis.view_ego_data import ViewEgoConfig, view_ego

# Example usage
if __name__ == "__main__":
    view_ego(
        tyro.cli(
            ViewEgoConfig,
            description="Visualize Ego Only dataset",
        )
    )
