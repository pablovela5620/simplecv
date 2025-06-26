import tyro

from simplecv.apis.view_exoego import VisualizeConfig, visualize_exo_ego

# Example usage
if __name__ == "__main__":
    visualize_exo_ego(
        tyro.cli(
            VisualizeConfig,
            description="Visualize Ego Only dataset",
        )
    )
