import tyro

from simplecv.apis.quest3_oakd_loader import Quest3OakDVisualizeConfig, main

# Example usage
if __name__ == "__main__":
    main(
        tyro.cli(
            Quest3OakDVisualizeConfig,
            description="Visualize Quest3 OakD datasets.",
        )
    )
