import tyro

from simplecv.apis.ingest_exoego_recording import IngestConfig, main

# Example usage
if __name__ == "__main__":
    main(
        tyro.cli(
            IngestConfig,
            description="Given a directory with ego/exo recordings, save them to RRD and visualize them with Rerun.",
        )
    )
