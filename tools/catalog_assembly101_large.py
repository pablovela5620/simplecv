import tyro

from simplecv.apis.exoego_forge_catalog import Assembly101LargeIndexConfig, main_large_index

if __name__ == "__main__":
    main_large_index(
        tyro.cli(
            Assembly101LargeIndexConfig,
            description="Host a lightweight URL index for the generated full-size Assembly101 RRDs.",
        )
    )
