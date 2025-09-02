import os

# Enable beartype runtime checking for the test suite in dev environment.
if os.environ.get("PIXI_ENVIRONMENT_NAME") == "dev":
    try:
        from beartype.claw import beartype_this_package

        beartype_this_package()
    except Exception:
        # Be permissive if beartype isn't available in this environment.
        pass

