"""Real-time multimodal transient classification demo.

See docs/demo-plan.md. Entry points:

    python -m demo.run_consumer --mode live     # ingest + classify
    python -m demo.run_consumer --mode replay  # ingest + classify (replay from saved data when there is no live stream at night telescope doesn't observe images)
    python -m demo.run_api                      # dashboard + REST API
"""

__version__ = "1.0.0"
