"""Capture a pinned (topic, partition, offset) set for a reproducible demo.

Kafka's own rewind (``fink_consumer -start_at earliest``) gives you *a* replay,
not *the same* replay: a shared consumer group's committed offsets move between
runs, so two "replays" can cover different alerts. Pinning explicit offsets and
committing the manifest to git is what turns "the demo is reproducible" into a
claim you can defend — same offsets, same alerts, same predictions, every time.

    # record the current head positions
    python scripts/record_replay_manifest.py --limit 60

    # then, any time, replay exactly that window
    python -m demo.run_consumer --mode replay

By default it pins offsets ``--limit`` behind the current head, so the manifest
points at alerts that already exist rather than at a position the stream has not
reached yet.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from demo.adapters.offsets import PinnedOffset, ReplayManifest  # noqa: E402
from demo.config import get_settings  # noqa: E402

log = logging.getLogger("manifest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--limit", type=int, default=60,
        help="total alerts the replay should consume (default 60)",
    )
    parser.add_argument(
        "--behind", type=int, default=None,
        help="offsets to rewind per partition (default: --limit)",
    )
    parser.add_argument("--out", default=None, help="manifest path")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s %(message)s", stream=sys.stdout
    )
    settings = get_settings(refresh=True)
    behind = args.behind if args.behind is not None else args.limit
    out = Path(args.out) if args.out else settings.replay_manifest

    try:
        from fink_client.consumer import AlertConsumer

        from demo.adapters.fink_kafka import load_kafka_config
    except ImportError as exc:
        log.error(
            "fink-client and confluent-kafka are required to record a manifest "
            "(%s). Install with: pip install -r requirements-demo.txt",
            exc,
        )
        return 2

    try:
        # Shared with the adapter so there is one place that knows how to
        # translate ztf_credentials.yml into a Kafka config.
        conf = load_kafka_config(settings.topics)
    except Exception as exc:
        log.error(
            "could not load ~/.finkclient/ztf_credentials.yml (%s). Register "
            "first: fink_client_register -survey ztf -username ... ",
            exc,
        )
        return 2

    topics = list(settings.topics)
    log.info("connecting to record head offsets for %d topic(s)", len(topics))
    consumer = AlertConsumer(topics, conf, "ztf")

    # An assignment only materialises after the first poll, so provoke one.
    consumer._consumer.poll(min(settings.poll_timeout_s, 10.0))
    assignment = consumer._consumer.assignment()
    if not assignment:
        log.error(
            "no partitions were assigned. Either the topics are empty or the "
            "credentials do not grant access to them."
        )
        consumer.close()
        return 1

    pinned: list[PinnedOffset] = []
    for tp in assignment:
        try:
            low, high = consumer._consumer.get_watermark_offsets(
                tp, timeout=10.0, cached=False
            )
        except Exception as exc:
            log.warning("watermark query failed for %s[%s]: %s", tp.topic, tp.partition, exc)
            continue
        start = max(int(low), int(high) - behind)
        pinned.append(PinnedOffset(tp.topic, tp.partition, start))
        log.info(
            "%s[%s] low=%s high=%s -> pinned at %s (%s alerts available)",
            tp.topic, tp.partition, low, high, start, max(0, int(high) - start),
        )
    consumer.close()

    if not pinned:
        log.error("nothing to pin")
        return 1

    manifest = ReplayManifest.new(pinned, total_limit=args.limit)
    manifest.save(out)
    log.info("wrote %s (%d partition(s), limit %s)", out, len(pinned), args.limit)
    print()
    print("Commit this file so the demo is reproducible from a clean checkout:")
    print(f"  git add {out.relative_to(REPO_ROOT)}")
    print()
    print("Replay it with:")
    print("  python -m demo.run_consumer --mode replay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
