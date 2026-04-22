import argparse
import json
from collections import Counter
from pathlib import Path


DFA_TOPICS: frozenset[str] = frozenset(
    [
        "Design Patterns",
        "Gaming",
        "Personalisation",
        "Influencer Marketing",
        "Subscriptions",
        "Unfair Pricing",
        "Other Unfair Contract Terms",
    ]
)


def title_case(text: str) -> str:
    if not text:
        return text
    return " ".join(w[0].upper() + w[1:] if w else w for w in text.strip().split())


def process_file(path: Path, apply: bool) -> tuple[int, int, int]:
    """
    Returns (items_total, normalised, relabelled).
    normalised  — topics whose text changed after title-casing
    relabelled  — items whose topic_source changed from 'dfa' to 'new'
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        doc = json.loads(raw)
    except Exception as e:
        print(f"  SKIP {path.name}: {e}")
        return 0, 0, 0

    items_total = normalised = relabelled = 0
    dirty = False
    paras = (doc.get("extractions") or {}).get("paragraphs") or []

    for para in paras:
        for item in para.get("items") or []:
            items_total += 1
            original_topic = item.get("topic") or ""

            normed = title_case(original_topic)
            if normed != original_topic:
                item["topic"] = normed
                normalised += 1
                dirty = True

            if (
                item.get("topic_source") == "dfa"
                and item.get("topic") not in DFA_TOPICS
            ):
                item["topic_source"] = "new"
                relabelled += 1
                dirty = True

            if (
                item.get("topic_source") == "new"
                and item.get("topic") in DFA_TOPICS
            ):
                item["topic_source"] = "dfa"
                relabelled += 1
                dirty = True

    if dirty and apply:
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")

    return items_total, normalised, relabelled


def process_directory(directory: Path, apply: bool) -> None:
    exts = (".paragraph.json", ".consolidated.json")
    files = sorted(
        p for p in directory.iterdir() if p.is_file() and p.name.endswith(exts)
    )

    if not files:
        print(f"  No matching JSON files in {directory}")
        return

    total_items = total_normalised = total_relabelled = 0
    normalisation_changes: Counter[tuple[str, str]] = Counter()
    relabel_topics: Counter[str] = Counter()

    for path in files:
        if not apply:
            try:
                doc = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                for para in (doc.get("extractions") or {}).get("paragraphs") or []:
                    for item in para.get("items") or []:
                        total_items += 1
                        orig = item.get("topic") or ""
                        normed = title_case(orig)
                        if normed != orig:
                            total_normalised += 1
                            normalisation_changes[(orig, normed)] += 1
                        effective = normed if normed != orig else orig
                        if (
                            item.get("topic_source") == "dfa"
                            and effective not in DFA_TOPICS
                        ):
                            total_relabelled += 1
                            relabel_topics[effective or orig] += 1
            except Exception:
                pass
        else:
            items, norm, relab = process_file(path, apply=True)
            total_items += items
            total_normalised += norm
            total_relabelled += relab

    action = "Applied" if apply else "Would apply"
    print(
        f"  {len(files):,} files · {total_items:,} items · "
        f"{action}: {total_normalised:,} title-case fixes · {total_relabelled:,} source relabels"
    )

    if normalisation_changes and not apply:
        print("  Title-case changes (top 20):")
        for (before, after), count in normalisation_changes.most_common(20):
            print(f"    {count:5d}×  {before!r}  →  {after!r}")

    if relabel_topics and not apply:
        print("  Topics relabelled dfa→new after normalisation:")
        for topic, count in relabel_topics.most_common():
            print(f"    {count:5d}×  {topic!r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalise and fix topic labels in consolidated JSONs."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Write changes to disk. Without this flag the script only reports.",
    )
    parser.add_argument(
        "--root",
        default="../../data/consolidated_outputs",
        help="Path to the consolidated_outputs directory (default: ../../data/consolidated_outputs)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: root not found: {root.resolve()}")
        return

    mode = "APPLY" if args.apply else "DRY RUN"
    print(f"=== fix_topic_source — {mode} ===")
    print(f"Root: {root.resolve()}")
    print(f"Canonical DFA topics ({len(DFA_TOPICS)}): {', '.join(sorted(DFA_TOPICS))}")
    print()

    subdirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not subdirs:
        print("No subdirectories found — nothing to do.")
        return

    for subdir in subdirs:
        print(f"[{subdir.name}]")
        process_directory(subdir, apply=args.apply)
        print()

    if not args.apply:
        print("Dry run complete. Run with --apply to write changes.")
    else:
        print("Done.")


if __name__ == "__main__":
    main()
