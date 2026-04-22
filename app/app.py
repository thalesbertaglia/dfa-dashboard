import html as html_mod
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# =========================
# Config
# =========================


@dataclass(frozen=True)
class AppPaths:
    dataset_csv: Path
    outputs_dir: Path
    cache_dir: Path


def get_paths() -> AppPaths:
    dataset_csv = Path(os.getenv("DFA_DATASET_CSV", "dataset.csv"))
    outputs_dir = Path(os.getenv("DFA_OUTPUTS_DIR", "data/consolidated_outputs"))
    cache_dir = Path(os.getenv("DFA_CACHE_DIR", ".cache"))
    return AppPaths(
        dataset_csv=dataset_csv, outputs_dir=outputs_dir, cache_dir=cache_dir
    )


def discover_datasets(outputs_dir: Path) -> dict[str, Path]:
    """Return ordered {label: path} for every subdirectory that has a label.txt."""
    result: dict[str, Path] = {}
    if not outputs_dir.exists():
        return result
    for d in sorted(outputs_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        label_file = d / "label.txt"
        if label_file.exists():
            label = label_file.read_text(encoding="utf-8").strip()
        else:
            label = d.name  # fallback to directory name
        if label:
            result[label] = d
    return result


@st.cache_data(show_spinner=False)
def load_provenance(enriched_dir: Path) -> dict[str, Any]:
    """Read extraction_run metadata from the first available JSON file."""
    files = list_enriched_files(enriched_dir)
    for path in files:
        try:
            doc = read_json(path)
            run = doc.get("extraction_run")
            if run:
                return run
        except Exception:
            continue
    return {}


# =========================
# CSS
# =========================

CSS = """
<style>
/* ---- quote cards ---- */
.qcard {
    border-left: 4px solid #3b82f6;
    padding: 12px 16px;
    margin: 8px 0;
    background: #f8fafc;
    border-radius: 0 6px 6px 0;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.qcard.src-new { border-left-color: #f97316; }
.qtext {
    font-style: italic;
    color: #1e293b;
    line-height: 1.65;
    margin: 0 0 8px 0;
    font-size: 0.93em;
}
.qattr {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    align-items: center;
}
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.78em;
    background: #f1f5f9;
    color: #475569;
    white-space: nowrap;
}
.badge-type { background: #dbeafe; color: #1e40af; font-weight: 600; }
.badge-topic { background: #ede9fe; color: #5b21b6; }
.badge-topic-new { background: #ffedd5; color: #9a3412; }

/* ---- topic chips ---- */
.chips { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0; }
.chip {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.78em;
    font-weight: 500;
    white-space: nowrap;
}
.chip-dfa { background: #dbeafe; color: #1d4ed8; }
.chip-new { background: #fed7aa; color: #c2410c; }

/* ---- cluster cards ---- */
.ccard {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px;
    background: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    height: 100%;
}
.ccard h4 { margin: 6px 0 8px 0; color: #0f172a; font-size: 1em; line-height: 1.35; }
.ccard-stats { color: #64748b; font-size: 0.82em; margin-bottom: 10px; }
.ccard-quote {
    font-style: italic;
    color: #475569;
    font-size: 0.82em;
    border-left: 3px solid #cbd5e1;
    padding-left: 8px;
    margin: 0;
    line-height: 1.55;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
}

/* ---- source card (topic evidence view) ---- */
.src-card {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    margin: 10px 0;
    overflow: hidden;
    background: white;
}
.src-card-header {
    padding: 8px 12px;
    background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    align-items: center;
}
.src-card-text {
    padding: 10px 14px;
    color: #475569;
    font-size: 0.88em;
    line-height: 1.75;
    border-bottom: 1px solid #f1f5f9;
    max-height: 130px;
    overflow-y: auto;
}
.src-card-quotes { padding: 10px 12px 12px; }
.src-card-quotes .qcard {
    margin: 6px 0 0 0;
    background: #f0f7ff;
    border-left-color: #93c5fd;
    box-shadow: none;
}
.src-card-quotes .qcard:first-child { margin-top: 0; }
.src-card-quotes .qtext { color: #1e3a5f; font-size: 0.90em; }

/* ---- submission header card ---- */
.sub-header {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 14px;
}
.sub-header h3 { margin: 0 0 8px 0; color: #0f172a; }
.sub-meta { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 6px; }
.sub-feedback {
    color: #334155;
    font-size: 0.88em;
    line-height: 1.65;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #e2e8f0;
}
</style>
"""


# =========================
# Helpers
# =========================


def normalise_doc_id(raw: Any) -> str:
    s = "" if raw is None else str(raw).strip()
    if not s:
        return ""
    if s.endswith(".clean"):
        s = s[: -len(".clean")]
    return s


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def list_enriched_files(enriched_dir: Path) -> list[Path]:
    if not enriched_dir.exists():
        return []
    exts = (".paragraph.json", ".consolidated.json")
    return sorted(
        [p for p in enriched_dir.iterdir() if p.is_file() and p.name.endswith(exts)]
    )


@st.cache_data(show_spinner=False)
def get_processed_doc_ids(enriched_dir: Path) -> set[str]:
    ids: set[str] = set()
    for p in list_enriched_files(enriched_dir):
        stem = normalise_doc_id(p.name.split(".")[0])
        if stem:
            ids.add(stem)
        try:
            doc = read_json(p)
            did = normalise_doc_id(doc.get("doc_id"))
            if did:
                ids.add(did)
        except Exception:
            continue
    return ids


def _parse_quote(q: Any) -> str | None:
    """Handle both string and dict evidence_quote formats."""
    if isinstance(q, str):
        return q.strip() or None
    if isinstance(q, dict):
        return (q.get("quote") or q.get("text") or "").strip() or None
    return None


# =========================
# Data loading
# =========================


@st.cache_data(show_spinner=True)
def load_dataset_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "feedback_reference" not in df.columns:
        raise ValueError("dataset.csv must contain a 'feedback_reference' column")
    df = df.rename(columns={"feedback_reference": "doc_id"})
    df["doc_id"] = df["doc_id"].map(normalise_doc_id)
    return df


@st.cache_data(show_spinner=True)
def load_items_long(enriched_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in list_enriched_files(enriched_dir):
        doc = read_json(path)
        doc_id = normalise_doc_id(doc.get("doc_id") or path.name.split(".")[0])
        paras = (doc.get("extractions") or {}).get("paragraphs") or []
        for p in paras:
            para_id = p.get("para_id")
            text = p.get("text") or ""
            for item in p.get("items") or []:
                topic = item.get("topic")
                topic_source = item.get("topic_source")
                quotes = item.get("evidence_quotes") or []
                if not quotes:
                    rows.append(
                        dict(
                            doc_id=doc_id,
                            para_id=para_id,
                            topic=topic,
                            topic_source=topic_source,
                            evidence_quote=None,
                            text=text,
                            _primary=True,
                        )
                    )
                    continue
                for i, q in enumerate(quotes):
                    rows.append(
                        dict(
                            doc_id=doc_id,
                            para_id=para_id,
                            topic=topic,
                            topic_source=topic_source,
                            evidence_quote=_parse_quote(q),
                            text=text,
                            _primary=(i == 0),  # True only for the first quote per item
                        )
                    )
    df = pd.DataFrame.from_records(rows)
    if df.empty:
        return df
    df["doc_id"] = df["doc_id"].map(normalise_doc_id)
    df["para_id"] = df["para_id"].fillna("").astype(str)
    df["topic"] = df["topic"].astype(str)
    df["topic_source"] = df["topic_source"].fillna("").astype(str)
    return df


def make_submission_label(row: pd.Series) -> str:
    doc_id = row.get("doc_id", "")
    org = row.get("organization", "")
    lang = row.get("feedback_language", "")
    country = row.get("country", "")
    date = row.get("date_feedback", row.get("feedback_date", ""))
    parts = [
        str(p).strip()
        for p in [doc_id, org, country, lang, date]
        if str(p).strip() and str(p).strip() != "nan"
    ]
    return " · ".join(parts)


def compute_topic_summary(items: pd.DataFrame) -> pd.DataFrame:
    if items.empty:
        return pd.DataFrame()
    # Filter to one row per item: load_items_long marks the first quote of each

    primary_col = "_primary" if "_primary" in items.columns else None
    if primary_col:
        base = items[items["_primary"]].dropna(subset=["topic"]).copy()
    else:
        base = items.dropna(subset=["topic"]).copy()
    return (
        base.groupby(["topic", "topic_source"], dropna=False)
        .agg(mentions=("topic", "size"), submissions=("doc_id", "nunique"))
        .reset_index()
        .sort_values(["submissions", "mentions"], ascending=False)
    )


def build_doc_paragraph_view(items: pd.DataFrame, doc_id: str) -> pd.DataFrame:
    df = items[items["doc_id"] == doc_id].copy()
    if df.empty:
        return df

    def agg_quotes(s: pd.Series) -> list[str]:
        return sorted({str(x).strip() for x in s.dropna() if str(x).strip()})

    return (
        df.groupby(["para_id", "topic", "topic_source"], dropna=False)
        .agg(text=("text", "first"), evidence_quotes=("evidence_quote", agg_quotes))
        .reset_index()
        .sort_values(["para_id", "topic_source", "topic"])
    )


# =========================
# UI components
# =========================


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _badge(text: str, cls: str = "") -> str:
    return f'<span class="badge {cls}">{html_mod.escape(str(text))}</span>'


def _chip(text: str, source: str) -> str:
    cls = "chip-new" if source == "new" else "chip-dfa"
    return f'<span class="chip {cls}">{html_mod.escape(str(text))}</span>'


def render_quote_card(
    quote: str,
    user_type: str = "",
    country: str = "",
    org: str = "",
    topic: str = "",
    topic_source: str = "",
) -> None:
    src_cls = "src-new" if topic_source == "new" else ""
    q = html_mod.escape(quote)
    badges: list[str] = []
    if user_type:
        badges.append(_badge(user_type, "badge-type"))
    if country:
        badges.append(_badge(country))
    if org and org != "nan":
        badges.append(_badge(org))
    if topic:
        t_cls = "badge-topic-new" if topic_source == "new" else "badge-topic"
        badges.append(_badge(topic, t_cls))
    attr = f'<div class="qattr">{"".join(badges)}</div>' if badges else ""
    st.markdown(
        f'<div class="qcard {src_cls}"><p class="qtext">&ldquo;{q}&rdquo;</p>{attr}</div>',
        unsafe_allow_html=True,
    )


def render_quotes_feed(
    view: pd.DataFrame,
    max_quotes: int = 50,
    show_topic: bool = False,
) -> None:
    quotes = view.dropna(subset=["evidence_quote"]).copy()
    quotes = quotes[quotes["evidence_quote"].str.strip() != ""]
    if quotes.empty:
        st.info("No evidence quotes available for this selection.")
        return
    total = len(quotes)
    if total > max_quotes:
        st.caption(f"Showing {max_quotes:,} of {total:,} quotes.")
        quotes = quotes.head(max_quotes)
    for _, row in quotes.iterrows():
        org = row.get("organization", "")
        render_quote_card(
            quote=row["evidence_quote"],
            user_type=str(row.get("user_type", "") or ""),
            country=str(row.get("country", "") or ""),
            org=str(org) if pd.notna(org) else "",
            topic=str(row.get("topic", "") or "") if show_topic else "",
            topic_source=str(row.get("topic_source", "") or ""),
        )


def render_topic_chips(topics_df: pd.DataFrame) -> None:
    if topics_df.empty:
        return
    chips = [
        _chip(r["topic"], r.get("topic_source", "dfa")) for _, r in topics_df.iterrows()
    ]
    st.markdown(f'<div class="chips">{"".join(chips)}</div>', unsafe_allow_html=True)


# =========================
# Sidebar / filters
# =========================


def render_provenance(enriched_dir: Path) -> None:
    prov = load_provenance(enriched_dir)
    if not prov:
        return
    llm = prov.get("llm_config") or {}
    with st.sidebar.expander("Pipeline provenance", expanded=False):
        rows = [
            ("Model", llm.get("model", "—")),
            ("Schema", prov.get("schema", "—")),
            ("Granularity", prov.get("granularity", "—")),
            ("Temperature", llm.get("temperature", "—")),
            ("Max tokens", llm.get("max_completion_tokens", "—")),
            ("Started", (prov.get("started_at") or "—")[:10]),
            ("Finished", (prov.get("finished_at") or "—")[:10]),
        ]
        for k, v in rows:
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:0.82em;padding:2px 0;border-bottom:1px solid #f1f5f9;">'
                f'<span style="color:#64748b;">{k}</span>'
                f'<span style="color:#0f172a;font-weight:500;">{html_mod.escape(str(v))}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )


def sidebar_filters(
    docs: pd.DataFrame,
    datasets: dict[str, Path],
    default_dir: Path,
) -> tuple[dict[str, Any], Path]:
    selected_dir = default_dir

    logo_path = Path(__file__).parent.parent / "data" / "human-ads_logo.png"
    if logo_path.exists():
        st.sidebar.image(str(logo_path), use_container_width=True)
        st.sidebar.divider()

    st.sidebar.header("Dataset")
    if len(datasets) > 1:
        default_label = next(
            (k for k, v in datasets.items() if v == default_dir), list(datasets)[0]
        )
        chosen = st.sidebar.selectbox(
            "Version",
            options=list(datasets.keys()),
            index=list(datasets.keys()).index(default_label),
        )
        selected_dir = datasets[chosen]
    else:
        label = next(iter(datasets), selected_dir.name)
        st.sidebar.caption(label)

    render_provenance(selected_dir)
    st.sidebar.divider()

    st.sidebar.header("Filters")

    def multi(col: str, label: str) -> list[str]:
        vals = sorted({str(x).strip() for x in docs[col].dropna() if str(x).strip()})
        return st.sidebar.multiselect(label, options=vals, default=[])

    f: dict[str, Any] = {
        "user_type": multi("user_type", "Stakeholder type"),
        "organization": st.sidebar.text_input(
            "Organisation contains", value=""
        ).strip(),
        "scope": multi("scope", "Scope"),
        "governance_level": multi("governance_level", "Governance level"),
        "company_size": multi("company_size", "Company size"),
        "country": multi("country", "Country"),
        "feedback_language": multi("feedback_language", "Language"),
    }
    return f, selected_dir


def filter_docs(docs: pd.DataFrame, f: dict[str, Any]) -> pd.DataFrame:
    out = docs.copy()
    for col in [
        "user_type",
        "scope",
        "governance_level",
        "company_size",
        "country",
        "feedback_language",
    ]:
        sel = f.get(col) or []
        if sel:
            out = out[out[col].isin(sel)]
    org_q = f.get("organization", "")
    if org_q:
        out = out[
            out["organization"].fillna("").str.contains(org_q, case=False, regex=False)
        ]
    return out


def apply_filters(
    items: pd.DataFrame, docs: pd.DataFrame, f: dict[str, Any]
) -> pd.DataFrame:
    """Left-join items with docs to add metadata, then apply active filters.

    Topics come from JSONs; all items are kept when no filters are active.
    When a filter is active, items from non-matching docs (NaN metadata) are
    excluded because they can't be confirmed to match the filter criteria.
    """
    if items.empty:
        return items.copy()
    with_meta = items.merge(docs, on="doc_id", how="left", suffixes=("", "_doc"))
    for col in [
        "user_type",
        "scope",
        "governance_level",
        "company_size",
        "country",
        "feedback_language",
    ]:
        sel = f.get(col) or []
        if sel:
            with_meta = with_meta[with_meta[col].isin(sel)]
    org_q = f.get("organization", "")
    if org_q:
        with_meta = with_meta[
            with_meta["organization"]
            .fillna("")
            .str.contains(org_q, case=False, regex=False)
        ]
    return with_meta


# =========================
# Pages
# =========================


def _pick_anchor_quote(items: pd.DataFrame, topic: str) -> str:
    """Pick a representative quote for a topic: aim for 80–220 chars."""
    pool = items[items["topic"] == topic]["evidence_quote"].dropna()
    pool = pool[pool.str.strip() != ""]
    sized = pool[pool.str.len().between(80, 220)]
    src = sized if not sized.empty else pool
    return src.iloc[0] if not src.empty else ""


def page_overview(items_filtered: pd.DataFrame) -> None:
    if items_filtered.empty:
        st.info("No extracted items match the current filters.")
        return

    summary = compute_topic_summary(items_filtered)
    n_topics = summary["topic"].nunique()
    n_subs = items_filtered["doc_id"].nunique()
    n_countries = (
        items_filtered["country"].dropna().nunique()
        if "country" in items_filtered.columns
        else 0
    )
    n_new = (summary["topic_source"] == "new").sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Submissions", f"{n_subs:,}")
    c2.metric("Topics", f"{n_topics:,}")
    c3.metric("Countries", f"{n_countries:,}")
    c4.metric("Emergent topics", f"{n_new:,}")

    st.divider()

    records = summary.to_dict("records")
    n_cols = 3 if len(records) >= 3 else len(records)

    # Split into DFA vs new for section headings
    dfa_records = [r for r in records if r.get("topic_source") != "new"]
    new_records = [r for r in records if r.get("topic_source") == "new"]

    def _render_cards(recs: list[dict], source: str) -> None:
        for i in range(0, len(recs), n_cols):
            batch = recs[i : i + n_cols]
            cols = st.columns(n_cols)
            for col, row in zip(cols, batch):
                with col:
                    topic = row["topic"]
                    subs = row["submissions"]
                    ments = row["mentions"]
                    anchor = _pick_anchor_quote(items_filtered, topic)
                    q_block = (
                        f'<p style="color:#94a3b8;font-size:0.72em;font-weight:600;'
                        f'letter-spacing:0.06em;text-transform:uppercase;margin:0 0 4px 0;">'
                        f"Example quote</p>"
                        f'<p class="ccard-quote">&ldquo;{html_mod.escape(anchor[:250])}&rdquo;</p>'
                        if anchor
                        else ""
                    )
                    chip = _chip(source.upper(), source)
                    st.markdown(
                        f"""<div class="ccard">
  {chip}
  <h4>{html_mod.escape(topic)}</h4>
  <div class="ccard-stats">{subs:,} submissions &nbsp;·&nbsp; {ments:,} mentions</div>
  {q_block}
</div>""",
                        unsafe_allow_html=True,
                    )
                    st.write("")

    if dfa_records:
        st.markdown("### Main topics covered by the DFA")
        _render_cards(dfa_records, "dfa")

    if new_records:
        st.divider()
        st.markdown("### More specific topics identified by the model")
        st.caption(
            f"{len(new_records):,} granular topics extracted from submissions beyond the core DFA framework."
        )
        # Render top cards, then a ranked table for the long tail
        card_limit = 9
        _render_cards(new_records[:card_limit], "new")
        if len(new_records) > card_limit:
            with st.expander(f"All {len(new_records):,} emergent topics"):
                tail_df = pd.DataFrame(new_records).drop(columns=["topic_source"])
                st.dataframe(
                    tail_df.rename(
                        columns={
                            "topic": "Topic",
                            "submissions": "Submissions",
                            "mentions": "Mentions",
                        }
                    ),
                    width="stretch",
                    hide_index=True,
                )

    if not dfa_records and not new_records:
        st.dataframe(summary, width="stretch", hide_index=True)


def render_topic_evidence(view: pd.DataFrame, max_paragraphs: int = 30) -> None:
    """Group by source paragraph: show submitter info + truncated text + evidence quotes."""
    # Work with unique (doc_id, para_id) groups to avoid repeating the text per quote
    groups = list(view.groupby(["doc_id", "para_id"], sort=False))
    total = len(groups)
    shown_groups = groups[:max_paragraphs]

    if total > max_paragraphs:
        st.caption(f"Showing {max_paragraphs} of {total:,} source paragraphs.")

    for (doc_id, para_id), group in shown_groups:
        first = group.iloc[0]
        user_type = str(first.get("user_type", "") or "")
        country = str(first.get("country", "") or "")
        org = str(first.get("organization", "") or "")
        text = str(first.get("text", "") or "").strip()

        quotes = [q for q in group["evidence_quote"].dropna() if str(q).strip()]

        # Header badges
        badges_html = "".join(
            _badge(b, cls)
            for b, cls in [
                (doc_id, ""),
                (user_type, "badge-type"),
                (country, ""),
                (org, ""),
            ]
            if b and b != "nan"
        )
        header = (
            f'<div class="src-card-header">{badges_html}</div>' if badges_html else ""
        )

        # Paragraph text — skip only when text is nearly identical to the single quote
        text_block = ""
        if text:
            skip = len(quotes) == 1 and len(text) < len(quotes[0]) * 1.4
            if not skip:
                text_block = f'<div class="src-card-text">{html_mod.escape(text)}</div>'

        # Quote cards
        quote_cards = "".join(
            f'<div class="qcard"><p class="qtext">&ldquo;{html_mod.escape(q)}&rdquo;</p></div>'
            for q in quotes
        )
        quotes_block = (
            f'<div class="src-card-quotes">{quote_cards}</div>' if quote_cards else ""
        )

        if header or text_block or quotes_block:
            st.markdown(
                f'<div class="src-card">{header}{text_block}{quotes_block}</div>',
                unsafe_allow_html=True,
            )


def page_topic(items_filtered: pd.DataFrame) -> None:
    if items_filtered.empty:
        st.info("No extracted items match the current filters.")
        return

    topics = (
        items_filtered[["topic", "topic_source"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["topic_source", "topic"])
        .reset_index(drop=True)
    )

    choice = st.selectbox(
        "Topic",
        options=list(range(len(topics))),
        format_func=lambda i: (
            f"{topics.iloc[i]['topic']}  ({topics.iloc[i]['topic_source']})"
        ),
    )
    topic = topics.iloc[int(choice)]["topic"]
    topic_source = topics.iloc[int(choice)]["topic_source"]

    view = items_filtered[
        (items_filtered["topic"] == topic)
        & (items_filtered["topic_source"] == topic_source)
    ]

    n_subs = view["doc_id"].nunique()
    n_ments = len(view.dropna(subset=["evidence_quote"]))
    st.caption(
        f"**{n_subs:,}** submissions &nbsp;·&nbsp; **{n_ments:,}** evidence quotes"
    )

    col_feed, col_break = st.columns([2, 1])

    with col_feed:
        render_topic_evidence(view)

    with col_break:
        if "user_type" in view.columns and view["user_type"].notna().any():
            counts = (
                view.groupby("user_type")["doc_id"]
                .nunique()
                .reset_index(name="submissions")
                .sort_values("submissions", ascending=False)
            )
            chart = (
                alt.Chart(counts)
                .mark_bar(color="#3b82f6")
                .encode(
                    x=alt.X("submissions:Q", title=""),
                    y=alt.Y("user_type:N", sort="-x", title=""),
                    tooltip=["user_type", "submissions"],
                )
                .properties(
                    height=max(80, len(counts) * 28), title="Stakeholder breakdown"
                )
            )
            st.altair_chart(chart, width="stretch")

        if "country" in view.columns and view["country"].notna().any():
            top_countries = (
                view.groupby("country")["doc_id"]
                .nunique()
                .reset_index(name="submissions")
                .sort_values("submissions", ascending=False)
                .head(10)
            )
            st.markdown("**Top countries**")
            for _, r in top_countries.iterrows():
                st.caption(f"{r['country']} — {r['submissions']:,}")


def page_search(items_filtered: pd.DataFrame) -> None:
    query = st.text_input(
        "Search quotes",
        placeholder="Search across all evidence quotes…",
        label_visibility="collapsed",
    )
    if not query.strip():
        st.caption("Enter a search term to find relevant quotes.")
        return

    mask = (
        items_filtered["evidence_quote"]
        .fillna("")
        .str.contains(query.strip(), case=False, regex=False)
    )
    results = items_filtered[mask].copy()

    st.caption(
        f"**{len(results):,}** matches across **{results['doc_id'].nunique():,}** submissions."
    )
    render_quotes_feed(results, max_quotes=60, show_topic=True)


def page_landscape(items_filtered: pd.DataFrame) -> None:
    st.caption(
        "Each cell shows how many unique submissions mention the topic for that group."
    )

    if items_filtered.empty:
        st.info("No extracted items match the current filters.")
        return

    dim = st.selectbox(
        "Compare topics by",
        options=[
            "user_type",
            "country",
            "governance_level",
            "company_size",
            "feedback_language",
            "scope",
        ],
        index=0,
    )
    top_n = st.slider(
        "Show top N topics", min_value=10, max_value=200, value=30, step=10
    )

    base = items_filtered.dropna(subset=["topic"]).copy()
    if dim not in base.columns:
        st.info("This dimension is not available in the current dataset.")
        return

    base[dim] = base[dim].fillna("").astype(str).str.strip()
    base = base[base[dim] != ""]

    if base.empty:
        st.info("No data for this dimension.")
        return

    pivot = (
        base.groupby(["topic", dim], dropna=False)
        .agg(submissions=("doc_id", "nunique"))
        .reset_index()
        .pivot(index="topic", columns=dim, values="submissions")
        .fillna(0)
        .astype(int)
    )
    pivot["__total__"] = pivot.sum(axis=1)
    pivot = (
        pivot.sort_values("__total__", ascending=False)
        .drop(columns=["__total__"])
        .head(top_n)
    )
    pivot = pivot.loc[:, pivot.sum(axis=0).sort_values(ascending=False).index]

    styled = pivot.style.background_gradient(axis=None)
    st.dataframe(styled, width="stretch")

    st.divider()
    st.markdown("### Evidence for cell")

    if pivot.empty:
        return

    topic_choice = st.selectbox(
        "Topic", options=pivot.index.tolist(), key="landscape_topic"
    )
    dim_choice = st.selectbox(dim, options=pivot.columns.tolist(), key="landscape_dim")

    view = items_filtered[
        (items_filtered["topic"] == topic_choice)
        & (
            items_filtered[dim].fillna("").astype(str).str.strip()
            == str(dim_choice).strip()
        )
    ].copy()

    st.caption(
        f"**{view['doc_id'].nunique():,}** submissions · **{len(view.dropna(subset=['evidence_quote'])):,}** quotes"
    )
    render_quotes_feed(view, max_quotes=30, show_topic=False)


def page_submission(docs_filtered: pd.DataFrame, items: pd.DataFrame) -> None:
    docs_for_select = (
        docs_filtered.dropna(subset=["doc_id"])
        .copy()
        .assign(doc_id=lambda df: df["doc_id"].astype(str).str.strip())
        .query("doc_id != ''")
        .drop_duplicates(subset=["doc_id"])
    )

    if docs_for_select.empty:
        st.info("No submissions match the current filters.")
        return

    labels = [make_submission_label(row) for _, row in docs_for_select.iterrows()]
    picked_idx = st.selectbox(
        "Submission",
        options=list(range(len(labels))),
        format_func=lambda i: labels[i],
    )
    selected = docs_for_select.iloc[int(picked_idx)]["doc_id"]

    meta = docs_filtered[docs_filtered["doc_id"] == selected].head(1)
    if not meta.empty:
        row = meta.iloc[0].to_dict()
        org = str(row.get("organization", "") or "").strip()
        country = str(row.get("country", "") or "").strip()
        user_type = str(row.get("user_type", "") or "").strip()
        lang = str(row.get("feedback_language", "") or "").strip()
        date = str(row.get("feedback_date", row.get("date_feedback", "")) or "").strip()
        feedback_text = str(row.get("feedback", "") or "").strip()

        badges_html = "".join(
            [
                _badge(b, cls)
                for b, cls in [
                    (user_type, "badge-type"),
                    (country, ""),
                    (lang, ""),
                    (date, ""),
                ]
                if b and b != "nan"
            ]
        )
        meta_div = f'<div class="sub-meta">{badges_html}</div>' if badges_html else ""
        fb_div = (
            f'<div class="sub-feedback">{html_mod.escape(feedback_text)}</div>'
            if feedback_text
            else ""
        )
        heading = org if org and org != "nan" else selected
        st.markdown(
            f'<div class="sub-header"><h3>{html_mod.escape(heading)}</h3>{meta_div}{fb_div}</div>',
            unsafe_allow_html=True,
        )

    # Topic chips across full submission
    sub_items = items[items["doc_id"] == selected]
    if not sub_items.empty:
        all_topics = sub_items[["topic", "topic_source"]].drop_duplicates()
        render_topic_chips(all_topics)
        st.write("")

    doc_view = build_doc_paragraph_view(items, selected)
    if doc_view.empty:
        st.info("No extracted items found for this submission.")
        return

    for para_id in doc_view["para_id"].dropna().unique():
        block = doc_view[doc_view["para_id"] == para_id]
        text = block["text"].iloc[0] if "text" in block.columns else ""
        n_t = block["topic"].nunique()

        with st.expander(
            f"**{para_id}** · {n_t} topic{'s' if n_t != 1 else ''}", expanded=False
        ):
            if text:
                st.markdown(
                    f'<div style="color:#475569;font-size:0.88em;line-height:1.7;margin-bottom:14px;">'
                    f"{html_mod.escape(text)}</div>",
                    unsafe_allow_html=True,
                )
                st.divider()

            for _, r in block.iterrows():
                topic = r.get("topic", "")
                src = str(r.get("topic_source", "") or "")
                st.markdown(_chip(topic, src), unsafe_allow_html=True)
                for q in r.get("evidence_quotes") or []:
                    render_quote_card(quote=q, topic_source=src)
                st.write("")


# =========================
# Main
# =========================


def main() -> None:
    st.set_page_config(page_title="DFA Consultation Dashboard", layout="wide")
    inject_css()

    paths = get_paths()

    if not paths.dataset_csv.exists():
        st.error(f"Dataset CSV not found: {paths.dataset_csv}")
        st.stop()

    docs = load_dataset_csv(paths.dataset_csv)

    if not paths.outputs_dir.exists():
        st.error(f"Outputs directory not found: {paths.outputs_dir}")
        st.stop()

    datasets = discover_datasets(paths.outputs_dir)
    if not datasets:
        st.error(f"No labelled dataset folders found in: {paths.outputs_dir}")
        st.stop()

    default_dir = next(iter(datasets.values()))
    f, active_dir = sidebar_filters(docs, datasets, default_dir)

    if not active_dir.exists():
        st.error(f"Enriched extractions dir not found: {active_dir}")
        st.stop()

    items = load_items_long(active_dir)
    processed_ids = get_processed_doc_ids(active_dir)

    docs_active = docs[docs["doc_id"].isin(processed_ids)].copy()
    if docs_active.empty:
        st.warning("No dataset rows match the processed extraction files.")
        st.stop()

    items_active = items[items["doc_id"].isin(processed_ids)].copy()
    if items_active.empty:
        st.warning("No items loaded. Check the enriched extraction files.")
        st.stop()

    docs_filtered = filter_docs(docs_active, f)
    items_filtered = apply_filters(items, docs_active, f)

    active_label = next(
        (k for k, v in datasets.items() if v == active_dir), active_dir.name
    )

    n_item_mentions = (
        items_filtered[items_filtered["_primary"]].shape[0]
        if not items_filtered.empty and "_primary" in items_filtered.columns
        else (items_filtered.shape[0] if not items_filtered.empty else 0)
    )
    st.title("DFA Consultation Dashboard")
    st.caption(
        f"Dataset: **{active_label}** &nbsp;·&nbsp; "
        f"{len(docs_filtered):,} submissions &nbsp;·&nbsp; "
        f"{items_filtered['doc_id'].nunique() if not items_filtered.empty else 0:,} with extractions &nbsp;·&nbsp; "
        f"{n_item_mentions:,} topic mentions"
    )

    tab_overview, tab_topic, tab_search, tab_landscape, tab_submission = st.tabs(
        ["Overview", "Topics", "Search", "Landscape", "Submissions"]
    )

    with tab_overview:
        page_overview(items_filtered)

    with tab_topic:
        page_topic(items_filtered)

    with tab_search:
        page_search(items_filtered)

    with tab_landscape:
        page_landscape(items_filtered)

    with tab_submission:
        page_submission(docs_filtered=docs_filtered, items=items_active)


if __name__ == "__main__":
    main()
