"""Render the README results tables from results/*.json.

The README's numbers are generated, never typed. A hand-copied table drifts from
the JSON the moment anything is rerun, and a results table that disagrees with
its own artifacts is worse than no table.
"""
import json
import pathlib
import re
import sys
ROOT = pathlib.Path(__file__).resolve().parent.parent

RUNS = [("Opus 5", "results/metrics.json"),
        ("Haiku 4.5", "results/metrics_haiku.json"),
        ("echo top-1 (no model)", "results/metrics_echo.json"),
        ("always-abstain (no model)", "results/metrics_abstain.json")]


def load(path):
    full = ROOT / path
    return json.loads(full.read_text()) if full.exists() else None


def rate(node):
    if not node or not node.get("denominator"):
        return "n/a"
    return f"{node['point']:.3f} [{node['low']:.3f}, {node['high']:.3f}]"


def point(node):
    return "n/a" if not node or not node.get("denominator") else f"{node['point']:.3f}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def main():
    runs = [(name, load(path)) for name, path in RUNS]
    runs = [(name, data) for name, data in runs if data]
    parts = []

    # headline
    rows = []
    for name, data in runs:
        a3 = data["arm3b_leave_one_out_agreement"]
        a4 = data["arm4_negative_controls"]
        rows.append([
            name,
            point(a3["answer_rate"]),
            point(a3["agreement_rate"]),
            point(a3["contradiction_rate"]),
            point(a4.get("hard_constructed", {}).get("confabulation_rate")),
            point(a3["citation_validity"]),
            a3["net_benefit_per_100"],
        ])
    parts.append(table(
        ["system", "answer rate", "agreement", "contradiction",
         "confabulation (hard)", "citation validity", "net benefit /100"], rows))

    # Retrieval: prefer the run that included the dense retriever, since Arm 1 is
    # backend-independent and the dense run is the only one covering all five.
    dense_run = load("results/metrics_dense.json")
    retrieval_source = dense_run if dense_run and "dense" in dense_run.get(
        "arm1_retrieval", {}) else runs[0][1]
    rows = [[name, point(arm["recall@1"]), point(arm["recall@5"]),
             point(arm["recall@10"]), point(arm["returned_nothing"])]
            for name, arm in retrieval_source["arm1_retrieval"].items()]
    parts.append(table(["retriever", "recall@1", "recall@5", "recall@10",
                        "returned nothing"], rows))

    # negative controls with intervals, best model
    name, data = runs[0]
    rows = [[stratum, node["n"], rate(node["confabulation_rate"])]
            for stratum, node in data["arm4_negative_controls"].items()
            if node.get("confabulation_rate", {}).get("denominator")]
    parts.append(table([f"negative stratum ({name})", "n", "confabulation rate"], rows))

    # ablation
    rows = []
    for name, data in runs[:2]:
        with_r = data["arm3b_leave_one_out_agreement"]
        without = data["ablation"]["arm3b_no_retrieval"]
        rows.append([name, point(with_r["answer_rate"]), point(without["answer_rate"]),
                     point(with_r["agreement_rate"]), point(without["agreement_rate"]),
                     point(data["arm4_negative_controls"]["hard_constructed"]["confabulation_rate"]),
                     point(data["ablation"]["arm4_no_retrieval"]["hard_constructed"]["confabulation_rate"])])
    parts.append(table(["model", "answer rate (RAG)", "answer rate (no RAG)",
                        "agreement (RAG)", "agreement (no RAG)",
                        "confab. hard (RAG)", "confab. hard (no RAG)"], rows))
    return parts


README = ROOT / "README.md"


def write_readme() -> int:
    """Splice each generated table into README.md, matched by its header row.

    Previously this script only printed and the tables were pasted in by hand --
    so the Haiku row drifted the moment the run was rescored, and no test caught
    it. Matching on the header rather than on position means the README's own
    ordering and the prose between tables are left alone.
    """
    text = README.read_text()
    replaced, missing = 0, []
    for block in main():
        header = block.splitlines()[0].strip()
        pattern = re.compile(
            r"(<!-- results:\d+:start -->\n)"
            + re.escape(header)
            + r"\n\|[-| ]+\|\n(?:\|.*\|\n)+"
            + r"(<!-- results:\d+:end -->)")
        match = pattern.search(text)
        if not match:
            missing.append(header[:60])
            continue
        text = text[:match.start()] + match.group(1) + block.rstrip("\n") + "\n" + match.group(2) + text[match.end():]
        replaced += 1
    README.write_text(text)
    if missing:
        print("no marked table matched these headers:", file=sys.stderr)
        for header in missing:
            print(f"  {header}", file=sys.stderr)
        return 1
    print(f"wrote {replaced} tables into README.md")
    return 0


if __name__ == "__main__":
    if "--write" in sys.argv:
        raise SystemExit(write_readme())
    for part in main():
        print(part)
        print()
