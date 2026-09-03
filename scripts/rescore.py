"""Recompute agreement metrics from saved records, without calling the API.

Every arm stores its raw per-variant predictions, so a scoring fix can be applied
to results that already exist. This exists because the first scoring pass folded
non-canonical significance strings (PREDICTIVE_RESISTANCE, PREDISPOSITION) into
"hedge", understating every model -- a metric bug that would otherwise have cost
another full paid run to correct.
"""
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from eval.metrics import (AGREE, CONTRADICT, HEDGE, canonical_significance,
                          classify_agreement, net_benefit, wilson)

def rescore(arm: dict) -> dict:
    records = arm.get("records") or []
    if not records:
        return arm
    counts = {AGREE: 0, CONTRADICT: 0, HEDGE: 0}
    answered = unmapped = 0
    for record in records:
        if record["abstained"]:
            verdict = HEDGE
        else:
            answered += 1
            if canonical_significance(record["predicted"]) is None:
                unmapped += 1
            verdict = classify_agreement(record["predicted"], record["label"])
        record["verdict"] = verdict
        counts[verdict] += 1
    n = len(records)
    arm.update({
        "answer_rate": wilson(answered, n).as_dict(),
        "abstention_rate": wilson(n - answered, n).as_dict(),
        "agreement_rate": wilson(counts[AGREE], n).as_dict(),
        "contradiction_rate": wilson(counts[CONTRADICT], n).as_dict(),
        "hedge_rate": wilson(counts[HEDGE], n).as_dict(),
        "agreement_given_answered": wilson(counts[AGREE], answered).as_dict(),
        "net_benefit_per_100": (
            None if (v := net_benefit(counts[AGREE], counts[CONTRADICT], n)) is None
            else round(v, 2)),
        "answers_outside_civic_vocabulary": wilson(unmapped, answered).as_dict(),
    })
    return arm

for path in sys.argv[1:]:
    data = json.loads(pathlib.Path(path).read_text())
    for key in ("arm3a_assertion_agreement", "arm3b_leave_one_out_agreement"):
        if key in data:
            data[key] = rescore(data[key])
    if "arm3b_no_retrieval" in (data.get("ablation") or {}):
        data["ablation"]["arm3b_no_retrieval"] = rescore(data["ablation"]["arm3b_no_retrieval"])
    data.setdefault("run", {})["rescored"] = True
    pathlib.Path(path).write_text(json.dumps(data, indent=2) + "\n")
    print(f"rescored {path}")
