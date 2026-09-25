"""Agreement between primary labels and the 20% second review.

Reports raw agreement and Cohen's kappa on `relevant`, lists disagreements
to resolve, and writes labels/reference_labels.csv once every disagreement
has a `resolved` value in labels/disagreements.csv.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "labels"


def load(name):
    return {r["pair_id"]: r for r in csv.DictReader(open(ROOT / name, encoding="utf-8"))}


def kappa(pairs):
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    pa1 = sum(a == "1" for a, _ in pairs) / n
    pb1 = sum(b == "1" for _, b in pairs) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return po, (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    prim, sec = load("primary_labels.csv"), load("second_review.csv")
    missing = [k for k, r in prim.items() if r["relevant"] not in ("0", "1")]
    if missing:
        print(f"WARNING: {len(missing)} primary pairs unlabeled (e.g. {missing[:3]})")
    pairs, dis = [], []
    for k, s in sec.items():
        if s["relevant"] in ("0", "1") and prim[k]["relevant"] in ("0", "1"):
            pairs.append((prim[k]["relevant"], s["relevant"]))
            if prim[k]["relevant"] != s["relevant"]:
                dis.append({"pair_id": k, "primary": prim[k]["relevant"], "primary_reason": prim[k]["reason_code"],
                            "second": s["relevant"], "second_reason": s["reason_code"], "resolved": "", "resolution_note": ""})
    if not pairs:
        raise SystemExit("no overlapping labeled pairs yet")
    po, k = kappa(pairs)
    print(f"overlap pairs: {len(pairs)}  agreement: {po:.1%}  Cohen's kappa: {k:.2f}  disagreements: {len(dis)}")
    dpath = ROOT / "disagreements.csv"
    if dis and not dpath.exists():
        with open(dpath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dis[0].keys()))
            w.writeheader()
            w.writerows(dis)
        print(f"wrote {dpath.name}; fill `resolved` (0/1) then rerun")
        return
    resolved = {r["pair_id"]: r["resolved"] for r in csv.DictReader(open(dpath, encoding="utf-8"))} if dpath.exists() else {}
    if any(v not in ("0", "1") for v in resolved.values()):
        raise SystemExit("some disagreements still unresolved")
    with open(ROOT / "reference_labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "resume_id", "job_id", "relevant", "source"])
        for kid, r in prim.items():
            rel, src = (resolved[kid], "resolved") if kid in resolved else (r["relevant"], "primary")
            w.writerow([kid, r["resume_id"], r["job_id"], rel, src])
    print("wrote reference_labels.csv")


if __name__ == "__main__":
    main()
