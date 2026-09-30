"""Rules vs zero-shot Laya vs hybrid, on labelled messages.  Run: python -m eval.compare_classifiers

Two decisions are compared:
  area  - which life area a message is about (drives graph retrieval)
  keep  - is the message worth storing as long-term memory (drives the extraction gate)
Half of the area examples deliberately contain no keyword the rules know (paraphrases).
"""
import json
import logging
import statistics
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from laya import Router  # noqa: E402

from app.memory.classifier import RulesClassifier  # noqa: E402
from app.memory.laya_classifier import _AREA_CRITERIA, LayaClassifier  # noqa: E402
from app.models import LIFE_AREAS  # noqa: E402

# (message, gold_area)   -- "general" = none of the specific areas
AREA = [
    # keyword-friendly
    ("What should I focus on for my career?", "career"), ("Will I get a promotion this year?", "career"),
    ("Is this a good time to start a business?", "career"), ("Should I quit my job for a startup?", "career"),
    ("Will I find love and marriage soon?", "relationships"), ("Is my partner the right one for me?", "relationships"),
    ("How is my health going to be this year?", "health"), ("I've been stressed and sleeping badly, any guidance?", "health"),
    ("Will my finances improve soon?", "finance"), ("Is it a good time to invest my savings?", "finance"),
    ("Which course should I pick at university?", "education"), ("Will I clear my exams this year?", "education"),
    ("How will things go with my parents?", "family"), ("Will my sister's wedding go smoothly?", "family"),
    ("Which gemstone or mantra should I follow?", "spirituality"), ("What does my karma say about this phase?", "spirituality"),
    ("How can I build better habits and more confidence?", "personal_growth"), ("What is my purpose in life?", "personal_growth"),
    ("Tell me about the stars today", "general"), ("What is my lucky colour?", "general"),
    # paraphrases, no obvious keywords
    ("Should I accept the offer from the new firm?", "career"), ("My manager keeps ignoring my ideas, what now?", "career"),
    ("I keep attracting the wrong people romantically", "relationships"), ("We argue constantly and I'm not sure we'll last", "relationships"),
    ("My back pain won't go away", "health"), ("I feel drained and low on energy lately", "health"),
    ("Can I afford to buy a flat soon?", "finance"), ("Will my rent and bills ever feel easy?", "finance"),
    ("Should I apply for a master's abroad?", "education"), ("I'm struggling to focus on my syllabus", "education"),
    ("My mother is unwell and I'm worried about her", "family"), ("Planning to have a baby next year, is the timing good?", "family"),
    ("I want to feel more at peace inside", "spirituality"), ("Should I do a puja before the big day?", "spirituality"),
    ("I want to stop procrastinating", "personal_growth"), ("How do I become a more disciplined person?", "personal_growth"),
    ("What does the moon mean for me?", "general"), ("Good morning, what's in store?", "general"),
]

# (message, worth_storing)
KEEP = [
    ("I'm planning to switch jobs next year.", True), ("My name is Rahul and I was born in Delhi.", True),
    ("I want to start my own business in 2027.", True), ("Actually I was born in Mumbai, not Delhi.", True),
    ("I'm preparing for a product management interview next month.", True), ("I prefer getting answers in Hindi.", True),
    ("I'm interested in entrepreneurship and astrology.", True), ("My mother was diagnosed with diabetes last month.", True),
    ("I got engaged last week!", True), ("I decided not to switch jobs after all.", True),
    ("I'm saving up to buy a house in two years.", True), ("I was born on 3 March 1992 at 6:30 am.", True),
    ("I recently moved to Bangalore for work.", True), ("Please always reply briefly.", True),
    ("I'm a software engineer with five years of experience.", True), ("We are expecting our first child in December.", True),
    ("thanks", False), ("ok cool", False), ("Why do you say that?", False), ("What should I focus on for my career?", False),
    ("Can you explain that again?", False), ("hello", False), ("Will I find love soon?", False),
    ("What does Leo mean?", False), ("great, that helps", False), ("hmm interesting", False),
    ("Tell me more about that.", False), ("Is today auspicious?", False), ("What is my sun sign?", False),
    ("lol", False), ("How does Mercury retrograde work?", False), ("sounds good", False),
]


def main():
    rules = RulesClassifier()
    router = Router()
    hybrid = LayaClassifier(fallback=rules, router=router)
    router.predict("warm up", {"k": {"type": "noul", "instructions": "x"}})  # load weights before timing

    def laya_area(m):
        r = router.predict(m, {"area": {"type": "choice", "instructions": "Which life area is this message about?",
                                        "criteria": {a: _AREA_CRITERIA[a] for a in LIFE_AREAS}}})
        return r["answers"]["area"]["choice"]

    def laya_keep_p(m):
        r = router.predict(m, {"keep": {"type": "noul", "instructions":
                                        "Does the user state a durable personal fact, goal, plan, preference or "
                                        "correction worth remembering long-term?"}})
        return float(r["answers"]["keep"]["noul"])

    def timed(fn, items):
        out, ts = [], []
        for x in items:
            t = time.perf_counter(); out.append(fn(x)); ts.append((time.perf_counter() - t) * 1000)
        return out, statistics.median(ts)

    msgs = [m for m, _ in AREA]
    gold = [g for _, g in AREA]
    paraphrase = [i >= 20 and i < 38 for i in range(len(AREA))]

    def rules_area(m):
        a = rules.life_areas(m); return a[0] if a else "general"

    def hybrid_area(m):
        a = hybrid.life_areas(m); return a[0] if a else "general"

    def rules_hit(m, g):  # rules may return several areas: correct if the gold one is among them
        a = rules.life_areas(m); return (g in a) if g != "general" else not a

    res = {"area": {}, "keep": {}}
    rp, rt = timed(lambda m: rules.life_areas(m), msgs)
    lp, lt = timed(laya_area, msgs)
    hp, ht = timed(hybrid_area, msgs)
    rows = {
        "rules": ([rules_hit(m, g) for m, g in AREA], rt),
        "laya (zero-shot)": ([p == g for p, g in zip(lp, gold)], lt),
        "hybrid": ([p == g for p, g in zip(hp, gold)], ht),
    }
    print(f"\n== Life-area routing ({len(AREA)} msgs, {sum(paraphrase)} keyword-free paraphrases) ==")
    print(f"{'method':<20}{'overall':>9}{'keyword':>9}{'paraphrase':>12}{'median ms':>11}")
    for name, (hits, ms) in rows.items():
        kw = [h for h, p in zip(hits, paraphrase) if not p]; pa = [h for h, p in zip(hits, paraphrase) if p]
        res["area"][name] = {"overall": sum(hits) / len(hits), "keyword": sum(kw) / len(kw),
                             "paraphrase": sum(pa) / len(pa), "median_ms": round(ms, 2)}
        print(f"{name:<20}{sum(hits)/len(hits):>9.2f}{sum(kw)/len(kw):>9.2f}{sum(pa)/len(pa):>12.2f}{ms:>11.1f}")
    print("laya misses:", [(m[:38], g, p) for (m, g), p in zip(AREA, lp) if p != g][:8])

    km = [m for m, _ in KEEP]; kg = [g for _, g in KEEP]
    rk, rkt = timed(rules.worth_remembering, km)
    lk, lkt = timed(laya_keep_p, km)
    hk, hkt = timed(hybrid.worth_remembering, km)

    def metrics(pred):
        tp = sum(p and g for p, g in zip(pred, kg)); fp = sum(p and not g for p, g in zip(pred, kg))
        fn = sum((not p) and g for p, g in zip(pred, kg))
        prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
        return {"accuracy": sum(p == g for p, g in zip(pred, kg)) / len(kg), "precision": prec, "recall": rec}

    def auc(scores):  # probability a random positive outscores a random negative
        pos = [s for s, g in zip(scores, kg) if g]; neg = [s for s, g in zip(scores, kg) if not g]
        return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))

    krows = {"rules": (metrics(rk), rkt), "laya (zero-shot, p>=0.5)": (metrics([p >= 0.5 for p in lk]), lkt),
             "hybrid": (metrics(hk), hkt)}
    print(f"\n== Worth-remembering gate ({len(KEEP)} msgs: {sum(kg)} keep / {len(kg)-sum(kg)} skip) ==")
    print(f"{'method':<26}{'acc':>6}{'prec':>7}{'recall':>8}{'median ms':>11}")
    for name, (m, ms) in krows.items():
        res["keep"][name] = {**{k: round(v, 3) for k, v in m.items()}, "median_ms": round(ms, 2)}
        print(f"{name:<26}{m['accuracy']:>6.2f}{m['precision']:>7.2f}{m['recall']:>8.2f}{ms:>11.1f}")
    a = auc(lk); res["keep"]["laya_auc"] = round(a, 3)
    print(f"laya ranking quality (AUC, 0.5 = chance): {a:.2f}   mean p(keep) on keepers {statistics.mean(p for p,g in zip(lk,kg) if g):.2f}"
          f" vs skips {statistics.mean(p for p,g in zip(lk,kg) if not g):.2f}")
    Path("eval/classifier_comparison.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
