"""Optional zero-shot Laya backend (pip install -r requirements-laya.txt; CLASSIFIER=laya).

Life-area routing is a hybrid: rules decide keyword-clear cases, Laya only the keyword-free ones
(measured 0.63 -> 0.82 accuracy, see README), and any Laya error falls back to rules. The
worth-remembering gate stays rule-based: zero-shot Laya measured poorly there.
"""
import logging

from app.memory.classifier import Classifier, RulesClassifier
from app.models import LIFE_AREAS

log = logging.getLogger(__name__)

_AREA_CRITERIA = {
    "career": "work, jobs, career changes, interviews, business, promotions",
    "relationships": "love, dating, marriage, partners",
    "health": "physical or mental health, fitness, illness",
    "finance": "money, investments, income, debt",
    "education": "studies, exams, degrees, courses",
    "family": "parents, siblings, children, relatives",
    "spirituality": "spiritual practice, remedies, karma, rituals",
    "personal_growth": "habits, confidence, purpose, self-improvement",
    "general": "none of the above",
}


class LayaClassifier(Classifier):
    def __init__(self, fallback: RulesClassifier | None = None, router=None, threshold: float = 0.5):
        self.rules = fallback or RulesClassifier()
        self.threshold = threshold
        if router is None:
            from laya import Router  # ImportError -> factory falls back to rules
            router = Router()
        self.router = router

    def life_areas(self, message: str) -> list[str]:
        areas = self.rules.life_areas(message)
        if areas:
            return areas  # unambiguous: skip the model
        try:
            res = self.router.predict(message, {"area": {
                "type": "choice", "instructions": "Which life area is this message about?",
                "criteria": {a: _AREA_CRITERIA[a] for a in LIFE_AREAS}}})
            choice = res["answers"]["area"]["choice"]
            return [choice] if choice in LIFE_AREAS and choice != "general" else []
        except Exception as e:
            log.warning("Laya life_areas failed (%s); rules result used", e)
            return []

    def worth_remembering(self, message: str) -> bool:
        # Measured (eval/compare_classifiers.py): zero-shot Laya is a poor keep/skip gate (recall 0.19,
        # AUC 0.59 vs rules 0.97 accuracy), so the gate stays rule-based until Laya is fine-tuned.
        return self.rules.worth_remembering(message)
