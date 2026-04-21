from scipy import stats 
from typing import Optional
from App.config import MIN_DAYS, THRESHOLD


TEMPLATES = {
    ("journaled", "tasks_done", "positive"):             "You complete more tasks on days you journal.",
    ("journaled", "tasks_done", "negative"):             "You tend to do fewer tasks on days you skip journaling.",
    ("journaled", "mood", "positive"):                   "Your mood tends to be higher on days you journal.",
    ("journaled", "mood", "negative"):                   "Days without journaling often show lower mood scores.",
    ("journaled", "focus_mins", "positive"):             "You focus longer on days you journal.",
    ("journaled", "focus_mins", "negative"):             "Skipping your journal tends to shorten your focus sessions.",
    ("mood", "tasks_done", "positive"):                  "You get more done on days your mood is high.",
    ("mood", "tasks_done", "negative"):                  "Low mood days tend to reduce how much you complete.",
    ("mood", "focus_mins", "positive"):                  "You focus for longer when you start the day in a good mood.",
    ("mood", "focus_mins", "negative"):                  "Low mood days tend to shorten your focus sessions.",
    ("tasks_done", "focus_mins", "positive"):            "Days with more focus time also tend to have more tasks done.",
    ("tasks_done", "focus_mins", "negative"):            "More tasks done does not seem to mean more focus time for you.",
    ("sentiment_confidence", "mood", "positive"):        "Days when your journal tone is more certain tend to have higher mood.",
    ("sentiment_confidence", "mood", "negative"):        "Lower confidence in your journal tone often aligns with lower mood.",
    ("sentiment_confidence", "tasks_done", "positive"):  "A more confident journal tone tends to go with more tasks done.",
    ("sentiment_confidence", "tasks_done", "negative"):  "Lower journal confidence often lines up with fewer tasks completed.",
    ("sentiment_confidence", "focus_mins", "positive"):  "You seem to focus longer on days your journal tone is more certain.",
    ("sentiment_confidence", "focus_mins", "negative"):  "Lower confidence in your journal tone tends to shorten focus sessions.",
}


def _direction(r: float) -> str:
    return "positive" if r > 0 else "negative"


def _pearson(x: list, y: list) -> Optional[float]:
    if len(x) < MIN_DAYS:
        return None
    r, _ = stats.pearsonr(x, y)
    return r

def _point_biserial(binary: list, continuous: list) -> Optional[float]:
    if len(binary) < MIN_DAYS:
        return None
    r, _ = stats.pointbiserialr(binary, continuous)
    return r

def generate_insights(logs: list) -> dict:
    if len(logs) < MIN_DAYS:
        return {
            "insights": [],
            "days_analyzed": len(logs),
            "message": f"Need at least {MIN_DAYS} days of data. You sent {len(logs)}."
        }
 
    journaled  = [log.journaled   for log in logs]
    tasks_done = [log.tasks_done  for log in logs]
    mood       = [log.mood        for log in logs]
    focus_mins = [log.focus_mins  for log in logs]
 
    pairs = [
        ("journaled",  "tasks_done", _point_biserial(journaled, tasks_done)),
        ("journaled",  "mood",       _point_biserial(journaled, mood)),
        ("journaled",  "focus_mins", _point_biserial(journaled, focus_mins)),
        ("mood",       "tasks_done", _pearson(mood, tasks_done)),
        ("mood",       "focus_mins", _pearson(mood, focus_mins)),
        ("tasks_done", "focus_mins", _pearson(tasks_done, focus_mins)),
    ]
 
    sentiment_logs = [log for log in logs if log.sentiment is not None]
    if len(sentiment_logs) >= MIN_DAYS:
        s_conf  = [log.sentiment.confidence for log in sentiment_logs]
        s_mood  = [log.mood       for log in sentiment_logs]
        s_tasks = [log.tasks_done for log in sentiment_logs]
        s_focus = [log.focus_mins for log in sentiment_logs]
        pairs += [
            ("sentiment_confidence", "mood",       _pearson(s_conf, s_mood)),
            ("sentiment_confidence", "tasks_done", _pearson(s_conf, s_tasks)),
            ("sentiment_confidence", "focus_mins", _pearson(s_conf, s_focus)),
        ]
 
    strong = sorted(
        [(a, b, r) for a, b, r in pairs if r is not None and abs(r) >= THRESHOLD],
        key=lambda x: abs(x[2]),
        reverse=True
    )
 
    sentences = []
    for a, b, r in strong[:5]:
        sentence = TEMPLATES.get((a, b, _direction(r)))
        if sentence:
            sentences.append(sentence)
 
    if not sentences:
        return {
            "insights": [],
            "days_analyzed": len(logs),
            "message": "No strong patterns found yet. Keep logging more days."
        }
 
    return {
        "insights": sentences,
        "days_analyzed": len(logs),
        "message": None
    }

