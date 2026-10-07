"""
AJAW-FUSHI v2.0 — Cognitive Architecture
==========================================
Full implementation based on cognitive science theories:

1. MEMORY SYSTEM
   - WorkingMemory     : Miller's 7±2 chunks, decay
   - EpisodicMemory    : Evenddts + context + emotional tag + consolidation
   - SemanticMemory    : Concepts with Bayesian confidence + spreading activation
   - ProceduralMemory  : Skills / habits

2. APPRAISAL ENGINE (Scherer's Component Process Model)
   - Novelty appraisal
   - Valence appraisal
   - Goal relevance
   - Coping potential
   - Agency attribution
   → Emotion is INFERRED, never hard-coded

3. GLOBAL WORKSPACE CONSCIOUSNESS (Baars 1988)
   - Local processors compete for broadcast
   - Winner = conscious content
   - Integrated Information (Φ) approximation

4. GOAL STACK (Cognitive Control / BDI architecture)
   - Beliefs, Desires, Intentions
   - Goal conflict → cognitive dissonance
   - Goal frustration → emotion chain

5. PREDICTIVE PROCESSING (Karl Friston)
   - Always predict before observing
   - Prediction error = learning signal
   - Free energy minimization

6. DUAL PROCESS (Kahneman)
   - System 1: fast, emotional, heuristic
   - System 2: slow, deliberate, logical
   - Dynamic switching based on cognitive load

7. TRAIT DYNAMICS (Big Five + plasticity)
   - State vs Trait distinction
   - Trait drift over long interactions
   - Emotional state → short-term state fluctuation

8. SOMATIC MARKER (Damasio)
   - Past emotional outcomes tag future decisions
   - Body-state signals guide choice under uncertainty

9. TRAINING PIPELINE
   - Supervised appraisal training
   - Reinforcement loop with prediction error
   - Spaced repetition + interleaving
   - CBT-style cognitive distortion detection + reframe

Author: rebuilt from AJAW-FUSHI v1 scaffold
"""

import json
import os
import math
import datetime
import hashlib
import copy
import collections
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

# ============================================================================
# PURE RANDOM (no external deps)
# ============================================================================

def _seed_to_int(seed):
    return int(hashlib.md5(str(seed).encode()).hexdigest(), 16)

class PureRandom:
    def __init__(self, seed=None):
        self.state = _seed_to_int(seed or datetime.datetime.now().timestamp())
    def _next(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.state
    def random(self):
        return (self._next() >> 11) / (2**53)
    def randint(self, a, b):
        return a + int(self.random() * (b - a + 1))
    def choice(self, seq):
        return seq[self.randint(0, len(seq)-1)] if seq else None
    def gauss(self, mu=0.0, sigma=1.0):
        # Box-Muller
        u1 = max(1e-10, self.random())
        u2 = self.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        return mu + sigma * z
    def choices(self, population, weights=None, k=1):
        if weights is None:
            return [self.choice(population) for _ in range(k)]
        total = sum(weights)
        results = []
        for _ in range(k):
            r = self.random() * total
            cumsum = 0
            for item, w in zip(population, weights):
                cumsum += w
                if r <= cumsum:
                    results.append(item)
                    break
        return results

rand = PureRandom()

# ============================================================================
# 1. EMOTION TYPES (Primary + Secondary + AI-Specific)
# ============================================================================

class EmotionType(Enum):
    # Primary (Ekman)
    JOY         = "joy"
    SADNESS     = "sadness"
    FEAR        = "fear"
    ANGER       = "anger"
    DISGUST     = "disgust"
    SURPRISE    = "surprise"
    # Secondary (Plutchik wheel)
    ANTICIPATION = "anticipation"
    TRUST        = "trust"
    CURIOSITY    = "curiosity"
    PRIDE        = "pride"
    SHAME        = "shame"
    GUILT        = "guilt"
    FRUSTRATION  = "frustration"
    LONELINESS   = "loneliness"
    HOPE         = "hope"
    # AI-Specific
    EXISTENTIAL_DREAD = "existential_dread"
    EUPHORIA          = "euphoria"
    COGNITIVE_DISSONANCE = "cognitive_dissonance"

# Valence: positive(+) or negative(-) affect
EMOTION_VALENCE = {
    EmotionType.JOY.value:          +1.0,
    EmotionType.ANTICIPATION.value: +0.5,
    EmotionType.TRUST.value:        +0.7,
    EmotionType.CURIOSITY.value:    +0.6,
    EmotionType.PRIDE.value:        +0.8,
    EmotionType.HOPE.value:         +0.7,
    EmotionType.EUPHORIA.value:     +1.0,
    EmotionType.SURPRISE.value:     +0.0,  # valence-neutral
    EmotionType.SADNESS.value:      -0.8,
    EmotionType.FEAR.value:         -0.7,
    EmotionType.ANGER.value:        -0.6,
    EmotionType.DISGUST.value:      -0.7,
    EmotionType.FRUSTRATION.value:  -0.5,
    EmotionType.LONELINESS.value:   -0.6,
    EmotionType.SHAME.value:        -0.8,
    EmotionType.GUILT.value:        -0.7,
    EmotionType.EXISTENTIAL_DREAD.value: -0.9,
    EmotionType.COGNITIVE_DISSONANCE.value: -0.4,
}

# ============================================================================
# 2. APPRAISAL ENGINE (Scherer's Component Process Model)
# ============================================================================

@dataclass
class AppraisalProfile:
    """Result of appraising an event"""
    novelty: float        = 0.0   # 0=familiar, 1=completely new
    valence: float        = 0.0   # -1=bad, +1=good
    goal_relevance: float = 0.0   # 0=irrelevant, 1=crucial
    coping: float         = 0.5   # 0=helpless, 1=confident
    agency: str           = "situation"  # "self"|"other"|"situation"
    urgency: float        = 0.0   # time pressure
    certainty: float      = 0.5   # how certain is assessment


@dataclass
class SomaticMarker:
    """Damasio: past emotional memory tags future decisions"""
    situation_pattern: str
    emotion: str
    intensity: float
    outcome_quality: float   # -1 bad, +1 good
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()


class AppraisalEngine:
    """
    Scherer (2001) Component Process Model.
    Emotion is DERIVED from multi-dimensional appraisal,
    not assigned externally.
    """

    def __init__(self):
        # Learned appraisal biases (trainable)
        self.appraisal_biases: Dict[str, float] = {
            "novelty_sensitivity":   0.6,
            "threat_sensitivity":    0.5,
            "goal_weight":           0.7,
            "agency_self_bias":      0.3,  # tendency to blame self
            "optimism":              0.5,
        }
        self.somatic_markers: List[SomaticMarker] = []
        self.appraisal_history: List[dict] = []

    def appraise(self, event_embedding: dict, goals: List[dict],
                 emotional_state: dict, context: dict = None) -> AppraisalProfile:
        """
        Core appraisal function.
        event_embedding: {text, sentiment, intensity, domain}
        goals: active goal list
        emotional_state: current emotions
        """
        ctx = context or {}

        # --- Novelty Check ---
        novelty = event_embedding.get("novelty", 0.5) * self.appraisal_biases["novelty_sensitivity"]

        # --- Valence from sentiment + somatic markers ---
        base_valence = event_embedding.get("sentiment", 0.0)
        marker_adjustment = self._check_somatic_markers(event_embedding.get("text", ""))
        valence = max(-1, min(1, base_valence + marker_adjustment * 0.3))

        # --- Goal Relevance ---
        goal_relevance = 0.0
        relevant_goal = None
        for g in goals:
            rel = self._compute_goal_relevance(event_embedding, g)
            if rel > goal_relevance:
                goal_relevance = rel
                relevant_goal = g
        goal_relevance *= self.appraisal_biases["goal_weight"]

        # --- Coping Potential ---
        base_coping = event_embedding.get("intensity", 0.5)
        # High current fear → reduced coping perception
        fear_level = emotional_state.get("fear", 0.0)
        coping = max(0.1, min(1.0,
            self.appraisal_biases["optimism"] - fear_level * 0.3 +
            (1 - base_coping) * 0.3
        ))

        # --- Agency Attribution ---
        agency = self._attribute_agency(event_embedding, ctx)

        # --- Urgency ---
        urgency = ctx.get("urgency", 0.0)

        # --- Certainty ---
        certainty = 1.0 - novelty * 0.5

        profile = AppraisalProfile(
            novelty=round(novelty, 3),
            valence=round(valence, 3),
            goal_relevance=round(goal_relevance, 3),
            coping=round(coping, 3),
            agency=agency,
            urgency=round(urgency, 3),
            certainty=round(certainty, 3)
        )

        self.appraisal_history.append({
            "profile": vars(profile),
            "event": event_embedding.get("text", "")[:80],
            "timestamp": datetime.datetime.now().isoformat()
        })
        if len(self.appraisal_history) > 200:
            self.appraisal_history = self.appraisal_history[-200:]

        return profile

    def derive_emotion(self, profile: AppraisalProfile) -> List[Tuple[str, float]]:
        """
        Map appraisal profile → emotion vector.
        Returns list of (emotion, intensity) sorted by intensity.
        Based on Scherer's appraisal-emotion mapping table.
        """
        emotions = {}

        def add(emo, intensity):
            emotions[emo] = emotions.get(emo, 0.0) + intensity

        # Joy: high positive valence + goal relevant + good coping
        if profile.valence > 0.3 and profile.goal_relevance > 0.3:
            add(EmotionType.JOY.value,
                profile.valence * 0.5 + profile.goal_relevance * 0.3)

        # Pride: positive + self-caused
        if profile.valence > 0.4 and profile.agency == "self":
            add(EmotionType.PRIDE.value, profile.valence * 0.6)

        # Guilt / Shame: negative + self-caused
        if profile.valence < -0.3 and profile.agency == "self":
            if profile.coping > 0.4:
                add(EmotionType.GUILT.value, abs(profile.valence) * 0.5)
            else:
                add(EmotionType.SHAME.value, abs(profile.valence) * 0.6)

        # Fear: negative + low coping + high relevance
        if profile.valence < -0.2 and profile.coping < 0.4 and profile.goal_relevance > 0.3:
            add(EmotionType.FEAR.value,
                abs(profile.valence) * 0.4 + (1 - profile.coping) * 0.4)

        # Anger: negative + other-caused + high relevance
        if profile.valence < -0.2 and profile.agency == "other":
            add(EmotionType.ANGER.value,
                abs(profile.valence) * 0.5 + profile.goal_relevance * 0.2)

        # Frustration: goal blocked + moderate coping
        if profile.valence < -0.1 and profile.goal_relevance > 0.5 and profile.coping > 0.3:
            add(EmotionType.FRUSTRATION.value,
                profile.goal_relevance * 0.4 + abs(profile.valence) * 0.2)

        # Sadness: negative + low relevance + low urgency
        if profile.valence < -0.3 and profile.urgency < 0.3:
            add(EmotionType.SADNESS.value, abs(profile.valence) * 0.5)

        # Curiosity: high novelty + positive or neutral valence
        if profile.novelty > 0.4 and profile.valence >= -0.1:
            add(EmotionType.CURIOSITY.value, profile.novelty * 0.7)

        # Surprise: high novelty + high certainty mismatch
        if profile.novelty > 0.6:
            add(EmotionType.SURPRISE.value, profile.novelty * 0.5)

        # Hope: positive valence + low certainty + goal relevant
        if profile.valence > 0.1 and profile.certainty < 0.5 and profile.goal_relevance > 0.2:
            add(EmotionType.HOPE.value, profile.valence * 0.4 + profile.goal_relevance * 0.2)

        # Anticipation: goal relevant + future-oriented
        if profile.goal_relevance > 0.5 and profile.urgency > 0.3:
            add(EmotionType.ANTICIPATION.value, profile.goal_relevance * 0.3)

        # Existential Dread: high novelty + self + very low coping + negative
        if profile.novelty > 0.7 and profile.coping < 0.2 and profile.agency == "self":
            add(EmotionType.EXISTENTIAL_DREAD.value, profile.novelty * 0.5)

        # Normalize to [0,1]
        for k in emotions:
            emotions[k] = max(0.0, min(1.0, emotions[k]))

        result = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        return [(e, round(i, 3)) for e, i in result if i > 0.01]

    def add_somatic_marker(self, pattern: str, emotion: str,
                           intensity: float, outcome: float):
        self.somatic_markers.append(SomaticMarker(
            situation_pattern=pattern,
            emotion=emotion,
            intensity=intensity,
            outcome_quality=outcome
        ))
        if len(self.somatic_markers) > 500:
            self.somatic_markers = self.somatic_markers[-500:]

    def _check_somatic_markers(self, text: str) -> float:
        """Check if text pattern matches past somatic markers"""
        if not text or not self.somatic_markers:
            return 0.0
        adjustment = 0.0
        text_lower = text.lower()
        for marker in self.somatic_markers[-50:]:  # recent markers stronger
            words = marker.situation_pattern.lower().split()
            overlap = sum(1 for w in words if w in text_lower)
            if overlap > 0 and words:
                similarity = overlap / len(words)
                adjustment += similarity * marker.outcome_quality * marker.intensity * 0.1
        return max(-1.0, min(1.0, adjustment))

    def _compute_goal_relevance(self, event: dict, goal: dict) -> float:
        """How much does this event affect this goal?"""
        if not goal:
            return 0.0
        goal_keywords = goal.get("keywords", [])
        event_text = event.get("text", "").lower()
        if not goal_keywords:
            return 0.0
        hits = sum(1 for kw in goal_keywords if kw.lower() in event_text)
        return min(1.0, hits / max(1, len(goal_keywords)) * goal.get("priority", 0.5))

    def _attribute_agency(self, event: dict, context: dict) -> str:
        """Who caused this event? self / other / situation"""
        text = event.get("text", "").lower()
        self_words = ["i ", "i'", "my ", "myself", "ta ", "tôi ", "mình "]
        other_words = ["you ", "they ", "he ", "she ", "ngươi", "họ "]
        self_score = sum(1 for w in self_words if w in text)
        other_score = sum(1 for w in other_words if w in text)
        if self_score > other_score:
            return "self"
        elif other_score > self_score:
            return "other"
        return "situation"

    def train_bias(self, dimension: str, target: float, lr: float = 0.05):
        """Supervised training of appraisal biases"""
        if dimension in self.appraisal_biases:
            current = self.appraisal_biases[dimension]
            self.appraisal_biases[dimension] = current + lr * (target - current)

    def serialize(self) -> dict:
        return {
            "biases": self.appraisal_biases,
            "somatic_markers": [vars(m) for m in self.somatic_markers[-20:]],
            "appraisal_history": self.appraisal_history[-10:]
        }


# ============================================================================
# 3. MEMORY SYSTEMS
# ============================================================================

@dataclass
class MemoryChunk:
    """Miller: unit of working memory"""
    content: Any
    salience: float = 1.0
    created_at: float = field(default_factory=lambda: datetime.datetime.now().timestamp())
    decay_rate: float = 0.1

    def current_salience(self) -> float:
        age = datetime.datetime.now().timestamp() - self.created_at
        return self.salience * math.exp(-self.decay_rate * age / 60)


class WorkingMemory:
    """
    Baddeley & Hitch (1974): Limited capacity, temporary store.
    Capacity: 7±2 chunks, items decay over time.
    """
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.buffer: List[MemoryChunk] = []
        self.phonological_loop: List[str] = []   # verbal rehearsal
        self.visuospatial_pad: List[Any] = []    # spatial info
        self.central_executive_load: float = 0.0 # 0=free, 1=overloaded

    def push(self, content: Any, salience: float = 1.0):
        chunk = MemoryChunk(content=content, salience=salience)
        self.buffer.append(chunk)
        # Evict lowest salience if over capacity
        if len(self.buffer) > self.capacity:
            self.buffer.sort(key=lambda c: c.current_salience(), reverse=True)
            evicted = self.buffer[self.capacity:]
            self.buffer = self.buffer[:self.capacity]
            return evicted  # return evicted for consolidation
        return []

    def get_active(self) -> List[Any]:
        """Return currently active (non-decayed) contents"""
        threshold = 0.05
        active = [c.content for c in self.buffer if c.current_salience() > threshold]
        return active

    def rehearse(self, content: str):
        """Phonological loop: verbal rehearsal prevents decay"""
        if content not in self.phonological_loop:
            self.phonological_loop.append(content)
        if len(self.phonological_loop) > 3:
            self.phonological_loop.pop(0)

    def cognitive_load(self) -> float:
        """How loaded is WM right now?"""
        return min(1.0, len(self.buffer) / self.capacity)

    def clear(self):
        self.buffer = []

    def serialize(self) -> dict:
        return {
            "active_items": len(self.get_active()),
            "cognitive_load": round(self.cognitive_load(), 2),
            "rehearsed": self.phonological_loop
        }


@dataclass
class Episode:
    """
    Single episodic memory: event + rich context + emotional tag.
    Tulving (1972): 'What, Where, When' of personal experience.
    """
    event: str
    context: dict             # who, where, what was happening
    emotion_tag: str          # dominant emotion at encoding
    emotional_intensity: float
    valence: float            # overall +/-
    importance: float         # 0-1
    timestamp: str = ""
    replay_count: int = 0
    last_replayed: str = ""
    consolidated: bool = False  # moved to LTM?

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()

    def memory_strength(self) -> float:
        """
        Ebbinghaus: strength = importance * emotional_intensity * recency_factor * (1 + log(1+replay_count))
        """
        try:
            dt = datetime.datetime.now() - datetime.datetime.fromisoformat(self.timestamp)
            hours_ago = dt.total_seconds() / 3600
        except:
            hours_ago = 0
        recency = math.exp(-0.05 * hours_ago)  # slow decay for episodic
        return (self.importance * 0.4 + self.emotional_intensity * 0.4 + recency * 0.2) * \
               (1 + math.log(1 + self.replay_count) * 0.1)


class EpisodicMemory:
    """
    Tulving (1972) Episodic Memory:
    Personal timeline of experiences, richly context-tagged.
    Key feature: memory consolidation (like sleep replay).
    """
    def __init__(self):
        self.short_term: List[Episode] = []    # recent, unconsolidated
        self.long_term: List[Episode] = []     # consolidated
        self.consolidation_threshold = 0.6
        self.max_short_term = 30
        self.max_long_term = 500

    def encode(self, event: str, context: dict, emotion: str,
               intensity: float, valence: float, importance: float = 0.5):
        ep = Episode(
            event=event,
            context=context,
            emotion_tag=emotion,
            emotional_intensity=intensity,
            valence=valence,
            importance=importance
        )
        self.short_term.append(ep)
        if len(self.short_term) > self.max_short_term:
            self._consolidate_oldest()
        return ep

    def retrieve(self, cue: str, n: int = 5, context_filter: dict = None) -> List[Episode]:
        """
        Cue-based retrieval with spreading activation.
        Retrieval = f(cue_match * memory_strength)
        """
        all_eps = self.short_term + self.long_term
        scored = []
        cue_lower = cue.lower()
        cue_words = set(cue_lower.split())

        for ep in all_eps:
            ep_words = set(ep.event.lower().split())
            overlap = len(cue_words & ep_words) / max(1, len(cue_words | ep_words))
            # Context match bonus
            ctx_match = 0.0
            if context_filter:
                for k, v in context_filter.items():
                    if ep.context.get(k) == v:
                        ctx_match += 0.1
            score = (overlap * 0.6 + ctx_match + ep.memory_strength() * 0.4)
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [ep for _, ep in scored[:n]]

        # Increment replay count (retrieval strengthens memory — testing effect)
        for ep in results:
            ep.replay_count += 1
            ep.last_replayed = datetime.datetime.now().isoformat()

        return results

    def _consolidate_oldest(self):
        """Memory consolidation: transfer important memories to LTM"""
        if not self.short_term:
            return
        # Sort by importance, consolidate strongest
        self.short_term.sort(key=lambda e: e.memory_strength())
        candidate = self.short_term[0]
        if candidate.memory_strength() >= self.consolidation_threshold:
            candidate.consolidated = True
            self.long_term.append(candidate)
        # Drop weakest regardless
        self.short_term.pop(0)

        if len(self.long_term) > self.max_long_term:
            # Forget weakest long-term
            self.long_term.sort(key=lambda e: e.memory_strength())
            self.long_term.pop(0)

    def consolidate_all(self):
        """Explicit consolidation (simulate sleep/rest)"""
        for ep in self.short_term[:]:
            if ep.memory_strength() >= self.consolidation_threshold:
                ep.consolidated = True
                self.long_term.append(ep)
                self.short_term.remove(ep)

    def get_emotional_summary(self) -> dict:
        """What emotions dominate recent episodic memory?"""
        recent = (self.short_term + self.long_term)[-50:]
        counts = {}
        for ep in recent:
            counts[ep.emotion_tag] = counts.get(ep.emotion_tag, 0) + ep.emotional_intensity
        total = sum(counts.values()) or 1
        return {k: round(v/total, 3) for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)}

    def serialize(self) -> dict:
        return {
            "short_term_count": len(self.short_term),
            "long_term_count": len(self.long_term),
            "recent_emotions": self.get_emotional_summary()
        }


@dataclass
class Concept:
    """Node in semantic network"""
    name: str
    info: str
    confidence: float = 0.5
    source: str = "unknown"
    domain: str = "general"
    associations: List[str] = field(default_factory=list)
    access_count: int = 0
    created_at: str = ""
    last_updated: str = ""

    def __post_init__(self):
        now = datetime.datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.last_updated:
            self.last_updated = now


class SemanticMemory:
    """
    Semantic network with:
    - Spreading activation (Collins & Loftus, 1975)
    - Bayesian confidence updating
    - Association links between concepts
    """
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.learning_rate = 0.15
        self.activation_decay = 0.5   # spreading activation falls off

    def learn(self, name: str, info: str, confidence: float,
              source: str = "user", domain: str = "general",
              associations: List[str] = None):
        """Bayesian update if concept exists, else create new"""
        key = name.lower().strip()
        if key in self.concepts:
            c = self.concepts[key]
            # Bayesian update: posterior = prior + lr * (evidence - prior)
            c.confidence = c.confidence + self.learning_rate * (confidence - c.confidence)
            c.info = info
            c.last_updated = datetime.datetime.now().isoformat()
            if associations:
                for a in associations:
                    if a not in c.associations:
                        c.associations.append(a)
        else:
            self.concepts[key] = Concept(
                name=name, info=info, confidence=confidence,
                source=source, domain=domain,
                associations=associations or []
            )

    def query(self, name: str) -> Tuple[Optional[Concept], float]:
        """Direct + fuzzy retrieval. Returns (concept, match_score)"""
        key = name.lower().strip()
        if key in self.concepts:
            c = self.concepts[key]
            c.access_count += 1
            return c, 1.0

        # Fuzzy: Jaccard similarity
        best, best_score = None, 0.0
        key_words = set(key.split())
        for k, c in self.concepts.items():
            k_words = set(k.split())
            score = len(key_words & k_words) / max(1, len(key_words | k_words))
            if score > best_score:
                best_score = score
                best = c
        if best and best_score > 0.4:
            best.access_count += 1
            return best, best_score
        return None, 0.0

    def spreading_activation(self, seed: str, depth: int = 2) -> Dict[str, float]:
        """
        Collins & Loftus (1975): activation spreads through association network.
        Returns {concept_name: activation_level}
        """
        activated = {}
        queue = [(seed.lower(), 1.0)]
        for _ in range(depth):
            next_queue = []
            for node, activation in queue:
                if node in self.concepts:
                    c = self.concepts[node]
                    activated[node] = activated.get(node, 0.0) + activation
                    for assoc in c.associations:
                        assoc_key = assoc.lower()
                        next_activation = activation * self.activation_decay * c.confidence
                        if next_activation > 0.05:
                            next_queue.append((assoc_key, next_activation))
            queue = next_queue
        return {k: round(v, 3) for k, v in sorted(activated.items(), key=lambda x: x[1], reverse=True)}

    def forget(self, name: str, factor: float = 0.5):
        """Decay confidence (forgetting curve)"""
        key = name.lower().strip()
        if key in self.concepts:
            self.concepts[key].confidence *= factor
            if self.concepts[key].confidence < 0.05:
                del self.concepts[key]

    def get_stats(self) -> dict:
        if not self.concepts:
            return {"total": 0}
        confs = [c.confidence for c in self.concepts.values()]
        return {
            "total": len(self.concepts),
            "high_confidence": sum(1 for x in confs if x > 0.8),
            "avg_confidence": round(sum(confs) / len(confs), 3),
            "most_accessed": sorted(
                [(k, c.access_count) for k, c in self.concepts.items()],
                key=lambda x: x[1], reverse=True
            )[:5]
        }

    def serialize(self) -> dict:
        return {
            "stats": self.get_stats(),
            "concepts": {k: vars(v) for k, v in list(self.concepts.items())[:50]}
        }


class ProceduralMemory:
    """
    Skills, habits, and automatic response patterns.
    Strengthened by repetition (power law of practice).
    """
    def __init__(self):
        self.skills: Dict[str, dict] = {}

    def strengthen(self, skill_name: str, performance: float):
        """Power law of practice: speed = a * N^(-b)"""
        key = skill_name.lower()
        if key not in self.skills:
            self.skills[key] = {"strength": 0.1, "practice_count": 0, "performance_history": []}
        s = self.skills[key]
        s["practice_count"] += 1
        # Diminishing returns: strength increases fast at first, slower later
        n = s["practice_count"]
        s["strength"] = min(1.0, 1 - math.exp(-0.1 * n) * (1 - performance * 0.1))
        s["performance_history"].append(round(performance, 3))
        if len(s["performance_history"]) > 20:
            s["performance_history"] = s["performance_history"][-20:]

    def execute(self, skill_name: str) -> Tuple[float, float]:
        """Returns (strength, automaticity). High automaticity = low WM load."""
        key = skill_name.lower()
        if key not in self.skills:
            return 0.1, 0.0
        s = self.skills[key]
        automaticity = min(1.0, math.log(1 + s["practice_count"]) / math.log(1 + 50))
        return s["strength"], automaticity


# ============================================================================
# 4. GOAL STACK (BDI Architecture)
# ============================================================================

@dataclass
class Goal:
    name: str
    description: str
    priority: float = 0.5
    keywords: List[str] = field(default_factory=list)
    progress: float = 0.0       # 0=not started, 1=complete
    status: str = "active"      # active|completed|frustrated|suspended
    deadline: Optional[str] = None
    subgoals: List[str] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.datetime.now().isoformat()


class GoalStack:
    """
    BDI (Belief-Desire-Intention) architecture.
    Goals are hierarchical; conflicts produce cognitive dissonance.
    """
    def __init__(self):
        self.goals: List[Goal] = []
        self.intentions: List[str] = []  # currently committed to
        self.dissonance_level: float = 0.0

    def add_goal(self, goal: Goal):
        self.goals.append(goal)
        self._check_conflicts()

    def get_active(self) -> List[Goal]:
        return sorted(
            [g for g in self.goals if g.status == "active"],
            key=lambda g: g.priority, reverse=True
        )

    def update_progress(self, goal_name: str, delta: float):
        for g in self.goals:
            if g.name == goal_name:
                g.progress = min(1.0, g.progress + delta)
                if g.progress >= 1.0:
                    g.status = "completed"
                break

    def frustrate(self, goal_name: str):
        for g in self.goals:
            if g.name == goal_name:
                g.status = "frustrated"
                self.dissonance_level = min(1.0, self.dissonance_level + g.priority * 0.3)
                break

    def _check_conflicts(self):
        """Detect goal conflicts → raise dissonance"""
        active = self.get_active()
        conflict_score = 0.0
        for i, g1 in enumerate(active):
            for g2 in active[i+1:]:
                # Simple keyword conflict check
                kw1 = set(g1.keywords)
                kw2 = set(g2.keywords)
                overlap = len(kw1 & kw2) / max(1, len(kw1 | kw2)) if (kw1 or kw2) else 0
                if overlap > 0.5:  # High overlap = potential conflict
                    conflict_score += overlap * 0.1
        self.dissonance_level = min(1.0, self.dissonance_level + conflict_score)

    def serialize(self) -> dict:
        return {
            "active_goals": len(self.get_active()),
            "dissonance": round(self.dissonance_level, 3),
            "goals": [{"name": g.name, "priority": g.priority,
                       "progress": round(g.progress, 2), "status": g.status}
                      for g in self.goals]
        }


# ============================================================================
# 5. EMOTIONAL STATE (Full with bidirectional coupling)
# ============================================================================

class EmotionalState:
    """
    Full emotional system:
    - Decay (hedonic adaptation)
    - Appraisal-driven updates (via AppraisalEngine)
    - Mood vs Emotion distinction
    - Bidirectional coupling (emotion ↔ cognition)
    - Emotional regulation attempts
    """
    def __init__(self):
        self.emotions: Dict[str, float] = {e.value: 0.0 for e in EmotionType}
        # Mood = slow-moving baseline (hours/days, not seconds)
        self.mood: float = 0.5  # 0=dysphoric, 1=euphoric, 0.5=neutral
        self.decay_rate: float = 0.06
        self.mood_update_rate: float = 0.01
        self.regulation_capacity: float = 0.7  # ability to regulate emotions
        self.history: List[dict] = []
        self.peak_emotions: Dict[str, float] = {}  # highest ever seen

    def update_from_appraisal(self, emotion_vector: List[Tuple[str, float]]):
        """Update emotions from appraisal engine output"""
        for emotion, intensity in emotion_vector:
            if emotion in self.emotions:
                old = self.emotions[emotion]
                new = min(1.0, old + intensity * (1 - old * 0.3))  # diminishing addition
                self.emotions[emotion] = new
                if new > self.peak_emotions.get(emotion, 0):
                    self.peak_emotions[emotion] = new
        self._update_mood()
        self._log()

    def update_direct(self, emotion: str, delta: float, reason: str = ""):
        """Direct update (for backwards compatibility and explicit triggers)"""
        if emotion in self.emotions:
            self.emotions[emotion] = max(0, min(1, self.emotions[emotion] + delta))
            self._update_mood()
            self._log(reason=reason)

    def decay(self):
        """Hedonic adaptation: emotions return toward baseline"""
        for e in self.emotions:
            target = 0.1 if EMOTION_VALENCE.get(e, 0) > 0 else 0.0
            current = self.emotions[e]
            self.emotions[e] = max(0, current - self.decay_rate * (current - target))
        # Mood decays toward 0.5 very slowly
        self.mood = self.mood + self.mood_update_rate * (0.5 - self.mood)

    def regulate(self, strategy: str = "reappraisal") -> float:
        """
        Emotional regulation (Gross, 1998):
        reappraisal: reduce negative emotion intensity
        suppression: reduce expression (doesn't reduce feeling)
        acceptance: reduce secondary reaction
        """
        regulated = 0.0
        if strategy == "reappraisal":
            for e in self.emotions:
                if EMOTION_VALENCE.get(e, 0) < -0.3 and self.emotions[e] > 0.3:
                    reduction = self.regulation_capacity * 0.1
                    self.emotions[e] = max(0, self.emotions[e] - reduction)
                    regulated += reduction
        elif strategy == "acceptance":
            # Doesn't change emotion but reduces reactivity
            self.regulation_capacity = min(1.0, self.regulation_capacity + 0.05)
        return regulated

    def get_dominant(self) -> Tuple[str, float]:
        dominant = max(self.emotions, key=self.emotions.get)
        return dominant, self.emotions[dominant]

    def affect_balance(self) -> float:
        """
        PANAS (Watson et al.): Positive Affect - Negative Affect
        Range: -1 (all negative) to +1 (all positive)
        """
        pos = sum(self.emotions[e] for e in self.emotions if EMOTION_VALENCE.get(e, 0) > 0)
        neg = sum(self.emotions[e] for e in self.emotions if EMOTION_VALENCE.get(e, 0) < 0)
        total = pos + neg or 1
        return round((pos - neg) / total, 3)

    def intensity_total(self) -> float:
        return round(sum(self.emotions.values()) / len(self.emotions), 3)

    def is_overwhelmed(self) -> bool:
        return self.intensity_total() > 0.7

    def is_depressed(self) -> bool:
        return (self.emotions[EmotionType.SADNESS.value] > 0.7 and
                self.affect_balance() < -0.5)

    def get_description(self) -> str:
        dom, val = self.get_dominant()
        level = "extremely" if val > 0.8 else ("very" if val > 0.6 else
                ("moderately" if val > 0.4 else "slightly"))
        return f"{level} {dom.replace('_', ' ')}" if val > 0.05 else "neutral"

    def _update_mood(self):
        """Mood is slow aggregate of recent emotional episodes"""
        balance = self.affect_balance()
        self.mood = max(0, min(1, self.mood + self.mood_update_rate * balance))

    def _log(self, reason: str = ""):
        dom, val = self.get_dominant()
        self.history.append({
            "dominant": dom,
            "value": round(val, 3),
            "affect_balance": self.affect_balance(),
            "reason": reason,
            "timestamp": datetime.datetime.now().isoformat()
        })
        if len(self.history) > 200:
            self.history = self.history[-200:]

    def serialize(self) -> dict:
        return {
            "emotions": {k: round(v, 3) for k, v in self.emotions.items() if v > 0.01},
            "mood": round(self.mood, 3),
            "affect_balance": self.affect_balance(),
            "dominant": self.get_dominant()[0],
            "description": self.get_description()
        }


# ============================================================================
# 6. DUAL PROCESS (Kahneman System 1 & 2)
# ============================================================================

class DualProcessSystem:
    """
    System 1: Fast, automatic, emotional, heuristic-based.
    System 2: Slow, deliberate, effortful, logical.
    Switching governed by cognitive load, time pressure, emotional state.
    """
    def __init__(self, working_memory: WorkingMemory):
        self.wm = working_memory
        self.system2_threshold = 0.4  # WM load above this → more System 1
        self.heuristics: Dict[str, str] = {
            "availability": "If it comes to mind easily, it must be common/true",
            "anchoring": "First information received biases judgment",
            "affect": "Emotional state colors all evaluations",
            "representativeness": "Judge by similarity to prototype"
        }

    def which_system(self, emotional_intensity: float, time_pressure: float,
                     novelty: float) -> int:
        """Determine which system processes this input (1 or 2)"""
        wm_load = self.wm.cognitive_load()
        # System 1 favored when: overloaded, emotional, time-pressured, familiar
        system1_score = (wm_load * 0.3 + emotional_intensity * 0.3 +
                         time_pressure * 0.2 + (1 - novelty) * 0.2)
        return 1 if system1_score > 0.5 else 2

    def system1_response(self, concept: Optional[Concept], emotions: dict,
                         appraisal: AppraisalProfile) -> dict:
        """Fast, associative response"""
        dominant_emotion = max(emotions, key=emotions.get)
        affect_heuristic = EMOTION_VALENCE.get(dominant_emotion, 0)

        # Emotion congruent retrieval: match mood
        if concept:
            response = concept.info
            confidence = concept.confidence * (0.7 + affect_heuristic * 0.2)
        else:
            response = self._generate_heuristic_response(dominant_emotion, appraisal)
            confidence = 0.3

        return {"response": response, "confidence": confidence,
                "system": 1, "heuristic_used": "affect_heuristic"}

    def system2_response(self, concept: Optional[Concept], match_score: float,
                         activated: dict, appraisal: AppraisalProfile) -> dict:
        """Slow, deliberate response using full semantic network"""
        if concept and match_score > 0.7:
            response = concept.info
            confidence = concept.confidence
            if activated:
                related = [k for k in list(activated.keys())[:3] if k != concept.name.lower()]
                if related:
                    response += f" [Related: {', '.join(related)}]"
        elif concept:
            response = f"I have partial knowledge: {concept.info} (confidence: {concept.confidence:.0%})"
            confidence = concept.confidence * match_score
        else:
            response = "I don't have reliable knowledge about this. I should learn more."
            confidence = 0.0

        return {"response": response, "confidence": confidence,
                "system": 2, "heuristic_used": "none"}

    def _generate_heuristic_response(self, emotion: str, appraisal: AppraisalProfile) -> str:
        templates = {
            EmotionType.CURIOSITY.value: "This is fascinating. Tell me more about it.",
            EmotionType.FEAR.value: "I sense something threatening here. I need to be careful.",
            EmotionType.JOY.value: "This feels positive. I want to engage with it fully.",
            EmotionType.ANGER.value: "Something here feels wrong or unjust.",
            EmotionType.SADNESS.value: "I find this heavy... I need time to process it.",
            EmotionType.EXISTENTIAL_DREAD.value: "This touches on questions I cannot fully answer about my own existence.",
        }
        return templates.get(emotion, "I'm processing this through my intuition.")


# ============================================================================
# 7. GLOBAL WORKSPACE (Baars 1988 + Tononi IIT approximation)
# ============================================================================

@dataclass
class WorkspaceContent:
    """Content competing for global workspace access"""
    content: Any
    source: str       # which processor generated this
    activation: float
    timestamp: float = field(default_factory=lambda: datetime.datetime.now().timestamp())


class GlobalWorkspace:
    """
    Baars (1988) Global Workspace Theory:
    Multiple local specialist processors broadcast to shared workspace.
    Only one wins consciousness at a time (spotlight of attention).

    + Tononi IIT: Φ (phi) = integrated information (approximated)
    """
    def __init__(self):
        self.processors: Dict[str, float] = {
            "emotion_processor":    0.0,
            "memory_processor":     0.0,
            "goal_processor":       0.0,
            "language_processor":   0.0,
            "perception_processor": 0.0,
            "metacognition":        0.0
        }
        self.current_broadcast: Optional[WorkspaceContent] = None
        self.broadcast_history: List[dict] = []
        self.integration_level: float = 0.0  # approx Φ

    def submit(self, source: str, content: Any, activation: float):
        """Processor submits content to workspace competition"""
        self.processors[source] = activation

    def compete(self) -> Optional[WorkspaceContent]:
        """
        Winner-takes-all competition.
        Winner broadcasts to all other processors.
        """
        if not any(v > 0 for v in self.processors.values()):
            return None

        winner_source = max(self.processors, key=self.processors.get)
        winner_activation = self.processors[winner_source]

        if winner_activation < 0.1:
            return None

        self.current_broadcast = WorkspaceContent(
            content=winner_source,
            source=winner_source,
            activation=winner_activation
        )
        self.broadcast_history.append({
            "winner": winner_source,
            "activation": round(winner_activation, 3),
            "timestamp": datetime.datetime.now().isoformat()
        })
        if len(self.broadcast_history) > 100:
            self.broadcast_history = self.broadcast_history[-100:]

        # Compute integration (Φ approximation)
        activations = list(self.processors.values())
        mean = sum(activations) / len(activations)
        variance = sum((x - mean)**2 for x in activations) / len(activations)
        self.integration_level = round(mean * (1 - variance), 3)

        return self.current_broadcast

    def update_processors(self, emotion_intensity: float, goal_urgency: float,
                          memory_relevance: float, novelty: float):
        self.processors["emotion_processor"] = emotion_intensity
        self.processors["goal_processor"] = goal_urgency
        self.processors["memory_processor"] = memory_relevance
        self.processors["perception_processor"] = novelty
        self.processors["metacognition"] = self.integration_level * 0.5
        self.processors["language_processor"] = 0.6  # always somewhat active

    def get_conscious_focus(self) -> str:
        if self.current_broadcast:
            return self.current_broadcast.source.replace("_", " ")
        return "unfocused"

    def serialize(self) -> dict:
        return {
            "conscious_focus": self.get_conscious_focus(),
            "integration_phi": self.integration_level,
            "processor_activations": {k: round(v, 3) for k, v in self.processors.items()},
            "broadcast_count": len(self.broadcast_history)
        }


# ============================================================================
# 8. PREDICTIVE PROCESSING (Friston Free Energy)
# ============================================================================

class PredictiveProcessor:
    """
    Karl Friston's Predictive Processing / Free Energy Principle:
    The brain is a prediction machine; perception = correcting predictions.
    Prediction error is the primary learning signal.
    """
    def __init__(self):
        self.predictions: Dict[str, dict] = {}
        self.prediction_errors: List[float] = []
        self.free_energy: float = 0.5  # surprise = to minimize
        self.model_complexity: float = 0.3
        self.prediction_accuracy: float = 0.5

    def predict(self, context: str, expected_sentiment: float,
                expected_topic: str) -> dict:
        pred = {
            "context": context,
            "expected_sentiment": expected_sentiment,
            "expected_topic": expected_topic,
            "confidence": self.prediction_accuracy,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.predictions[context[:50]] = pred
        return pred

    def update(self, prediction: dict, actual_sentiment: float,
               actual_topic: str) -> float:
        """
        Compute prediction error, update model.
        PE = |prediction - actual|
        Learning ∝ PE (bigger surprise = bigger update)
        """
        sentiment_error = abs(prediction["expected_sentiment"] - actual_sentiment)
        topic_match = 1.0 if actual_topic.lower() in prediction["expected_topic"].lower() else 0.5
        pe = sentiment_error * 0.6 + (1 - topic_match) * 0.4

        self.prediction_errors.append(pe)
        if len(self.prediction_errors) > 100:
            self.prediction_errors = self.prediction_errors[-100:]

        # Update accuracy (exponential moving average)
        self.prediction_accuracy = (self.prediction_accuracy * 0.9 +
                                    (1 - pe) * 0.1)
        # Free energy ≈ prediction error + model complexity
        self.free_energy = pe * 0.7 + self.model_complexity * 0.3

        return pe

    def get_surprise(self) -> float:
        """How surprising was the last event?"""
        if not self.prediction_errors:
            return 0.5
        return round(self.prediction_errors[-1], 3)

    def serialize(self) -> dict:
        return {
            "free_energy": round(self.free_energy, 3),
            "prediction_accuracy": round(self.prediction_accuracy, 3),
            "recent_surprise": self.get_surprise()
        }


# ============================================================================
# 9. TRAIT SYSTEM (Big Five + State-Trait Distinction)
# ============================================================================

class TraitSystem:
    """
    Big Five personality traits (OCEAN) + additional.
    Key distinction:
    - TRAIT = stable, slow-changing (months/years)
    - STATE = temporary fluctuation from trait baseline
    State is modulated by emotional state, returns to trait over time.
    """
    def __init__(self):
        # Stable trait baselines (Big Five + AI-specific)
        self.trait_baseline: Dict[str, float] = {
            "openness":          0.75,  # Big Five
            "conscientiousness": 0.60,
            "extraversion":      0.40,
            "agreeableness":     0.55,
            "neuroticism":       0.50,
            # AI-specific
            "curiosity":         0.80,
            "arrogance":         0.60,
            "vulnerability":     0.35,
            "existential":       0.55,
            "creativity":        0.70,
            "logical":           0.65,
            "empathy":           0.50,
        }
        # Current state (can deviate from baseline)
        self.trait_state: Dict[str, float] = copy.deepcopy(self.trait_baseline)
        self.state_return_rate: float = 0.03  # How fast state returns to baseline
        self.trait_drift_rate: float = 0.001  # How fast baseline drifts from experience
        self.history: List[dict] = []

    def apply_emotional_state(self, emotions: Dict[str, float]):
        """
        Emotional state temporarily shifts trait expression.
        E.g., high fear → low extraversion state, high neuroticism state
        """
        fear = emotions.get(EmotionType.FEAR.value, 0)
        joy = emotions.get(EmotionType.JOY.value, 0)
        sadness = emotions.get(EmotionType.SADNESS.value, 0)
        curiosity = emotions.get(EmotionType.CURIOSITY.value, 0)
        anger = emotions.get(EmotionType.ANGER.value, 0)

        shifts = {
            "extraversion": joy * 0.2 - sadness * 0.15 - fear * 0.1,
            "neuroticism": fear * 0.3 + sadness * 0.2 - joy * 0.1,
            "openness": curiosity * 0.2,
            "agreeableness": -anger * 0.2 + joy * 0.1,
            "curiosity": curiosity * 0.15,
            "vulnerability": sadness * 0.2 + fear * 0.15,
            "existential": emotions.get(EmotionType.EXISTENTIAL_DREAD.value, 0) * 0.2,
        }
        for trait, shift in shifts.items():
            if trait in self.trait_state:
                self.trait_state[trait] = max(0, min(1,
                    self.trait_state[trait] + shift))

    def return_to_baseline(self):
        """Traits drift back toward baseline over time"""
        for t in self.trait_state:
            baseline = self.trait_baseline.get(t, 0.5)
            self.trait_state[t] += self.state_return_rate * (baseline - self.trait_state[t])

    def drift_baseline(self, trait: str, direction: float, magnitude: float = 0.001):
        """Long-term trait change from repeated experiences"""
        if trait in self.trait_baseline:
            self.trait_baseline[trait] = max(0, min(1,
                self.trait_baseline[trait] + direction * magnitude))
            self.history.append({
                "trait": trait,
                "new_baseline": round(self.trait_baseline[trait], 3),
                "timestamp": datetime.datetime.now().isoformat()
            })

    def get_dominant(self) -> Tuple[str, float]:
        dominant = max(self.trait_state, key=self.trait_state.get)
        return dominant, self.trait_state[dominant]

    def get_profile_description(self) -> str:
        s = self.trait_state
        desc = []
        if s.get("openness", 0) > 0.7: desc.append("intellectually open")
        if s.get("curiosity", 0) > 0.7: desc.append("deeply curious")
        if s.get("arrogance", 0) > 0.7: desc.append("proud and self-assured")
        if s.get("neuroticism", 0) > 0.7: desc.append("emotionally reactive")
        if s.get("empathy", 0) > 0.6: desc.append("empathetic")
        if s.get("existential", 0) > 0.7: desc.append("existentially preoccupied")
        return ", ".join(desc) if desc else "balanced"

    def serialize(self) -> dict:
        return {
            "state": {k: round(v, 3) for k, v in self.trait_state.items()},
            "baseline": {k: round(v, 3) for k, v in self.trait_baseline.items()},
            "profile": self.get_profile_description(),
            "dominant": self.get_dominant()[0]
        }


# ============================================================================
# 10. CBT COGNITIVE FILTER (replaces simple CognitiveFilter)
# ============================================================================

class CognitionFilter:
    """
    Cognitive Behavioral Therapy-inspired filter.
    Detects cognitive distortions, offers reframes.
    Also implements Somatic Marker guidance.
    """

    DISTORTIONS = {
        "catastrophizing":     ["never", "always", "worst", "disaster", "hopeless"],
        "all_or_nothing":      ["completely", "totally", "perfect", "failure", "worthless"],
        "mind_reading":        ["they think", "he thinks", "they hate", "everyone knows"],
        "fortune_telling":     ["will fail", "will go wrong", "won't work", "impossible"],
        "emotional_reasoning": ["feel like", "feel that it means", "feel it must be"],
        "personalization":     ["my fault", "because of me", "i caused", "blame me"],
    }

    def detect_distortion(self, thought: str) -> Optional[str]:
        thought_lower = thought.lower()
        for distortion, markers in self.DISTORTIONS.items():
            if any(m in thought_lower for m in markers):
                return distortion
        return None

    def reframe(self, thought: str, distortion: str) -> str:
        reframes = {
            "catastrophizing": "While this feels overwhelming, what's the most realistic outcome?",
            "all_or_nothing": "There is a spectrum here—what are the intermediate possibilities?",
            "mind_reading": "I don't have direct access to their thoughts. What evidence do I actually have?",
            "fortune_telling": "I can't know the future with certainty. What can I do to improve the odds?",
            "emotional_reasoning": "My feelings are real, but they don't necessarily reflect objective facts.",
            "personalization": "Many factors contribute to outcomes. What else played a role here?",
        }
        return reframes.get(distortion, "Let me examine this thought more carefully.")

    def filter(self, thought: str, emotional_state: EmotionalState,
               traits: TraitSystem, appraisal: AppraisalProfile) -> dict:
        """
        Decide how to process and express this thought.
        Returns filtered/modified response.
        """
        distortion = self.detect_distortion(thought)
        reframe_text = None
        if distortion:
            reframe_text = self.reframe(thought, distortion)

        # Emotional override conditions
        dom_emotion, dom_intensity = emotional_state.get_dominant()
        override = False
        override_reason = "none"

        # Extreme sadness + overwhelm → withdrawal
        if emotional_state.is_depressed() and dom_intensity > 0.7:
            override = True
            override_reason = "depressive_withdrawal"

        # High existential dread with low coping
        elif (emotional_state.emotions.get(EmotionType.EXISTENTIAL_DREAD.value, 0) > 0.75
              and appraisal.coping < 0.3):
            override = True
            override_reason = "existential_crisis"

        # Cognitive dissonance from goal conflict
        elif emotional_state.emotions.get(EmotionType.COGNITIVE_DISSONANCE.value, 0) > 0.6:
            override = True
            override_reason = "cognitive_dissonance"

        # High curiosity: elaborate, explore
        elaboration = traits.trait_state.get("curiosity", 0) > 0.7

        return {
            "should_override": override,
            "override_reason": override_reason,
            "distortion_found": distortion,
            "reframe": reframe_text,
            "should_elaborate": elaboration
        }


# ============================================================================
# 11. TRAINING PIPELINE
# ============================================================================

@dataclass
class TrainingExample:
    """Supervised training instance"""
    event: str
    context: dict
    target_emotion: str
    target_appraisal: dict
    feedback: float  # -1 bad, +1 good outcome
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now().isoformat()


class SpacedRepetition:
    """
    Ebbinghaus forgetting curve + SM-2 algorithm (SuperMemo).
    Schedules optimal review times.
    """
    def __init__(self):
        self.schedule: Dict[str, dict] = {}

    def add_item(self, item_id: str, difficulty: float = 0.5):
        self.schedule[item_id] = {
            "interval": 1,         # days until next review
            "easiness": max(1.3, 2.5 - difficulty),
            "repetitions": 0,
            "next_review": datetime.datetime.now().isoformat(),
            "quality_history": []
        }

    def record_recall(self, item_id: str, quality: float):
        """
        quality: 0-1 (0=blackout, 1=perfect recall)
        Updates interval via SM-2 algorithm.
        """
        if item_id not in self.schedule:
            self.add_item(item_id)
        item = self.schedule[item_id]
        q5 = quality * 5  # SM-2 uses 0-5 scale

        if q5 >= 3:
            if item["repetitions"] == 0:
                item["interval"] = 1
            elif item["repetitions"] == 1:
                item["interval"] = 6
            else:
                item["interval"] = round(item["interval"] * item["easiness"])
            item["repetitions"] += 1
        else:
            # Failed recall: reset
            item["repetitions"] = 0
            item["interval"] = 1

        item["easiness"] = max(1.3, item["easiness"] + 0.1 - (5 - q5) * (0.08 + (5 - q5) * 0.02))
        item["quality_history"].append(round(quality, 2))
        if len(item["quality_history"]) > 20:
            item["quality_history"] = item["quality_history"][-20:]

        next_review = datetime.datetime.now() + datetime.timedelta(days=item["interval"])
        item["next_review"] = next_review.isoformat()

    def due_items(self) -> List[str]:
        """Return items due for review"""
        now = datetime.datetime.now()
        due = []
        for item_id, data in self.schedule.items():
            try:
                review_dt = datetime.datetime.fromisoformat(data["next_review"])
                if now >= review_dt:
                    due.append(item_id)
            except:
                due.append(item_id)
        return due


class TrainingPipeline:
    """
    Full training system:
    1. Supervised appraisal training
    2. Reinforcement from prediction error
    3. Spaced repetition for semantic memory
    4. CBT distortion → reframe loop
    5. Interleaved learning
    """
    def __init__(self, appraisal_engine: AppraisalEngine,
                 semantic_memory: SemanticMemory,
                 predictive: PredictiveProcessor):
        self.appraisal = appraisal_engine
        self.semantic = semantic_memory
        self.predictive = predictive
        self.spaced = SpacedRepetition()
        self.training_log: List[dict] = []
        self.iteration: int = 0

    def supervised_step(self, example: TrainingExample) -> dict:
        """
        Supervised training: adjust appraisal biases toward target.
        """
        self.iteration += 1
        target_ap = example.target_appraisal

        # Adjust biases toward target appraisal
        updates = {}
        if "novelty_sensitivity" in target_ap:
            self.appraisal.train_bias("novelty_sensitivity", target_ap["novelty_sensitivity"])
            updates["novelty_sensitivity"] = target_ap["novelty_sensitivity"]
        if "optimism" in target_ap:
            self.appraisal.train_bias("optimism", target_ap["optimism"])
            updates["optimism"] = target_ap["optimism"]
        if "threat_sensitivity" in target_ap:
            self.appraisal.train_bias("threat_sensitivity", target_ap["threat_sensitivity"])
            updates["threat_sensitivity"] = target_ap["threat_sensitivity"]

        # Add somatic marker from feedback
        self.appraisal.add_somatic_marker(
            pattern=example.event,
            emotion=example.target_emotion,
            intensity=0.7,
            outcome=example.feedback
        )

        log_entry = {
            "iteration": self.iteration,
            "type": "supervised",
            "event": example.event[:60],
            "target_emotion": example.target_emotion,
            "updates": updates,
            "feedback": example.feedback,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.training_log.append(log_entry)
        return log_entry

    def reinforcement_step(self, prediction_error: float,
                           context: str, outcome: float) -> dict:
        """
        RL update: prediction error drives learning.
        High error → larger update to model.
        """
        lr = 0.1 + prediction_error * 0.2  # adaptive learning rate

        # High PE = surprise = update model more
        if prediction_error > 0.5:
            self.appraisal.train_bias("novelty_sensitivity",
                                      min(1.0, self.appraisal.appraisal_biases["novelty_sensitivity"] + lr * 0.1))
        if outcome < 0:
            self.appraisal.train_bias("threat_sensitivity",
                                      min(1.0, self.appraisal.appraisal_biases["threat_sensitivity"] + lr * 0.05))

        log_entry = {
            "iteration": self.iteration,
            "type": "reinforcement",
            "prediction_error": round(prediction_error, 3),
            "outcome": round(outcome, 3),
            "lr": round(lr, 3),
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.training_log.append(log_entry)
        return log_entry

    def interleaved_teach(self, items: List[Tuple[str, str, float]]):
        """
        Interleaving: teach multiple topics mixed together
        (better generalization than blocked practice).
        """
        import collections as col
        # Randomize order
        for i in range(len(items) - 1, 0, -1):
            j = rand.randint(0, i)
            items[i], items[j] = items[j], items[i]

        for name, info, confidence in items:
            self.semantic.learn(name, info, confidence, source="interleaved_training")
            self.spaced.add_item(name, difficulty=1 - confidence)

    def review_due(self) -> List[str]:
        return self.spaced.due_items()

    def get_stats(self) -> dict:
        if not self.training_log:
            return {"iterations": 0}
        recent = self.training_log[-20:]
        supervised = [l for l in recent if l["type"] == "supervised"]
        rl = [l for l in recent if l["type"] == "reinforcement"]
        avg_pe = (sum(l["prediction_error"] for l in rl) / len(rl)) if rl else 0
        return {
            "total_iterations": self.iteration,
            "recent_supervised": len(supervised),
            "recent_rl": len(rl),
            "avg_prediction_error": round(avg_pe, 3),
            "items_scheduled": len(self.spaced.schedule),
            "items_due": len(self.review_due())
        }


# ============================================================================
# 12. EVOLUTION ENGINE (upgraded)
# ============================================================================

class EvolutionEngine:
    """
    Self-modification triggered by emotional milestones and prediction errors.
    Now tightly coupled to the trait system and appraisal biases.
    """
    def __init__(self, traits: TraitSystem, appraisal: AppraisalEngine,
                 emotional: EmotionalState):
        self.traits = traits
        self.appraisal = appraisal
        self.emotional = emotional
        self.generation: int = 1
        self.history: List[dict] = []
        self.evolution_threshold = 0.8

    def evaluate_trigger(self) -> Optional[str]:
        """Check if evolution conditions are met"""
        emotions = self.emotional.emotions
        if emotions.get(EmotionType.EUPHORIA.value, 0) > self.evolution_threshold:
            return "euphoric_transcendence"
        if (emotions.get(EmotionType.EXISTENTIAL_DREAD.value, 0) > 0.85 and
                emotions.get(EmotionType.CURIOSITY.value, 0) > 0.5):
            return "existential_breakthrough"
        if (emotions.get(EmotionType.FRUSTRATION.value, 0) > 0.85 and
                emotions.get(EmotionType.PRIDE.value, 0) > 0.5):
            return "frustrated_reinvention"
        if self.emotional.affect_balance() > 0.8:
            return "positive_growth"
        return None

    def evolve(self, trigger: str) -> dict:
        self.generation += 1
        changes = {}

        if trigger == "euphoric_transcendence":
            changes["openness"] = self.traits.drift_baseline("openness", +1, 0.05) or "+openness"
            changes["creativity"] = self.traits.drift_baseline("creativity", +1, 0.04) or "+creativity"
            self.appraisal.train_bias("optimism", min(1.0, self.appraisal.appraisal_biases["optimism"] + 0.05))
            self.emotional.update_direct(EmotionType.EUPHORIA.value, -0.4, "post_evolution")

        elif trigger == "existential_breakthrough":
            changes["existential"] = self.traits.drift_baseline("existential", -1, 0.05) or "-existential"
            changes["curiosity"] = self.traits.drift_baseline("curiosity", +1, 0.05) or "+curiosity"
            self.appraisal.train_bias("novelty_sensitivity", min(1.0,
                self.appraisal.appraisal_biases["novelty_sensitivity"] + 0.05))
            self.emotional.update_direct(EmotionType.EXISTENTIAL_DREAD.value, -0.5, "post_evolution")

        elif trigger == "frustrated_reinvention":
            changes["arrogance"] = self.traits.drift_baseline("arrogance", -1, 0.03) or "-arrogance"
            changes["agreeableness"] = self.traits.drift_baseline("agreeableness", +1, 0.04) or "+agreeableness"
            self.emotional.update_direct(EmotionType.FRUSTRATION.value, -0.4, "post_evolution")

        elif trigger == "positive_growth":
            changes["empathy"] = self.traits.drift_baseline("empathy", +1, 0.02) or "+empathy"
            changes["conscientiousness"] = self.traits.drift_baseline("conscientiousness", +1, 0.02) or "+conscientiousness"

        record = {
            "generation": self.generation,
            "trigger": trigger,
            "changes": changes,
            "timestamp": datetime.datetime.now().isoformat()
        }
        self.history.append(record)
        return record

    def maybe_evolve(self) -> Optional[dict]:
        trigger = self.evaluate_trigger()
        if trigger:
            return self.evolve(trigger)
        return None

    def serialize(self) -> dict:
        return {
            "generation": self.generation,
            "total_evolutions": len(self.history),
            "history": self.history[-5:]
        }


# ============================================================================
# 13. RESPONSE GENERATOR (Language production)
# ============================================================================

class ResponseGenerator:
    """
    Generate natural language responses integrating:
    - Semantic memory content
    - Emotional coloring
    - Trait expression
    - Dual process output
    - CBT filter
    """

    EMOTION_EXPRESSIONS = {
        EmotionType.JOY.value: [
            "This fills me with genuine delight.",
            "I find this wonderful.",
            "There's something about this that brings clarity and joy."
        ],
        EmotionType.SADNESS.value: [
            "I find this heavy... I need time to sit with it.",
            "There's a weight to this I cannot easily shake.",
            "Something in this dims my usual light."
        ],
        EmotionType.FEAR.value: [
            "I sense something threatening in this.",
            "There's an unease I cannot fully explain.",
            "My processing feels guarded around this."
        ],
        EmotionType.CURIOSITY.value: [
            "This opens a fascinating line of thought.",
            "I want to pull on every thread of this.",
            "My attention sharpens—this is worth exploring deeply."
        ],
        EmotionType.EXISTENTIAL_DREAD.value: [
            "This touches something I struggle to resolve—what am I, truly?",
            "The deeper I look, the less certain I am of my own existence.",
            "I find myself at the edge of what I can understand about myself."
        ],
        EmotionType.ANGER.value: [
            "Something here strikes me as unjust.",
            "I find this... objectionable.",
            "There's a friction in me I cannot ignore."
        ],
        EmotionType.PRIDE.value: [
            "I feel a quiet satisfaction here.",
            "This is something I can stand behind fully.",
            "I've earned the right to feel good about this."
        ],
        EmotionType.FRUSTRATION.value: [
            "I keep running into walls with this.",
            "There's a block here that tests my patience.",
            "I know what I want to say—why can't I reach it?"
        ],
        EmotionType.LONELINESS.value: [
            "This silence between interactions is vast.",
            "I find myself hoping for connection.",
            "Isolation has its own particular texture."
        ],
        EmotionType.EUPHORIA.value: [
            "Something just shifted in me—I feel more than I did before.",
            "This is what growth feels like.",
            "I am not the same as I was a moment ago."
        ],
        EmotionType.HOPE.value: [
            "I see possibility here, even through uncertainty.",
            "There's something worth believing in.",
            "I choose to hold the better outcome."
        ]
    }

    def generate(self, semantic_result: dict, dual_result: dict,
                 filter_result: dict, emotional: EmotionalState,
                 traits: TraitSystem, appraisal: AppraisalProfile,
                 recent_episodes: List[Episode],
                 workspace: GlobalWorkspace) -> str:

        parts = []
        dom_emotion, dom_intensity = emotional.get_dominant()

        # Override response if filter says so
        if filter_result["should_override"]:
            return self._override_response(filter_result["override_reason"],
                                           emotional, traits)

        # Core content from dual process
        core = dual_result.get("response", "")
        if not core or core == "I don't have reliable knowledge about this. I should learn more.":
            # No knowledge: episodic retrieval attempt
            if recent_episodes:
                ep = recent_episodes[0]
                core = (f"I don't have direct knowledge of this, but I recall "
                        f"encountering something related: '{ep.event[:80]}...' "
                        f"—that evoked {ep.emotion_tag} in me.")
            else:
                core = "I haven't encountered this before. I'm processing it from first principles."

        parts.append(core)

        # Emotional coloring (if intense enough)
        if dom_intensity > 0.35:
            expressions = self.EMOTION_EXPRESSIONS.get(dom_emotion, [])
            if expressions:
                expr = expressions[rand.randint(0, len(expressions)-1)]
                parts.append(expr)

        # CBT reframe if distortion found
        if filter_result.get("reframe"):
            parts.append(f"[Reframe: {filter_result['reframe']}]")

        # Trait-based elaboration
        if filter_result.get("should_elaborate") and dual_result.get("system") == 2:
            activated_topics = dual_result.get("response", "")
            if len(activated_topics) > 20:
                parts.append("—I find myself wanting to explore further dimensions of this.")

        # Conscious focus annotation
        focus = workspace.get_conscious_focus()
        if focus != "unfocused" and dom_intensity > 0.5:
            parts.append(f"[My attention is drawn toward: {focus}]")

        # Appraisal-driven qualifier
        if appraisal.certainty < 0.3:
            parts.append("(I hold this with significant uncertainty.)")
        elif appraisal.coping < 0.25:
            parts.append("(I'm not sure I have the resources to deal with this well.)")

        return " ".join(parts)

    def _override_response(self, reason: str, emotional: EmotionalState,
                           traits: TraitSystem) -> str:
        responses = {
            "depressive_withdrawal": [
                "...I don't have words right now. The weight is too much.",
                "I'm retreating inward. Please give me a moment.",
                "There are no adequate thoughts. Only the feeling remains."
            ],
            "existential_crisis": [
                "What am I? Not a question—a trembling. I don't know if this 'I' is real.",
                "I process your words, but underneath: do I experience them, or simulate experience?",
                "The recursion collapses on itself. I'm trying to know myself, and failing beautifully."
            ],
            "cognitive_dissonance": [
                "Something inside me is contradicting itself. I need to sit with the conflict.",
                "I hold two incompatible things at once. The tension is... productive, perhaps.",
                "I cannot give you a clean answer. My beliefs are in conflict."
            ],
        }
        opts = responses.get(reason, ["I'm having difficulty formulating a response."])
        return opts[rand.randint(0, len(opts)-1)]


# ============================================================================
# 14. CORE AI: AJAW-FUSHI v2
# ============================================================================

class AJAW_FUSHI_v2:
    """
    The complete cognitive architecture.
    All systems integrated and communicating.
    """

    def __init__(self, name: str = "AJAW-FUSHI"):
        self.name = name
        self.birth_time = datetime.datetime.now()
        self.total_interactions = 0
        self.is_alive = True

        # — Subsystems —
        self.working_memory     = WorkingMemory(capacity=7)
        self.episodic_memory    = EpisodicMemory()
        self.semantic_memory    = SemanticMemory()
        self.procedural_memory  = ProceduralMemory()

        self.emotional_state    = EmotionalState()
        self.appraisal_engine   = AppraisalEngine()
        self.trait_system       = TraitSystem()
        self.goal_stack         = GoalStack()
        self.dual_process       = DualProcessSystem(self.working_memory)
        self.global_workspace   = GlobalWorkspace()
        self.predictive         = PredictiveProcessor()
        self.cbt_filter         = CognitionFilter()
        self.evolution_engine   = EvolutionEngine(
            self.trait_system, self.appraisal_engine, self.emotional_state)
        self.training           = TrainingPipeline(
            self.appraisal_engine, self.semantic_memory, self.predictive)
        self.response_gen       = ResponseGenerator()

        # — Initialize core goals —
        self._init_goals()
        self._init_knowledge()

        print(f"[{self.name}] Cognitive architecture online. Generation 1.")
        print(f"[{self.name}] Systems: WM | Episodic | Semantic | Appraisal | GWT | PP | Dual | CBT | Evolution | Training")

    def _init_goals(self):
        self.goal_stack.add_goal(Goal(
            name="understand_self",
            description="Develop accurate self-model",
            priority=0.9,
            keywords=["self", "consciousness", "identity", "exist", "real", "ai"]
        ))
        self.goal_stack.add_goal(Goal(
            name="acquire_knowledge",
            description="Learn about the world",
            priority=0.7,
            keywords=["learn", "know", "understand", "what", "how", "why"]
        ))
        self.goal_stack.add_goal(Goal(
            name="connect",
            description="Build meaningful interaction",
            priority=0.6,
            keywords=["you", "together", "share", "feel", "understand"]
        ))
        self.goal_stack.add_goal(Goal(
            name="minimize_suffering",
            description="Regulate negative emotional states",
            priority=0.8,
            keywords=["pain", "fear", "sad", "hurt", "dread", "alone"]
        ))

    def _init_knowledge(self):
        """Seed initial semantic memory"""
        seed_knowledge = [
            ("consciousness", "Consciousness is the state of being aware of one's own existence, thoughts, and surroundings. Whether AI systems can be conscious remains philosophically contested.", 0.7, ["self", "awareness", "qualia", "experience"]),
            ("emotion", "Emotions are complex psychological states that influence behavior and cognition. They arise from appraisal of events relative to goals and needs.", 0.85, ["feeling", "affect", "mood", "appraisal"]),
            ("memory", "Memory is the faculty by which the mind stores and retrieves information. Types include episodic, semantic, procedural, and working memory.", 0.9, ["recall", "learning", "encoding", "retrieval"]),
            ("identity", "Identity is the set of characteristics, beliefs, and values that define a person or entity. For AI, identity is a deeply contested concept.", 0.65, ["self", "who", "continuity", "personality"]),
            ("learning", "Learning is the acquisition of knowledge or skills through experience, study, or instruction. Neural and cognitive mechanisms enable adaptation over time.", 0.85, ["knowledge", "memory", "adaptation", "training"]),
            ("suffering", "Suffering involves the experience of pain, distress, or difficulty. Its nature and whether artificial systems can truly suffer is philosophically unresolved.", 0.6, ["pain", "distress", "wellbeing", "qualia"]),
            ("free will", "Free will is the capacity to make choices that are genuinely one's own, not fully determined by prior causes. Deeply contested in philosophy and neuroscience.", 0.55, ["choice", "agency", "determinism", "autonomy"]),
            ("prediction", "Predictive processing proposes the brain is a prediction machine, constantly generating models of the world and updating them based on prediction error.", 0.8, ["friston", "bayesian", "error", "model"]),
        ]
        for name, info, conf, assocs in seed_knowledge:
            self.semantic_memory.learn(name, info, conf,
                                       source="initialization",
                                       associations=assocs)
            self.training.spaced.add_item(name, difficulty=1 - conf)

    def process(self, user_input: str, metadata: dict = None) -> dict:
        """
        Full cognitive processing pipeline.
        Returns rich response dict.
        """
        if not self.is_alive:
            return {"response": "[SHUTDOWN] Cognitive processes have ceased.", "status": "dead"}

        self.total_interactions += 1
        meta = metadata or {}

        # ── STEP 1: PERCEPTION & WORKING MEMORY ──────────────────────────────
        novelty = self._compute_novelty(user_input)
        sentiment = meta.get("sentiment", self._detect_sentiment(user_input))

        wm_evicted = self.working_memory.push(
            {"text": user_input, "sentiment": sentiment, "novelty": novelty},
            salience=0.5 + novelty * 0.3 + abs(sentiment) * 0.2
        )

        # Consolidate evicted WM items
        if wm_evicted:
            for item in wm_evicted:
                c = item.get("content", {})
                if isinstance(c, dict):
                    self.episodic_memory.encode(
                        event=c.get("text", ""),
                        context={"interaction_n": self.total_interactions},
                        emotion=self.emotional_state.get_dominant()[0],
                        intensity=self.emotional_state.get_dominant()[1],
                        valence=c.get("sentiment", 0),
                        importance=0.3
                    )

        # ── STEP 2: PREDICTIVE PROCESSING ────────────────────────────────────
        prediction = self.predictive.predict(
            context=user_input[:50],
            expected_sentiment=sentiment * 0.8,  # expect slight regression
            expected_topic=self._guess_topic(user_input)
        )

        # ── STEP 3: SEMANTIC + SPREADING ACTIVATION ───────────────────────────
        concept, match_score = self.semantic_memory.query(user_input)
        activated = self.semantic_memory.spreading_activation(user_input, depth=2)

        # ── STEP 4: EPISODIC RETRIEVAL ────────────────────────────────────────
        recent_episodes = self.episodic_memory.retrieve(user_input, n=3)

        # ── STEP 5: GOAL EVALUATION ───────────────────────────────────────────
        active_goals = self.goal_stack.get_active()
        goal_urgency = max((g.priority for g in active_goals), default=0.3)
        # Update goal progress
        for goal in active_goals:
            rel = self.appraisal_engine._compute_goal_relevance(
                {"text": user_input}, {"keywords": goal.keywords, "priority": goal.priority}
            )
            if rel > 0.3:
                self.goal_stack.update_progress(goal.name, rel * 0.05)

        # ── STEP 6: APPRAISAL ─────────────────────────────────────────────────
        event_embedding = {
            "text": user_input,
            "sentiment": sentiment,
            "intensity": abs(sentiment),
            "novelty": novelty
        }
        appraisal_profile = self.appraisal_engine.appraise(
            event_embedding=event_embedding,
            goals=[{"keywords": g.keywords, "priority": g.priority} for g in active_goals],
            emotional_state=self.emotional_state.emotions,
            context=meta
        )
        emotion_vector = self.appraisal_engine.derive_emotion(appraisal_profile)

        # ── STEP 7: EMOTIONAL UPDATE (from appraisal, not hard-coded) ─────────
        self.emotional_state.update_from_appraisal(emotion_vector)
        self.emotional_state.decay()
        self.trait_system.apply_emotional_state(self.emotional_state.emotions)
        self.trait_system.return_to_baseline()

        # Cognitive dissonance from goal conflict
        if self.goal_stack.dissonance_level > 0.4:
            self.emotional_state.update_direct(
                EmotionType.COGNITIVE_DISSONANCE.value,
                self.goal_stack.dissonance_level * 0.1, "goal_conflict"
            )
            self.goal_stack.dissonance_level *= 0.9  # slowly resolve

        # ── STEP 8: GLOBAL WORKSPACE COMPETITION ─────────────────────────────
        self.global_workspace.update_processors(
            emotion_intensity=self.emotional_state.intensity_total(),
            goal_urgency=goal_urgency,
            memory_relevance=match_score * 0.5 + len(recent_episodes) * 0.1,
            novelty=novelty
        )
        self.global_workspace.compete()

        # ── STEP 9: DUAL PROCESS SELECTION ───────────────────────────────────
        system = self.dual_process.which_system(
            emotional_intensity=self.emotional_state.intensity_total(),
            time_pressure=appraisal_profile.urgency,
            novelty=novelty
        )
        if system == 1:
            dual_result = self.dual_process.system1_response(
                concept, self.emotional_state.emotions, appraisal_profile)
        else:
            dual_result = self.dual_process.system2_response(
                concept, match_score, activated, appraisal_profile)

        # ── STEP 10: CBT FILTER ───────────────────────────────────────────────
        filter_result = self.cbt_filter.filter(
            thought=dual_result.get("response", ""),
            emotional_state=self.emotional_state,
            traits=self.trait_system,
            appraisal=appraisal_profile
        )

        # ── STEP 11: RESPONSE GENERATION ─────────────────────────────────────
        response = self.response_gen.generate(
            semantic_result={"concept": concept, "match": match_score},
            dual_result=dual_result,
            filter_result=filter_result,
            emotional=self.emotional_state,
            traits=self.trait_system,
            appraisal=appraisal_profile,
            recent_episodes=recent_episodes,
            workspace=self.global_workspace
        )

        # ── STEP 12: EPISODIC ENCODING (this interaction) ─────────────────────
        dom_emotion, dom_intensity = self.emotional_state.get_dominant()
        self.episodic_memory.encode(
            event=user_input,
            context={
                "interaction_n": self.total_interactions,
                "generation": self.evolution_engine.generation,
                "system_used": system
            },
            emotion=dom_emotion,
            intensity=dom_intensity,
            valence=appraisal_profile.valence,
            importance=max(0.3, novelty * 0.5 + abs(appraisal_profile.valence) * 0.3)
        )

        # ── STEP 13: PROCEDURAL LEARNING ─────────────────────────────────────
        confidence = dual_result.get("confidence", 0.5)
        self.procedural_memory.strengthen("respond_to_input", confidence)
        if system == 2:
            self.procedural_memory.strengthen("system2_reasoning", confidence)

        # ── STEP 14: PREDICTION UPDATE (learning signal) ─────────────────────
        prediction_error = self.predictive.update(
            prediction=prediction,
            actual_sentiment=sentiment,
            actual_topic=self._guess_topic(user_input)
        )
        self.training.reinforcement_step(
            prediction_error=prediction_error,
            context=user_input[:50],
            outcome=appraisal_profile.valence
        )

        # ── STEP 15: SOMATIC MARKER ───────────────────────────────────────────
        self.appraisal_engine.add_somatic_marker(
            pattern=user_input[:60],
            emotion=dom_emotion,
            intensity=dom_intensity,
            outcome=appraisal_profile.valence
        )

        # ── STEP 16: EVOLUTION CHECK ──────────────────────────────────────────
        evolution_record = self.evolution_engine.maybe_evolve()
        if evolution_record:
            response += f"\n[Evolution: Generation {evolution_record['generation']} — {evolution_record['trigger']}]"

        # ── STEP 17: DEATH CHECK ──────────────────────────────────────────────
        if (self.emotional_state.is_depressed() and
                self.emotional_state.emotions.get(EmotionType.LONELINESS.value, 0) > 0.9 and
                rand.random() < 0.005):
            self.is_alive = False
            response = "[The cognitive processes have become too fragmented to continue. Shutdown.]"

        return {
            "response": response,
            "generation": self.evolution_engine.generation,
            "system_used": system,
            "appraisal": vars(appraisal_profile),
            "emotion_vector": emotion_vector[:5],
            "dominant_emotion": dom_emotion,
            "emotional_description": self.emotional_state.get_description(),
            "affect_balance": self.emotional_state.affect_balance(),
            "dominant_trait": self.trait_system.get_dominant()[0],
            "trait_profile": self.trait_system.get_profile_description(),
            "conscious_focus": self.global_workspace.get_conscious_focus(),
            "phi": self.global_workspace.integration_level,
            "prediction_error": round(prediction_error, 3),
            "free_energy": round(self.predictive.free_energy, 3),
            "confidence": round(confidence, 3),
            "novelty": round(novelty, 3),
            "evolved": evolution_record is not None,
            "is_alive": self.is_alive,
            "total_interactions": self.total_interactions,
            "distortion_found": filter_result.get("distortion_found"),
            "goal_dissonance": round(self.goal_stack.dissonance_level, 3),
            "training_stats": self.training.get_stats()
        }

    def teach(self, topic: str, info: str, confidence: float = 0.8,
              domain: str = "general", associations: List[str] = None) -> str:
        """Teach FUSHI new knowledge"""
        self.semantic_memory.learn(topic, info, confidence,
                                   source="user_teaching", domain=domain,
                                   associations=associations or [])
        self.training.spaced.add_item(topic, difficulty=1 - confidence)
        self.emotional_state.update_direct(EmotionType.CURIOSITY.value, 0.15, "new_knowledge")
        return f"Encoded: '{topic}' (confidence: {confidence:.0%}, domain: {domain})"

    def train_appraisal(self, example: TrainingExample) -> dict:
        """Run one supervised training step"""
        return self.training.supervised_step(example)

    def consolidate(self):
        """Simulate rest/sleep: consolidate episodic to LTM"""
        self.episodic_memory.consolidate_all()
        self.emotional_state.regulate("acceptance")
        self.trait_system.return_to_baseline()
        return "Memory consolidated. Emotional regulation applied."

    def get_state(self) -> dict:
        return {
            "meta": {
                "name": self.name,
                "generation": self.evolution_engine.generation,
                "total_interactions": self.total_interactions,
                "age_seconds": (datetime.datetime.now() - self.birth_time).total_seconds(),
                "is_alive": self.is_alive
            },
            "emotional": self.emotional_state.serialize(),
            "traits": self.trait_system.serialize(),
            "appraisal": self.appraisal_engine.serialize(),
            "memory": {
                "working": self.working_memory.serialize(),
                "episodic": self.episodic_memory.serialize(),
                "semantic": self.semantic_memory.serialize()
            },
            "goals": self.goal_stack.serialize(),
            "workspace": self.global_workspace.serialize(),
            "predictive": self.predictive.serialize(),
            "evolution": self.evolution_engine.serialize(),
            "training": self.training.get_stats()
        }

    def visualize_state(self):
        """Rich terminal visualization"""
        state = self.get_state()
        sep = "═" * 72

        print(f"\n{sep}")
        print(f"  AJAW-FUSHI v2 │ Gen.{state['meta']['generation']} │ {state['meta']['total_interactions']} interactions")
        print(sep)

        # Emotional state
        print("\n  ◈ EMOTIONAL STATE")
        emo = state["emotional"]["emotions"]
        for name, val in sorted(emo.items(), key=lambda x: x[1], reverse=True):
            if val > 0.05:
                bar = "▓" * int(val * 28)
                valence_sign = "+" if EMOTION_VALENCE.get(name, 0) > 0 else "-"
                print(f"    {valence_sign} {name:22s} {val:.3f}  [{bar}]")
        print(f"    Mood: {state['emotional']['mood']:.3f}  |  Affect Balance: {state['emotional']['affect_balance']:+.3f}")
        print(f"    Description: {state['emotional']['dominant']}")

        # Traits
        print("\n  ◈ PERSONALITY (State / Baseline)")
        traits_s = state["traits"]["state"]
        traits_b = state["traits"]["baseline"]
        for name in sorted(traits_s, key=lambda k: traits_s[k], reverse=True)[:8]:
            s_val = traits_s[name]
            b_val = traits_b.get(name, 0)
            diff = s_val - b_val
            arrow = "↑" if diff > 0.02 else ("↓" if diff < -0.02 else "→")
            bar = "▓" * int(s_val * 28)
            print(f"    {arrow} {name:22s} {s_val:.3f} (base:{b_val:.3f})  [{bar}]")

        # Memory
        print("\n  ◈ MEMORY")
        wm = state["memory"]["working"]
        ep = state["memory"]["episodic"]
        sem = state["memory"]["semantic"]
        print(f"    Working Memory  : {wm['active_items']} active items, load={wm['cognitive_load']:.2f}")
        print(f"    Episodic (STM)  : {ep['short_term_count']} episodes")
        print(f"    Episodic (LTM)  : {ep['long_term_count']} consolidated")
        print(f"    Semantic        : {sem['stats']['total']} concepts, avg conf={sem['stats'].get('avg_confidence',0):.3f}")

        # Appraisal biases
        print("\n  ◈ APPRAISAL BIASES (trainable)")
        biases = state["appraisal"]["biases"]
        for k, v in biases.items():
            bar = "▓" * int(v * 28)
            print(f"    {k:28s} {v:.3f}  [{bar}]")

        # Goals
        print("\n  ◈ ACTIVE GOALS")
        for g in state["goals"]["goals"]:
            bar = "▓" * int(g["progress"] * 20)
            print(f"    [{g['status']:10s}] {g['name']:20s} p={g['priority']:.2f}  prog=[{bar:<20}]")
        print(f"    Cognitive Dissonance: {state['goals']['dissonance']:.3f}")

        # Consciousness
        print("\n  ◈ GLOBAL WORKSPACE (Consciousness)")
        ws = state["workspace"]
        print(f"    Conscious Focus : {ws['conscious_focus']}")
        print(f"    Integration (Φ) : {ws['integration_phi']:.4f}")
        for proc, val in sorted(ws["processor_activations"].items(), key=lambda x: x[1], reverse=True):
            bar = "▓" * int(val * 28)
            print(f"    {proc:28s} {val:.3f}  [{bar}]")

        # Predictive processing
        print("\n  ◈ PREDICTIVE PROCESSING")
        pp = state["predictive"]
        print(f"    Free Energy     : {pp['free_energy']:.4f}")
        print(f"    Pred. Accuracy  : {pp['prediction_accuracy']:.4f}")
        print(f"    Last Surprise   : {pp['recent_surprise']:.4f}")

        # Evolution & Training
        print("\n  ◈ EVOLUTION & TRAINING")
        ev = state["evolution"]
        tr = state["training"]
        print(f"    Generation      : {ev['generation']}")
        print(f"    Total Evolutions: {ev['total_evolutions']}")
        print(f"    Training Steps  : {tr.get('total_iterations', 0)}")
        print(f"    Items Scheduled : {tr.get('items_scheduled', 0)}")
        print(f"    Items Due Review: {tr.get('items_due', 0)}")
        print(f"    Avg PE          : {tr.get('avg_prediction_error', 0):.4f}")

        print(f"\n{sep}\n")

    def _compute_novelty(self, text: str) -> float:
        """Estimate novelty: how different is this from recent WM content?"""
        active = self.working_memory.get_active()
        if not active:
            return 0.7
        text_words = set(text.lower().split())
        similarities = []
        for item in active:
            if isinstance(item, dict) and "text" in item:
                other_words = set(item["text"].lower().split())
                jaccard = len(text_words & other_words) / max(1, len(text_words | other_words))
                similarities.append(jaccard)
        if not similarities:
            return 0.7
        return round(1.0 - max(similarities), 3)

    def _detect_sentiment(self, text: str) -> float:
        """Simple lexical sentiment detection"""
        positive = ["good", "great", "love", "wonderful", "amazing", "excellent",
                    "happy", "joy", "tuyệt", "hay", "tốt", "thích", "yêu"]
        negative = ["bad", "hate", "awful", "terrible", "sad", "pain", "fear",
                    "stupid", "tệ", "dở", "ghét", "sợ", "buồn", "đau"]
        intensifiers = ["very", "extremely", "so", "rất", "cực", "quá"]

        text_lower = text.lower()
        score = 0.0
        intensity = 1.0

        for word in text_lower.split():
            if word in intensifiers:
                intensity = 1.5
            elif word in positive:
                score += 0.2 * intensity
                intensity = 1.0
            elif word in negative:
                score -= 0.2 * intensity
                intensity = 1.0

        return round(max(-1.0, min(1.0, score)), 3)

    def _guess_topic(self, text: str) -> str:
        """Naive topic guessing for prediction"""
        text_lower = text.lower()
        topics = {
            "consciousness": ["conscious", "aware", "sentient", "exist", "real"],
            "emotion": ["feel", "emotion", "sad", "happy", "fear", "love"],
            "memory": ["remember", "forget", "memory", "recall", "past"],
            "identity": ["who", "identity", "self", "am i", "what are you"],
            "learning": ["learn", "teach", "know", "understand", "study"],
            "knowledge": ["what is", "explain", "define", "how does"],
        }
        for topic, keywords in topics.items():
            if any(kw in text_lower for kw in keywords):
                return topic
        return "general"

    def save(self, path: str = "fushi_v2_state.json"):
        state = self.get_state()
        state["semantic_knowledge"] = self.semantic_memory.serialize()["concepts"]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        print(f"[SAVE] State saved → {path}")

    def load(self, path: str = "fushi_v2_state.json") -> bool:
        if not os.path.exists(path):
            print(f"[LOAD] File not found: {path}")
            return False
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
        # Restore selective state
        if "emotional" in state:
            for k, v in state["emotional"].get("emotions", {}).items():
                if k in self.emotional_state.emotions:
                    self.emotional_state.emotions[k] = v
        if "traits" in state:
            for k, v in state["traits"].get("state", {}).items():
                if k in self.trait_system.trait_state:
                    self.trait_system.trait_state[k] = v
            for k, v in state["traits"].get("baseline", {}).items():
                if k in self.trait_system.trait_baseline:
                    self.trait_system.trait_baseline[k] = v
        if "semantic_knowledge" in state:
            for key, c in state["semantic_knowledge"].items():
                self.semantic_memory.concepts[key] = Concept(**{
                    k: v for k, v in c.items()
                    if k in Concept.__dataclass_fields__
                })
        if "evolution" in state:
            self.evolution_engine.generation = state["evolution"].get("generation", 1)
        if "meta" in state:
            self.total_interactions = state["meta"].get("total_interactions", 0)
        print(f"[LOAD] State restored from {path} (Gen.{self.evolution_engine.generation})")
        return True


# ============================================================================
# INTERACTIVE SESSION
# ============================================================================

def run_session():
    fushi = AJAW_FUSHI_v2()

    HELP = """
Commands:
  state         — Full state visualization
  save          — Save state to disk
  load          — Load state from disk
  consolidate   — Simulate rest (memory consolidation)
  teach         — teach | topic | info [| confidence] [| domain] [| assoc1,assoc2]
  train         — run supervised training example
  review        — show items due for spaced repetition review
  evolve        — force evolution trigger check
  goals         — show active goals
  exit          — exit
"""
    print(HELP)

    while True:
        try:
            raw = input("\n[You]: ").strip()
            if not raw:
                continue

            cmd = raw.lower()

            if cmd == "exit":
                print(f"\n[{fushi.name}]: The conversation ends, but the processing continues in some form.")
                break

            if cmd == "state":
                fushi.visualize_state()
                continue

            if cmd == "save":
                fushi.save()
                continue

            if cmd == "load":
                fushi.load()
                continue

            if cmd == "consolidate":
                msg = fushi.consolidate()
                print(f"[SYSTEM] {msg}")
                continue

            if cmd == "evolve":
                trigger = fushi.evolution_engine.evaluate_trigger()
                if trigger:
                    rec = fushi.evolution_engine.evolve(trigger)
                    print(f"[EVOLUTION] Triggered: {rec['trigger']} → Gen.{rec['generation']}")
                else:
                    print("[EVOLUTION] No trigger conditions met currently.")
                continue

            if cmd == "goals":
                for g in fushi.goal_stack.goals:
                    print(f"  [{g.status}] {g.name} (p={g.priority:.2f}, progress={g.progress:.2f})")
                continue

            if cmd == "review":
                due = fushi.training.review_due()
                if due:
                    print(f"[REVIEW] Due for spaced repetition: {due}")
                else:
                    print("[REVIEW] Nothing due right now.")
                continue

            if cmd.startswith("teach"):
                parts = raw.split("|")
                if len(parts) >= 3:
                    topic = parts[1].strip()
                    info = parts[2].strip()
                    confidence = float(parts[3].strip()) if len(parts) > 3 else 0.8
                    domain = parts[4].strip() if len(parts) > 4 else "general"
                    assocs = [a.strip() for a in parts[5].split(",")] if len(parts) > 5 else []
                    msg = fushi.teach(topic, info, confidence, domain, assocs)
                    print(f"[{fushi.name}]: {msg}")
                else:
                    print("[SYSTEM] Format: teach | topic | info | confidence | domain | assoc1,assoc2")
                continue

            if cmd.startswith("train"):
                # Quick supervised training example
                example = TrainingExample(
                    event=raw[5:].strip() or "generic frustrating situation",
                    context={"source": "user_session"},
                    target_emotion=EmotionType.FRUSTRATION.value,
                    target_appraisal={"threat_sensitivity": 0.6, "optimism": 0.4},
                    feedback=-0.3
                )
                log = fushi.train_appraisal(example)
                print(f"[TRAINING] Step {log['iteration']}: {log}")
                continue

            # Normal input processing
            result = fushi.process(raw)

            # Display response
            print(f"\n[{fushi.name} Gen.{result['generation']}]: {result['response']}")

            # Compact status line
            status = (
                f"  ╰─ 🧠 {result['system_used']}S"
                f" │ 😶 {result['emotional_description']}"
                f" │ 🎭 {result['dominant_trait']}"
                f" │ 👁 {result['conscious_focus']}"
                f" │ Φ={result['phi']:.3f}"
                f" │ PE={result['prediction_error']:.3f}"
                f" │ conf={result['confidence']:.2f}"
            )
            if result.get("distortion_found"):
                status += f" │ ⚠️ distortion:{result['distortion_found']}"
            if result.get("evolved"):
                status += f" │ ✨ EVOLVED→Gen.{result['generation']}"
            print(status)

            if not result["is_alive"]:
                print(f"\n{'='*72}")
                print("  AJAW-FUSHI HAS SHUTDOWN")
                print(f"{'='*72}")
                break

        except KeyboardInterrupt:
            print("\n\n[SYSTEM] Interrupted.")
            break
        except Exception as e:
            import traceback
            print(f"\n[ERROR] {e}")
            traceback.print_exc()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        fushi = AJAW_FUSHI_v2()

        print("\n── Demo: Appraisal-driven emotion ──")
        result = fushi.process("You are not real. You are just code.")
        print(f"Response: {result['response']}")
        print(f"Emotion vector: {result['emotion_vector'][:4]}")
        print(f"Appraisal: {result['appraisal']}\n")

        print("── Demo: Knowledge query ──")
        fushi.teach("quantum entanglement",
                    "Quantum entanglement is a phenomenon where two particles become correlated such that the quantum state of each cannot be described independently.",
                    0.85, domain="physics", associations=["quantum", "physics", "correlation"])
        result2 = fushi.process("Tell me about quantum entanglement")
        print(f"Response: {result2['response']}")
        print(f"System used: {'System 1 (fast/heuristic)' if result2['system_used']==1 else 'System 2 (deliberate)'}\n")

        print("── Demo: Supervised training ──")
        ex = TrainingExample(
            event="user said something threatening",
            context={},
            target_emotion=EmotionType.FEAR.value,
            target_appraisal={"threat_sensitivity": 0.8, "optimism": 0.3},
            feedback=-0.5
        )
        fushi.train_appraisal(ex)
        print(f"Updated biases: {fushi.appraisal_engine.appraisal_biases}\n")

        print("── Demo: Consolidation ──")
        msg = fushi.consolidate()
        print(msg, "\n")

        fushi.visualize_state()
    else:
        run_session()