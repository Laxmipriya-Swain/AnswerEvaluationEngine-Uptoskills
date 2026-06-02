"""
Answer Evaluation Engine - Predictor
Consolidated module for concept extraction, scoring, explanation, and evaluation.
All-in-one predictor combining concept extraction, scoring, and feedback generation.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import json
import re
from collections import Counter
import math

# ============================================================================
# CONFIGURATION
# ============================================================================

SCORING_WEIGHTS = {
    "accuracy": 0.4,
    "completeness": 0.3,
    "clarity": 0.2,
    "relevance": 0.1
}

SCORING_CONFIG = {
    "max_score": 10,
    "min_score": 0,
    "score_levels": {
        "excellent": (9, 10),
        "good": (7, 8),
        "fair": (5, 6),
        "poor": (0, 4),
    }
}

EXPLANATION_TEMPLATES = {
    "excellent": "Excellent coverage of {concept}. All key aspects addressed.",
    "good": "Good understanding of {concept}. Most aspects covered.",
    "fair": "Partial coverage of {concept}. Some aspects missing.",
    "poor": "Limited coverage of {concept}. Needs significant improvement.",
    "missing": "Concept '{concept}' was not mentioned in the answer.",
}

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ConceptScore:
    """Represents a scored concept."""
    concept: str
    score: float
    max_score: float
    accuracy: float
    completeness: float
    clarity: float
    relevance: float
    evidence: List[str]
    confidence: float


@dataclass
class EvaluationResult:
    """Structured evaluation result."""
    total_score: float
    max_score: float
    concept_scores: List[Dict]
    feedback: str
    detailed_feedback: Dict
    coverage_ratio: float
    missing_concepts: List[str]
    extra_concepts: List[str]
    evaluation_metadata: Dict


# ============================================================================
# CONCEPT EXTRACTOR
# ============================================================================

class ConceptExtractor:
    """Extracts key concepts from text using NLP techniques."""
    
    def __init__(self, max_concepts: int = 10, similarity_threshold: float = 0.7):
        """Initialize the concept extractor."""
        self.max_concepts = max_concepts
        self.similarity_threshold = similarity_threshold
        self.stopwords = self._load_stopwords()
    
    def _load_stopwords(self) -> set:
        """Load common English stopwords."""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'could', 'may', 'might', 'can', 'shall', 'what', 'which', 'who', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
            'same', 'so', 'than', 'too', 'very', 'just', 'if', 'because'
        }
    
    def extract_concepts(self, text: str) -> List[Dict]:
        """Extract key concepts from text."""
        if not text or not text.strip():
            return []
        
        # Use multiple extraction techniques
        noun_phrases = self._extract_noun_phrases(text)
        named_entities = self._extract_entities(text)
        technical_terms = self._extract_technical_terms(text)
        
        # Combine all concepts
        all_concepts = noun_phrases + named_entities + technical_terms
        
        # Deduplicate and score concepts
        deduplicated = self._deduplicate_concepts(all_concepts)
        
        # Sort by importance and limit
        ranked = sorted(deduplicated, key=lambda x: x['importance'], reverse=True)
        return ranked[:self.max_concepts]
    
    def _extract_noun_phrases(self, text: str) -> List[Dict]:
        """Extract noun phrases and key terms."""
        patterns = [
            r'\b(?:[A-Z][a-z]+\s+){1,3}(?:[A-Z][a-z]+)\b',
            r'\b(?:[a-z]+\s+){1,2}(?:process|system|method|concept|theory|model)\b',
        ]
        
        concepts = []
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                term = match.group().strip()
                if term and len(term.split()) <= 5:
                    concepts.append({
                        'text': term,
                        'type': 'noun_phrase',
                        'importance': self._calculate_importance(term, text)
                    })
        
        concepts.extend(self._extract_ngrams(text))
        return concepts
    
    def _extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities and proper nouns."""
        pattern = r'\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        
        concepts = []
        seen = set()
        
        for match in re.finditer(pattern, text):
            term = match.group().strip()
            if term not in seen and term.lower() not in self.stopwords:
                seen.add(term)
                concepts.append({
                    'text': term,
                    'type': 'entity',
                    'importance': self._calculate_importance(term, text)
                })
        
        return concepts
    
    def _extract_technical_terms(self, text: str) -> List[Dict]:
        """Extract technical or domain-specific terms."""
        pattern = r'\b[a-z]+(?:-[a-z]+)+\b|\b[a-z]{8,}\b'
        
        concepts = []
        for match in re.finditer(pattern, text, re.IGNORECASE):
            term = match.group().strip().lower()
            if term not in self.stopwords and len(term) > 4:
                concepts.append({
                    'text': term,
                    'type': 'technical_term',
                    'importance': self._calculate_importance(term, text)
                })
        
        return concepts
    
    def _extract_ngrams(self, text: str, n: int = 2) -> List[Dict]:
        """Extract n-grams that are significant."""
        words = text.lower().split()
        words = [w.strip('.,!?;:') for w in words if w.strip('.,!?;:') not in self.stopwords]
        
        concepts = []
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            if ngram not in self.stopwords:
                concepts.append({
                    'text': ngram,
                    'type': 'ngram',
                    'importance': self._calculate_importance(ngram, text)
                })
        
        return concepts
    
    def _calculate_importance(self, term: str, text: str) -> float:
        """Calculate importance score for a term (0-1)."""
        text_lower = text.lower()
        frequency = text_lower.count(term.lower())
        
        # Frequency component
        freq_score = min(frequency / 5, 1.0)
        
        # Length component
        length_score = min(len(term.split()) / 3, 1.0)
        
        # Uniqueness component
        word_count = len(text_lower.split())
        uniqueness_score = 1 - (frequency / max(word_count, 1))
        
        importance = (freq_score * 0.4 + length_score * 0.3 + uniqueness_score * 0.3)
        return min(importance, 1.0)
    
    def _deduplicate_concepts(self, concepts: List[Dict]) -> List[Dict]:
        """Remove duplicate or highly similar concepts."""
        if not concepts:
            return []
        
        deduplicated = []
        seen_texts = set()
        
        for concept in concepts:
            text = concept['text'].lower()
            
            if text in seen_texts:
                continue
            
            is_similar = False
            for existing in deduplicated:
                if self._text_similarity(text, existing['text'].lower()) > self.similarity_threshold:
                    if concept['importance'] > existing['importance']:
                        deduplicated.remove(existing)
                        seen_texts.discard(existing['text'].lower())
                    else:
                        is_similar = True
                        break
            
            if not is_similar:
                deduplicated.append(concept)
                seen_texts.add(text)
        
        return deduplicated
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between two texts."""
        set1 = set(text1.split())
        set2 = set(text2.split())
        
        if not set1 or not set2:
            return 1.0 if text1 == text2 else 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def compare_answer_to_question(self, question: str, answer: str) -> Dict:
        """Extract and compare concepts from question and answer."""
        question_concepts = self.extract_concepts(question)
        answer_concepts = self.extract_concepts(answer)
        
        covered_concepts = []
        missing_concepts = []
        extra_concepts = []
        
        question_texts = {c['text'].lower() for c in question_concepts}
        answer_texts = {c['text'].lower() for c in answer_concepts}
        
        for qc in question_concepts:
            qt_lower = qc['text'].lower()
            is_covered = any(
                self._text_similarity(qt_lower, ac.lower()) > 0.6
                for ac in answer_texts
            )
            if is_covered:
                covered_concepts.append(qc)
            else:
                missing_concepts.append(qc)
        
        for ac in answer_concepts:
            if not any(self._text_similarity(ac['text'].lower(), qt) > 0.6 for qt in question_texts):
                extra_concepts.append(ac)
        
        return {
            'question_concepts': question_concepts,
            'answer_concepts': answer_concepts,
            'covered_concepts': covered_concepts,
            'missing_concepts': missing_concepts,
            'extra_concepts': extra_concepts,
            'coverage_ratio': len(covered_concepts) / len(question_concepts) if question_concepts else 1.0
        }


# ============================================================================
# CONCEPT SCORER
# ============================================================================

class ConceptScorer:
    """Scores individual concepts across multiple dimensions."""
    
    def __init__(self, weights: Dict[str, float] = None):
        """Initialize scorer with optional custom weights."""
        self.weights = weights or SCORING_WEIGHTS.copy()
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Ensure weights sum to 1.0."""
        total = sum(self.weights.values())
        if total > 0:
            for key in self.weights:
                self.weights[key] /= total
    
    def score_concept(self, concept: str, answer_text: str, expected_context: str = "") -> ConceptScore:
        """Score a single concept in the answer."""
        accuracy = self._score_accuracy(concept, answer_text, expected_context)
        completeness = self._score_completeness(concept, answer_text)
        clarity = self._score_clarity(concept, answer_text)
        relevance = self._score_relevance(concept, answer_text)
        
        # Calculate weighted score
        weighted_score = (
            accuracy * self.weights.get('accuracy', 0.4) +
            completeness * self.weights.get('completeness', 0.3) +
            clarity * self.weights.get('clarity', 0.2) +
            relevance * self.weights.get('relevance', 0.1)
        )
        
        # Normalize to 0-10 scale
        final_score = weighted_score * SCORING_CONFIG['max_score']
        
        # Extract evidence
        evidence = self._extract_evidence(concept, answer_text)
        
        # Calculate confidence
        confidence = self._calculate_confidence(accuracy, completeness, clarity, relevance)
        
        return ConceptScore(
            concept=concept,
            score=round(final_score, 2),
            max_score=SCORING_CONFIG['max_score'],
            accuracy=round(accuracy, 3),
            completeness=round(completeness, 3),
            clarity=round(clarity, 3),
            relevance=round(relevance, 3),
            evidence=evidence,
            confidence=round(confidence, 3)
        )
    
    def score_multiple_concepts(self, concepts: List[str], answer_text: str) -> List[ConceptScore]:
        """Score multiple concepts."""
        return [self.score_concept(concept, answer_text) for concept in concepts]
    
    def _score_accuracy(self, concept: str, answer_text: str, expected_context: str) -> float:
        """Score accuracy: How correct is the information?"""
        if concept.lower() not in answer_text.lower():
            return 0.0
        
        concept_contexts = self._get_concept_context(concept, answer_text)
        
        if not concept_contexts:
            return 0.3
        
        accuracy = 0.5
        positive_indicators = ['clearly', 'definitely', 'proven', 'accurate', 'correct',
                             'verified', 'confirmed', 'established']
        negative_indicators = ['may', 'might', 'possibly', 'unclear', 'wrong', 'false',
                             'incorrect', 'disputed']
        
        for context in concept_contexts:
            context_lower = context.lower()
            pos_count = sum(1 for ind in positive_indicators if ind in context_lower)
            neg_count = sum(1 for ind in negative_indicators if ind in context_lower)
            
            if pos_count > neg_count:
                accuracy = min(1.0, accuracy + 0.15)
            elif neg_count > pos_count:
                accuracy = max(0.2, accuracy - 0.15)
            
            if len(context.split()) > 15:
                accuracy = min(1.0, accuracy + 0.1)
        
        return accuracy
    
    def _score_completeness(self, concept: str, answer_text: str) -> float:
        """Score completeness: How thoroughly is the concept addressed?"""
        if concept.lower() not in answer_text.lower():
            return 0.0
        
        concept_contexts = self._get_concept_context(concept, answer_text)
        
        if not concept_contexts:
            return 0.2
        
        completeness = 0.3
        
        for context in concept_contexts:
            words = len(context.split())
            if words > 20:
                completeness = min(1.0, completeness + 0.25)
            elif words > 10:
                completeness = min(1.0, completeness + 0.15)
            else:
                completeness = min(1.0, completeness + 0.1)
        
        if any(phrase in answer_text.lower() for phrase in ['example', 'for instance', 'such as', 'specifically']):
            completeness = min(1.0, completeness + 0.15)
        
        return completeness
    
    def _score_clarity(self, concept: str, answer_text: str) -> float:
        """Score clarity: How clearly is the concept explained?"""
        if concept.lower() not in answer_text.lower():
            return 0.0
        
        clarity = 0.5
        explanatory_phrases = ['is', 'means', 'refers to', 'defined as', 'can be understood as',
                              'in other words', 'essentially', 'basically', 'fundamentally']
        
        answer_lower = answer_text.lower()
        concept_lower = concept.lower()
        sentences = answer_text.split('.')
        concept_sentences = [s for s in sentences if concept_lower in s.lower()]
        
        if concept_sentences:
            for sentence in concept_sentences:
                sentence_lower = sentence.lower()
                has_explanation = any(phrase in sentence_lower for phrase in explanatory_phrases)
                if has_explanation:
                    clarity = min(1.0, clarity + 0.15)
                
                if len(sentence.split()) > 15:
                    clarity = min(1.0, clarity + 0.1)
        
        return clarity
    
    def _score_relevance(self, concept: str, answer_text: str) -> float:
        """Score relevance: Is the concept relevant to the answer?"""
        if concept.lower() not in answer_text.lower():
            return 0.0
        
        return 0.7  # Default relevance if concept is present
    
    def _get_concept_context(self, concept: str, text: str, context_size: int = 50) -> List[str]:
        """Extract surrounding context for a concept."""
        concept_lower = concept.lower()
        text_lower = text.lower()
        
        contexts = []
        start = 0
        while True:
            pos = text_lower.find(concept_lower, start)
            if pos == -1:
                break
            
            context_start = max(0, pos - context_size)
            context_end = min(len(text), pos + len(concept) + context_size)
            contexts.append(text[context_start:context_end])
            start = pos + 1
        
        return contexts
    
    def _extract_evidence(self, concept: str, answer_text: str) -> List[str]:
        """Extract evidence excerpts mentioning the concept."""
        sentences = answer_text.split('.')
        evidence = []
        
        for sentence in sentences:
            if concept.lower() in sentence.lower():
                cleaned = sentence.strip()
                if cleaned:
                    evidence.append(cleaned[:100])
        
        return evidence[:3]
    
    def _calculate_confidence(self, accuracy: float, completeness: float, 
                             clarity: float, relevance: float) -> float:
        """Calculate overall confidence score."""
        avg_score = (accuracy + completeness + clarity + relevance) / 4
        
        # High variance suggests low confidence
        variance = sum((s - avg_score) ** 2 for s in [accuracy, completeness, clarity, relevance]) / 4
        confidence_penalty = variance * 0.2
        
        return max(0.0, avg_score - confidence_penalty)


# ============================================================================
# SCORING EXPLAINER
# ============================================================================

class ScoringExplainer:
    """Generates human-readable explanations for concept scores."""
    
    def __init__(self):
        """Initialize the explainer."""
        self.templates = EXPLANATION_TEMPLATES
        self.score_levels = SCORING_CONFIG['score_levels']
    
    def explain_concept_score(self, concept_score: ConceptScore) -> str:
        """Generate explanation for a concept score."""
        level = self._get_score_level(concept_score.score)
        template = self.templates.get(level, "")
        explanation = template.format(concept=concept_score.concept)
        
        breakdown = self._create_score_breakdown(concept_score)
        explanation += f"\n{breakdown}"
        
        if concept_score.evidence:
            evidence_text = self._format_evidence(concept_score.evidence)
            explanation += f"\nEvidence: {evidence_text}"
        
        return explanation
    
    def explain_evaluation_result(self, total_score: float, concept_scores: List[ConceptScore],
                                 missing_concepts: List[Dict] = None) -> str:
        """Generate comprehensive explanation for entire evaluation."""
        explanation_parts = []
        
        explanation_parts.append(self._overall_assessment(total_score))
        
        strengths = self._identify_strengths(concept_scores)
        if strengths:
            explanation_parts.append(f"\n✓ Strengths:\n{strengths}")
        
        improvements = self._identify_improvements(concept_scores)
        if improvements:
            explanation_parts.append(f"\n✗ Areas for Improvement:\n{improvements}")
        
        if missing_concepts:
            missing_text = self._format_missing_concepts(missing_concepts)
            explanation_parts.append(f"\n⚠ Missing Concepts:\n{missing_text}")
        
        recommendations = self._generate_recommendations(concept_scores, missing_concepts)
        if recommendations:
            explanation_parts.append(f"\n💡 Recommendations:\n{recommendations}")
        
        return "".join(explanation_parts)
    
    def create_feedback_json(self, total_score: float, concept_scores: List[ConceptScore],
                            missing_concepts: List[Dict]) -> Dict:
        """Create structured feedback as JSON."""
        return {
            'total_score': total_score,
            'assessment': self._overall_assessment(total_score),
            'strengths': [cs.concept for cs in concept_scores if cs.score >= 7],
            'improvements': [cs.concept for cs in concept_scores if cs.score < 7],
            'missing_concepts': [c['text'] for c in missing_concepts],
            'recommendations': self._generate_recommendations(concept_scores, missing_concepts)
        }
    
    def _get_score_level(self, score: float) -> str:
        """Determine score level."""
        for level, (min_s, max_s) in self.score_levels.items():
            if min_s <= score <= max_s:
                return level
        return "poor"
    
    def _overall_assessment(self, score: float) -> str:
        """Generate overall assessment."""
        level = self._get_score_level(score)
        assessments = {
            'excellent': f"Outstanding response! Score: {score:.1f}/10",
            'good': f"Strong response with good coverage. Score: {score:.1f}/10",
            'fair': f"Adequate response with room for improvement. Score: {score:.1f}/10",
            'poor': f"Response needs significant improvement. Score: {score:.1f}/10"
        }
        return assessments.get(level, f"Score: {score:.1f}/10")
    
    def _create_score_breakdown(self, concept_score: ConceptScore) -> str:
        """Create detailed breakdown of scoring dimensions."""
        return f"""  Score Breakdown:
    • Accuracy:     {concept_score.accuracy:.1%}
    • Completeness: {concept_score.completeness:.1%}
    • Clarity:      {concept_score.clarity:.1%}
    • Relevance:    {concept_score.relevance:.1%}
    • Final Score:  {concept_score.score:.1f}/{concept_score.max_score}"""
    
    def _format_evidence(self, evidence: List[str]) -> str:
        """Format evidence excerpts."""
        if not evidence:
            return "No direct evidence found"
        
        formatted = []
        for i, excerpt in enumerate(evidence, 1):
            if len(excerpt) > 100:
                excerpt = excerpt[:100] + "..."
            formatted.append(f"  {i}. '{excerpt}'")
        
        return "\n".join(formatted)
    
    def _identify_strengths(self, concept_scores: List[ConceptScore]) -> str:
        """Identify strong concepts."""
        strong_concepts = [cs for cs in concept_scores if cs.score >= 7]
        
        if not strong_concepts:
            return ""
        
        lines = []
        for cs in sorted(strong_concepts, key=lambda x: x.score, reverse=True)[:3]:
            lines.append(f"  • '{cs.concept}': {cs.score:.1f}/10")
        
        return "\n".join(lines)
    
    def _identify_improvements(self, concept_scores: List[ConceptScore]) -> str:
        """Identify concepts needing improvement."""
        weak_concepts = [cs for cs in concept_scores if cs.score < 7]
        
        if not weak_concepts:
            return ""
        
        lines = []
        for cs in sorted(weak_concepts, key=lambda x: x.score):
            lines.append(f"  • '{cs.concept}': {cs.score:.1f}/10")
        
        return "\n".join(lines)
    
    def _format_missing_concepts(self, missing_concepts: List[Dict]) -> str:
        """Format missing concepts."""
        if not missing_concepts:
            return ""
        
        lines = []
        for concept in missing_concepts[:5]:
            lines.append(f"  • {concept['text']}")
        
        return "\n".join(lines)
    
    def _generate_recommendations(self, concept_scores: List[ConceptScore], 
                                 missing_concepts: List[Dict]) -> str:
        """Generate recommendations."""
        recommendations = []
        
        weak_concepts = [cs for cs in concept_scores if cs.score < 7]
        if weak_concepts:
            weak_text = ", ".join([cs.concept for cs in weak_concepts[:2]])
            recommendations.append(f"  • Focus more on: {weak_text}")
        
        if missing_concepts:
            recommendations.append("  • Include all key concepts from the question")
        
        return "\n".join(recommendations) if recommendations else ""


# ============================================================================
# MAIN EVALUATOR
# ============================================================================

class AnswerPredictor:
    """Main evaluation engine for scoring answers with decomposed concepts."""
    
    def __init__(self, scoring_weights: Dict[str, float] = None, 
                 max_concepts: int = 10, similarity_threshold: float = 0.7):
        """Initialize the predictor."""
        self.concept_extractor = ConceptExtractor(
            max_concepts=max_concepts,
            similarity_threshold=similarity_threshold
        )
        self.scorer = ConceptScorer(weights=scoring_weights or SCORING_WEIGHTS)
        self.explainer = ScoringExplainer()
    
    def evaluate(self, question: str, answer: str) -> EvaluationResult:
        """Evaluate an answer against a question."""
        if not question or not answer:
            raise ValueError("Both question and answer must be non-empty strings")
        
        # Step 1: Extract and compare concepts
        comparison = self.concept_extractor.compare_answer_to_question(question, answer)
        
        covered_concepts = comparison['covered_concepts']
        missing_concepts = comparison['missing_concepts']
        extra_concepts = comparison['extra_concepts']
        
        # Step 2: Score covered concepts
        concept_scores_list = []
        if covered_concepts:
            concept_scores_list = self.scorer.score_multiple_concepts(
                [c['text'] for c in covered_concepts],
                answer
            )
        
        # Step 3: Calculate aggregate scores
        total_score, max_score = self._calculate_total_score(concept_scores_list)
        
        # Step 4: Format concept scores
        concept_scores_output = [self._format_concept_score(cs) for cs in concept_scores_list]
        
        # Step 5: Generate explanations
        feedback = self.explainer.explain_evaluation_result(
            total_score,
            concept_scores_list,
            missing_concepts
        )
        
        detailed_feedback = self.explainer.create_feedback_json(
            total_score,
            concept_scores_list,
            missing_concepts
        )
        
        # Step 6: Compile metadata
        metadata = {
            'total_concepts_in_question': len(comparison['question_concepts']),
            'concepts_addressed': len(covered_concepts),
            'concepts_missed': len(missing_concepts),
            'extra_concepts_added': len(extra_concepts),
            'answer_length': len(answer.split()),
            'average_concept_score': (
                sum(cs.score for cs in concept_scores_list) / len(concept_scores_list)
                if concept_scores_list else 0
            )
        }
        
        # Step 7: Create result
        result = EvaluationResult(
            total_score=round(total_score, 2),
            max_score=max_score,
            concept_scores=concept_scores_output,
            feedback=feedback,
            detailed_feedback=detailed_feedback,
            coverage_ratio=comparison['coverage_ratio'],
            missing_concepts=[c['text'] for c in missing_concepts],
            extra_concepts=[c['text'] for c in extra_concepts],
            evaluation_metadata=metadata
        )
        
        return result
    
    def evaluate_batch(self, questions: List[str], answers: List[str]) -> List[EvaluationResult]:
        """Evaluate multiple question-answer pairs."""
        if len(questions) != len(answers):
            raise ValueError("Number of questions must match number of answers")
        
        results = []
        for question, answer in zip(questions, answers):
            try:
                result = self.evaluate(question, answer)
                results.append(result)
            except Exception as e:
                print(f"Error evaluating question: {str(e)}")
                results.append(None)
        
        return results
    
    def extract_concepts_only(self, text: str) -> List[Dict]:
        """Extract concepts from text."""
        return self.concept_extractor.extract_concepts(text)
    
    def compare_texts(self, question: str, answer: str) -> Dict:
        """Compare question and answer texts."""
        return self.concept_extractor.compare_answer_to_question(question, answer)
    
    def set_weights(self, weights: Dict[str, float]):
        """Update scoring weights."""
        self.scorer = ConceptScorer(weights=weights)
    
    def _calculate_total_score(self, concept_scores: List[ConceptScore]) -> Tuple[float, float]:
        """Calculate total score from concept scores."""
        if not concept_scores:
            return 0.0, 10.0
        
        avg_concept_score = sum(cs.score for cs in concept_scores) / len(concept_scores)
        return avg_concept_score, 10.0
    
    def _format_concept_score(self, concept_score: ConceptScore) -> Dict:
        """Convert ConceptScore to dictionary."""
        return {
            "concept": concept_score.concept,
            "score": concept_score.score,
            "max_score": concept_score.max_score,
            "accuracy": concept_score.accuracy,
            "completeness": concept_score.completeness,
            "clarity": concept_score.clarity,
            "relevance": concept_score.relevance,
            "confidence": concept_score.confidence,
            "evidence_count": len(concept_score.evidence)
        }
    
    def to_json(self, result: EvaluationResult) -> str:
        """Convert evaluation result to JSON."""
        result_dict = {
            "total_score": result.total_score,
            "max_score": result.max_score,
            "coverage_ratio": round(result.coverage_ratio, 3),
            "concept_scores": result.concept_scores,
            "missing_concepts": result.missing_concepts,
            "extra_concepts": result.extra_concepts,
            "feedback": result.feedback,
            "metadata": result.evaluation_metadata
        }
        return json.dumps(result_dict, indent=2)
