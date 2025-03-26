import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np
from collections import Counter

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("⚠️ NLTK resource download failed. Some metrics may not work properly.")

# Keywords that indicate specific actionable feedback
ACTIONABLE_KEYWORDS = [
    "should", "could", "try", "consider", "use", "implement", "create", 
    "move", "rename", "extract", "separate", "refactor", "improve",
    "add", "remove", "replace", "modify", "update", "fix"
]

# Specific SOLID principle keywords for accuracy assessment
SOLID_PRINCIPLE_KEYWORDS = {
    "SRP": ["single responsibility", "responsibility", "cohesion", "one reason to change", 
            "multiple responsibilities", "too many methods"],
    "OCP": ["open closed", "extension", "modification", "extend", "modify", "inherit", "override"],
    "LSP": ["liskov", "substitution", "substitute", "inheritance", "polymorphism", "override", "behavior"],
    "ISP": ["interface segregation", "fat interface", "client-specific", "interface pollution", 
            "unnecessary dependency", "method stub"],
    "DIP": ["dependency inversion", "dependency injection", "abstraction", "concrete", "implementation",
            "high-level", "low-level", "constructor injection", "new operator"]
}

class FeedbackMetrics:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def evaluate_feedback(self, feedback_text, detected_violations=None, professor_feedback=None):
        """
        Evaluate feedback quality using multiple metrics.
        
        Args:
            feedback_text: The AI-generated feedback text
            detected_violations: List of detected violations from code analysis
            professor_feedback: Optional reference feedback from a professor
            
        Returns:
            Dictionary of quality metrics with scores
        """
        if not feedback_text:
            return {
                "specificity": 0,
                "actionability": 0,
                "accuracy": 0,
                "completeness": 0,
                "readability": 0,
                "overall_score": 0,
                "suggestions": ["Feedback text is empty or missing."]
            }
        
        # Calculate individual metrics
        specificity_score = self.measure_specificity(feedback_text)
        actionability_score = self.measure_actionability(feedback_text)
        accuracy_score = self.measure_accuracy(feedback_text, detected_violations)
        completeness_score = self.measure_completeness(feedback_text, detected_violations)
        readability_score = self.measure_readability(feedback_text)
        
        # Compare with professor feedback if available
        alignment_score = self.measure_professor_alignment(feedback_text, professor_feedback) if professor_feedback else None
        
        # Calculate overall score
        metrics = [specificity_score, actionability_score, accuracy_score, completeness_score, readability_score]
        if alignment_score is not None:
            metrics.append(alignment_score)
        
        overall_score = np.mean(metrics)
        
        # Generate improvement suggestions
        suggestions = self.generate_improvement_suggestions(
            specificity_score, 
            actionability_score, 
            accuracy_score, 
            completeness_score, 
            readability_score
        )
        
        return {
            "specificity": round(specificity_score, 2),
            "actionability": round(actionability_score, 2),
            "accuracy": round(accuracy_score, 2),
            "completeness": round(completeness_score, 2),
            "readability": round(readability_score, 2),
            "professor_alignment": round(alignment_score, 2) if alignment_score is not None else None,
            "overall_score": round(overall_score, 2),
            "suggestions": suggestions
        }
    
    def measure_specificity(self, text):
        """Measure how specific the feedback is (mentions specific code elements and locations)."""
        sentences = sent_tokenize(text)
        
        # Count sentences with specific code references
        specificity_indicators = 0
        total_sentences = len(sentences)
        
        # Regular expressions for detecting specificity
        class_reference = re.compile(r'\b[A-Z][a-zA-Z0-9]*\b')  # Class names
        method_reference = re.compile(r'\b[a-z][a-zA-Z0-9]*\([^)]*\)')  # Method calls
        file_reference = re.compile(r'\b[A-Za-z0-9]+\.java\b')  # Java files
        line_reference = re.compile(r'line\s+\d+|method\s+[a-zA-Z0-9_]+')  # Line number references
        
        for sentence in sentences:
            if (class_reference.search(sentence) or 
                method_reference.search(sentence) or 
                file_reference.search(sentence) or 
                line_reference.search(sentence)):
                specificity_indicators += 1
        
        # Include specific methods mentioned
        methods_mentioned = len(re.findall(r'\b[a-z][a-zA-Z0-9]*\(', text))
        
        # Calculate specificity score (0-1)
        if total_sentences == 0:
            return 0
        
        base_score = specificity_indicators / total_sentences
        method_bonus = min(methods_mentioned / 5, 0.3)  # Cap the bonus at 0.3
        
        return min(base_score + method_bonus, 1.0)
    
    def measure_actionability(self, text):
        """Measure how actionable the feedback is (provides concrete suggestions)."""
        sentences = sent_tokenize(text)
        total_sentences = len(sentences)
        
        if total_sentences == 0:
            return 0
        
        # Count actionable sentences (containing action verbs)
        actionable_sentences = 0
        for sentence in sentences:
            words = sentence.lower().split()
            if any(keyword in words for keyword in ACTIONABLE_KEYWORDS):
                actionable_sentences += 1
        
        # Count code examples or patterns
        code_examples = len(re.findall(r'```java|`[^`]+`', text))
        
        # Calculate actionability score (0-1)
        base_score = actionable_sentences / total_sentences
        example_bonus = min(code_examples * 0.1, 0.3)  # Cap the bonus at 0.3
        
        return min(base_score + example_bonus, 1.0)
    
    def measure_accuracy(self, text, detected_violations=None):
        """Measure accuracy based on SOLID principle coverage and alignment with detected violations."""
        # Count mentions of SOLID principles
        principle_mentions = {principle: 0 for principle in SOLID_PRINCIPLE_KEYWORDS}
        
        for principle, keywords in SOLID_PRINCIPLE_KEYWORDS.items():
            for keyword in keywords:
                principle_mentions[principle] += len(re.findall(r'\b' + re.escape(keyword) + r'\b', text.lower()))
        
        # Calculate principle coverage score
        total_mentions = sum(principle_mentions.values())
        solid_coverage = min(total_mentions / 10, 1.0)  # Cap at 1.0, expect at least 10 mentions for full score
        
        # Check alignment with detected violations if provided
        violation_alignment = 0.5  # Default middle score if no violations provided
        if detected_violations:
            # Get principles from detected violations
            detected_principles = [v["principle"] for v in detected_violations if "principle" in v]
            detected_principles = [p for p in detected_principles if p in SOLID_PRINCIPLE_KEYWORDS]
            
            # Calculate overlap with mentioned principles
            mentioned_principles = [p for p, count in principle_mentions.items() if count > 0]
            
            if detected_principles:
                # Calculate Jaccard similarity
                intersection = set(detected_principles) & set(mentioned_principles)
                union = set(detected_principles) | set(mentioned_principles)
                
                if union:
                    violation_alignment = len(intersection) / len(union)
                else:
                    violation_alignment = 0
        
        # Final accuracy score is a weighted combination
        return 0.7 * solid_coverage + 0.3 * violation_alignment
    
    def measure_completeness(self, text, detected_violations=None):
        """Measure completeness of the feedback based on sections and violation coverage."""
        # Check for key sections
        has_assessment = bool(re.search(r'overall assessment|assessment|evaluation', text.lower()))
        has_violations = bool(re.search(r'violations|issues|problems', text.lower()))
        has_suggestions = bool(re.search(r'suggestions|recommendations|improvements', text.lower()))
        
        # Calculate section completeness score
        sections_score = (has_assessment + has_violations + has_suggestions) / 3
        
        # Check violation coverage if violations provided
        violation_coverage = 0.5  # Default middle score
        if detected_violations:
            # Count mentioned class/method names from violations
            violation_elements = []
            for v in detected_violations:
                if "location" in v:
                    # Extract class names from location field
                    class_matches = re.findall(r'\b[A-Z][a-zA-Z0-9]*\b', v["location"])
                    violation_elements.extend(class_matches)
            
            # Count how many of these elements are mentioned in the feedback
            mentioned_count = 0
            for element in violation_elements:
                if re.search(r'\b' + re.escape(element) + r'\b', text):
                    mentioned_count += 1
            
            # Calculate violation coverage
            if violation_elements:
                violation_coverage = mentioned_count / len(violation_elements)
            else:
                violation_coverage = 1.0  # No violations to cover
        
        # Final completeness score is a weighted combination
        return 0.6 * sections_score + 0.4 * violation_coverage
    
    def measure_readability(self, text):
        """Measure readability based on sentence length and structure."""
        sentences = sent_tokenize(text)
        
        if not sentences:
            return 0
        
        # Calculate average sentence length
        sentence_lengths = [len(sentence.split()) for sentence in sentences]
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        
        # Ideal sentence length is between 10-20 words
        length_score = 1.0 - min(abs(avg_sentence_length - 15) / 15, 1.0)
        
        # Check for formatting elements
        has_headings = bool(re.search(r'#{1,3}\s+\w+|[A-Z][A-Za-z ]+:\s*$', text, re.MULTILINE))
        has_lists = bool(re.search(r'^\s*[*-]\s+\w+', text, re.MULTILINE))
        has_paragraphs = text.count('\n\n') > 0
        
        formatting_score = (has_headings + has_lists + has_paragraphs) / 3
        
        # Final readability score
        return 0.5 * length_score + 0.5 * formatting_score
    
    def measure_professor_alignment(self, ai_feedback, professor_feedback):
        """Measure alignment with professor's feedback if available."""
        if not professor_feedback:
            return None
        
        # Extract text from professor feedback if it's a dict
        if isinstance(professor_feedback, dict):
            prof_text = ""
            for key, value in professor_feedback.items():
                if value:
                    prof_text += f"{value} "
        else:
            prof_text = professor_feedback
        
        # Extract key terms from both feedbacks
        ai_terms = self._extract_key_terms(ai_feedback)
        prof_terms = self._extract_key_terms(prof_text)
        
        # Calculate term overlap (Jaccard similarity)
        if not ai_terms or not prof_terms:
            return 0.5  # Default middle score
        
        intersection = ai_terms.intersection(prof_terms)
        union = ai_terms.union(prof_terms)
        
        similarity = len(intersection) / len(union) if union else 0
        
        # Check if SOLID principles mentioned in professor feedback are also in AI feedback
        prof_principles = set()
        ai_principles = set()
        
        for principle, keywords in SOLID_PRINCIPLE_KEYWORDS.items():
            if any(keyword in prof_text.lower() for keyword in keywords):
                prof_principles.add(principle)
            if any(keyword in ai_feedback.lower() for keyword in keywords):
                ai_principles.add(principle)
        
        principle_overlap = 0
        if prof_principles:
            principle_overlap = len(prof_principles.intersection(ai_principles)) / len(prof_principles)
        
        # Final alignment score is a weighted combination
        return 0.6 * similarity + 0.4 * principle_overlap
    
    def _extract_key_terms(self, text):
        """Extract key terms from text for comparison."""
        # Tokenize and remove stop words
        words = nltk.word_tokenize(text.lower())
        words = [word for word in words if word.isalnum() and word not in self.stop_words]
        
        # Get key terms (nouns, verbs, adjectives are most important)
        # For simplicity, we're using frequency here, but could be improved with POS tagging
        counter = Counter(words)
        key_terms = set([word for word, count in counter.most_common(30)])
        
        return key_terms
    
    def generate_improvement_suggestions(self, specificity, actionability, accuracy, completeness, readability):
        """Generate suggestions to improve the feedback based on metric scores."""
        suggestions = []
        
        if specificity < 0.7:
            suggestions.append("Add more references to specific classes, methods, and code locations.")
        
        if actionability < 0.7:
            suggestions.append("Include more concrete suggestions using action verbs like 'refactor', 'extract', or 'implement'.")
        
        if accuracy < 0.7:
            suggestions.append("Improve accuracy by explaining SOLID principles more specifically and relating them to the detected issues.")
        
        if completeness < 0.7:
            suggestions.append("Ensure feedback includes an overall assessment, specific violations, and improvement suggestions.")
        
        if readability < 0.7:
            suggestions.append("Improve readability by using headings, bullet points, and keeping sentences concise.")
        
        if not suggestions:
            suggestions.append("Feedback quality is good across all metrics.")
        
        return suggestions

# Example usage
if __name__ == "__main__":
    metrics = FeedbackMetrics()
    
    # Test feedback
    test_feedback = """
    ## Overall Assessment
    The code violates the Single Responsibility Principle as the User class has too many responsibilities.
    
    ## SOLID Violations
    1. SRP - User.java: This class has 10 methods handling different concerns.
       - Consider extracting the calculation methods to a separate class.
    2. DIP - UserService.java: The class directly instantiates PostgresDriver.
       - Use constructor dependency injection instead.
    
    ## Improvement Suggestions
    - Extract calculation logic to a UserCalculator class
    - Implement dependency injection in UserService
    """
    
    # Sample detected violations
    detected_violations = [
        {"principle": "SRP", "location": "User.java", "description": "Class has 10 methods, suggesting too many responsibilities"},
        {"principle": "DIP", "location": "UserService.java", "description": "Directly instantiates PostgresDriver"}
    ]
    
    results = metrics.evaluate_feedback(test_feedback, detected_violations)
    print(json.dumps(results, indent=2))