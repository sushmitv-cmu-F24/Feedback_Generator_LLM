import re
import nltk
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize, sent_tokenize
from rouge import Rouge
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    print("⚠️ NLTK resource download failed. Some metrics may not work properly.")

class FeedbackEvaluation:
    def __init__(self):
        """Initialize evaluation resources"""
        self.rouge = Rouge()
        # Use a context manager for PyTorch operations
        with torch.no_grad():
            # Initialize with smaller batch size and CPU only
            self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('bert-base-uncased')
            
            # Force model to CPU mode to avoid CUDA issues
            self.bert_model.cpu()
            torch.set_num_threads(1)  # Limit the number of threads
        
    def calculate_rouge_scores(self, generated_feedback, reference_feedback):
        """
        Calculate ROUGE scores between generated and reference feedback.
        
        Args:
            generated_feedback (str): The AI-generated feedback text
            reference_feedback (str): The instructor's reference feedback text
            
        Returns:
            dict: Dictionary containing ROUGE-1, ROUGE-2, and ROUGE-L scores
        """
        try:
            # Ensure inputs are strings and not empty
            if not generated_feedback or not reference_feedback:
                print("Empty feedback text detected in ROUGE calculation")
                return {
                    'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                    'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                    'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
                }
                
            # Process the feedback texts to ensure they're in the right format
            generated_text = ' '.join(generated_feedback.split())
            reference_text = ' '.join(reference_feedback.split())
            
            # Check minimum length requirement - Rouge expects at least one word
            if len(generated_text.strip()) == 0 or len(reference_text.strip()) == 0:
                print("Empty processed text in ROUGE calculation")
                return {
                    'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                    'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                    'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
                }
            
            # Calculate ROUGE scores
            print(f"Calculating ROUGE with texts of length {len(generated_text)} and {len(reference_text)}")
            scores = self.rouge.get_scores(generated_text, reference_text)[0]
            return scores
        except Exception as e:
            print(f"⚠️ Error calculating ROUGE scores: {e}")
            import traceback
            traceback.print_exc()
            return {
                'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
                'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
            }
    
    def calculate_bert_score(self, generated_feedback, reference_feedback):
        """
        Calculate BERTScore between generated and reference feedback.
        
        Args:
            generated_feedback (str): The AI-generated feedback text
            reference_feedback (str): The instructor's reference feedback text
            
        Returns:
            float: BERTScore (F1) between the generated and reference feedback
        """
        try:
            # Ensure inputs are strings and not empty
            if not generated_feedback or not reference_feedback:
                print("Empty feedback text detected in BERTScore calculation")
                return 0.0
                
            # Tokenize and encode the texts
            gen_tokens = self.bert_tokenizer(generated_feedback, return_tensors='pt', 
                                            padding=True, truncation=True, max_length=512)
            ref_tokens = self.bert_tokenizer(reference_feedback, return_tensors='pt', 
                                            padding=True, truncation=True, max_length=512)
            
            # Get BERT embeddings
            with torch.no_grad():
                gen_outputs = self.bert_model(**gen_tokens)
                ref_outputs = self.bert_model(**ref_tokens)
            
            # Get the mean pooled vectors
            gen_embeddings = self.mean_pooling(gen_outputs, gen_tokens['attention_mask'])
            ref_embeddings = self.mean_pooling(ref_outputs, ref_tokens['attention_mask'])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(gen_embeddings.numpy(), ref_embeddings.numpy())[0][0]
            
            print(f"Calculated BERTScore: {similarity}")
            return float(similarity)
        except Exception as e:
            print(f"⚠️ Error calculating BERTScore: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling for BERT embeddings to get sentence-level embeddings.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def calculate_bleu_score(self, generated_feedback, reference_feedback):
        """
        Calculate BLEU score between generated and reference feedback.
        
        Args:
            generated_feedback (str): The AI-generated feedback text
            reference_feedback (str): The instructor's reference feedback text
            
        Returns:
            float: BLEU score between 0 and 1
        """
        try:
            if not generated_feedback or not reference_feedback:
                print("Empty feedback text detected in BLEU calculation")
                return 0.0
                
            # Tokenize the texts
            gen_tokens = word_tokenize(generated_feedback.lower())
            ref_tokens = word_tokenize(reference_feedback.lower())
            
            # Check if tokens are valid
            if not gen_tokens or not ref_tokens:
                print("Empty tokens after tokenization in BLEU calculation")
                return 0.0
                
            # Calculate BLEU score
            print(f"Calculating BLEU with {len(gen_tokens)} generated tokens and {len(ref_tokens)} reference tokens")
            score = sentence_bleu([ref_tokens], gen_tokens)
            print(f"Calculated BLEU score: {score}")
            return score
        except Exception as e:
            print(f"⚠️ Error calculating BLEU score: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def evaluate_feedback_quality(self, generated_feedback, instructor_feedback=None):
        """
        Evaluate feedback quality using multiple metrics, including alignment with instructor feedback.
        
        Args:
            generated_feedback (str): The AI-generated feedback text
            instructor_feedback (str, optional): The instructor's reference feedback
            
        Returns:
            dict: Dictionary of quality metrics including alignment scores
        """
        # Prepare feedback text
        if isinstance(generated_feedback, dict):
            generated_text = ""
            for section, content in generated_feedback.items():
                if content:
                    generated_text += f"{section}: {content}\n\n"
        else:
            generated_text = generated_feedback
            
        if isinstance(instructor_feedback, dict):
            instructor_text = ""
            for section, content in instructor_feedback.items():
                if content:
                    instructor_text += f"{section}: {content}\n\n"
        else:
            instructor_text = instructor_feedback
        
        # Add debugging to check text values
        print(f"Generated text length: {len(generated_text) if generated_text else 0}")
        print(f"Instructor text length: {len(instructor_text) if instructor_text else 0}")
        
        # Calculate rubric-based metrics (without instructor feedback)
        relevance_score = self._evaluate_relevance(generated_text)
        specificity_score = self._evaluate_specificity(generated_text)
        clarity_score = self._evaluate_clarity(generated_text)
        actionability_score = self._evaluate_actionability(generated_text)
        
        # Initialize alignment scores
        rouge_scores = {
            'rouge-1-f': 0.0, 
            'rouge-2-f': 0.0, 
            'rouge-l-f': 0.0
        }
        bert_score = 0.0
        bleu_score = 0.0
        
        # Calculate alignment scores if instructor feedback is available
        if instructor_text and len(instructor_text.strip()) > 0:
            print("Processing instructor feedback for alignment scores")
            
            # ROUGE scores
            try:
                rouge_result = self.calculate_rouge_scores(generated_text, instructor_text)
                print(f"ROUGE results: {rouge_result}")
                rouge_scores = {
                    'rouge-1-f': rouge_result['rouge-1']['f'], 
                    'rouge-2-f': rouge_result['rouge-2']['f'], 
                    'rouge-l-f': rouge_result['rouge-l']['f']
                }
            except Exception as e:
                print(f"Error calculating ROUGE scores: {e}")
            
            # BERTScore
            try:
                bert_score = self.calculate_bert_score(generated_text, instructor_text)
                print(f"BERT score: {bert_score}")
            except Exception as e:
                print(f"Error calculating BERT score: {e}")
            
            # BLEU score
            try:
                bleu_score = self.calculate_bleu_score(generated_text, instructor_text)
                print(f"BLEU score: {bleu_score}")
            except Exception as e:
                print(f"Error calculating BLEU score: {e}")
        else:
            print("No valid instructor feedback available for alignment scoring")
        
        # Calculate overall score
        metrics = [relevance_score, specificity_score, clarity_score, actionability_score]
        if instructor_text and len(instructor_text.strip()) > 0:
            # Add alignment scores to the overall calculation if available
            metrics.extend([rouge_scores['rouge-l-f'], bert_score])
        
        overall_score = np.mean(metrics)
        
        return {
            "relevance": round(relevance_score, 2),
            "specificity": round(specificity_score, 2),
            "clarity": round(clarity_score, 2),
            "actionability": round(actionability_score, 2),
            "alignment_scores": {
                "rouge": {k: round(v, 3) for k, v in rouge_scores.items()},
                "bert_score": round(bert_score, 3),
                "bleu_score": round(bleu_score, 3)
            },
            "overall_score": round(overall_score, 2)
        }
    
    def _evaluate_relevance(self, text):
        """Evaluate how relevant the feedback is to SOLID principles and software design."""
        if not text:
            return 0.0
            
        # Define keywords for relevance to SOLID and software design
        relevant_keywords = [
            "solid", "responsibility", "open", "closed", "liskov", "interface segregation", 
            "dependency", "inversion", "design", "architecture", "refactor", "pattern",
            "class", "method", "interface", "dependency injection", "coupling", "cohesion"
        ]
        
        # Count relevant keywords
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in relevant_keywords if keyword in text_lower)
        
        # Calculate relevance score based on keyword density
        words = text_lower.split()
        if len(words) == 0:
            return 0.0
            
        keyword_density = min(keyword_count / 10, 1.0)  # Cap at 1.0, expect at least 10 mentions for full score
        
        # Check for SOLID principle mentions (higher weight)
        solid_principles = ["single responsibility", "open closed", "liskov substitution", 
                           "interface segregation", "dependency inversion"]
        solid_mentions = sum(1 for principle in solid_principles if principle in text_lower)
        solid_score = min(solid_mentions / 5, 1.0)  # Cap at 1.0, all principles mentioned = full score
        
        # Final relevance score - weighted combination
        return 0.6 * keyword_density + 0.4 * solid_score
    
    def _evaluate_specificity(self, text):
        """Evaluate how specific the feedback is (mentions specific code elements and locations)."""
        if not text:
            return 0.0
            
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
    
    def _evaluate_clarity(self, text):
        """Evaluate clarity of feedback based on sentence structure, formatting, and organization."""
        if not text:
            return 0.0
            
        sentences = sent_tokenize(text)
        if not sentences:
            return 0.0
            
        # Calculate average sentence length - prefer 10-20 words per sentence
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
        avg_length = np.mean(sentence_lengths)
        
        # Score sentence length - penalize very short or very long sentences
        length_score = 1.0 - min(abs(avg_length - 15) / 15, 1.0)
        
        # Check for organization elements
        has_headings = bool(re.search(r'#{1,3}\s+\w+|[A-Z][A-Za-z ]+:', text, re.MULTILINE))
        has_paragraphs = text.count('\n\n') > 0
        has_structure = bool(re.search(r'(overall|assessment|violations|suggestions|improvements)', text.lower()))
        
        organization_score = (has_headings + has_paragraphs + has_structure) / 3
        
        # Check for cohesive transitions
        transition_words = ["therefore", "furthermore", "however", "additionally", "consequently", 
                           "because", "since", "for example", "specifically", "in contrast"]
        transitions_count = sum(1 for word in transition_words if word in text.lower())
        transition_score = min(transitions_count / 3, 1.0)  # Cap at 1.0
        
        # Final clarity score - weighted combination
        return 0.4 * length_score + 0.4 * organization_score + 0.2 * transition_score
    
    def _evaluate_actionability(self, text):
        """Evaluate how actionable the feedback is (provides concrete suggestions)."""
        if not text:
            return 0.0
            
        sentences = sent_tokenize(text)
        total_sentences = len(sentences)
        
        if total_sentences == 0:
            return 0.0
        
        # Define actionable verbs/phrases
        actionable_verbs = [
            "refactor", "extract", "implement", "move", "rename", "create", "separate", 
            "use", "replace", "modify", "add", "remove", "improve", "change", "apply", 
            "consider", "try", "should", "could", "introduce", "simplify", "restructure"
        ]
        
        # Count actionable sentences
        actionable_count = sum(1 for sentence in sentences 
                              if any(verb in sentence.lower() for verb in actionable_verbs))
        
        # Check for code examples or patterns (adds to actionability)
        code_examples = len(re.findall(r'```java|`[^`]+`', text))
        
        # Calculate actionability score
        base_score = actionable_count / total_sentences
        example_bonus = min(code_examples * 0.1, 0.3)  # Cap the bonus at 0.3
        
        return min(base_score + example_bonus, 1.0)

# Example usage
if __name__ == "__main__":
    evaluator = FeedbackEvaluation()
    
    # Sample feedback
    generated_feedback = """
    ## Overall Assessment
    The code violates several SOLID principles, particularly Single Responsibility Principle and Dependency Inversion Principle.
    
    ## SOLID Violations
    1. SRP - The UserService class has too many responsibilities including user management, validation, and direct database operations.
    2. DIP - The code directly instantiates PostgresDriver in UserService instead of using dependency injection.
    
    ## Improvement Suggestions
    - Refactor UserService to separate concerns into smaller classes
    - Implement constructor injection for the database driver dependency
    - Use the DatabaseDriver interface rather than concrete PostgresDriver implementation
    """
    
    instructor_feedback = """
    The student's code shows problems with SRP as UserService does too many things. 
    It would help if you had used dependency injection for the PostgresDriver instead of creating it directly.
    The refactoring for STEP 2 should focus on separating the different responsibilities.
    """
    
    results = evaluator.evaluate_feedback_quality(generated_feedback, instructor_feedback)
    print(results)