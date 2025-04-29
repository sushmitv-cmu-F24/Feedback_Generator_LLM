import os
import json
import shutil
import base64
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.utils import secure_filename
from evaluation_phase import extract_java_files_from_zip, evaluate_submission, inspect_faiss_index
from feedback_evaluation import FeedbackEvaluation
from reinforcement import FeedbackReinforcementLearning
import atexit
import sys
import gc
import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
import io

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'data/evaluation'
app.config['PROCESSED_DATA'] = 'data/processed_data.json'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['RL_CONFIG'] = 'config/reinforcement_config.json'  # Path to RL config

# Create folders if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('static/css', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('config', exist_ok=True)

# Initialize feedback metrics and reinforcement learning
reinforcement = FeedbackReinforcementLearning(config_file=app.config['RL_CONFIG'])

# Cache for generated feedback
feedback_cache = {}

# Storage for submissions data
submissions = {}

def load_submissions():
    """Load saved submissions data from disk."""
    global submissions
    submissions_file = os.path.join('data', 'submissions_data.json')
    
    if os.path.exists(submissions_file):
        try:
            with open(submissions_file, 'r') as f:
                submissions = json.load(f)
            print(f"Loaded {len(submissions)} submissions from disk")
        except Exception as e:
            print(f"Error loading submissions data: {e}")
            submissions = {}
    else:
        submissions = {}
    
    # Initialize from feedback cache for any missing entries
    for student_id, data in feedback_cache.items():
        if student_id not in submissions:
            submissions[student_id] = data

# Replace processed_data.json from backup on startup to avoid data leakage
def reset_processed_data():
    """Reset processed_data.json from processed_data_backup.json at application startup."""
    try:
        backup_file = os.path.join('data', 'processed_data_backup.json')
        processed_file = os.path.join('data', 'processed_data.json')
        
        if os.path.exists(backup_file):
            # Simply copy the backup file to processed_data.json
            shutil.copy2(backup_file, processed_file)
            print("✅ Reset processed_data.json from backup file")
        else:
            print("⚠️ Backup file (processed_data_backup.json) not found. Creating empty processed_data.json.")
            # Create an empty processed_data.json
            with open(processed_file, 'w') as f:
                json.dump({}, f)
    except Exception as e:
        print(f"⚠️ Error resetting processed data: {e}")
        import traceback
        traceback.print_exc()

def update_processed_data(student_id, evaluation_result):
    """
    Add newly generated feedback to processed_data.json with the same structure as existing entries,
    and update the FAISS index for similarity matching.
    """
    try:
        # Import necessary functions from training_phase.py and evaluation_phase.py
        from training_phase import analyze_sentiment, categorize_solid_violations, parse_java_code, extract_package_structure
        from evaluation_phase import extract_solid_violations_from_feedback
        
        processed_file = os.path.join('data', 'processed_data.json')
        
        # Load current processed data
        if os.path.exists(processed_file):
            with open(processed_file, 'r') as f:
                processed_data = json.load(f)
        else:
            processed_data = {}
        
        # Extract feedback text
        feedback_text = evaluation_result.get('generated_feedback', '')
        java_files = evaluation_result.get('java_files', {})
        
        # Divide feedback into structured sections like in other entries
        feedback_sections = extract_feedback_sections(feedback_text)
        
        # Parse all Java files for class structure
        parsed_classes = {}
        for file_path, content in java_files.items():
            file_parsed_classes = parse_java_code(content)
            parsed_classes.update(file_parsed_classes)
        
        # Extract package structure
        package_structure, class_packages = extract_package_structure(java_files)
        
        # Extract SOLID violations using function from training_phase.py
        solid_violations = categorize_solid_violations(feedback_text)
        
        # Analyze sentiment
        sentiment_label, sentiment_score = analyze_sentiment(feedback_text)
        
        # Get the detected violations in the same format as training_phase.py
        detected_violations = {
            "solid": [],
            "package": [],
            "dependency_injection": []
        }
        
        # Map the flat list of violations from evaluation_result to the categorized structure
        for violation in evaluation_result.get('detected_violations', []):
            principle = violation.get('principle', '')
            if principle in ['SRP', 'OCP', 'LSP', 'ISP', 'DIP']:
                detected_violations["solid"].append({
                    "principle": principle,
                    "class": violation.get('location', ''),
                    "reason": violation.get('description', '')
                })
            elif principle == 'package_structure':
                detected_violations["package"].append({
                    "type": "package_structure",
                    "description": violation.get('description', '')
                })
            elif principle == 'dependency_injection':
                detected_violations["dependency_injection"].append({
                    "type": "dependency_injection", 
                    "description": violation.get('description', '')
                })
        
        # Structure the entry to fully match the format in processed_data.json
        processed_data[student_id] = {
            "feedback": feedback_sections,  # Use the divided feedback sections
            "java_files": java_files,
            "parsed_classes": parsed_classes,
            "package_structure": package_structure,
            "solid_violations": solid_violations,
            "detected_violations": detected_violations,
            "sentiment": {"label": sentiment_label, "score": sentiment_score}
        }
        
        # Save the updated data
        with open(processed_file, 'w') as f:
            json.dump(processed_data, f, indent=4)
        
        print(f"✅ Updated processed_data.json with complete feedback entry for {student_id}")
        
        # Update the FAISS index with the new feedback
        update_faiss_index(student_id, evaluation_result)
        
        return True
    except Exception as e:
        print(f"⚠️ Error updating processed data: {e}")
        import traceback
        traceback.print_exc()
        return False

def reset_faiss_index_files():
    """Reset FAISS index files from backup at application startup."""
    try:
        # Files to reset
        faiss_file = "./data/feedback_embeddings.faiss"
        mapping_file = "./data/feedback_embeddings.faiss.json"
        
        # Backup files
        faiss_backup = "./data/feedback_embeddings_backup.faiss"
        mapping_backup = "./data/feedback_embeddings_backup.faiss.json"
        
        # Reset FAISS index if backup exists
        if os.path.exists(faiss_backup) and os.path.exists(mapping_backup):
            # Copy backup files to original locations
            shutil.copy2(faiss_backup, faiss_file)
            shutil.copy2(mapping_backup, mapping_file)
            
            print("✅ Reset FAISS index files from backups")
            
            # Reload the FAISS index in memory
            reload_faiss_index()
        else:
            print("⚠️ FAISS backup files not found. Index will not be reset.")
    except Exception as e:
        print(f"⚠️ Error resetting FAISS index files: {e}")
        import traceback
        traceback.print_exc()

# Add the extract_feedback_sections function from training_phase.py
def extract_feedback_sections(feedback_text):
    """Breaks feedback into structured categories: general_comments, solid_violations, and suggestions."""
    if not feedback_text:
        return {"general_comments": None, "solid_violations": None, "suggestions": None}
    
    # Try to identify sections based on common headers in the feedback
    sections = {"general_comments": None, "solid_violations": None, "suggestions": None}
    
    # Split the feedback into sections based on markdown headers
    parts = re.split(r'## ', feedback_text)
    
    # The first part is likely the overall assessment (general comments)
    if parts and len(parts) > 0:
        sections["general_comments"] = parts[0].strip()
    
    # Look for sections with specific keywords
    for part in parts:
        part = part.strip()
        if part.startswith("Overall Assessment") or part.startswith("Assessment"):
            sections["general_comments"] = part
        elif part.startswith("SOLID Violations") or part.startswith("Violations"):
            sections["solid_violations"] = part
        elif part.startswith("Improvement Suggestions") or part.startswith("Suggestions"):
            sections["suggestions"] = part
    
    # If we couldn't identify clear sections, use the whole text as general_comments
    if not any(sections.values()):
        sections["general_comments"] = feedback_text
    
    # Clean the text and remove section headers
    for key in sections:
        if sections[key]:
            sections[key] = re.sub(r'^(Overall Assessment|SOLID Violations|Improvement Suggestions)[\s:]*', '', sections[key]).strip()
    
    return sections

def update_faiss_index(student_id, evaluation_result):
    """
    Update the FAISS index with the embedding for newly generated feedback.
    
    Args:
        student_id: The student ID for the new feedback
        evaluation_result: The evaluation result containing feedback and Java files
    """
    try:
        import faiss
        from evaluation_phase import get_submission_embedding
        from training_phase import parse_java_code
        
        # Check if the FAISS index exists
        embeddings_file = "./data/feedback_embeddings.faiss"
        if not os.path.exists(embeddings_file):
            print(f"⚠️ FAISS index file not found at {embeddings_file}. Cannot update.")
            return False
        
        # Load the existing index
        faiss_index = faiss.read_index(embeddings_file)
        
        # Load the student ID mapping
        mapping_file = embeddings_file + ".json"
        if os.path.exists(mapping_file):
            with open(mapping_file, "r") as f:
                student_id_mapping = json.load(f)
        else:
            student_id_mapping = []
        
        # Get Java files
        java_files = evaluation_result.get('java_files', {})
        
        # Parse Java files for class structure in the format expected by get_submission_embedding
        parsed_classes = {}
        for file_path, content in java_files.items():
            file_parsed = parse_java_code(content)
            if file_parsed:  # Only add if parsing was successful
                parsed_classes[file_path] = file_parsed
        
        # Generate the embedding
        # Note: get_submission_embedding expects parsed_classes in a format where
        # keys are file paths and values are dictionaries of class info
        embedding = get_submission_embedding(java_files, parsed_classes)
        
        # Add the embedding to the FAISS index
        faiss_index.add(embedding)
        
        # Update the student ID mapping
        student_id_mapping.append(student_id)
        
        # Save the updated index and mapping
        faiss.write_index(faiss_index, embeddings_file)
        with open(mapping_file, "w") as f:
            json.dump(student_id_mapping, f)
        
        print(f"✅ Updated FAISS index with embedding for {student_id}")
        return True
    except Exception as e:
        print(f"⚠️ Error updating FAISS index: {e}")
        import traceback
        traceback.print_exc()
        return False

def reload_faiss_index():
    """
    Reload the FAISS index to ensure we're using the most up-to-date version.
    This should be called before searching for similar submissions.
    """
    try:
        import faiss
        
        # Path to the FAISS index file
        embeddings_file = "./data/feedback_embeddings.faiss"
        
        # Check if the file exists
        if not os.path.exists(embeddings_file):
            print(f"⚠️ FAISS index file not found at {embeddings_file}. Cannot reload.")
            return False
        
        # Import the FAISS variables from evaluation_phase.py
        import evaluation_phase
        
        # Reload the FAISS index
        evaluation_phase.faiss_index = faiss.read_index(embeddings_file)
        
        # Reload the student ID mapping
        mapping_file = embeddings_file + ".json"
        if os.path.exists(mapping_file):
            with open(mapping_file, "r") as f:
                evaluation_phase.student_id_mapping = json.load(f)
                
        print(f"✅ Reloaded FAISS index with {evaluation_phase.faiss_index.ntotal} vectors")
        return True
    except Exception as e:
        print(f"⚠️ Error reloading FAISS index: {e}")
        import traceback
        traceback.print_exc()
        return False

def update_processed_data_with_ril(student_id, instructor_feedback):
    """
    Update the feedback portion in processed_data.json when instructor provides feedback via RIL.
    Preserves all other metadata while replacing just the feedback.
    
    Args:
        student_id: The student ID for the feedback
        instructor_feedback: The new feedback text from instructor
    """
    try:
        # Path to processed data file
        processed_file = os.path.join('data', 'processed_data.json')
        
        # Check if file exists
        if not os.path.exists(processed_file):
            print(f"⚠️ Processed data file not found at {processed_file}")
            return False
        
        # Load current processed data
        with open(processed_file, 'r') as f:
            processed_data = json.load(f)
        
        # Check if student entry exists
        if student_id not in processed_data:
            print(f"⚠️ No entry found for student {student_id} in processed data")
            return False
        
        # Get the existing entry to preserve metadata
        student_entry = processed_data[student_id]
        
        # Extract feedback sections from the new feedback
        feedback_sections = extract_feedback_sections(instructor_feedback)
        
        # Update just the feedback portion
        student_entry['feedback'] = feedback_sections
        
        # Re-analyze SOLID violations if needed
        from training_phase import categorize_solid_violations
        solid_violations = categorize_solid_violations(instructor_feedback)
        student_entry['solid_violations'] = solid_violations
        
        # Save the updated data
        with open(processed_file, 'w') as f:
            json.dump(processed_data, f, indent=4)
        
        print(f"✅ Updated processed_data.json with new RIL feedback for {student_id}")
        return True
    except Exception as e:
        print(f"⚠️ Error updating processed data with RIL feedback: {e}")
        import traceback
        traceback.print_exc()
        return False

def format_ril_feedback(original_feedback, instructor_message, ai_response):
    """
    Format the RIL feedback to maintain the same structure as the original feedback.
    
    Args:
        original_feedback: The original feedback structure
        instructor_message: The message from the instructor
        ai_response: The AI's response to the instructor
        
    Returns:
        A formatted feedback text that maintains the original structure
    """
    try:
        # Check if original feedback is a dictionary
        if isinstance(original_feedback, dict):
            # Extract sections from original feedback
            general_comments = original_feedback.get('general_comments', '')
            solid_violations = original_feedback.get('solid_violations', '')
            suggestions = original_feedback.get('suggestions', '')
            
            # Analyze instructor message to determine which section they want to update
            instructor_msg_lower = instructor_message.lower()
            
            # Check for keywords to determine which section to update
            update_general = any(keyword in instructor_msg_lower for keyword in 
                               ['overall', 'assessment', 'general', 'introduction'])
            update_violations = any(keyword in instructor_msg_lower for keyword in 
                                  ['violation', 'solid', 'principle', 'srp', 'ocp', 'lsp', 'isp', 'dip'])
            update_suggestions = any(keyword in instructor_msg_lower for keyword in 
                                   ['suggestion', 'recommend', 'improvement', 'fix', 'solution'])
            
            # If no specific section is mentioned, determine based on content
            if not any([update_general, update_violations, update_suggestions]):
                # Extract sections from AI response
                ai_sections = extract_feedback_sections(ai_response)
                
                # If AI response has clear sections, use those
                if ai_sections.get('general_comments'):
                    general_comments = ai_sections.get('general_comments')
                if ai_sections.get('solid_violations'):
                    solid_violations = ai_sections.get('solid_violations')
                if ai_sections.get('suggestions'):
                    suggestions = ai_sections.get('suggestions')
            else:
                # Update specific sections based on instructor message
                if update_general:
                    general_comments = ai_response
                elif update_violations:
                    solid_violations = ai_response
                elif update_suggestions:
                    suggestions = ai_response
            
            # Format the feedback in the same structure as the original
            formatted_feedback = ""
            if general_comments:
                formatted_feedback += f"## Overall Assessment\n{general_comments}\n\n"
            if solid_violations:
                formatted_feedback += f"## SOLID Violations\n{solid_violations}\n\n"
            if suggestions:
                formatted_feedback += f"## Improvement Suggestions\n{suggestions}\n\n"
            
            return formatted_feedback
        else:
            # If original_feedback is not a dictionary, just return the AI response
            return ai_response
    except Exception as e:
        print(f"⚠️ Error formatting RIL feedback: {e}")
        return ai_response  # Return AI response as fallback

def clear_cache_on_startup():
    """Clear cached feedback and reset submissions on application startup."""
    global feedback_cache, submissions
    
    # Clear the global variables
    feedback_cache = {}
    submissions = {}
    
    # Delete or clear the submissions data file
    submissions_file = os.path.join('data', 'submissions_data.json')
    if os.path.exists(submissions_file):
        try:
            with open(submissions_file, 'w') as f:
                json.dump({}, f)
            
            print("✅ Cache and submissions data cleared on startup")
        except Exception as e:
            print(f"⚠️ Error clearing submissions data: {e}")
    
    # Reset processed_data.json from backup
    reset_processed_data()

    # Reset FAISS index files from backup
    reset_faiss_index_files()

def save_submissions():
    """Save submissions data to disk."""
    submissions_file = os.path.join('data', 'submissions_data.json')
    
    try:
        with open(submissions_file, 'w') as f:
            json.dump(submissions, f, indent=4)
        print(f"Saved {len(submissions)} submissions to disk")
    except Exception as e:
        print(f"Error saving submissions data: {e}")

# Clear cache on startup
clear_cache_on_startup()

# Load submissions data on startup
load_submissions()

@app.route('/')
def index():
    """Home page with upload form and list of submissions."""
    submissions_list = []
    
    # Get list of uploaded submissions
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.endswith('.zip'):
            student_id = filename.split('_')[0]
            submission_time = os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            submission_date = datetime.fromtimestamp(submission_time).strftime('%Y-%m-%d %H:%M')
            
            # Check if feedback has been generated
            has_feedback = student_id in feedback_cache or student_id in submissions
            
            submissions_list.append({
                'student_id': student_id,
                'filename': filename,
                'date': submission_date,
                'has_feedback': has_feedback
            })
    
    # Sort submissions by date (newest first)
    submissions_list.sort(key=lambda x: x['date'], reverse=True)
    
    return render_template('index.html', submissions=submissions_list)

@app.route('/upload', methods=['POST'])
def upload_submission():
    """Handle submission upload."""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and file.filename.endswith('.zip'):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash(f'Submission {filename} uploaded successfully!')
        return redirect(url_for('index'))
    else:
        flash('Invalid file. Please upload a ZIP file.')
        return redirect(request.url)

@app.route('/generate/<student_id>')
def generate_feedback(student_id):
    """Show loading page while generating feedback."""
    return render_template('loading.html', student_id=student_id)

def format_feedback_as_markdown(feedback_text):
    """Format the feedback as proper markdown with headers."""
    # Check if the feedback already has markdown formatting
    if "##" in feedback_text:
        # Make sure headers and content are properly separated
        feedback_text = re.sub(r'(## .+?)(\w)', r'\1\n\2', feedback_text)
        return feedback_text
        
    # Otherwise, add markdown formatting
    formatted_text = feedback_text
    
    # Add headers for sections if they don't already have them
    if "Overall Assessment:" in formatted_text and not "## Overall Assessment" in formatted_text:
        formatted_text = formatted_text.replace("Overall Assessment:", "## Overall Assessment")
    
    if "SOLID Violations:" in formatted_text and not "## SOLID Violations" in formatted_text:
        formatted_text = formatted_text.replace("SOLID Violations:", "## SOLID Violations")
    
    if "Improvement Suggestions:" in formatted_text and not "## Improvement Suggestions" in formatted_text:
        formatted_text = formatted_text.replace("Improvement Suggestions:", "## Improvement Suggestions")
    
    # Format list items if they're not already formatted
    lines = formatted_text.split('\n')
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        if (line_stripped.startswith('1.') or line_stripped.startswith('2.') or
            line_stripped.startswith('3.') or line_stripped.startswith('4.') or
            line_stripped.startswith('5.')) and not line_stripped.startswith('- '):
            # Replace the number with a markdown list item
            number_part = line_stripped.split('.', 1)[0]
            rest = line_stripped.split('.', 1)[1] if '.' in line_stripped else ''
            lines[i] = '- ' + rest.strip()
    
    return '\n'.join(lines)

def get_reference_feedback(student_id, closest_matches):
    """
    Get reference feedback from the most similar submission in processed_data.json.
    
    Args:
        student_id: The current student ID
        closest_matches: List of similar student IDs from FAISS
        
    Returns:
        Tuple of (reference_feedback, source_id)
    """
    # Load processed data
    try:
        with open("data/processed_data.json", "r") as f:
            processed_data = json.load(f)
    except Exception as e:
        print(f"⚠️ Error loading processed data: {e}")
        return None, None
    
    # Find the first match with feedback that's not the current student
    for match_id in closest_matches:
        if match_id == student_id:
            continue  # Skip the current student
            
        if match_id in processed_data and processed_data[match_id].get("feedback"):
            feedback = processed_data[match_id]["feedback"]
            # Extract text from the feedback structure
            if isinstance(feedback, dict):
                feedback_text = ""
                for section, content in feedback.items():
                    if content:
                        feedback_text += f"{content}\n\n"
                return feedback_text.strip(), match_id
            else:
                return feedback, match_id
    
    return None, None

@app.route('/student/<student_id>')
def student_view(student_id):
    """View generated feedback for a student."""
    
    # Check if feedback is already in cache or submissions
    if student_id in feedback_cache:
        return render_template(
            'student_view.html',
            student_id=student_id,
            feedback=feedback_cache[student_id],
            java_files=feedback_cache[student_id].get('java_files', {})
        )
    elif student_id in submissions:
        return render_template(
            'student_view.html',
            student_id=student_id,
            feedback=submissions[student_id],
            java_files=submissions[student_id].get('java_files', {})
        )
    
    # Find the ZIP file for this student
    zip_file = None
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.startswith(student_id) and filename.endswith('.zip'):
            zip_file = filename
            break
    
    if not zip_file:
        flash(f'No submission found for student {student_id}')
        return redirect(url_for('index'))
    
    # Extract Java files
    zip_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_file)
    java_files = extract_java_files_from_zip(zip_path)
    
    if not java_files:
        flash(f'No Java files found in {zip_file}')
        return redirect(url_for('index'))
    
    # Reload the FAISS index to ensure we have the latest data
    reload_faiss_index()
    
    # Generate feedback
    evaluation_result = evaluate_submission(java_files)
    evaluation_result['java_files'] = java_files
    
    # Get reference feedback from the most similar submission
    reference_feedback, reference_source = get_reference_feedback(
        student_id, 
        evaluation_result.get('closest_matches', [])
    )
    
    # Store the reference source
    evaluation_result['reference_feedback_source'] = reference_source
    
    # Add quality metrics
    feedback_text = evaluation_result['generated_feedback']
    
    # Calculate alignment with reference feedback if available
    if reference_feedback:
        try:
            evaluator = FeedbackEvaluation()
            evaluation_scores = evaluator.evaluate_feedback_quality(feedback_text, reference_feedback)
            evaluation_result['feedback_evaluation'] = evaluation_scores
            print(f"Calculated metrics using reference feedback from {reference_source}")
        except Exception as e:
            print(f"⚠️ Error calculating feedback metrics: {e}")
            import traceback
            traceback.print_exc()
    
    # Format the feedback as markdown with proper headers for nicer display
    formatted_feedback = format_feedback_as_markdown(evaluation_result['generated_feedback'])
    evaluation_result['generated_feedback'] = formatted_feedback
    
    # Store in cache and submissions
    feedback_cache[student_id] = evaluation_result
    submissions[student_id] = evaluation_result
    
    # Save the updated submissions
    save_submissions()
    
    # Update processed_data.json with the new feedback
    update_processed_data(student_id, evaluation_result)
    
    return render_template(
        'student_view.html',
        student_id=student_id,
        feedback=evaluation_result,
        java_files=java_files
    )

@app.route('/submit_student_rating/<student_id>', methods=['POST'])
def submit_student_rating(student_id):
    """Handle student rating submission with enhanced data capture for RIL dashboard."""
    if student_id not in submissions:
        flash(f"Student submission {student_id} not found.", "danger")
        return redirect(url_for('index'))
    
    # Get rating from form
    rating = request.form.get('rating', '0')
    rating_value = 1.0 if rating == '1' else 0.0
    
    # Add timestamp for tracking when feedback was received
    submission_date = datetime.now().strftime('%Y-%m-%d')
    
    # Update submission with student rating and date
    submissions[student_id]['student_rating'] = rating_value
    submissions[student_id]['submission_date'] = submission_date
    
    # Update cache if needed
    if student_id in feedback_cache:
        feedback_cache[student_id]['student_rating'] = rating_value
        feedback_cache[student_id]['submission_date'] = submission_date
    
    # Save submissions
    save_submissions()
    
    # Call reinforcement learning to store feedback pair if possible
    try:
        if student_id in submissions and 'java_files' in submissions[student_id] and 'generated_feedback' in submissions[student_id]:
            # Get original code (combine all Java files)
            java_files = submissions[student_id]['java_files']
            combined_code = ""
            for file_path, content in java_files.items():
                combined_code += f"// {file_path}\n{content}\n\n"
                
            # Get generated feedback
            generated_feedback = submissions[student_id]['generated_feedback']
            
            # Store in RL system - using student rating as feedback quality indicator
            # Note: We pass the same text for model_feedback and instructor_feedback since we're
            # not using instructor feedback anymore, just student ratings
            reinforcement.store_feedback_pair(
                submission_id=student_id,
                original_code=combined_code,
                model_feedback=generated_feedback,
                instructor_feedback=generated_feedback,  # Same as model feedback since we only use student ratings
                instructor_rating=rating_value,  # Using student rating as the quality indicator
                feedback_meta={
                    "rating_type": "student",
                    "submission_date": submission_date,
                    "has_alignment_scores": 'feedback_evaluation' in submissions[student_id]
                }
            )
            print(f"✅ Stored feedback with student rating ({rating_value}) for {student_id}")
    except Exception as e:
        print(f"⚠️ Error storing feedback pair in RL system: {e}")
        import traceback
        traceback.print_exc()
    
    # Flash appropriate message
    if rating_value == 1.0:
        flash("Thank you for rating the feedback as helpful!", "success")
    else:
        flash("Thank you for your feedback. We'll work to improve our feedback.", "info")
    
    return redirect(url_for('student_view', student_id=student_id))

def process_alignment_scores(submissions_dict):
    """
    Process alignment scores from all submissions to prepare data for visualizations.
    
    Args:
        submissions_dict: Dictionary of all submissions
        
    Returns:
        Dictionary with processed alignment data
    """
    # Initialize data containers
    rouge_data = []
    bleu_data = []
    bert_data = []
    
    # Find submissions with alignment scores
    for sid, data in submissions_dict.items():
        if 'feedback_evaluation' in data and 'alignment_scores' in data['feedback_evaluation']:
            scores = data['feedback_evaluation']['alignment_scores']
            
            # Extract ROUGE-L score
            if 'rouge' in scores and 'rouge-l-f' in scores['rouge']:
                rouge_score = scores['rouge']['rouge-l-f']
                rouge_data.append((sid, rouge_score))
            
            # Extract BLEU score
            if 'bleu_score' in scores:
                bleu_score = scores['bleu_score']
                bleu_data.append((sid, bleu_score))
            
            # Extract BERT score
            if 'bert_score' in scores:
                bert_score = scores['bert_score']
                bert_data.append((sid, bert_score))
    
    # Sort data by submission ID for consistency
    rouge_data.sort(key=lambda x: x[0])
    bleu_data.sort(key=lambda x: x[0])
    bert_data.sort(key=lambda x: x[0])
    
    return {
        'rouge_data': rouge_data,
        'bleu_data': bleu_data,
        'bert_data': bert_data,
        'submissions_with_scores': len(set([x[0] for x in rouge_data + bleu_data + bert_data]))
    }

def calculate_correlation_metrics(submissions_dict):
    """
    Calculate correlation between student ratings and alignment scores.
    
    Args:
        submissions_dict: Dictionary of all submissions
        
    Returns:
        Dictionary with correlation data
    """
    # Create lists to hold paired data
    student_ratings = []
    rouge_scores = []
    bleu_scores = []
    bert_scores = []
    
    # Find submissions with both student ratings and alignment scores
    for sid, data in submissions_dict.items():
        if 'student_rating' in data and 'feedback_evaluation' in data and 'alignment_scores' in data['feedback_evaluation']:
            student_rating = data['student_rating']
            scores = data['feedback_evaluation']['alignment_scores']
            
            # Add ROUGE-L score pair
            if 'rouge' in scores and 'rouge-l-f' in scores['rouge']:
                rouge_score = scores['rouge']['rouge-l-f']
                student_ratings.append(student_rating)
                rouge_scores.append(rouge_score)
            
            # Add BLEU score pair
            if 'bleu_score' in scores:
                bleu_score = scores['bleu_score']
                bleu_scores.append(bleu_score)
            
            # Add BERT score pair
            if 'bert_score' in scores:
                bert_score = scores['bert_score']
                bert_scores.append(bert_score)
    
    # Calculate correlations if we have enough data
    correlations = {}
    if len(student_ratings) >= 3:
        try:
            from scipy.stats import pearsonr
            
            # Calculate Pearson correlation
            if len(rouge_scores) >= 3:
                correlations['rouge_correlation'] = pearsonr(student_ratings[:len(rouge_scores)], rouge_scores)[0]
            
            if len(bleu_scores) >= 3:
                correlations['bleu_correlation'] = pearsonr(student_ratings[:len(bleu_scores)], bleu_scores)[0]
            
            if len(bert_scores) >= 3:
                correlations['bert_correlation'] = pearsonr(student_ratings[:len(bert_scores)], bert_scores)[0]
        except Exception as e:
            print(f"Error calculating correlations: {e}")
            correlations = {}
    
    return {
        'correlations': correlations,
        'samples': len(student_ratings)
    }

def generate_quality_metrics_chart(submissions_dict):
    """
    Generate a chart comparing quality metrics (ROUGE-L, BLEU, BERTScore) for submissions.
    
    Args:
        submissions_dict: Dictionary containing all submissions
        
    Returns:
        Tuple of (base64_image_data, has_data_flag)
    """
    try:
        # Process alignment scores
        alignment_data = process_alignment_scores(submissions_dict)
        rouge_data = alignment_data['rouge_data']
        bleu_data = alignment_data['bleu_data']
        bert_data = alignment_data['bert_data']
        
        # Check if we have enough data
        if not rouge_data and not bleu_data and not bert_data:
            return None, False
        
        # Get all submission IDs
        all_sids = set([x[0] for x in rouge_data + bleu_data + bert_data])
        
        # Simplify submission IDs for display
        sid_labels = [sid.split('_')[0] if '_' in sid else sid for sid in all_sids]
        sid_labels = [sid[:10] + '..' if len(sid) > 12 else sid for sid in sid_labels]
        
        # Prepare data for each metric
        metrics_data = {
            'ROUGE-L': {sid: score for sid, score in rouge_data},
            'BLEU': {sid: score for sid, score in bleu_data},
            'BERTScore': {sid: score for sid, score in bert_data}
        }
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Set up bar width and positions
        bar_width = 0.25
        x = np.arange(len(all_sids))
        
        # Plot bars for each metric
        colors = ['#4285F4', '#34A853', '#FBBC05']
        for i, (metric, scores_dict) in enumerate(metrics_data.items()):
            values = [scores_dict.get(sid, 0) for sid in all_sids]
            plt.bar(x + (i-1)*bar_width, values, bar_width, label=metric, color=colors[i], alpha=0.8)
        
        # Format plot
        plt.xlabel('Submissions')
        plt.ylabel('Score')
        plt.title('Feedback Quality Metrics')
        plt.xticks(x, sid_labels, rotation=45)
        plt.ylim(0, 1.1)
        plt.legend()
        plt.tight_layout()
        
        # Add correlation info if available
        correlation_info = calculate_correlation_metrics(submissions_dict)
        if correlation_info['samples'] >= 3 and correlation_info['correlations']:
            corr_text = "Correlation with student ratings:\n"
            for metric, corr in correlation_info['correlations'].items():
                metric_name = metric.split('_')[0].upper()
                corr_text += f"{metric_name}: {corr:.2f}  "
            plt.figtext(0.5, 0.01, corr_text, ha='center', fontsize=9, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
        
        # Save to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        return image_data, True
        
    except Exception as e:
        print(f"Error generating quality metrics chart: {e}")
        import traceback
        traceback.print_exc()
        return None, False

@app.route('/instructor/<student_id>')
def instructor_view(student_id):
    """Instructor view with chat interface for feedback refinement."""
    # Check if feedback is already in cache or submissions
    if student_id in feedback_cache:
        feedback_data = feedback_cache[student_id]
    elif student_id in submissions:
        feedback_data = submissions[student_id]
    else:
        # Try to load from processed_data if not in cache or submissions
        try:
            with open(app.config['PROCESSED_DATA'], "r") as f:
                processed_data = json.load(f)
                if student_id in processed_data:
                    feedback_data = processed_data[student_id]
                    # Add to submissions for future access
                    submissions[student_id] = feedback_data
                    save_submissions()
                else:
                    flash(f'No feedback found for student {student_id}')
                    return redirect(url_for('index'))
        except Exception as e:
            print(f"Error loading processed data: {e}")
            flash(f'No feedback found for student {student_id}')
            return redirect(url_for('index'))
    
    java_files = feedback_data.get('java_files', {})
    
    # Get or initialize chat history
    if 'chat_history' not in feedback_data:
        feedback_data['chat_history'] = []
    
    # Get/recalculate metrics using reference feedback from similar submissions
    if 'generated_feedback' in feedback_data:
        # Get reference feedback from closest matches
        closest_matches = []
        if 'closest_matches' in feedback_data:
            closest_matches = feedback_data['closest_matches']
        else:
            # Reload the FAISS index
            reload_faiss_index()
            
            # Get submission embedding
            from evaluation_phase import get_submission_embedding, find_closest_past_submissions
            
            # Parse all Java files for class structure if needed
            parsed_classes = feedback_data.get('parsed_classes', {})
            if not parsed_classes and 'java_files' in feedback_data:
                from training_phase import parse_java_code
                parsed_classes = {}
                for file_path, content in feedback_data['java_files'].items():
                    file_parsed = parse_java_code(content)
                    if file_parsed:  # Only add if parsing was successful
                        parsed_classes.update(file_parsed)
                feedback_data['parsed_classes'] = parsed_classes
            
            # Find similar submissions
            if 'java_files' in feedback_data and parsed_classes:
                embedding = get_submission_embedding(feedback_data['java_files'], parsed_classes)
                closest_matches = find_closest_past_submissions(embedding)
                feedback_data['closest_matches'] = closest_matches
        
        # Get reference feedback
        reference_feedback, reference_source = get_reference_feedback(student_id, closest_matches)
        feedback_data['reference_feedback_source'] = reference_source
        
        # Calculate metrics with reference feedback
        if reference_feedback:
            try:
                evaluator = FeedbackEvaluation()
                generated_feedback = feedback_data.get('generated_feedback', '')
                
                print(f"Calculating alignment scores with reference feedback from {reference_source}")
                
                # Calculate scores
                evaluation_results = evaluator.evaluate_feedback_quality(generated_feedback, reference_feedback)
                
                # Store the evaluation results
                feedback_data['feedback_evaluation'] = evaluation_results
                
                # Debug
                alignment_scores = evaluation_results.get('alignment_scores', {})
                print(f"Calculated ROUGE-L: {alignment_scores.get('rouge', {}).get('rouge-l-f', 0)}")
                print(f"Calculated BLEU: {alignment_scores.get('bleu_score', 0)}")
                print(f"Calculated BERT: {alignment_scores.get('bert_score', 0)}")
                
            except Exception as e:
                print(f"Error calculating feedback metrics: {e}")
                import traceback
                traceback.print_exc()
    
    # Update cache and save submissions
    if student_id in feedback_cache:
        feedback_cache[student_id] = feedback_data
    submissions[student_id] = feedback_data
    save_submissions()
    
    return render_template(
        'instructor_view.html',
        student_id=student_id,
        feedback=feedback_data,
        java_files=java_files
    )

@app.route('/submit_instructor_message/<student_id>', methods=['POST'])
def submit_instructor_message(student_id):
    """Handle instructor messages in the chat interface."""
    try:
        if student_id not in submissions:
            return jsonify({"success": False, "error": "Student submission not found"})
        
        # Get message from form
        message = request.form.get('message', '')
        
        # Safety check for empty messages
        if not message or message.strip() == '':
            return jsonify({"success": False, "error": "Message cannot be empty"})
        
        # Get submission data
        submission = submissions[student_id]
        
        # Initialize chat history if not exists
        if 'chat_history' not in submission:
            submission['chat_history'] = []
        
        # Add instructor message to chat history
        timestamp = datetime.now().strftime('%H:%M')
        submission['chat_history'].append({
            'sender': 'instructor',
            'content': message,
            'timestamp': timestamp
        })
        
        # Store message as instructor feedback for alignment calculation
        submission['instructor_feedback'] = message
        
        # Generate AI response
        ai_response = generate_ai_response(student_id, message, submission)
        
        # Add AI response to chat history
        submission['chat_history'].append({
            'sender': 'ai',
            'content': ai_response,
            'timestamp': timestamp
        })
        
        # Get original feedback structure
        original_feedback = None
        if 'processed_data.json' in app.config:
            try:
                with open(app.config['PROCESSED_DATA'], "r") as f:
                    processed_data = json.load(f)
                    if student_id in processed_data:
                        original_feedback = processed_data[student_id].get('feedback')
            except Exception as e:
                print(f"Error loading processed data: {e}")
        
        # Format RIL feedback to maintain the same structure
        formatted_feedback = format_ril_feedback(
            original_feedback or submission.get('feedback', {}),
            message,
            ai_response
        )
        
        # Update the feedback with the formatted version
        submission['formatted_ril_feedback'] = formatted_feedback
        
        # Update processed_data.json with the new feedback
        update_processed_data_with_ril(student_id, formatted_feedback)
        
        # Calculate alignment scores with instructor feedback
        try:
            evaluator = FeedbackEvaluation()
            
            # Use generated feedback and latest instructor message
            generated_feedback = submission.get('generated_feedback', '')
            
            # Calculate alignment scores
            print(f"Explicitly calculating alignment scores between generated feedback and instructor message")
            evaluation_results = evaluator.evaluate_feedback_quality(generated_feedback, message)
            
            # Keep any existing data in feedback_evaluation
            current_evaluation = submission.get('feedback_evaluation', {})
            for key, value in evaluation_results.items():
                current_evaluation[key] = value
                
            # Store the updated evaluation results
            submission['feedback_evaluation'] = current_evaluation
            
            print(f"Calculated ROUGE-L: {evaluation_results['alignment_scores']['rouge'].get('rouge-l-f', 0)}")
            print(f"Calculated BLEU: {evaluation_results['alignment_scores']['bleu_score']}")
            print(f"Calculated BERT: {evaluation_results['alignment_scores']['bert_score']}")
        except Exception as e:
            print(f"Error calculating feedback metrics: {e}")
            import traceback
            traceback.print_exc()
        
        # Update cache if needed
        if student_id in feedback_cache:
            feedback_cache[student_id] = submission
        
        # Save submissions
        save_submissions()
        
        return jsonify({
            "success": True,
            "response": ai_response
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})

def generate_ai_response(student_id, instructor_message, submission):
    """Generate AI response based on instructor message and context."""
    try:
        # Import the reinforcement learning module
        from reinforcement import FeedbackReinforcementLearning
        rl = FeedbackReinforcementLearning()
        
        # Get the original feedback
        original_feedback = submission.get('generated_feedback', '')
        
        # Check if this is a request to remove a specific violation
        remove_violation = False
        violation_type = None
        
        if "remove" in instructor_message.lower():
            remove_violation = True
            # Look for which violation to remove
            for violation in ["srp", "ocp", "dip", "isp"]:
                if violation in instructor_message.lower():
                    violation_type = violation.upper()
                    break
            
            # Look for specific principles
            if "single responsibility" in instructor_message.lower():
                violation_type = "SRP"
            elif "open" in instructor_message.lower() and "closed" in instructor_message.lower():
                violation_type = "OCP"
            elif "dependency" in instructor_message.lower() and "inversion" in instructor_message.lower():
                violation_type = "DIP"
            elif "interface" in instructor_message.lower() and "segregation" in instructor_message.lower():
                violation_type = "ISP"
        
        # Create the prompt
        instruction_text = ""
        if remove_violation and violation_type:
            instruction_text = f"Please completely remove all mention of {violation_type} from the feedback. Delete any bullet points or sections that discuss this principle."
        else:
            instruction_text = "Please update the feedback according to the instructor's request."
        
        system_prompt = f"""
        You are an expert Java code reviewer. 

        Here is the current feedback:

        {original_feedback}

        The instructor has requested: "{instructor_message}"

        {instruction_text}

        Return the complete updated feedback with proper markdown formatting. Remember to remove original feedback from the response.
        Ensure each header has a blank line after it.
        """
        
        try:
            # Generate feedback with Ollama
            response = rl.ollama_client.generate(
                model=rl.config["ollama_model"],
                prompt=system_prompt
            )
            
            feedback_text = response['response'].strip()
            
            # Manual post-processing to ensure the first header has a line break
            feedback_text = re.sub(r'^(## [^\n]+)([^\n])', r'\1\n\n\2', feedback_text)
            
            return feedback_text
            
        except Exception as e:
            print(f"Error generating with Ollama: {e}")
            # Create a manual response if Ollama fails
            if remove_violation and violation_type:
                # Manually remove violation sections
                lines = original_feedback.split('\n')
                filtered_lines = []
                skip_section = False
                
                for line in lines:
                    if violation_type in line:
                        skip_section = True
                    elif skip_section and line.startswith('-'):
                        continue  # Skip bullet points in violation section
                    elif skip_section and (line.strip() == '' or line.startswith('#')):
                        skip_section = False
                    
                    if not skip_section:
                        filtered_lines.append(line)
                
                return '\n'.join(filtered_lines)
            else:
                return "I apologize, but I couldn't process your request. Please try again with more specific instructions."
    
    except Exception as e:
        print(f"Error in generate_ai_response: {e}")
        import traceback
        traceback.print_exc()
        return "An error occurred while generating a response. Please try again later."

@app.route('/reinforcement_dashboard')
def reinforcement_dashboard():
    """Show reinforcement learning dashboard with student feedback stats and quality metrics."""
    try:
        # Get all submissions with feedback
        submissions_with_feedback = {sid: data for sid, data in submissions.items() 
                                  if 'student_rating' in data or 'feedback_evaluation' in data}
        
        # Collect student ratings
        student_ratings = []
        thumbs_up_count = 0
        thumbs_down_count = 0
        
        # Collect alignment scores
        rouge_scores = []
        bleu_scores = []
        bert_scores = []
        
        # Submission IDs for reference
        submission_ids = []
        
        for sid, data in submissions_with_feedback.items():
            # Add student rating if available
            if 'student_rating' in data:
                rating = data['student_rating']
                student_ratings.append(rating)
                
                # Count thumbs up and down
                if rating == 1.0:
                    thumbs_up_count += 1
                else:
                    thumbs_down_count += 1
                    
                submission_ids.append(sid)
            
            # Add alignment scores if available
            if 'feedback_evaluation' in data and 'alignment_scores' in data['feedback_evaluation']:
                scores = data['feedback_evaluation']['alignment_scores']
                
                # Get ROUGE-L score
                if 'rouge' in scores and 'rouge-l-f' in scores['rouge']:
                    rouge_scores.append((sid, scores['rouge']['rouge-l-f']))
                
                # Get BLEU score
                if 'bleu_score' in scores:
                    bleu_scores.append((sid, scores['bleu_score']))
                
                # Get BERT score
                if 'bert_score' in scores:
                    bert_scores.append((sid, scores['bert_score']))
        
        # Prepare stats dictionary
        stats = {
            "status": "success" if student_ratings or rouge_scores else "no_data",
            "total_samples": len(student_ratings),
            "thumbs_up_count": thumbs_up_count,
            "thumbs_down_count": thumbs_down_count,
            "avg_rating": np.mean(student_ratings) if student_ratings else 0,
            "ratings": student_ratings
        }
        
        # Generate pie chart for student ratings distribution
        pie_chart_data = None
        has_trend_data = False
        
        if stats['status'] == 'success' and len(student_ratings) > 0:
            try:
                # Create a figure for the pie chart - SMALLER SIZE
                plt.figure(figsize=(5, 4))
                
                # Data for pie chart
                labels = ['Helpful', 'Not Helpful']
                sizes = [thumbs_up_count, thumbs_down_count]
                
                # Skip if all values are 0
                if sum(sizes) > 0:
                    # Custom colors for pie chart
                    colors = ['#4CAF50', '#F44336']  # Green for helpful, red for not helpful
                    
                    # Plot pie chart - simplified style
                    wedges, texts, autotexts = plt.pie(
                        sizes, 
                        labels=None,  # No labels inside the chart
                        colors=colors, 
                        autopct='%1.1f%%', 
                        shadow=False, 
                        startangle=90,
                        wedgeprops={'linewidth': 0.5, 'edgecolor': 'white'},
                        textprops={'fontsize': 12}
                    )
                    
                    # Style the percentage text
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontsize(12)
                        autotext.set_fontweight('bold')
                    
                    # Add a legend instead of labels on the pie
                    plt.legend(
                        labels,
                        loc="center right",
                        bbox_to_anchor=(1.15, 0.5),
                        frameon=False
                    )
                    
                    # Equal aspect ratio ensures that pie is drawn as a circle
                    plt.axis('equal')
                    plt.title('Student Feedback Distribution', fontsize=14, pad=10)
                    plt.tight_layout()
                    
                    # Save to base64 string
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                    buffer.seek(0)
                    pie_chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    plt.close()
                    
                    has_trend_data = True
            except Exception as e:
                print(f"Error generating pie chart: {e}")
                import traceback
                traceback.print_exc()
        
        # Generate line graph for quality metrics
        quality_chart_data = None
        has_quality_data = False
        
        if rouge_scores or bleu_scores or bert_scores:
            try:
                # Create figure for line graph
                plt.figure(figsize=(10, 6))
                
                # Get all unique submission IDs
                all_sids = sorted(set([x[0] for x in rouge_scores + bleu_scores + bert_scores]))
                
                # Simplify submission IDs for display
                sid_labels = [sid.split('_')[0] if '_' in sid else sid for sid in all_sids]
                sid_labels = [sid[:12] if len(sid) > 12 else sid for sid in sid_labels]
                
                # Prepare x-axis positions
                x = np.arange(len(all_sids))
                
                # Prepare data for each metric as lines
                rouge_values = [dict(rouge_scores).get(sid, 0) for sid in all_sids]
                bleu_values = [dict(bleu_scores).get(sid, 0) for sid in all_sids]
                bert_values = [dict(bert_scores).get(sid, 0) for sid in all_sids]
                
                # Plot lines for each metric with markers
                if rouge_values:
                    plt.plot(x, rouge_values, 'o-', label='ROUGE-L', color='#4285F4', linewidth=2, markersize=8)
                if bleu_values:
                    plt.plot(x, bleu_values, 's-', label='BLEU', color='#34A853', linewidth=2, markersize=8)
                if bert_values:
                    plt.plot(x, bert_values, '^-', label='BERTScore', color='#FBBC05', linewidth=2, markersize=8)
                
                # Format plot
                plt.xlabel('Submissions', fontsize=12, fontweight='bold')
                plt.ylabel('Score', fontsize=12, fontweight='bold')
                plt.title('Feedback Quality Metrics', fontsize=16, fontweight='bold', pad=20)
                plt.xticks(x, sid_labels, rotation=45)
                plt.yticks(np.arange(0, 1.1, 0.1))
                plt.ylim(0, 1.05)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.legend(fontsize=12)
                
                # Add light background grid
                plt.grid(True, linestyle='--', alpha=0.3)
                
                # Tight layout
                plt.tight_layout()
                
                # Save to base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                buffer.seek(0)
                quality_chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                
                has_quality_data = True
            except Exception as e:
                print(f"Error generating quality metrics chart: {e}")
                import traceback
                traceback.print_exc()
        
        return render_template('reinforcement_dashboard.html', 
                              stats=stats,
                              has_trend_data=has_trend_data,
                              pie_chart_data=pie_chart_data,
                              has_quality_data=has_quality_data,
                              quality_chart_data=quality_chart_data)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Error loading reinforcement dashboard: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.context_processor
def inject_now():
    """Add the current timestamp to all templates."""
    return {'now': datetime.now()}

# Resource cleanup function
def cleanup_resources():
    """Clean up resources when the application exits"""
    print("Cleaning up resources...")
    # Clear PyTorch cache
    if 'torch' in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()

atexit.register(cleanup_resources)

if __name__ == '__main__':
    app.run(debug=True, port=5000)