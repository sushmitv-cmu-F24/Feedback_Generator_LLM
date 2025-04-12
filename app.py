import os
import json
import base64
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from evaluation_phase import extract_java_files_from_zip, evaluate_submission, inspect_faiss_index
from feedback_metrics import FeedbackMetrics
from feedback_evaluation import FeedbackEvaluation
from feedback_reinforcement import FeedbackReinforcementLearning
import atexit
import sys
import gc
import re
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np

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

# Initialize feedback metrics, evaluation, and reinforcement learning
metrics = FeedbackMetrics()
evaluator = FeedbackEvaluation()
reinforcement = FeedbackReinforcementLearning(config_file=app.config['RL_CONFIG'])

# Cache for generated feedback
feedback_cache = {}

@app.route('/')
def index():
    """Home page with upload form and list of submissions."""
    submissions = []
    
    # Get list of uploaded submissions
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.endswith('.zip'):
            student_id = filename.split('_')[0]
            submission_time = os.path.getmtime(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            submission_date = datetime.fromtimestamp(submission_time).strftime('%Y-%m-%d %H:%M')
            
            # Check if feedback has been generated
            has_feedback = student_id in feedback_cache
            
            submissions.append({
                'student_id': student_id,
                'filename': filename,
                'date': submission_date,
                'has_feedback': has_feedback
            })
    
    # Sort submissions by date (newest first)
    submissions.sort(key=lambda x: x['date'], reverse=True)
    
    return render_template('index.html', submissions=submissions)

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

@app.route('/view/<student_id>')
def view_feedback(student_id):
    """View generated feedback for a student."""
    try:
        import psutil
        print(f"Memory usage before processing: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    except ImportError:
        print("psutil not installed, skipping memory usage monitoring")
    
    # Check if feedback is already in cache
    if student_id in feedback_cache:
        return render_template(
            'view_feedback.html',
            student_id=student_id,
            feedback=feedback_cache[student_id],
            java_files=feedback_cache[student_id].get('java_files', {})
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
    
    # Generate feedback
    evaluation_result = evaluate_submission(java_files)
    evaluation_result['java_files'] = java_files
    
    # Add quality metrics
    feedback_text = evaluation_result['generated_feedback']
    quality_metrics = metrics.evaluate_feedback(
        feedback_text, 
        evaluation_result.get('detected_violations', [])
    )
    evaluation_result['quality_metrics'] = quality_metrics
    
    # Add alignment metrics if reference feedback is available
    reference_feedback = None
    if evaluation_result.get('retrieved_feedbacks') and len(evaluation_result['retrieved_feedbacks']) > 0:
        reference_feedback = evaluation_result['retrieved_feedbacks'][0]
        
        # Debug the reference feedback
        print(f"Reference feedback type: {type(reference_feedback)}")
        if isinstance(reference_feedback, dict):
            print(f"Reference feedback keys: {reference_feedback.keys()}")
            # Check if feedback is nested too deeply
            for key, value in reference_feedback.items():
                print(f"Key: {key}, Value type: {type(value)}")
                if isinstance(value, dict):
                    print(f"Nested keys: {value.keys()}")
        else:
            print(f"Reference feedback length: {len(reference_feedback) if reference_feedback else 0}")
    
    # Handle the case where reference_feedback might be a nested dictionary
    if isinstance(reference_feedback, dict) and 'overall_assessment' in reference_feedback:
        # This is already properly formatted
        pass
    elif isinstance(reference_feedback, dict) and 'feedback' in reference_feedback:
        # The feedback is nested one level deeper
        reference_feedback = reference_feedback['feedback']
    
    # Calculate feedback evaluation metrics including alignment scores
    feedback_evaluation = evaluator.evaluate_feedback_quality(feedback_text, reference_feedback)
    evaluation_result['feedback_evaluation'] = feedback_evaluation
    evaluation_result['has_reference_feedback'] = (reference_feedback is not None)
    
    # Format the feedback as markdown with proper headers for nicer display
    formatted_feedback = format_feedback_as_markdown(evaluation_result['generated_feedback'])
    evaluation_result['generated_feedback'] = formatted_feedback
    
    # Store in cache
    feedback_cache[student_id] = evaluation_result
    
    try:
        import psutil
        print(f"Memory usage after processing: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    except ImportError:
        pass
    
    return render_template(
        'view_feedback.html',
        student_id=student_id,
        feedback=evaluation_result,
        java_files=java_files
    )

@app.route('/review/<student_id>')
def review_feedback(student_id):
    """Show review interface for instructor to provide feedback on model output."""
    # Check if feedback is in cache
    if student_id not in feedback_cache:
        # Find the ZIP file
        zip_file = None
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.startswith(student_id) and filename.endswith('.zip'):
                zip_file = filename
                break
        
        if not zip_file:
            flash(f'No submission found for student {student_id}')
            return redirect(url_for('index'))
        
        # Generate feedback if not in cache
        zip_path = os.path.join(app.config['UPLOAD_FOLDER'], zip_file)
        java_files = extract_java_files_from_zip(zip_path)
        
        if not java_files:
            flash(f'No Java files found in {zip_file}')
            return redirect(url_for('index'))
        
        # Generate feedback
        evaluation_result = evaluate_submission(java_files)
        evaluation_result['java_files'] = java_files
        feedback_cache[student_id] = evaluation_result
    else:
        # Use cached feedback
        evaluation_result = feedback_cache[student_id]
        java_files = evaluation_result.get('java_files', {})
    
    # Extract model-generated feedback and metrics
    model_feedback = evaluation_result.get('generated_feedback', '')
    metrics_data = evaluation_result.get('quality_metrics', {})
    
    # Render review template
    return render_template(
        'review_feedback.html',
        student_id=student_id,
        model_feedback=model_feedback,
        java_files=java_files,
        metrics=metrics_data
    )

@app.route('/submit_feedback_review/<student_id>', methods=['POST'])
def submit_feedback_review(student_id):
    """Process instructor feedback for reinforcement learning."""
    # Extract form data
    rating = float(request.form.get('rating', 0)) / 100.0  # Convert to 0-1 scale
    corrected_feedback = request.form.get('corrected_feedback', '')
    additional_comments = request.form.get('additional_comments', '')
    issues = request.form.getlist('issues')
    
    # Extract detailed ratings
    accuracy_rating = int(request.form.get('accuracy_rating', 3))
    specificity_rating = int(request.form.get('specificity_rating', 3))
    actionability_rating = int(request.form.get('actionability_rating', 3))
    completeness_rating = int(request.form.get('completeness_rating', 3))
    
    # Get the original submission and feedback
    if student_id not in feedback_cache:
        flash('Submission not found in cache. Please try again.')
        return redirect(url_for('index'))
    
    evaluation_result = feedback_cache[student_id]
    
    # Prepare data for reinforcement learning
    original_code = ""
    for file_content in evaluation_result.get('java_files', {}).values():
        original_code += file_content + "\n\n"
    
    model_feedback = evaluation_result.get('generated_feedback', '')
    
    # Create metadata for training
    feedback_meta = {
        'accuracy_rating': accuracy_rating,
        'specificity_rating': specificity_rating,
        'actionability_rating': actionability_rating,
        'completeness_rating': completeness_rating,
        'issues': issues,
        'additional_comments': additional_comments
    }
    
    # Store the feedback pair for reinforcement learning
    reinforcement.store_feedback_pair(
        submission_id=student_id,
        original_code=original_code,
        model_feedback=model_feedback,
        instructor_feedback=corrected_feedback,
        instructor_rating=rating,
        feedback_meta=feedback_meta
    )
    
    # Flash success message
    flash('Feedback successfully submitted for reinforcement learning. Thank you!')
    
    # Redirect to dashboard
    return redirect(url_for('reinforcement_dashboard'))

@app.route('/reinforcement')
def reinforcement_dashboard():
    """Dashboard showing reinforcement learning progress and stats."""
    # Get learning statistics
    stats = reinforcement.get_learning_stats()
    
    # If there's enough data, prepare for visualization
    has_trend_data = False
    trend_data = None
    
    if stats.get('status') == 'success' and len(stats.get('ratings', [])) > 1:
        has_trend_data = True
        
        # Prepare trend chart data
        ratings = stats.get('ratings', [])
        timestamps = stats.get('timestamps', [])
        moving_avg = stats.get('moving_avg', [])
        
        # Create a simple plot with matplotlib
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(ratings)), ratings, 'o-', label='Instructor Ratings')
        
        if moving_avg:
            # Adjust x-coordinates for moving average (centered window)
            window_size = 5
            ma_x = [i + window_size//2 for i in range(len(moving_avg))]
            if len(ma_x) > 0 and len(moving_avg) > 0 and max(ma_x) < len(ratings):
                plt.plot(ma_x, moving_avg, 'r-', label='Moving Average')
        
        plt.xlabel('Submission Index')
        plt.ylabel('Rating (0-1)')
        plt.title('Feedback Rating Trend')
        plt.legend()
        plt.grid(True)
        
        # Convert plot to base64 for embedding in HTML
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        trend_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
    
    # Check if we have enough data for fine-tuning
    can_fine_tune = stats.get('samples_needed_for_training', 999) <= 0
    
    # Get distribution of ratings if available
    rating_distribution = None
    if stats.get('status') == 'success' and len(stats.get('ratings', [])) > 0:
        ratings = np.array(stats.get('ratings', []))
        bins = [0, 0.25, 0.5, 0.75, 1.0]
        bin_labels = ['Poor (0-0.25)', 'Fair (0.25-0.5)', 'Good (0.5-0.75)', 'Excellent (0.75-1.0)']
        
        hist, _ = np.histogram(ratings, bins=bins)
        rating_distribution = {
            'counts': hist.tolist(),
            'labels': bin_labels
        }
    
    return render_template(
        'reinforcement_dashboard.html',
        stats=stats,
        has_trend_data=has_trend_data,
        trend_data=trend_data,
        can_fine_tune=can_fine_tune,
        rating_distribution=rating_distribution
    )

@app.route('/start_fine_tuning', methods=['POST'])
def start_fine_tuning():
    """Start the fine-tuning process."""
    # Prepare data for fine-tuning
    prep_result = reinforcement.prepare_fine_tuning_data()
    
    if prep_result.get('status') == 'insufficient_data':
        flash(f"Not enough data for fine-tuning. Need {prep_result.get('samples_needed')} more samples.")
        return redirect(url_for('reinforcement_dashboard'))
    
    # Start fine-tuning (this would be implemented differently in production)
    fine_tune_result = reinforcement.fine_tune_model()
    
    # Store results
    session['fine_tune_result'] = fine_tune_result
    
    if fine_tune_result.get('status') == 'not_implemented':
        flash("Fine-tuning functionality requires additional setup. See documentation.")
    else:
        flash("Fine-tuning process started. This may take some time.")
    
    return redirect(url_for('reinforcement_dashboard'))

@app.route('/inspect_index')
def inspect_index():
    """Debug page to view FAISS index information."""
    results = []
    
    # Inspect the FAISS index
    inspect_faiss_index()
    
    # Check processed data
    try:
        with open(app.config['PROCESSED_DATA'], 'r') as f:
            processed_data = json.load(f)
            results.append(f"Processed data contains {len(processed_data)} entries")
            if len(processed_data) > 0:
                student_ids = list(processed_data.keys())
                results.append(f"Sample student IDs: {', '.join(student_ids[:5])}")
                
                # Check if any submission has feedback
                has_feedback = False
                for student_id, data in processed_data.items():
                    if data.get('feedback'):
                        has_feedback = True
                        results.append(f"Found feedback for {student_id}")
                        break
                
                if not has_feedback:
                    results.append("⚠️ No feedback found in any processed data entry")
    except Exception as e:
        results.append(f"Error reading processed data: {e}")
    
    return render_template('debug.html', results=results)

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