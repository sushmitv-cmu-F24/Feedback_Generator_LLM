import os
import json
import base64
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from evaluation_phase import extract_java_files_from_zip, evaluate_submission, inspect_faiss_index
from feedback_metrics import FeedbackMetrics
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
metrics = FeedbackMetrics()
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

def save_submissions():
    """Save submissions data to disk."""
    submissions_file = os.path.join('data', 'submissions_data.json')
    
    try:
        with open(submissions_file, 'w') as f:
            json.dump(submissions, f, indent=4)
        print(f"Saved {len(submissions)} submissions to disk")
    except Exception as e:
        print(f"Error saving submissions data: {e}")

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

@app.route('/view/<student_id>')
def view_feedback(student_id):
    """View generated feedback for a student."""
    try:
        import psutil
        print(f"Memory usage before processing: {psutil.Process().memory_info().rss / 1024 / 1024:.2f} MB")
    except ImportError:
        print("psutil not installed, skipping memory usage monitoring")
    
    # Check if feedback is already in cache or submissions
    if student_id in feedback_cache:
        return render_template(
            'view_feedback.html',
            student_id=student_id,
            feedback=feedback_cache[student_id],
            java_files=feedback_cache[student_id].get('java_files', {})
        )
    elif student_id in submissions:
        return render_template(
            'view_feedback.html',
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
    
    # Format the feedback as markdown with proper headers for nicer display
    formatted_feedback = format_feedback_as_markdown(evaluation_result['generated_feedback'])
    evaluation_result['generated_feedback'] = formatted_feedback
    
    # Store in cache and submissions
    feedback_cache[student_id] = evaluation_result
    submissions[student_id] = evaluation_result
    
    # Save the updated submissions
    save_submissions()
    
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
    """Show simplified review interface for instructor to provide feedback on model output."""
    # Check if feedback is in cache or submissions
    if student_id not in feedback_cache and student_id not in submissions:
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
        submissions[student_id] = evaluation_result
        save_submissions()
    else:
        # Use cached feedback
        evaluation_result = feedback_cache[student_id] if student_id in feedback_cache else submissions[student_id]
        java_files = evaluation_result.get('java_files', {})
    
    # Extract model-generated feedback 
    model_feedback = evaluation_result.get('generated_feedback', '')
    
    # Render simplified review template
    return render_template(
        'review_feedback.html',
        student_id=student_id,
        model_feedback=model_feedback,
        java_files=java_files
    )

@app.route('/submit_feedback_review/<student_id>', methods=['POST'])
def submit_feedback_review(student_id):
    """Simplified handler for instructor feedback submission."""
    if student_id not in submissions:
        flash(f"Student submission {student_id} not found.", "danger")
        return redirect(url_for('index'))
    
    # Get form data - simplified to just the essentials
    rating = int(request.form.get('rating', 0)) / 100.0  # Convert 0-100 to 0-1 scale
    corrected_feedback = request.form.get('corrected_feedback', '')
    
    # Get the submission data
    submission = submissions[student_id]
    
    # Save the feedback to the submission
    submission['instructor_feedback'] = corrected_feedback
    submission['instructor_rating'] = rating
    
    # Store feedback for reinforcement learning
    try:
        # Get the original code and model feedback
        original_code = ""
        for file_content in submission.get('java_files', {}).values():
            original_code += file_content + "\n\n"
            
        model_feedback = submission.get('generated_feedback', '')
        
        # Initialize RL module and store the feedback pair
        rl = FeedbackReinforcementLearning()
        rl.store_feedback_pair(
            submission_id=student_id,
            original_code=original_code,
            model_feedback=model_feedback,
            instructor_feedback=corrected_feedback,
            instructor_rating=rating
        )
        
        # Check if we have enough samples for improvement
        stats = rl.get_learning_stats()
        if stats['status'] == 'success':
            if stats['samples_needed_for_improvement'] <= 0:
                flash("Enough feedback collected! Future feedback will be improved based on your input.", "success")
            else:
                flash(f"Feedback stored. Need {stats['samples_needed_for_improvement']} more examples for improvement.", "info")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Error storing feedback: {str(e)}", "warning")
    
    # Save all submissions
    save_submissions()
    
    flash("Feedback review submitted successfully!", "success")
    return redirect(url_for('index'))

@app.route('/reinforcement_dashboard')
def reinforcement_dashboard():
    """Show simplified reinforcement learning dashboard."""
    try:
        # Initialize the reinforcement learning module
        rl = FeedbackReinforcementLearning()
        
        # Get learning stats
        stats = rl.get_learning_stats()
        
        # Generate a simple trend chart if we have timestamps and ratings
        trend_data = None
        has_trend_data = False
        
        if stats['status'] == 'success' and len(stats.get('timestamps', [])) > 1:
            try:
                import matplotlib
                matplotlib.use('Agg')  # Use non-interactive backend
                import matplotlib.pyplot as plt
                import io
                import base64
                from datetime import datetime
                
                # Parse timestamps and format for display
                dates = [datetime.fromisoformat(ts) for ts in stats['timestamps']]
                formatted_dates = [dt.strftime('%m/%d') for dt in dates]
                
                # Create a simple figure and plot data
                plt.figure(figsize=(8, 4))
                plt.plot(formatted_dates, stats['ratings'], marker='o', linestyle='-', color='#0d6efd')
                
                # Add moving average
                if len(stats.get('moving_avg', [])) > 0:
                    plt.plot(formatted_dates, stats['moving_avg'], linestyle='--', color='#dc3545', label='Average')
                
                # Format plot
                plt.xlabel('Date')
                plt.ylabel('Rating')
                plt.title('Feedback Ratings')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.ylim(0, 1.1)
                plt.legend()
                
                # Save to base64 string
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                trend_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                plt.close()
                
                has_trend_data = True
            except Exception as e:
                print(f"Error generating trend chart: {e}")
        
        return render_template('reinforcement_dashboard.html', 
                              stats=stats,
                              has_trend_data=has_trend_data,
                              trend_data=trend_data)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Error loading reinforcement dashboard: {str(e)}", "danger")
        return redirect(url_for('index'))

@app.route('/generate_improved_feedback/<student_id>', methods=['GET'])
def generate_improved_feedback(student_id):
    """Generate feedback using the reinforcement learning system."""
    try:
        # Check if we have a valid student submission
        if student_id not in submissions:
            flash(f"Student submission {student_id} not found.", "danger")
            return redirect(url_for('index'))
        
        # Get the submission data
        submission = submissions[student_id]
        
        # Extract the Java files
        java_files = submission.get('java_files', {})
        if not java_files:
            flash("No Java files found for this submission.", "warning")
            return redirect(url_for('view_feedback', student_id=student_id))
        
        # Combine all Java files into one for analysis
        combined_code = ""
        for file_path, content in java_files.items():
            combined_code += f"// {file_path}\n{content}\n\n"
        
        # Get detected violations if available
        detected_violations = submission.get('detected_violations', [])
        
        # Initialize RL module
        rl = FeedbackReinforcementLearning()
        
        # Generate improved feedback
        improved_feedback = rl.generate_improved_feedback(combined_code, detected_violations)
        
        if improved_feedback:
            # Update the submission with the improved feedback
            submission['improved_feedback'] = improved_feedback
            submission['has_improved_feedback'] = True
            
            # Save submissions data
            save_submissions()
            
            flash("Improved feedback generated successfully!", "success")
        else:
            flash("Failed to generate improved feedback.", "warning")
        
        # Redirect to the feedback view
        return redirect(url_for('view_feedback', student_id=student_id))
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        flash(f"Error generating improved feedback: {str(e)}", "danger")
        return redirect(url_for('index'))

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