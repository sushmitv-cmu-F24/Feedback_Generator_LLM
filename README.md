# Automated Assignment Feedback System

This system uses NLP and LLMs to automatically generate feedback for Java programming assignments based on SOLID principles.

## Folder Structure

```
automated_assignment_feedback/
├── app.py                    # Web interface for instructors
├── training_phase.py         # Processes past assignments and feedbacks
├── evaluation_phase.py       # Evaluates new submissions
├── feedback_metrics.py       # Analyzes quality of generated feedback
├── data/                     # Data storage
│   ├── comments/             # Instructor feedback PDFs
│   ├── submissions/          # Past student submissions (ZIPs)
│   ├── evaluation/           # New submissions to evaluate
│   ├── processed_data.json   # Database of processed submissions
│   └── feedback_embeddings.faiss # Vector embeddings for similarity search
│   ├── evaluation_results.json   # Database of newly evaluated submissions
└── templates/                # Web interface templates
    ├── index.html            # Main page listing submissions
    ├── loading.html          # Loading page while generating feedback
    └── view_feedback.html    # Detailed feedback view
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/automated_assignment_feedback.git
   cd automated_assignment_feedback
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install Ollama (required for CodeLlama):
   - Follow instructions at https://ollama.ai/download
   - Pull the CodeLlama model: `ollama pull codellama:13b`
   - Run Ollama: `ollama serve`

5. Create necessary directories:
   ```
   mkdir -p data/comments data/submissions data/evaluation
   ```

## Usage

### Training Phase

1. Place instructor feedback PDFs in `data/comments/`
2. Place past student submissions (ZIPs) in `data/submissions/`
3. Run the training phase:
   ```
   python training_phase.py
   ```

### Web Interface

1. Start the web application:
   ```
   python app.py
   ```

2. Open your browser and go to: http://localhost:5000

3. Upload new submissions through the interface and generate feedback

### Command Line Evaluation

To evaluate all submissions in the evaluation folder:
```
python evaluation_phase.py
```

To evaluate a specific submission:
```
python evaluation_phase.py path/to/submission.zip
```

## Requirements

- Python 3.8+
- PyPDF2, spaCy, NLTK, FAISS, sentence-transformers
- Flask for web interface
- Ollama for LLM inference

## Next Steps

1. Optimize the evaluation process
2. Fine tune the model for feedback generation
3. Add a heading using index.html
4. Dockerize and deploy the service
5. Add metrics like BLEU, create list of 4-5 metrics