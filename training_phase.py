import os
import re
import json
import zipfile
import spacy
import javalang
import nltk
import faiss
import numpy as np
from PyPDF2 import PdfReader
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer

# Load models
nltk.download('vader_lexicon', quiet=True)
nlp = spacy.load("en_core_web_sm")
sia = SentimentIntensityAnalyzer()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # FAISS embeddings

CONFIG = {
    "comments_folder": "./data/comments/",
    "submissions_folder": "./data/submissions/",
    "output_file": "./data/processed_data.json",
    "embeddings_file": "./data/feedback_embeddings.faiss",
    "professor_name": "Leonardo da Silva Sousa",
    "feedback_patterns": {
        "general_comments": r"(.*?)(?=The refactoring for STEP|The incomplete refactored version)",
        "solid_violations": r"(The refactoring for STEP.*?)(?=It would help if you had used|With DI, UserService)",
        "suggestions": r"(It would help if you had used.*)"
    },
    # Assignment-specific requirements taken from professor feedback
    "assignment_requirements": {
        "package_structure": {
            "required_packages": ["solid.persistence.drivers"],
            "interface_locations": {
                "DatabaseDriver": "solid.persistence"
            },
            "implementation_locations": {
                "PostgresDriver": "solid.persistence.drivers"
            }
        },
        "dependency_injection": {
            "services_requiring_di": ["UserService"],
            "dependencies": {
                "UserService": ["DatabaseDriver"]
            }
        },
        "solid_principles": {
            "srp": {
                "max_methods": 7,  # Threshold for SRP violation
                "max_concerns": 2   # Max number of different concerns in a class
            }
        }
    }
}

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs(CONFIG["comments_folder"], exist_ok=True)
os.makedirs(CONFIG["submissions_folder"], exist_ok=True)

### --- STEP 1: Extract & Preprocess Feedback --- ###
def clean_text(text):
    """Removes timestamps & extra whitespace from feedback."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\(\w+ \d{1,2}, \d{4} at \d{1,2}:\d{2}(am|pm)\)', '', text)
    return text

def extract_feedback_sections(feedback_text):
    """Breaks feedback into structured categories using regex patterns."""
    if not feedback_text:
        return {key: None for key in CONFIG["feedback_patterns"]}
        
    sections = {key: None for key in CONFIG["feedback_patterns"]}
    for key, pattern in CONFIG["feedback_patterns"].items():
        match = re.search(pattern, feedback_text, re.DOTALL)
        sections[key] = clean_text(match.group(1)) if match else None
    if not any(sections.values()):
        sections["general_comments"] = clean_text(feedback_text)
    return sections

def extract_student_info_from_pdf(pdf_path):
    """Extracts student name and structured feedback from PDFs."""
    try:
        reader = PdfReader(pdf_path)
        pdf_text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        
        name_match = re.search(r"Student:\s+(.+)", pdf_text)
        student_name = name_match.group(1).strip() if name_match else None

        feedback_start = pdf_text.find(CONFIG["professor_name"])
        raw_feedback = pdf_text[feedback_start:].strip() if feedback_start != -1 else None

        structured_feedback = extract_feedback_sections(raw_feedback) if raw_feedback else None
        return student_name, structured_feedback
    except Exception as e:
        print(f"⚠️ Error processing PDF {pdf_path}: {e}")
        return None, None

def extract_name_from_zip(filename):
    """Extract student name from ZIP filename (lastnamefirstname format)."""
    try:
        return filename.split("_")[0]
    except IndexError:
        print(f"⚠️ Could not extract student name from {filename}")
        return filename.replace(".zip", "")

def extract_java_files_from_zip(zip_path):
    """Extract Java files from ZIP submissions while filtering out system files."""
    java_files = {}
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith(".java") and not file.startswith("__MACOSX"):
                    with zip_ref.open(file) as f:
                        try:
                            content = f.read().decode("utf-8", errors="ignore").strip()
                            if content:  # Ensure file is not empty
                                java_files[file] = content
                        except Exception as e:
                            print(f"⚠️ Skipping file {file} due to decoding error: {e}")
        return java_files
    except Exception as e:
        print(f"⚠️ Error extracting files from {zip_path}: {e}")
        return {}

### --- STEP 2: AST-Based Java Code Parsing --- ###
def parse_java_code(java_code):
    """Extracts class and method names using AST parsing while tracking parent classes."""
    try:
        if "class " not in java_code and "interface " not in java_code:
            return {}  # Skip files that aren't Java classes
            
        tree = javalang.parse.parse(java_code)
        classes = {}  # Stores class → methods mapping
        current_class = None  # Track last encountered class

        for path, node in tree:
            if isinstance(node, javalang.tree.ClassDeclaration):
                current_class = node.name
                classes[current_class] = {
                    "methods": [],
                    "fields": [],
                    "extends": node.extends.name if node.extends else None,
                    "implements": [i.name for i in node.implements] if node.implements else []
                }
                
            elif isinstance(node, javalang.tree.InterfaceDeclaration):
                current_class = node.name
                classes[current_class] = {
                    "methods": [],
                    "type": "interface",
                    "extends": [e.name for e in node.extends] if node.extends else []
                }

            elif isinstance(node, javalang.tree.MethodDeclaration):
                if current_class:  # Associate method with last seen class
                    method_info = {
                        "name": node.name,
                        "return_type": str(node.return_type) if node.return_type else "void",
                        "modifiers": [str(m) for m in node.modifiers],
                        "parameters": [str(p.type) for p in node.parameters]
                    }
                    classes[current_class]["methods"].append(method_info)
                    
            elif isinstance(node, javalang.tree.FieldDeclaration):
                if current_class and "fields" in classes[current_class]:
                    for declarator in node.declarators:
                        field_info = {
                            "name": declarator.name,
                            "type": str(node.type),
                            "modifiers": [str(m) for m in node.modifiers]
                        }
                        classes[current_class]["fields"].append(field_info)
                        
            # Extract object instantiations to detect DI violations
            elif isinstance(node, javalang.tree.ClassCreator):
                if current_class and node.type.name == "PostgresDriver":
                    # Track where PostgresDriver is instantiated
                    classes[current_class]["instantiates_postgres"] = True
        
        return classes
    except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError) as e:
        print(f"⚠️ Java parsing error: {e}")
        return {}
    except Exception as e:
        print(f"⚠️ Unexpected error parsing Java code: {e}")
        return {}

def extract_package_structure(java_files):
    """Extract package structure from Java files."""
    package_structure = {}
    class_packages = {}
    
    for file_path, content in java_files.items():
        try:
            # Extract package declaration
            package_match = re.search(r'package\s+([\w.]+);', content)
            if package_match:
                package_name = package_match.group(1)
                
                # Extract class or interface name
                class_match = re.search(r'(class|interface)\s+(\w+)', content)
                if class_match:
                    entity_type = class_match.group(1)
                    class_name = class_match.group(2)
                    
                    if package_name not in package_structure:
                        package_structure[package_name] = {"classes": [], "interfaces": []}
                    
                    if entity_type == "class":
                        package_structure[package_name]["classes"].append(class_name)
                    else:
                        package_structure[package_name]["interfaces"].append(class_name)
                    
                    # Map class to package
                    class_packages[class_name] = package_name
        except Exception as e:
            print(f"⚠️ Error extracting package structure from {file_path}: {e}")
    
    return package_structure, class_packages

def extract_class_and_method_names(text):
    """Extract class and method names from feedback using regex and NLP."""
    class_pattern = re.findall(r"\b[A-Z][a-zA-Z0-9_]*\b", text)
    method_pattern = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*\(", text)
    return list(set(class_pattern)), [m[:-1] for m in list(set(method_pattern))]

### --- STEP 3: Sentiment & SOLID Analysis --- ###
def analyze_sentiment(feedback_text):
    """Analyze sentiment of feedback text."""
    if not feedback_text:
        return "neutral", 0.0
        
    sentiment_score = sia.polarity_scores(feedback_text)
    sentiment_label = "positive" if sentiment_score['compound'] > 0.05 else \
                      "negative" if sentiment_score['compound'] < -0.05 else "neutral"
    return sentiment_label, sentiment_score['compound']

def categorize_solid_violations(feedback_text):
    """Identify SOLID principle violations from feedback text."""
    if not feedback_text:
        return []
        
    solid_categories = {
        "SRP": ["single responsibility", "too many responsibilities", "multiple reasons to change", "separate concerns"],
        "OCP": ["open/closed", "new functionality", "modify existing code", "extends without changing", "extension"],
        "LSP": ["liskov", "inheritance", "substitutable", "override", "subclass behavior", "polymorphism"],
        "ISP": ["interface segregation", "unrelated methods", "client-specific", "fat interface", "small interfaces"],
        "DIP": ["dependency inversion", "abstract", "tight coupling", "concrete implementation", "dependency injection"]
    }
    
    violations = []
    for principle, keywords in solid_categories.items():
        for keyword in keywords:
            if keyword.lower() in feedback_text.lower():
                violations.append(principle)
                break  # Only add each principle once
    
    return violations

def detect_package_structure_violations(package_structure, class_packages):
    """Detect violations in package structure based on assignment requirements."""
    violations = []
    
    # Check if required packages exist
    for required_package in CONFIG["assignment_requirements"]["package_structure"]["required_packages"]:
        if required_package not in package_structure:
            violations.append({
                "type": "package_structure",
                "description": f"Missing required package: {required_package}"
            })
    
    # Check if interfaces and implementations are in correct packages
    for interface, expected_package in CONFIG["assignment_requirements"]["package_structure"]["interface_locations"].items():
        if interface in class_packages:
            actual_package = class_packages[interface]
            if actual_package != expected_package:
                violations.append({
                    "type": "package_structure",
                    "description": f"Interface {interface} should be in package {expected_package}, but found in {actual_package}"
                })
    
    for implementation, expected_package in CONFIG["assignment_requirements"]["package_structure"]["implementation_locations"].items():
        if implementation in class_packages:
            actual_package = class_packages[implementation]
            if actual_package != expected_package:
                violations.append({
                    "type": "package_structure",
                    "description": f"Implementation {implementation} should be in package {expected_package}, but found in {actual_package}"
                })
    
    return violations

def detect_di_violations(parsed_classes, java_files):
    """Detect dependency injection violations."""
    violations = []
    
    # Check if services are using constructor injection for dependencies
    for service_name in CONFIG["assignment_requirements"]["dependency_injection"]["services_requiring_di"]:
        service_found = False
        
        # Scan all files for the service
        for file_path, content in java_files.items():
            if service_name in content:
                service_found = True
                # Check if service directly instantiates dependencies
                for dependency in CONFIG["assignment_requirements"]["dependency_injection"]["dependencies"].get(service_name, []):
                    # Look for patterns like "new PostgresDriver()" or "private PostgresDriver"
                    if f"new {dependency.replace('Driver', 'Driver()')}" in content or f"private {dependency.replace('Driver', 'Driver ')}" in content:
                        violations.append({
                            "type": "dependency_injection",
                            "description": f"{service_name} directly instantiates {dependency} instead of using dependency injection"
                        })
                        break
                
                # Check for constructor injection pattern
                if not re.search(fr"private\s+\w+\s+\w+Driver\s*;.*public\s+{service_name}\s*\(\s*\w+Driver", content, re.DOTALL):
                    for dependency in CONFIG["assignment_requirements"]["dependency_injection"]["dependencies"].get(service_name, []):
                        violations.append({
                            "type": "dependency_injection",
                            "description": f"{service_name} missing constructor injection for {dependency}"
                        })
                break
        
        if not service_found:
            violations.append({
                "type": "dependency_injection",
                "description": f"Required service {service_name} not found"
            })
    
    return violations

def detect_solid_violations_from_code(parsed_classes):
    """Detect SOLID violations from parsed code structure."""
    violations = []
    
    # Check for SRP violations based on method count and concern diversity
    for class_name, class_info in parsed_classes.items():
        if "type" in class_info and class_info["type"] == "interface":
            continue  # Skip interfaces
        
        # Check method count
        method_count = len(class_info.get("methods", []))
        if method_count > CONFIG["assignment_requirements"]["solid_principles"]["srp"]["max_methods"]:
            violations.append({
                "principle": "SRP",
                "class": class_name,
                "reason": f"Class has {method_count} methods, exceeding the maximum of {CONFIG['assignment_requirements']['solid_principles']['srp']['max_methods']}"
            })
        
        # Analyze method names to detect multiple concerns
        method_concerns = {}
        for method in class_info.get("methods", []):
            method_name = method["name"]
            if method_name.startswith("get") or method_name.startswith("set"):
                method_concerns["accessor"] = method_concerns.get("accessor", 0) + 1
            elif method_name.startswith("save") or method_name.startswith("delete") or method_name.startswith("update") or method_name.startswith("query"):
                method_concerns["persistence"] = method_concerns.get("persistence", 0) + 1
            elif method_name.startswith("validate") or method_name.startswith("check"):
                method_concerns["validation"] = method_concerns.get("validation", 0) + 1
            elif method_name.startswith("calculate") or method_name.startswith("compute"):
                method_concerns["calculation"] = method_concerns.get("calculation", 0) + 1
            elif method_name.startswith("format") or method_name.startswith("display") or method_name.startswith("print"):
                method_concerns["presentation"] = method_concerns.get("presentation", 0) + 1
            elif method_name.startswith("send") or method_name.startswith("notify"):
                method_concerns["notification"] = method_concerns.get("notification", 0) + 1
        
        # If a class has too many different concerns, flag it
        if len(method_concerns) > CONFIG["assignment_requirements"]["solid_principles"]["srp"]["max_concerns"]:
            concerns = [f"{concern}:{count}" for concern, count in method_concerns.items()]
            violations.append({
                "principle": "SRP",
                "class": class_name,
                "reason": f"Class has multiple concerns: {', '.join(concerns)}"
            })
        
        # Check for DIP violations based on field types and instantiations
        for field in class_info.get("fields", []):
            field_type = field["type"]
            # Check if a high-level module directly depends on a concrete implementation
            if "PostgresDriver" in field_type and "DatabaseDriver" not in field_type:
                violations.append({
                    "principle": "DIP",
                    "class": class_name,
                    "reason": f"Class depends directly on concrete PostgresDriver implementation instead of DatabaseDriver abstraction"
                })
                break
        
        # Check if the class directly instantiates dependencies
        if class_info.get("instantiates_postgres", False):
            violations.append({
                "principle": "DIP",
                "class": class_name,
                "reason": f"Class directly instantiates PostgresDriver instead of using dependency injection"
            })
    
    return violations

### --- STEP 4: FAISS Embedding Storage --- ###
def get_improved_submission_embedding(java_files, parsed_classes):
    """Generate improved embedding for a submission with better text representation"""
    # Create a more comprehensive text representation
    java_text = " ".join(java_files.values())
    
    # Extract package structure
    packages = set()
    for content in java_files.values():
        package_match = re.search(r'package\s+([\w.]+);', content)
        if package_match:
            packages.add(package_match.group(1))
    
    # Include detailed class structure
    class_info = []
    for class_name, class_info_dict in parsed_classes.items():
        # Class type
        class_type = "interface" if class_info_dict.get("type") == "interface" else "class"
        class_info.append(f"{class_type} {class_name}")
        
        # Add inheritance info
        if class_info_dict.get("extends"):
            class_info.append(f"{class_name} extends {class_info_dict.get('extends')}")
        
        if class_info_dict.get("implements"):
            implements_list = class_info_dict.get("implements", [])
            if implements_list:
                implements = " ".join(implements_list)
                class_info.append(f"{class_name} implements {implements}")
        
        # Add methods (with extra weight for method names)
        for method in class_info_dict.get("methods", []):
            signature = f"{method.get('name')} {method.get('name')} {method.get('return_type', '')}"
            class_info.append(signature)
    
    # Add special focus on SOLID-related terms
    solid_terms = [
        "responsibility", "single responsibility", "open closed", "liskov", 
        "interface segregation", "dependency injection", "abstraction",
        "DatabaseDriver", "PostgresDriver", "UserService", "dependency"
    ]
    
    # Add SOLID terms to the embedding if they appear in the code
    solid_text = " ".join([term for term in solid_terms if term.lower() in java_text.lower()])
    
    # Combine all text elements with differential weighting
    combined_text = (
        java_text + " " +
        " ".join(packages) + " " + " ".join(packages) + " " +  # Double weight for packages
        " ".join(class_info) + " " +
        solid_text + " " + solid_text + " " + solid_text  # Triple weight for SOLID terms
    )
    
    # Generate embedding
    print(f"Generating embedding from text of length: {len(combined_text)}")
    embedding = embedding_model.encode([combined_text], normalize_embeddings=True)
    return np.array(embedding, dtype=np.float32)[0]  # Return the first embedding vector, not in a batch

def store_feedback_embeddings(processed_data):
    """Create and store embeddings for feedback using FAISS with improved embedding generation."""
    embeddings = []
    student_ids = []
    
    print(f"Building embeddings for {len(processed_data)} submissions...")
    
    for student_id, data in processed_data.items():
        # Only process submissions with feedback and valid Java files
        if data.get("feedback") and data.get("java_files"):
            try:
                # Parse the Java files if not already parsed
                parsed_classes = data.get("parsed_classes", {})
                if not parsed_classes:
                    parsed_classes = {}
                    for file_path, content in data["java_files"].items():
                        file_parsed_classes = parse_java_code(content)
                        parsed_classes.update(file_parsed_classes)
                
                # Generate improved embedding
                embedding = get_improved_submission_embedding(data["java_files"], parsed_classes)
                
                embeddings.append(embedding)
                student_ids.append(student_id)
                print(f"✓ Created embedding for {student_id}")
            except Exception as e:
                print(f"⚠️ Error creating embedding for {student_id}: {e}")
    
    if not embeddings:
        print("⚠️ No valid embeddings generated. Skipping FAISS indexing.")
        return
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Create FAISS index
    dimension = embeddings_array.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings_array)
    
    # Save the index and mapping for later retrieval
    faiss.write_index(faiss_index, CONFIG["embeddings_file"])
    
    # Save mapping of index to student_id
    with open(CONFIG["embeddings_file"] + ".json", "w") as f:
        json.dump(student_ids, f)
    
    print(f"✅ Feedback embeddings stored in {CONFIG['embeddings_file']} for {embeddings_array.shape[0]} submissions")

def inspect_faiss_index():
    """Diagnose issues with the FAISS index and processed data."""
    # Check if index exists
    if not os.path.exists(CONFIG["embeddings_file"]):
        print(f"⚠️ FAISS index file not found at {CONFIG['embeddings_file']}")
        return False
    
    # Load the index
    try:
        faiss_index = faiss.read_index(CONFIG["embeddings_file"])
        print(f"FAISS index info: dimension={faiss_index.d}, total vectors={faiss_index.ntotal}")
        
        if faiss_index.ntotal == 0:
            print("⚠️ FAISS index is empty! You need to add vectors to it.")
            return False
        
        # Load the student ID mapping
        try:
            with open(CONFIG["embeddings_file"] + ".json", "r") as f:
                student_id_mapping = json.load(f)
            
            print(f"Student ID mapping contains {len(student_id_mapping)} entries")
            
            # Check if mappings align
            if len(student_id_mapping) != faiss_index.ntotal:
                print(f"⚠️ Mismatch: mapping has {len(student_id_mapping)} entries but FAISS has {faiss_index.ntotal} vectors")
                return False
            
            # Load processed data
            try:
                with open(CONFIG["output_file"], "r") as f:
                    processed_data = json.load(f)
                
                print(f"Processed data contains {len(processed_data)} entries")
                
                # Check for feedback
                feedback_count = 0
                for student_id in student_id_mapping:
                    if student_id in processed_data and processed_data[student_id].get("feedback"):
                        feedback_count += 1
                
                print(f"Found feedback for {feedback_count} out of {len(student_id_mapping)} indexed submissions")
                
                if feedback_count == 0:
                    print("⚠️ No feedback found in any indexed submissions!")
                    return False
                
                # Print sample indexed student IDs
                if student_id_mapping:
                    print("Sample indexed student IDs:")
                    for i, student_id in enumerate(student_id_mapping[:5]):
                        print(f"  {i}: {student_id}")
                
                return feedback_count > 0
                
            except Exception as e:
                print(f"⚠️ Error reading processed data: {e}")
                return False
                
        except Exception as e:
            print(f"⚠️ Error loading student ID mapping: {e}")
            return False
            
    except Exception as e:
        print(f"⚠️ Error loading FAISS index: {e}")
        return False

### --- STEP 5: Main Processing Pipeline --- ###
def main():
    # Check existing FAISS index first
    index_is_valid = inspect_faiss_index() if os.path.exists(CONFIG["embeddings_file"]) else False
    
    # Load existing processed data if available
    processed_data = {}
    if os.path.exists(CONFIG["output_file"]):
        try:
            with open(CONFIG["output_file"], "r", encoding="utf-8") as f:
                processed_data = json.load(f)
            print(f"Loaded {len(processed_data)} existing submissions from {CONFIG['output_file']}")
        except Exception as e:
            print(f"⚠️ Error loading existing data: {e}")
    
    # Process PDF feedback files
    pdf_students = {}
    for pdf_file in os.listdir(CONFIG["comments_folder"]):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(CONFIG["comments_folder"], pdf_file)
            student_name, structured_feedback = extract_student_info_from_pdf(pdf_path)
            if student_name:
                try:
                    first_name, last_name = student_name.lower().split(" ")
                    formatted_name = last_name+first_name
                    pdf_students[formatted_name] = structured_feedback
                except ValueError:
                    print(f"⚠️ Could not parse student name correctly: {student_name}")
                    pdf_students[student_name.lower().replace(" ", "")] = structured_feedback
    
    # Process ZIP submission files
    new_or_updated_submissions = 0
    for zip_file in os.listdir(CONFIG["submissions_folder"]):
        if zip_file.endswith(".zip"):
            student_name = extract_name_from_zip(zip_file)
            
            # Skip if already processed and not in pdf_students (no new feedback)
            if student_name in processed_data and student_name not in pdf_students:
                continue
            
            zip_path = os.path.join(CONFIG["submissions_folder"], zip_file)
            extracted_java_files = extract_java_files_from_zip(zip_path)
            
            if not extracted_java_files:
                print(f"⚠️ No Java files extracted from {zip_file}")
                continue
            
            # Extract package structure
            package_structure, class_packages = extract_package_structure(extracted_java_files)
            
            # Parse all Java files in the submission
            parsed_classes = {}
            for file, content in extracted_java_files.items():
                file_parsed_classes = parse_java_code(content)
                for class_name, class_info in file_parsed_classes.items():
                    parsed_classes[class_name] = class_info
            
            # Get feedback if available
            feedback = pdf_students.get(student_name, None)
            
            # Analyze sentiment and SOLID violations
            sentiment_label, sentiment_score = "neutral", 0.0
            solid_violations = []
            
            if feedback:
                feedback_text = json.dumps(feedback)
                sentiment_label, sentiment_score = analyze_sentiment(feedback_text)
                solid_violations = categorize_solid_violations(feedback_text)
            
            # Detect violations from code structure
            code_violations = detect_solid_violations_from_code(parsed_classes)
            
            # Detect package structure violations
            package_violations = detect_package_structure_violations(package_structure, class_packages)
            
            # Detect dependency injection violations
            di_violations = detect_di_violations(parsed_classes, extracted_java_files)
            
            processed_data[student_name] = {
                "feedback": feedback,
                "java_files": extracted_java_files,
                "parsed_classes": parsed_classes,
                "package_structure": package_structure,
                "solid_violations": solid_violations,
                "detected_violations": {
                    "solid": code_violations,
                    "package": package_violations,
                    "dependency_injection": di_violations
                },
                "sentiment": {"label": sentiment_label, "score": sentiment_score}
            }
            
            new_or_updated_submissions += 1
            print(f"✓ Processed submission: {student_name}")
    
    # Save processed data
    with open(CONFIG["output_file"], "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4)
    
    print(f"✅ Processed {new_or_updated_submissions} new or updated submissions")
    print(f"✅ Total data entries: {len(processed_data)}")
    print(f"✅ Data saved to '{CONFIG['output_file']}'")
    
    # Rebuild embeddings if there are new or updated submissions or if index is invalid
    if new_or_updated_submissions > 0 or not index_is_valid:
        print("Rebuilding FAISS index with updated data...")
        store_feedback_embeddings(processed_data)
    else:
        print("No new submissions and existing index is valid. Skipping FAISS indexing.")
    
    print(f"✅ Training Phase Complete!")

if __name__ == "__main__":
    main()