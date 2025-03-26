import os
import gc
import sys
import re
import json
import zipfile
import faiss
import numpy as np
import javalang
import time
import torch
from sentence_transformers import SentenceTransformer
from ollama import Client

# Lazy-loaded globals
embedding_model = None
faiss_index = None
codellama = None
student_id_mapping = []
past_data = {}

CONFIG = {
    "evaluation_folder": "./data/evaluation/",
    "output_file": "./data/evaluation_results.json",
    "processed_data_file": "./data/processed_data.json",
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
                "max_methods": 7,
                "max_concerns": 2
            }
        }
    }
}

def initialize_resources():
    global embedding_model, faiss_index, codellama, student_id_mapping, past_data

    if embedding_model is None:
        print("\u23F3 Loading embedding model...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    if faiss_index is None:
        try:
            print("\u23F3 Loading FAISS index...")
            faiss_index = faiss.read_index("./data/feedback_embeddings.faiss")
        except Exception as e:
            print(f"⚠️ Failed to load FAISS index: {e}")
            faiss_index = faiss.IndexFlatL2(384)

    if not student_id_mapping:
        try:
            with open("./data/feedback_embeddings.faiss.json", "r") as f:
                student_id_mapping = json.load(f)
        except FileNotFoundError:
            print("⚠️ Student ID mapping file not found.")
            student_id_mapping = []

    if not past_data:
        try:
            with open(CONFIG["processed_data_file"], "r") as f:
                past_data = json.load(f)
        except FileNotFoundError:
            print("⚠️ Past data file not found. Creating new data structure.")
            past_data = {}

    if codellama is None:
        print("\u23F3 Connecting to Ollama...")
        codellama = connect_to_ollama()

def connect_to_ollama(retries=5, delay=5):
    """Establish connection to Ollama with retries."""
    ollama_host = os.environ.get('OLLAMA_HOST', 'http://ollama:11434')
    print(f"Attempting to connect to Ollama at {ollama_host}")
    
    for attempt in range(retries):
        try:
            client = Client(host=ollama_host)
            # Test the connection
            models = client.list()
            print(f"Successfully connected to Ollama. Available models: {models}")
            return client
        except Exception as e:
            print(f"Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
    
    print("Failed to connect to Ollama after multiple attempts")
    return None

def extract_java_files_from_zip(zip_path):
    """Extract Java files from a ZIP submission and clean content."""
    java_files = {}
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith(".java") and not file.startswith("__MACOSX"): 
                    with zip_ref.open(file) as f:
                        content = f.read().decode("utf-8", errors="ignore").strip()
                        # Remove non-printable characters (metadata corruption fix)
                        content = re.sub(r'[^\x00-\x7F]+', '', content)
                        if content:
                            java_files[file] = content
        return java_files
    except Exception as e:
        print(f"⚠️ Error extracting files from {zip_path}: {e}")
        return {}


def parse_java_code(java_code):
    """Parses Java code using javalang and extracts detailed structure."""
    try:
        if "class " not in java_code and "interface " not in java_code:
            return {}  # Skip files that aren't Java classes
        
        tree = javalang.parse.parse(java_code)
        classes = {}
        current_class = None
        
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
            elif isinstance(node, javalang.tree.ClassCreator) and current_class:
                if node.type.name == "PostgresDriver":
                    if "instantiates" not in classes[current_class]:
                        classes[current_class]["instantiates"] = []
                    classes[current_class]["instantiates"].append(node.type.name)
        
        return classes
    except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError) as e:
        print(f"⚠️ Skipping invalid Java file due to parsing error: {e}")
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


def preprocess_code_for_llm(java_files):
    """Preprocesses Java files to create a clean, concise representation for the LLM."""
    # Filter out system files and focus on most relevant files
    relevant_files = {k: v for k, v in java_files.items() 
                     if not k.startswith("__MACOSX") and k.endswith(".java")}
    
    # Create a structured representation of the code
    structured_code = []
    for filename, content in relevant_files.items():
        # Simplify path for readability
        structured_code.append(f"// File: {filename}\n{content}\n\n")
    
    # Limit total tokens if needed
    combined_code = "".join(structured_code)
    if len(combined_code) > 8000:  # arbitrary limit to avoid token issues
        combined_code = combined_code[:8000] + "\n// [Code truncated for length]"
    
    return combined_code


def get_submission_embedding(java_files, parsed_classes):
    """Generate a normalized embedding for a new submission with improved representation."""
    # Create a more comprehensive text representation that captures structural elements
    
    # 1. Start with basic code text
    java_text = " ".join(java_files.values())
    
    # 2. Extract package structure
    packages = set()
    for content in java_files.values():
        package_match = re.search(r'package\s+([\w.]+);', content)
        if package_match:
            packages.add(package_match.group(1))
    
    # 3. Include more detailed class structure
    class_info = []
    for file_path, classes in parsed_classes.items():
        for class_name, info in classes.items():
            # Add class name and type info
            class_type = "interface" if info.get("type") == "interface" else "class"
            class_info.append(f"{class_type} {class_name}")
            
            # Add inheritance info
            if info.get("extends"):
                class_info.append(f"{class_name} extends {info.get('extends')}")
            
            if info.get("implements"):
                implements = " ".join(info.get("implements", []))
                class_info.append(f"{class_name} implements {implements}")
            
            # Add method signatures (more weight to method names)
            for method in info.get("methods", []):
                signature = f"{method.get('name')} {method.get('name')} {method.get('return_type')}"
                class_info.append(signature)
    
    # 4. Add special focus on SOLID-related terms
    solid_terms = [
        "responsibility", "single responsibility", "open closed", "liskov", 
        "interface segregation", "dependency injection", "abstraction",
        "DatabaseDriver", "PostgresDriver", "UserService", "dependency"
    ]
    
    # Add SOLID terms to the embedding if they appear in the code
    solid_text = " ".join([term for term in solid_terms if term.lower() in java_text.lower()])
    
    # 5. Combine all text elements, giving more weight to important structural elements
    combined_text = (
        java_text + " " +
        " ".join(packages) + " " + " ".join(packages) + " " +  # Double weight for packages
        " ".join(class_info) + " " +
        solid_text + " " + solid_text + " " + solid_text  # Triple weight for SOLID terms
    )
    
    # Generate embedding
    print(f"Generating embedding from text of length: {len(combined_text)}")
    embedding = embedding_model.encode([combined_text], normalize_embeddings=True)
    return np.array(embedding, dtype=np.float32)


def find_closest_past_submissions(embedding, top_k=10, threshold=0.5):
    """
    Finds the closest past submissions using FAISS with improved matching.
    
    Args:
        embedding: The embedding of the current submission
        top_k: Maximum number of matches to return
        threshold: Minimum similarity threshold (lowered from 0.6 to 0.5)
        
    Returns:
        List of student IDs for similar past submissions
    """
    # Sanity check on FAISS index
    if faiss_index.ntotal == 0:
        print("⚠️ FAISS index is empty! No past submissions available for comparison.")
        return []
    
    print(f"Searching for similar submissions among {faiss_index.ntotal} indexed submissions")
    
    # Search for similar vectors
    distances, indices = faiss_index.search(embedding, min(top_k, faiss_index.ntotal))
    
    # Convert L2 distances to similarity scores (0-1 range)
    # For normalized vectors, L2 distance of 2 = completely dissimilar
    max_distance = 2.0
    similarity_scores = [(1.0 - (dist / max_distance)) for dist in distances[0]]
    
    # Debug: Print all potential matches before threshold filtering
    all_matches = []
    for i, similarity in enumerate(similarity_scores):
        if i < len(indices[0]):
            idx = indices[0][i]
            student_id = "unknown"
            
            # Try to get student ID from mapping
            if student_id_mapping and idx < len(student_id_mapping):
                student_id = student_id_mapping[idx]
            elif idx < len(list(past_data.keys())):
                student_id = list(past_data.keys())[idx]
                
            all_matches.append((student_id, similarity))
    
    print(f"All potential matches (before threshold): {all_matches}")
    
    # Filter by threshold
    filtered_matches = []
    for i, similarity in enumerate(similarity_scores):
        if similarity >= threshold and i < len(indices[0]):
            idx = indices[0][i]
            if student_id_mapping and idx < len(student_id_mapping):
                student_id = student_id_mapping[idx]
            elif idx < len(list(past_data.keys())):
                student_id = list(past_data.keys())[idx]
            else:
                continue  # Skip if index is out of bounds
                
            if student_id in past_data:
                filtered_matches.append((student_id, similarity))
    
    # Sort by similarity (highest first)
    filtered_matches.sort(key=lambda x: x[1], reverse=True)
    
    print(f"Filtered matches (similarity >= {threshold}): {filtered_matches}")
    return [match[0] for match in filtered_matches]


# Add this function to manually inspect the FAISS index and past_data
def inspect_faiss_index():
    """Diagnose issues with the FAISS index and past_data."""
    initialize_resources()
    print(f"FAISS index info: dimension={faiss_index.d}, total vectors={faiss_index.ntotal}")
    
    if faiss_index.ntotal == 0:
        print("⚠️ FAISS index is empty! You need to add vectors to it.")
        return
    
    print(f"past_data contains {len(past_data)} entries")
    
    if student_id_mapping:
        print(f"student_id_mapping contains {len(student_id_mapping)} entries")
        
        # Check if mappings align
        if len(student_id_mapping) != faiss_index.ntotal:
            print(f"⚠️ Mismatch: student_id_mapping has {len(student_id_mapping)} entries but FAISS has {faiss_index.ntotal} vectors")
    else:
        print("⚠️ student_id_mapping is empty or not loaded")
    
    # Print some sample student IDs
    if past_data:
        print("Sample student IDs in past_data:")
        for i, student_id in enumerate(list(past_data.keys())[:5]):
            print(f"  {i}: {student_id}")
    
    # Check if any student has feedback
    has_feedback = False
    for student_id, data in past_data.items():
        if data.get('feedback'):
            has_feedback = True
            print(f"Sample feedback available for {student_id}")
            if isinstance(data['feedback'], dict):
                print(f"  Feedback keys: {list(data['feedback'].keys())}")
            else:
                print(f"  Feedback type: {type(data['feedback'])}")
            break
    
    if not has_feedback:
        print("⚠️ No feedback found in any past_data entries!")


def detect_package_structure_violations(package_structure, class_packages):
    """Detect violations in package structure based on assignment requirements."""
    violations = []
    
    # Check if required packages exist
    for required_package in CONFIG["assignment_requirements"]["package_structure"]["required_packages"]:
        if required_package not in package_structure:
            violations.append({
                "principle": "package_structure",
                "location": "Project Structure",
                "description": f"Missing required package: {required_package}"
            })
    
    # Check if interfaces and implementations are in correct packages
    for interface, expected_package in CONFIG["assignment_requirements"]["package_structure"]["interface_locations"].items():
        if interface in class_packages:
            actual_package = class_packages[interface]
            if actual_package != expected_package:
                violations.append({
                    "principle": "package_structure",
                    "location": f"{interface} in {actual_package}",
                    "description": f"Interface {interface} should be in package {expected_package}, but found in {actual_package}"
                })
    
    for implementation, expected_package in CONFIG["assignment_requirements"]["package_structure"]["implementation_locations"].items():
        if implementation in class_packages:
            actual_package = class_packages[implementation]
            if actual_package != expected_package:
                violations.append({
                    "principle": "package_structure",
                    "location": f"{implementation} in {actual_package}",
                    "description": f"Implementation {implementation} should be in package {expected_package}, but found in {actual_package}"
                })
    
    return violations


def detect_solid_violations(parsed_classes, java_files, class_packages):
    """Detects potential SOLID principle violations from the code structure."""
    violations = []
    
    # ---- SRP violations ----
    # Check for classes with too many methods
    for file_path, classes in parsed_classes.items():
        file_name = file_path.split("/")[-1] if "/" in file_path else file_path
        for class_name, class_info in classes.items():
            if "type" in class_info and class_info["type"] == "interface":
                continue  # Skip interfaces for SRP violations
                
            # Check method count
            if "methods" in class_info and len(class_info["methods"]) > CONFIG["assignment_requirements"]["solid_principles"]["srp"]["max_methods"]:
                violations.append({
                    "principle": "SRP",
                    "location": f"{file_name} - {class_name}",
                    "description": f"Class has {len(class_info['methods'])} methods, suggesting too many responsibilities"
                })
            
            # Check method names for multiple concerns
            method_concerns = {}
            for method in class_info.get("methods", []):
                method_name = method["name"]
                # Categorize methods by their prefix/concern
                if method_name.startswith("get") or method_name.startswith("set"):
                    method_concerns["accessor"] = method_concerns.get("accessor", 0) + 1
                elif method_name.startswith("save") or method_name.startswith("delete") or method_name.startswith("update"):
                    method_concerns["persistence"] = method_concerns.get("persistence", 0) + 1
                elif method_name.startswith("validate") or method_name.startswith("check"):
                    method_concerns["validation"] = method_concerns.get("validation", 0) + 1
                elif method_name.startswith("calculate") or method_name.startswith("compute"):
                    method_concerns["calculation"] = method_concerns.get("calculation", 0) + 1
                elif method_name.startswith("format") or method_name.startswith("display"):
                    method_concerns["presentation"] = method_concerns.get("presentation", 0) + 1
                elif method_name.startswith("send") or method_name.startswith("notify"):
                    method_concerns["notification"] = method_concerns.get("notification", 0) + 1
            
            # If a class has multiple significant concerns, flag it
            significant_concerns = []
            for concern, count in method_concerns.items():
                if count > 1:  # Only count concerns with multiple methods
                    significant_concerns.append(f"{concern}:{count}")
            
            if len(significant_concerns) > CONFIG["assignment_requirements"]["solid_principles"]["srp"]["max_concerns"]:
                violations.append({
                    "principle": "SRP",
                    "location": f"{file_name} - {class_name}",
                    "description": f"Class appears to have multiple concerns: {', '.join(significant_concerns)}"
                })
    
    # ---- DIP violations ----
    # Check for direct instantiation and tight coupling
    for file_path, content in java_files.items():
        file_name = file_path.split("/")[-1] if "/" in file_path else file_path
        
        # Check UserService for DIP violations
        if "UserService" in file_path:
            # Check for direct instantiation of PostgresDriver
            if "new PostgresDriver" in content:
                violations.append({
                    "principle": "DIP",
                    "location": file_name,
                    "description": "Directly instantiates PostgresDriver instead of using dependency injection"
                })
            
            # Check for tight coupling to concrete implementation
            if "private PostgresDriver" in content:
                violations.append({
                    "principle": "DIP",
                    "location": file_name,
                    "description": "Tightly coupled to PostgresDriver concrete class instead of depending on an abstraction"
                })
                
            # Check if constructor injection is used
            if not re.search(r'private\s+\w+Driver.*?;.*?public\s+\w+Service\s*\(\s*\w+Driver', content, re.DOTALL):
                violations.append({
                    "principle": "DIP",
                    "location": file_name,
                    "description": "Missing constructor injection for database driver dependency"
                })
    
    # ---- ISP violations ----
    # Check for interfaces with too many methods (fat interfaces)
    for file_path, classes in parsed_classes.items():
        file_name = file_path.split("/")[-1] if "/" in file_path else file_path
        for class_name, class_info in classes.items():
            if class_info.get("type") == "interface" and len(class_info.get("methods", [])) > 5:
                violations.append({
                    "principle": "ISP",
                    "location": f"{file_name} - {class_name}",
                    "description": f"Interface has {len(class_info['methods'])} methods, might be a 'fat' interface"
                })
    
    return violations


def extract_solid_violations_from_feedback(feedback_text):
    """Extract SOLID principle violations from feedback text."""
    solid_categories = {
        "SRP": ["single responsibility", "too many responsibilities", "multiple reasons to change", "separate concerns"],
        "OCP": ["open/closed", "new functionality", "modify existing code", "extends without changing", "extension"],
        "LSP": ["liskov", "inheritance", "substitutable", "override", "subclass behavior", "polymorphism"],
        "ISP": ["interface segregation", "unrelated methods", "client-specific", "fat interface", "small interfaces"],
        "DIP": ["dependency inversion", "dependency injection", "abstract", "tight coupling", "concrete implementation", "high-level", "low-level", "constructor injection", "new operator"]
    }
    
    violations = []
    for principle, keywords in solid_categories.items():
        for keyword in keywords:
            if keyword in feedback_text.lower():
                violations.append(principle)
                break
    
    return list(set(violations))  # Remove duplicates


def generate_codellama_feedback(code_snippet, detected_violations, package_violations, past_feedback):
    """Generates structured feedback using Ollama with CodeLlama."""
    
    if codellama is None:
        return "Error: Could not connect to the Ollama service. Please ensure it's running."
    

    # Format detected violations for better prompt clarity
    formatted_violations = []
    confidence_levels = {
        "SRP": "high" if any(v["principle"] == "SRP" for v in detected_violations) else "low",
        "OCP": "medium",
        "LSP": "low",
        "ISP": "medium" if any(v["principle"] == "ISP" for v in detected_violations) else "low",
        "DIP": "high" if any(v["principle"] == "DIP" for v in detected_violations) else "medium",
    }
    
    for violation in detected_violations:
        formatted_violations.append(f"- {violation['principle']}: {violation['location']} - {violation['description']} (confidence: {confidence_levels[violation['principle']]})")
    
    # Format package violations
    for violation in package_violations:
        formatted_violations.append(f"- Package Structure: {violation['location']} - {violation['description']} (confidence: high)")
    
    # Convert past feedback to a more useful format
    past_feedback_sections = []
    if past_feedback:
        for key, value in past_feedback.items():
            if value:
                past_feedback_sections.append(f"## {key.replace('_', ' ').title()}\n{value}")
    
    past_feedback_text = "\n\n".join(past_feedback_sections) if past_feedback_sections else "No specific feedback available from similar submissions."
    
    # Create a structured prompt with specific SOLID principle explanations
    prompt = f"""
    You are an expert Java instructor evaluating code submissions for an assignment on SOLID principles.
    
    Your task is to provide detailed, constructive feedback on how well the code adheres to the SOLID principles:
    1. Single Responsibility Principle (SRP): A class should have only one reason to change
    2. Open/Closed Principle (OCP): Software entities should be open for extension but closed for modification 
    3. Liskov Substitution Principle (LSP): Objects of a superclass should be replaceable with objects of subclasses without breaking the application
    4. Interface Segregation Principle (ISP): Many client-specific interfaces are better than one general-purpose interface
    5. Dependency Inversion Principle (DIP): High-level modules should not depend on low-level modules; both should depend on abstractions
    
    Assignment-specific requirements:
    - The DatabaseDriver interface should be in the solid.persistence package
    - The PostgresDriver implementation should be in the solid.persistence.drivers package 
    - UserService should use dependency injection (constructor injection) for DatabaseDriver
    - Classes should follow SRP with focused responsibilities
    
    Code to analyze:
    ```java
    {code_snippet}
    ```
    
    Detected violations (with confidence levels):
    {chr(10).join(formatted_violations)}
    
    Previous instructor feedback for similar submissions:
    {past_feedback_text}
    
    Based on your analysis and the detected violations, provide the following:
    
    ## Overall Assessment
    [Provide a brief assessment of the code's adherence to SOLID principles. Be accurate but focus on major issues.]
    
    ## SOLID Violations
    [List specific SOLID violations, focusing on those with high confidence. For each violation, include:
    1. The principle being violated
    2. The specific location (class/method)
    3. A clear explanation of the issue
    4. A specific suggestion to fix it]
    
    ## Improvement Suggestions
    [Provide 2-3 concrete suggestions to improve the code's adherence to SOLID principles]
    
    Important: 
    - Focus on the most important issues rather than listing every minor problem
    - Reference the specific instructor feedback patterns from previous submissions
    - Be constructive and educational in your feedback
    - If there are few or no real issues in the code, acknowledge good design decisions
    """

    try:
        print(f"Attempting to generate feedback with model: codellama:7b")
        response = codellama.generate(model="codellama:7b", prompt=prompt)
        print(f"Successfully generated feedback")
        feedback_text = response['response'].strip()
        del response
        gc.collect()
        return feedback_text
        #return response['response'].strip()
    except Exception as e:
        print(f"Error generating feedback with CodeLlama: {e}")
        import traceback
        traceback.print_exc()  # This will print the full stack trace
        return f"Error generating feedback. Please check if the Ollama service is running correctly. Error: {str(e)}"


def evaluate_submission(java_files):
    """Evaluates a new submission using past feedback and LLM feedback generation."""
    initialize_resources()
    try:
        # Parse all Java files for class structure
        parsed_classes = {}
        for file, content in java_files.items():
            parsed_result = parse_java_code(content)
            if parsed_result:
                parsed_classes[file] = parsed_result
        
        # Extract package structure
        package_structure, class_packages = extract_package_structure(java_files)
        
        # Analyze code structure to detect SOLID violations
        detected_violations = detect_solid_violations(parsed_classes, java_files, class_packages)
        
        # Detect package structure violations
        package_violations = detect_package_structure_violations(package_structure, class_packages)
        
        # Create embedding and find similar past submissions
        embedding = get_submission_embedding(java_files, parsed_classes)
        closest_matches = find_closest_past_submissions(embedding, top_k=10)
        
        # Retrieve past feedback for all closest matches
        past_feedbacks = []
        for match in closest_matches:
            if match in past_data and past_data[match].get("feedback"):
                past_feedbacks.append(past_data[match]["feedback"])
        
        # Use the first past feedback as reference if available
        reference_feedback = past_feedbacks[0] if past_feedbacks else None
        
        # Prepare code for LLM analysis
        processed_code = preprocess_code_for_llm(java_files)
        
        # Generate feedback based on code, detected violations, and past feedback
        structured_feedback = generate_codellama_feedback(
            processed_code, 
            detected_violations,
            package_violations,
            reference_feedback
        )
        
        # Extract SOLID violations from the generated feedback
        solid_violations = extract_solid_violations_from_feedback(structured_feedback)
        
        # If "good job" or positive feedback is present in the generated text,
        # and there are no detected violations, don't add violations artificially
        if "good job" in structured_feedback.lower() or "well done" in structured_feedback.lower():
            if len(detected_violations) < 2 and len(package_violations) == 0:
                solid_violations = []
        
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        gc.collect()
        
        return {
            "closest_matches": closest_matches,
            "retrieved_feedbacks": past_feedbacks,
            "generated_feedback": structured_feedback,
            "parsed_classes": parsed_classes,
            "detected_violations": detected_violations + package_violations,
            "solid_violations": solid_violations if solid_violations else ["No SOLID violations detected"]
        }
    except Exception as e:
        import traceback
        print(f"ERROR in evaluate_submission: {e}")
        traceback.print_exc()
        return {
            "error": str(e),
            "generated_feedback": "Error generating feedback. Please try again later.",
            "closest_matches": [],
            "detected_violations": [],
            "solid_violations": []
        }

# Modify main() function to support both modes
def main():
    """Run in command-line mode to evaluate all submissions in the folder."""
    initialize_resources()
    # Add inspection of the FAISS index to diagnose issues
    inspect_faiss_index()
    
    evaluation_results = {}
    
    # Check if a specific file is provided as argument
    if len(sys.argv) > 1:
        specific_file = sys.argv[1]
        if os.path.exists(specific_file) and specific_file.endswith('.zip'):
            student_name = os.path.basename(specific_file).split('_')[0]
            print(f"Processing single submission: {specific_file}")
            java_files = extract_java_files_from_zip(specific_file)
            
            if not java_files:
                print(f"⚠️ No Java files found in {specific_file}. Skipping evaluation.")
            else:
                evaluation_results[student_name] = evaluate_submission(java_files)
                print(f"✓ Evaluated submission: {student_name}")
        else:
            print(f"⚠️ Invalid file: {specific_file}. Please provide a valid ZIP file.")
    else:
        # Process all files in the evaluation folder
        for zip_file in os.listdir(CONFIG["evaluation_folder"]):
            if zip_file.endswith(".zip"):
                print(f"Processing submission: {zip_file}")
                zip_path = os.path.join(CONFIG["evaluation_folder"], zip_file)
                student_name = zip_file.split("_")[0]
                java_files = extract_java_files_from_zip(zip_path)
                
                if not java_files:
                    print(f"⚠️ No Java files found in {zip_file}. Skipping evaluation.")
                    continue
                    
                evaluation_results[student_name] = evaluate_submission(java_files)
                print(f"✓ Evaluated submission: {student_name}")
    
    # Save evaluation results
    with open(CONFIG["output_file"], "w") as f:
        json.dump(evaluation_results, f, indent=4)
    print(f"✅ Evaluation completed! Results saved to '{CONFIG['output_file']}'")
    
    return evaluation_results

if __name__ == "__main__":
    main()