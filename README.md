# Production-Level Fine-Tuning Strategy for Code Models on AWS SageMaker

## Comprehensive Implementation Guide (Updated April 2025)

This document provides detailed instructions for fine-tuning a code model like LLaMA 4 on AWS SageMaker for bug detection and code generation tasks, including hardware requirements, cost analysis, synthetic data generation strategies, and step-by-step implementation.

> **IMPORTANT NOTE**: This implementation is designed to be run **DIRECTLY** on Amazon SageMaker without any third-party platforms. All resources, processing, and deployment will be contained within the AWS ecosystem to maintain security, compliance, and streamlined workflow management.

## 1. Hardware Requirements Analysis

For a 500MB-1GB codebase (used as training data), the following considerations apply:

### 1.1. Model Memory Requirements

LLaMA 4 variants have different parameter counts:
- LLaMA 4 Scout 17B (with 16 experts): ~34GB in FP16
- LLaMA 4 Maverick 17B (with 128 experts): ~34GB in FP16, but leverages mixture of experts architecture

Additional model options include:
- Mistral 7B: ~14GB in FP16
- Code Llama 7B: ~14GB in FP16
- IBM Granite Code 8B: ~16GB in FP16
- LLaMA 3.1 8B: ~16GB in FP16

When fine-tuning with Parameter-Efficient Fine-Tuning (PEFT) methods like QLoRA, memory requirements are significantly reduced:
- 17B models: ~25-30GB total memory needed (model + training overhead)
- 7B-8B models: ~15-20GB total memory needed (model + training overhead)
- Further reductions possible with 4-bit quantization and optimized attention mechanisms

### 1.2. SageMaker Instance Selection Table (Updated April 2025)

| Instance Type | vCPUs | GPU | GPU Memory | RAM | Cost/Hour | Suitable For | Notes |
|---------------|-------|-----|------------|-----|-----------|-------------|-------|
| ml.g4dn.xlarge | 4 | 1x T4 | 16 GB | 16 GB | $0.736 | Inference/testing | Insufficient for full model training |
| ml.g4dn.2xlarge | 8 | 1x T4 | 16 GB | 32 GB | $0.94 | Small model QLoRA tuning | Good for small PEFT |
| ml.g5.xlarge | 4 | 1x A10G | 24 GB | 16 GB | $1.24 | Small models or inference | Entry-level for 8B models |
| ml.g5.2xlarge | 8 | 1x A10G | 24 GB | 32 GB | $1.444 | 17B models with PEFT | Recommended for LLaMA 4 with QLoRA |
| ml.g5.4xlarge | 16 | 1x A10G | 24 GB | 64 GB | $1.866 | Larger data processing | More CPU for preprocessing |
| ml.p3.2xlarge | 8 | 1x V100 | 16 GB | 61 GB | $3.825 | Premium small-model option | High cost but good performance |
| ml.inf1.xlarge | 4 | AWS Inferentia | - | 8 GB | $0.585 | Optimized inference | Cost-effective for deployment |
| ml.p4d.24xlarge | 96 | 8x A100 (40GB) | 320 GB | 1152 GB | $32.77 | Large models, distributed training | For full fine-tuning |
| ml.p4de.24xlarge | 96 | 8x A100 (80GB) | 640 GB | 1152 GB | $40.96 | Very large models | Overkill for this task |

### 1.3. Recommended Instance

**For this particular task (fine-tuning LLaMA 4 on 500MB-1GB codebase):**
- **Primary Recommendation**: ml.g5.2xlarge for Scout/Maverick 17B models with QLoRA
- **Alternative**: ml.g4dn.2xlarge if cost is a major concern (though may require additional optimization)
- **For inference**: ml.inf1.xlarge for cost-efficient deployment
- **For larger models or faster training**: ml.p4d.24xlarge (can be shared across multiple fine-tuning jobs)

**For 7B-8B parameter models (Mistral, Code Llama, Granite Code, LLaMA 3.1):**
- **Primary Recommendation**: ml.g4dn.2xlarge with QLoRA (most cost-efficient)
- **Faster Alternative**: ml.g5.xlarge for better training speed
- **For inference**: ml.inf1.xlarge or ml.g4dn.xlarge for cost-efficient deployment

## 2. Model Selection Guide: Beyond LLaMA 4

When selecting a code model for fine-tuning, several factors should be considered: model size, context window, performance on code tasks, and cost. Here's a comparison of top models available on SageMaker JumpStart:

### 2.1. Code-Optimized Models Comparison

| Model | Parameters | Context Window | Code Performance | Fine-Tuning Cost* | Inference Cost* | Key Strengths |
|-------|------------|--------------|-----------------|-----------------|----------------|--------------|
| **Code Llama 7B** | 7B | 100K tokens | Strong | $224 | $0.736/hr | Excellent Python/C++, infilling capability |
| **Code Llama 13B** | 13B | 100K tokens | Very Strong | $414 | $1.444/hr | Better than 7B for complex code tasks |
| **Code Llama 70B** | 70B | 100K tokens | Excellent | $3,277 | $3.825/hr | State-of-the-art code generation |
| **Mistral 7B** | 7B | 32K tokens | Good | $224 | $0.736/hr | Fast inference, strong general reasoning |
| **Mistral 8x7B (MoE)** | 8x7B (MoE) | 32K tokens | Very Good | $788 | $1.444/hr | Excellent reasoning capabilities |
| **IBM Granite Code 3B** | 3B | 128K tokens | Good | $133 | $0.585/hr | Very compact, enterprise-focused |
| **IBM Granite Code 8B** | 8B | 128K tokens | Very Good | $252 | $0.736/hr | Excellent for enterprise code tasks, RAG |
| **LLaMA 3.1 8B** | 8B | 128K tokens | Very Good | $252 | $0.736/hr | Well-rounded performance, long context |
| **LLaMA 4 Scout 17B** | 17B | 10M tokens | Excellent | $560 | $1.444/hr | Massive context window, strong reasoning |

*Fine-tuning cost based on 24 hours of training on recommended instances. Inference cost based on hourly rate for appropriate instance.

### 2.2. Model Recommendations by Use Case

**For Bug Detection and Code Fixing:**
- **Budget Constrained**: Code Llama 7B or IBM Granite Code 8B
- **Best Performance**: Code Llama 13B or Mistral 8x7B
- **Enterprise Grade**: IBM Granite Code 8B (includes safety features)

**For Code Generation and Completion:**
- **Budget Constrained**: LLaMA 3.1 8B
- **Best Performance**: Code Llama 13B
- **Massive Context Needs**: LLaMA 4 Scout 17B (10M token context)

**For Long Code Context Processing:**
- **Budget Constrained**: IBM Granite Code 8B (128K tokens)
- **Best Performance**: LLaMA 4 Scout 17B (10M tokens)

## 3. Data Preparation and Synthetic Data Generation

### 3.1. Data Collection Strategies

#### 3.1.1. Real-World Code Repositories

```python
# Initialize repositories and storage
import os, subprocess, json
from pathlib import Path

# Create directories
os.makedirs("code_data/raw", exist_ok=True)
os.makedirs("code_data/processed", exist_ok=True)

# Define repositories based on target language
# For C/C++ focused fine-tuning:
cpp_repositories = [
    "https://github.com/llvm/llvm-project.git",
    "https://github.com/tensorflow/tensorflow.git",  # Has significant C++ components
    "https://github.com/electron/electron.git",
    "https://github.com/protocolbuffers/protobuf.git",
    "https://github.com/google/googletest.git",
    "https://github.com/opencv/opencv.git",
    "https://github.com/facebook/folly.git"
]

python_repositories = [
    "https://github.com/tensorflow/tensorflow.git",
    "https://github.com/pytorch/pytorch.git",
    "https://github.com/scikit-learn/scikit-learn.git",
    "https://github.com/django/django.git",
    "https://github.com/flask/flask.git"
]

# Select the appropriate repositories based on your target language
# For this example, we'll use C/C++ repositories
repositories = cpp_repositories

for repo in repositories:
    repo_name = repo.split("/")[-1].replace(".git", "")
    # Use depth=1 to avoid downloading entire history
    subprocess.run(f"git clone --depth 1 {repo} code_data/raw/{repo_name}", shell=True)
    
print("Repository cloning complete. Starting file extraction...")
```

#### 3.1.2. Synthetic Data Generation

Synthetic data generation can significantly enhance the quality and diversity of your training dataset. Research shows synthetic data can boost model performance by 25-45% when combined with real data.

```python
import openai
import json
import random
import time
from tqdm import tqdm

# Configure API key
openai.api_key = "your_api_key"  # Replace with your API key or use environment variables

def generate_synthetic_code_examples(prompt, n_examples=5, model="gpt-4", temperature=0.7):
    """Generate synthetic code examples using an external LLM."""
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert programmer generating diverse, high-quality code examples."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            n=n_examples,
            max_tokens=2048
        )
        return [choice.message.content for choice in response.choices]
    except Exception as e:
        print(f"Error generating synthetic examples: {e}")
        return []

# Example prompt for generating Python functions with bugs
bug_prompt = """
Generate {n} different Python functions that contain subtle bugs.
For each function:
1. First, write the buggy version of the function
2. Then, on a separate line starting with '# Corrected version:', write the corrected version
3. Then, on a separate line starting with '# Bug explanation:', explain what the bug was

Make the functions diverse in purpose, complexity, and the type of bugs they contain.
Each function should be between 5-20 lines of code.
"""

# Generate synthetic buggy code examples
synthetic_buggy_examples = generate_synthetic_code_examples(
    bug_prompt.format(n=10),
    n_examples=5
)

# Process and save synthetic examples
synthetic_data = []
for i, example in enumerate(synthetic_buggy_examples):
    # Parse the example to extract buggy and fixed code
    parts = example.split('# Corrected version:')
    if len(parts) != 2:
        continue
        
    buggy_code = parts[0].strip()
    fixed_parts = parts[1].split('# Bug explanation:')
    
    if len(fixed_parts) != 2:
        continue
        
    fixed_code = fixed_parts[0].strip()
    bug_explanation = fixed_parts[1].strip()
    
    synthetic_data.append({
        "task": "Bug Detection and Fix",
        "input_code": buggy_code,
        "instruction": f"Identify and fix the bug in this code. {bug_explanation}",
        "output_code": fixed_code,
        "language": "python",
        "source": "synthetic"
    })

print(f"Generated {len(synthetic_data)} synthetic examples")

# Save synthetic examples
with open("code_data/processed/synthetic_examples.json", "w") as f:
    json.dump(synthetic_data, f, indent=2)
```

#### 3.1.3. Combining Real and Synthetic Data

Research indicates that a blend of real and synthetic data yields the best results. For code models, an optimal ratio is approximately 70% real data and 30% synthetic data, with the synthetic examples focused on:

1. Underrepresented edge cases
2. Complex bug patterns that are rare in real repositories
3. Domain-specific coding patterns

### 3.2. Language Detection and Code Extraction

```python
import glob
import re
import os
from pathlib import Path
from collections import Counter

def detect_language(file_path):
    """Detect programming language based on file extension and content."""
    # Map extensions to languages
    extension_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.h': 'cpp_header',
        '.hpp': 'cpp_header',
        '.cs': 'csharp',
        '.go': 'go',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.ts': 'typescript',
        '.scala': 'scala',
        '.rs': 'rust'
    }
    
    extension = os.path.splitext(file_path)[1].lower()
    
    if extension in extension_map:
        # Basic extension-based detection
        detected_lang = extension_map[extension]
        
        # For C/C++ header files, perform additional analysis to confirm
        if detected_lang == 'cpp_header':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Check for C++ specific patterns
                cpp_patterns = ['class', 'namespace', 'template', 'std::', 'public:', 'private:', 'protected:']
                c_patterns = ['typedef struct', '#include <stdio.h>', 'NULL']
                
                cpp_matches = sum(1 for pattern in cpp_patterns if pattern in content)
                c_matches = sum(1 for pattern in c_patterns if pattern in content)
                
                if cpp_matches > c_matches:
                    return 'cpp'
                else:
                    return 'c'
        return detected_lang
    else:
        # Try content-based detection for ambiguous files
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(1000)  # Read first 1000 chars for quick detection
                
                # Look for language-specific patterns
                if '#include <iostream>' in content or 'std::' in content:
                    return 'cpp'
                elif '#include <stdio.h>' in content:
                    return 'c'
                elif 'import tensorflow as tf' in content:
                    return 'python'
                elif 'public class' in content or 'private void' in content:
                    return 'java'
                # Add more patterns as needed
                
        except:
            pass
    
    # Default to "unknown" if detection fails
    return "unknown"

def extract_code_files(directory, target_languages=None):
    """Extract all code files for specified languages from a directory."""
    if target_languages is None:
        # If no specific languages are provided, look for common code file extensions
        extensions = ['.py', '.js', '.java', '.c', '.cpp', '.cc', '.h', '.hpp', '.cs', '.go']
    else:
        # Map languages to their respective extensions
        language_extensions = {
            'python': ['.py'],
            'javascript': ['.js'],
            'java': ['.java'],
            'c': ['.c', '.h'],
            'cpp': ['.cpp', '.cc', '.hpp', '.h'],
            'csharp': ['.cs'],
            'go': ['.go']
        }
        
        # Collect extensions for the target languages
        extensions = []
        for lang in target_languages:
            if lang in language_extensions:
                extensions.extend(language_extensions[lang])
    
    # Find all files with specified extensions
    code_files = []
    for ext in extensions:
        code_files.extend(glob.glob(f"{directory}/**/*{ext}", recursive=True))
    
    # Classify files by detected language
    classified_files = {}
    for file_path in code_files:
        lang = detect_language(file_path)
        if lang not in classified_files:
            classified_files[lang] = []
        classified_files[lang].append(file_path)
    
    # Print language distribution statistics
    print("Language distribution in codebase:")
    for lang, files in classified_files.items():
        print(f"  {lang}: {len(files)} files")
    
    return code_files, classified_files

# Define enhanced filter criteria for different languages
def filter_files(file_path, min_size=100, max_size=20000, language=None):
    """Filter files based on size, quality criteria, and language-specific rules."""
    try:
        size = os.path.getsize(file_path)
        if size < min_size or size > max_size:
            return False
        
        # Comment patterns for different languages
        comment_patterns = {
            'python': ['#'],
            'cpp': ['//', '/*', '*', '*/'],
            'c': ['//', '/*', '*', '*/'],
            'java': ['//', '/*', '*', '*/'],
            'javascript': ['//', '/*', '*', '*/'],
            'go': ['//'],
            'ruby': ['#'],
            'php': ['//', '#', '/*', '*', '*/'],
        }
        
        # Default comment patterns if language not specified or not in our map
        line_comment_starts = comment_patterns.get(language, ['#', '//', '/*', '*', '*/'])
            
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Language-specific filtering logic
            if language in ['c', 'cpp']:
                # Check for C/C++ specific quality indicators
                if '#include' not in content:
                    # Likely not a proper C/C++ file
                    return False
                
                # For C/C++ header files, ensure they have function prototypes or class definitions
                if file_path.endswith(('.h', '.hpp')):
                    # Look for function prototypes, class definitions, or struct definitions
                    if not (re.search(r'\w+\s+\w+\s*\([^)]*\)\s*;', content) or  # Function prototype
                            'class ' in content or 'struct ' in content or 'enum ' in content):
                        return False
            
            # Generic code quality check
            code_lines = [
                line for line in content.split('\n') 
                if line.strip() and 
                not any(line.strip().startswith(pattern) for pattern in line_comment_starts) and 
                len(line.strip()) > 5
            ]
            
            # Require at least 5 lines of actual code
            if len(code_lines) < 5:
                return False
                
            return True
    except Exception as e:
        print(f"Error filtering file {file_path}: {e}")
        return False

# Extract and classify code files from repositories
all_code_files = []
language_classified_files = {}

# Specify target languages - for C/C++ focused model
target_languages = ['c', 'cpp']

for repo_dir in Path("code_data/raw").iterdir():
    if repo_dir.is_dir():
        print(f"Processing repository: {repo_dir}")
        repo_files, repo_classified = extract_code_files(str(repo_dir), target_languages)
        
        # Merge classification results
        for lang, files in repo_classified.items():
            if lang not in language_classified_files:
                language_classified_files[lang] = []
            # Apply language-specific filtering
            filtered_files = [f for f in files if filter_files(f, language=lang)]
            language_classified_files[lang].extend(filtered_files)
            all_code_files.extend(filtered_files)

print(f"Collected {len(all_code_files)} code files after filtering")
print("Language distribution after filtering:")
for lang, files in language_classified_files.items():
    print(f"  {lang}: {len(files)} files")
```

### 3.3. Semantic Analysis and Function Extraction

```python
import re
import ast
import subprocess
import tempfile
import json
import os
import clang.cindex
from typing import List, Dict, Tuple, Optional, Any

# Initialize clang for C/C++ parsing (requires libclang)
# You may need to install: pip install clang
try:
    clang.cindex.Config.set_library_file('/usr/lib/llvm-10/lib/libclang.so.1')  # Adjust path as needed
except:
    print("Warning: libclang not found. C/C++ semantic analysis will be limited.")

class SemanticCodeAnalyzer:
    """A class to perform semantic analysis on code files of different languages."""
    
    def __init__(self):
        self.parsers = {
            'python': self.parse_python,
            'c': self.parse_c_cpp,
            'cpp': self.parse_c_cpp,
            'java': self.parse_java,
            'javascript': self.parse_javascript
        }
        
        self.function_extractors = {
            'python': self.extract_python_functions,
            'c': self.extract_c_cpp_functions,
            'cpp': self.extract_c_cpp_functions,
            'java': self.extract_java_functions,
            'javascript': self.extract_javascript_functions
        }
    
    def analyze_file(self, file_path: str, language: str = None) -> Dict[str, Any]:
        """Analyze a code file and return its semantic information."""
        if language is None:
            # Detect language from file extension if not specified
            ext = os.path.splitext(file_path)[1].lower()
            ext_to_lang = {
                '.py': 'python', '.c': 'c', '.cpp': 'cpp', '.cc': 'cpp',
                '.h': 'c', '.hpp': 'cpp', '.java': 'java', '.js': 'javascript'
            }
            language = ext_to_lang.get(ext, 'unknown')
        
        if language == 'unknown':
            return {'error': 'Unsupported language or could not detect language'}
        
        parser = self.parsers.get(language, None)
        if not parser:
            return {'error': f'No parser available for language: {language}'}
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            return parser(content, file_path)
        except Exception as e:
            return {'error': f'Analysis failed: {str(e)}'}
    
    def parse_python(self, content: str, file_path: str) -> Dict[str, Any]:
        """Parse Python code using the ast module."""
        try:
            tree = ast.parse(content)
            
            # Extract top-level information
            imports = []
            classes = []
            functions = []
            global_vars = []
            
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    imports.append(ast.unparse(node).strip())
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        'name': node.name,
                        'line': node.lineno,
                        'docstring': ast.get_docstring(node) or ""
                    })
                elif isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'line': node.lineno,
                        'args': [a.arg for a in node.args.args],
                        'docstring': ast.get_docstring(node) or "",
                        'source': ast.unparse(node)
                    })
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            global_vars.append({
                                'name': target.id,
                                'line': node.lineno
                            })
            
            return {
                'language': 'python',
                'imports': imports,
                'classes': classes,
                'functions': functions,
                'global_vars': global_vars,
                'ast': tree  # Store the AST for further analysis
            }
        except Exception as e:
            return {'error': f'Python parsing failed: {str(e)}'}
    
    def parse_c_cpp(self, content: str, file_path: str) -> Dict[str, Any]:
        """Parse C/C++ code using clang."""
        try:
            # Create a temporary file to enable clang parsing
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(file_path)[1], delete=False) as tmp:
                tmp.write(content.encode('utf-8'))
                tmp_name = tmp.name
            
            # Parse the file using clang
            index = clang.cindex.Index.create()
            tu = index.parse(tmp_name)
            
            # Clean up temp file
            os.unlink(tmp_name)
            
            # Extract top-level information
            includes = []
            functions = []
            structs_classes = []
            typedefs = []
            
            def extract_comment(node):
                """Extract comment for a node if available."""
                comments = []
                for token in node.get_tokens():
                    if token.kind == clang.cindex.TokenKind.COMMENT:
                        comments.append(token.spelling)
                return '\n'.join(comments) if comments else ""
            
            for node in tu.cursor.get_children():
                if node.kind == clang.cindex.CursorKind.INCLUSION_DIRECTIVE:
                    includes.append(node.displayname)
                elif node.kind == clang.cindex.CursorKind.FUNCTION_DECL:
                    functions.append({
                        'name': node.spelling,
                        'line': node.location.line,
                        'type': node.type.spelling,
                        'comment': extract_comment(node),
                        'is_definition': node.is_definition()
                    })
                elif node.kind in [clang.cindex.CursorKind.STRUCT_DECL, clang.cindex.CursorKind.CLASS_DECL]:
                    structs_classes.append({
                        'name': node.spelling,
                        'line': node.location.line,
                        'kind': 'struct' if node.kind == clang.cindex.CursorKind.STRUCT_DECL else 'class',
                        'comment': extract_comment(node)
                    })
                elif node.kind == clang.cindex.CursorKind.TYPEDEF_DECL:
                    typedefs.append({
                        'name': node.spelling,
                        'line': node.location.line
                    })
            
            return {
                'language': 'cpp' if file_path.endswith(('.cpp', '.cc', '.hpp')) else 'c',
                'includes': includes,
                'functions': functions,
                'structs_classes': structs_classes,
                'typedefs': typedefs,
                'tu': tu  # Store the translation unit for further analysis
            }
        except Exception as e:
            # Fallback to regex-based parsing
            return self._regex_parse_c_cpp(content)
    
    def _regex_parse_c_cpp(self, content: str) -> Dict[str, Any]:
        """Fallback parser for C/C++ using regex when clang fails."""
        # Basic regex patterns for C/C++
        include_pattern = r'#include\s+[<"]([^>"]+)[>"]'
        function_pattern = r'(\w+)\s+(\w+)\s*\(([^)]*)\)\s*{'
        class_struct_pattern = r'(class|struct)\s+(\w+)(?:\s*:\s*(?:public|private|protected)\s+\w+)?\s*{'
        
        includes = re.findall(include_pattern, content)
        
        functions = []
        for match in re.finditer(function_pattern, content):
            return_type, name, args = match.groups()
            functions.append({
                'name': name,
                'return_type': return_type,
                'args': args.strip()
            })
        
        structs_classes = []
        for match in re.finditer(class_struct_pattern, content):
            kind, name = match.groups()
            structs_classes.append({
                'kind': kind,
                'name': name
            })
        
        return {
            'language': 'cpp' if 'class' in content or 'namespace' in content else 'c',
            'includes': includes,
            'functions': functions,
            'structs_classes': structs_classes
        }
    
    def parse_java(self, content: str, file_path: str) -> Dict[str, Any]:
        """Basic parser for Java code using regex."""
        # This is a simplified parser - for production, consider using a proper Java parser
        package_pattern = r'package\s+([\w.]+);'
        import_pattern = r'import\s+([\w.*]+);'
        class_pattern = r'(public|private|protected)?\s+class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?'
        method_pattern = r'(public|private|protected)?\s+(?:static\s+)?(\w+)\s+(\w+)\s*\(([^)]*)\)\s*(?:throws\s+[\w,\s]+)?'
        
        package = re.search(package_pattern, content)
        imports = re.findall(import_pattern, content)
        classes = []
        for match in re.finditer(class_pattern, content):
            classes.append(match.group(2))
        
        methods = []
        for match in re.finditer(method_pattern, content):
            methods.append({
                'visibility': match.group(1),
                'return_type': match.group(2),
                'name': match.group(3),
                'params': match.group(4)
            })
        
        return {
            'language': 'java',
            'package': package.group(1) if package else None,
            'imports': imports,
            'classes': classes,
            'methods': methods
        }
    
    def parse_javascript(self, content: str, file_path: str) -> Dict[str, Any]:
        """Basic parser for JavaScript code using regex."""
        # This is a simplified parser - for production, consider using a proper JS parser
        function_pattern = r'function\s+(\w+)\s*\(([^)]*)\)'
        arrow_function_pattern = r'(?:const|let|var)?\s*(\w+)\s*=\s*\(([^)]*)\)\s*=>'
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+\w+)?'
        import_pattern = r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]'
        
        functions = []
        for match in re.finditer(function_pattern, content):
            functions.append({
                'name': match.group(1),
                'params': match.group(2)
            })
        
        for match in re.finditer(arrow_function_pattern, content):
            functions.append({
                'name': match.group(1),
                'params': match.group(2),
                'type': 'arrow'
            })
        
        classes = []
        for match in re.finditer(class_pattern, content):
            classes.append(match.group(1))
        
        imports = re.findall(import_pattern, content)
        
        return {
            'language': 'javascript',
            'functions': functions,
            'classes': classes,
            'imports': imports
        }
    
    # Function extraction methods for each language
    
    def extract_python_functions(self, content: str) -> List[Dict[str, Any]]:
        """Extract functions from Python code."""
        analysis = self.parse_python(content, "temp.py")
        if 'error' in analysis:
            return []
        return analysis.get('functions', [])
    
    def extract_c_cpp_functions(self, content: str) -> List[Dict[str, Any]]:
        """Extract functions from C/C++ code."""
        analysis = self.parse_c_cpp(content, "temp.cpp" if "class" in content else "temp.c")
        if 'error' in analysis:
            return []
        return analysis.get('functions', [])
    
    def extract_java_functions(self, content: str) -> List[Dict[str, Any]]:
        """Extract functions from Java code."""
        analysis = self.parse_java(content, "temp.java")
        if 'error' in analysis:
            return []
        return analysis.get('methods', [])
    
    def extract_javascript_functions(self, content: str) -> List[Dict[str, Any]]:
        """Extract functions from JavaScript code."""
        analysis = self.parse_javascript(content, "temp.js")
        if 'error' in analysis:
            return []
        return analysis.get('functions', [])
    
    # Template creation methods
    
    def create_function_template(self, function_info: Dict[str, Any], language: str) -> str:
        """Create a function template based on the function info and language."""
        if language == 'python':
            args = ', '.join(function_info.get('args', []))
            template = f"def {function_info['name']}({args}):\n"
            if function_info.get('docstring'):
                template += f"    \"\"\"{function_info['docstring']}\"\"\"\n"
            template += "    # TODO: Implement function\n    pass"
            return template
        
        elif language in ['c', 'cpp']:
            return_type = function_info.get('type', '').split('(')[0].strip()
            if not return_type:
                return_type = function_info.get('return_type', 'void')
            
            name = function_info['name']
            args = function_info.get('args', '')
            
            if isinstance(args, list):
                args = ', '.join(args)
            
            template = f"{return_type} {name}({args}) {{\n"
            if function_info.get('comment'):
                comment_lines = function_info['comment'].split('\n')
                for line in comment_lines:
                    template += f"    // {line.strip('/* ')}\n"
            template += "    // TODO: Implement function\n"
            
            # Add return statement for non-void functions
            if return_type != 'void':
                if 'int' in return_type or 'bool' in return_type:
                    template += "    return 0;\n"
                elif 'char' in return_type and '*' in return_type:
                    template += "    return NULL;\n"
                elif 'float' in return_type or 'double' in return_type:
                    template += "    return 0.0;\n"
                else:
                    template += f"    return ({return_type})0;\n"
            
            template += "}"
            return template
        
        elif language == 'java':
            visibility = function_info.get('visibility', 'public')
            return_type = function_info.get('return_type', 'void')
            name = function_info['name']
            params = function_info.get('params', '')
            
            template = f"{visibility} {return_type} {name}({params}) {{\n"
            template += "    // TODO: Implement method\n"
            
            # Add return statement for non-void methods
            if return_type != 'void':
                if return_type in ['int', 'byte', 'short', 'long']:
                    template += "    return 0;\n"
                elif return_type == 'boolean':
                    template += "    return false;\n"
                elif return_type in ['float', 'double']:
                    template += "    return 0.0;\n"
                elif return_type == 'char':
                    template += "    return '\\0';\n"
                else:
                    template += "    return null;\n"
            
            template += "}"
            return template
        
        else:
            # Generic template for unsupported languages
            return f"Function: {function_info['name']}\n// TODO: Implement function"


def extract_function_template(code_content, language='python'):
    """Extract function templates using semantic analysis."""
    analyzer = SemanticCodeAnalyzer()
    
    # Get functions from the code content
    if language == 'python':
        functions = analyzer.extract_python_functions(code_content)
    elif language in ['c', 'cpp']:
        functions = analyzer.extract_c_cpp_functions(code_content)
    elif language == 'java':
        functions = analyzer.extract_java_functions(code_content)
    elif language == 'javascript':
        functions = analyzer.extract_javascript_functions(code_content)
    else:
        # Fallback to regex-based extraction for unsupported languages
        if language not in ['python', 'c', 'cpp', 'java', 'javascript']:
            print(f"Warning: Unsupported language '{language}'. Falling back to regex parsing.")
        
        # Simple regex-based extraction
        if language == 'python':
            function_match = re.search(r"def\s+(\w+)\s*\((.*?)\):", code_content, re.DOTALL)
            if function_match:
                func_name = function_match.group(1)
                args = function_match.group(2)
                return f"def {func_name}({args}):\n    # TODO: Implement function\n    pass"
        elif language in ['c', 'cpp']:
            function_match = re.search(r"(\w+)\s+(\w+)\s*\((.*?)\)\s*{", code_content, re.DOTALL)
            if function_match:
                return_type, func_name, args = function_match.groups()
                return f"{return_type} {func_name}({args}) {{\n    // TODO: Implement function\n}}"
        
        return None
    
    # Create templates for the extracted functions
    templates = []
    for func in functions:
        template = analyzer.create_function_template(func, language)
        if template:
            templates.append((template, func.get('name', ''), func.get('docstring', '')))
    
    # Return the first valid template or None
    return templates[0][0] if templates else None

def extract_function_description(code_content, language='python'):
    """Extract function description using semantic analysis."""
    analyzer = SemanticCodeAnalyzer()
    
    # Get functions from the code content
    if language == 'python':
        functions = analyzer.extract_python_functions(code_content)
        if functions:
            func = functions[0]
            docstring = func.get('docstring', '')
            if docstring:
                # Return first line of docstring as description
                return docstring.split('\n')[0]
            
            # If no docstring, try to infer from function name
            func_name = func.get('name', '')
            if func_name:
                # Convert snake_case to natural language
                name_desc = ' '.join(func_name.split('_')).capitalize()
                return f"{name_desc} function"
    
    elif language in ['c', 'cpp']:
        functions = analyzer.extract_c_cpp_functions(code_content)
        if functions:
            func = functions[0]
            comment = func.get('comment', '')
            if comment:
                # Clean up the comment and return the first line
                clean_comment = comment.replace('/*', '').replace('*/', '').replace('*', '').strip()
                return clean_comment.split('\n')[0]
            
            # If no comment, use function name
            func_name = func.get('name', '')
            if func_name:
                return f"{func_name} function in {language}"
    
    elif language == 'java':
        methods = analyzer.extract_java_functions(code_content)
        if methods:
            method = methods[0]
            return f"{method.get('name', '')} method"
    
    # Fallback to regex extraction
    if language == 'python':
        function_match = re.search(r"def\s+(\w+)\s*\(", code_content)
        if function_match:
            func_name = function_match.group(1)
            name_desc = ' '.join(func_name.split('_')).capitalize()
            return f"{name_desc} function"
    elif language in ['c', 'cpp']:
        # Look for a comment block before a function definition
        comment_and_function = re.search(r"/\*\*(.*?)\*/\s*(\w+)\s+(\w+)\s*\(", code_content, re.DOTALL)
        if comment_and_function:
            comment = comment_and_function.group(1).strip()
            if comment:
                # Clean up and return first line
                clean_comment = re.sub(r'\n\s*\*\s*', ' ', comment).strip()
                return clean_comment.split('.')[0]
            
            # If no useful comment, use function name
            func_name = comment_and_function.group(3)
            return f"{func_name} function in {language}"
    
    return f"Implement a {language} function"
```

### 3.4. Language-Specific Bug Introduction and Training Example Creation

```python
import random
import re

def introduce_bug(code, bug_type, language='python'):
    """
    Introduce a specific type of bug into the code based on language.
    Supports Python, C, and C++.
    """
    # Language-agnostic bugs
    if bug_type == "syntax":
        # Define language-specific syntax elements to remove
        syntax_elements = {
            'python': [":", ")", "(", "\"", "'", ",", ".", "[", "]"],
            'c': [";", ")", "(", "\"", "'", ",", ".", "{", "}", "*"],
            'cpp': [";", ")", "(", "\"", "'", ",", ".", "{", "}", "*", "::"]
        }
        
        patterns = syntax_elements.get(language, syntax_elements['python'])
        for pattern in patterns:
            if pattern in code:
                return code.replace(pattern, "", 1)
    
    # Language-specific bugs
    if language == 'python':
        if bug_type == "logical":
            # Change logical operators
            replacements = [
                ("<=", "<"), (">=", ">"), 
                ("==", "!="), ("!=", "=="),
                ("+", "-"), ("-", "+"),
                ("True", "False"), ("False", "True"),
                ("and", "or"), ("or", "and"),
                ("in", "not in"), ("not in", "in")
            ]
            
            for old, new in replacements:
                if old in code:
                    return code.replace(old, new, 1)
        
        elif bug_type == "variable":
            # Change variable name
            var_match = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", code)
            if var_match:
                var_name = var_match.group(1)
                # Skip common keywords
                if var_name not in ["def", "if", "else", "return", "for", "while", "import", "from", "as", "class"]:
                    return code.replace(var_name, var_name + "_typo", 1)
        
        elif bug_type == "indentation":
            # Introduce indentation error
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if line.startswith('    '):
                    lines[i] = line[2:]  # Remove 2 spaces
                    return '\n'.join(lines)
    
    elif language in ['c', 'cpp']:
        if bug_type == "logical":
            # Change logical operators
            replacements = [
                ("<=", "<"), (">=", ">"), 
                ("==", "!="), ("!=", "=="),
                ("+", "-"), ("-", "+"),
                ("++", "--"), ("--", "++"),
                ("+=", "-="), ("-=", "+="),
                ("*=", "/="), ("/=", "*="),
                ("&&", "||"), ("||", "&&")
            ]
            
            for old, new in replacements:
                if old in code:
                    return code.replace(old, new, 1)
        
        elif bug_type == "memory":
            # Introduce memory bugs (C/C++ specific)
            
            # 1. Missing free/delete
            if 'malloc(' in code and 'free(' in code:
                return code.replace('free(', '// free(', 1)
            
            if 'new ' in code and 'delete ' in code:
                return code.replace('delete ', '// delete ', 1)
            
            # 2. Buffer overflow
            array_decl = re.search(r'(\w+)\s*\[(\d+)\]', code)
            if array_decl:
                array_name = array_decl.group(1)
                array_size = int(array_decl.group(2))
                # Find a place where this array is used with an index
                array_usage = re.search(fr'{array_name}\s*\[\s*(\w+)\s*\]', code)
                if array_usage:
                    index_var = array_usage.group(1)
                    # If the index is a numeric constant
                    if index_var.isdigit():
                        new_index = str(array_size)  # Out of bounds access
                        return code.replace(f'{array_name}[{index_var}]', f'{array_name}[{new_index}]', 1)
                    else:
                        # If the index is a variable, try to modify its bounds check
                        bounds_check = re.search(fr'if\s*\(\s*{index_var}\s*<\s*(\d+|\w+)\s*\)', code)
                        if bounds_check:
                            limit = bounds_check.group(1)
                            return code.replace(f'{index_var} < {limit}', f'{index_var} <= {limit}', 1)
            
            # 3. Null pointer dereference
            ptr_check = re.search(r'if\s*\(\s*(\w+)\s*!=\s*NULL\s*\)', code)
            if ptr_check:
                ptr_name = ptr_check.group(1)
                return code.replace(f'if ({ptr_name} != NULL)', f'if (1)', 1)
        
        elif bug_type == "variable":
            # Change variable name, but avoid C/C++ keywords
            var_match = re.search(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", code)
            if var_match:
                var_name = var_match.group(1)
                # Skip common keywords
                c_cpp_keywords = ["if", "else", "return", "for", "while", "int", "char", "void", 
                                 "struct", "class", "public", "private", "protected", "template",
                                 "typename", "const", "static", "extern", "sizeof", "typedef"]
                if var_name not in c_cpp_keywords:
                    # Make sure we're not inside a string
                    pattern = r'\b' + re.escape(var_name) + r'\b'
                    matches = list(re.finditer(pattern, code))
                    if matches:
                        match = random.choice(matches)
                        start, end = match.span()
                        # Check if we're inside a string
                        string_regions = [(m.start(), m.end()) for m in re.finditer(r'"[^"]*"', code)]
                        if not any(s <= start and end <= e for s, e in string_regions):
                            return code[:start] + var_name + "_typo" + code[end:]
        
        elif bug_type == "off_by_one":
            # Introduce off-by-one errors
            loop_match = re.search(r'for\s*\(\s*\w+\s*=\s*\d+\s*;\s*\w+\s*([<>]=?)\s*(\w+)\s*;', code)
            if loop_match:
                operator = loop_match.group(1)
                if operator == "<":
                    return code.replace(operator, "<=", 1)
                elif operator == "<=":
                    return code.replace(operator, "<", 1)
                elif operator == ">":
                    return code.replace(operator, ">=", 1)
                elif operator == ">=":
                    return code.replace(operator, ">", 1)
    
    # Fallback - if no specific bug was introduced, try generic methods
    if "if" in code:
        # Negate a condition
        return code.replace("if (", "if (!", 1)
    
    # Return original code if no bugs introduced
    return code


def create_language_specific_bugs(code, language):
    """Create a list of language-specific bugs to introduce."""
    common_bugs = ["syntax", "logical", "variable"]
    
    if language == 'python':
        return common_bugs + ["indentation"]
    
    elif language in ['c', 'cpp']:
        return common_bugs + ["memory", "off_by_one"]
    
    return common_bugs  # Fallback for other languages

# Process files to create training examples
training_examples = []

# Process real code examples by language
analyzer = SemanticCodeAnalyzer()

for lang, files in language_classified_files.items():
    print(f"Processing {len(files)} {lang} files for training examples...")
    
    # Get language-specific bug types
    bug_types = create_language_specific_bugs("", lang)
    
    for file_path in files[:2000]:  # Limit per language - adjust based on your dataset size
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            try:
                content = f.read()
                
                # Size check - for manageable examples
                if 100 <= len(content) <= 5000:  # Increased max size for C/C++ which tends to have longer files
                    # Perform semantic analysis on the file
                    semantic_info = analyzer.analyze_file(file_path, language=lang)
                    
                    # Skip files with parsing errors
                    if 'error' in semantic_info:
                        continue
                    
                    # 1. Create bug detection example
                    bug_type = random.choice(bug_types)
                    buggy_code = introduce_bug(content, bug_type, language=lang)
                    
                    # Only add example if bug was successfully introduced
                    if buggy_code != content:
                        # For C/C++, add more detailed bug descriptions
                        instruction = f"Identify and fix any {bug_type} errors in the above code."
                        
                        if lang in ['c', 'cpp'] and bug_type == "memory":
                            instruction = "Identify and fix any memory-related issues in the code (e.g., memory leaks, buffer overflows, null pointer dereferences)."
                        elif lang in ['c', 'cpp'] and bug_type == "off_by_one":
                            instruction = "Identify and fix any off-by-one errors in loop conditions or array accesses."
                        
                        example = {
                            "task": "Bug Detection and Fix",
                            "input_code": buggy_code,
                            "instruction": instruction,
                            "output_code": content,
                            "language": lang,
                            "source": "real"
                        }
                        training_examples.append(example)
                    
                    # 2. Create template completion examples
                    function_template = extract_function_template(content, language=lang)
                    function_desc = extract_function_description(content, language=lang)
                    
                    if function_template and function_desc:
                        # For C/C++, add more context about function purpose and expected behavior
                        if lang in ['c', 'cpp']:
                            # Extract information about function parameters and return values if available
                            functions = semantic_info.get('functions', [])
                            if functions:
                                func = functions[0]
                                func_name = func.get('name', '')
                                func_type = func.get('type', '').split('(')[0].strip()
                                
                                if func_name and func_type:
                                    function_desc += f". The function takes parameters as specified in the template and should return a {func_type} value."
                        
                        template_example = {
                            "task": "Fill Template with Functional Code",
                            "template": function_template,
                            "instruction": f"Fill in the template to implement a {function_desc}",
                            "output_code": content,
                            "language": lang,
                            "source": "real"
                        }
                        training_examples.append(template_example)
                    
                    # 3. For C/C++ specifically, add examples for common patterns
                    if lang in ['c', 'cpp']:
                        # Look for memory allocation patterns
                        if 'malloc(' in content or 'new ' in content:
                            memory_management_example = {
                                "task": "Memory Management Implementation",
                                "input_code": re.sub(r'(free\([^)]+\)|delete\s+[^;]+;)', '// TODO: Clean up resources', content),
                                "instruction": "Complete the memory management code by properly freeing/deleting all allocated resources.",
                                "output_code": content,
                                "language": lang,
                                "source": "real"
                            }
                            training_examples.append(memory_management_example)
                        
                        # Look for error handling patterns
                        if 'if' in content and ('return' in content or 'exit' in content):
                            error_handling_example = {
                                "task": "Error Handling Implementation",
                                "input_code": re.sub(r'if\s*\([^)]+\)\s*{\s*[^}]*\s*(return|exit)[^;]*;\s*}', '// TODO: Add error handling', content),
                                "instruction": "Implement proper error handling for the function.",
                                "output_code": content,
                                "language": lang,
                                "source": "real"
                            }
                            training_examples.append(error_handling_example)
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

# Generate C/C++ specific synthetic data
def generate_cpp_synthetic_examples(n_examples=200):
    """Generate C/C++ specific synthetic code examples."""
    prompts = [
        # Memory management
        "Generate a C++ function that demonstrates proper memory management with new/delete. Include a bug where memory is leaked, then show the correct version with the bug fixed.",
        "Create a C function that uses malloc/free to handle dynamic memory. Include a memory leak bug and its fix.",
        "Write a C++ class that manages resources and needs a proper destructor. Show a buggy version where resources aren't properly freed, then fix it.",
        
        # Pointers and references
        "Write a C++ function that demonstrates common pointer mistakes (null dereferencing, dangling pointers) and then show the corrected version.",
        "Create a C function that incorrectly uses pointers resulting in buffer overflow, then show the fixed version.",
        "Generate a C++ example showing incorrect reference handling and its fix.",
        
        # Templates and generics
        "Write a C++ template function with a subtle bug, then show the corrected version.",
        "Create a C++ generic data structure with a bug in its implementation, then fix it.",
        
        # Concurrency
        "Generate a C++ multithreaded function with a race condition, then show how to fix it using proper synchronization.",
        "Write a C function that uses threads but has a deadlock issue, then show the corrected version.",
        
        # STL usage
        "Create a C++ function that misuses STL containers leading to undefined behavior, then fix it.",
        "Write a C++ example showing incorrect iterator usage and the proper way to use iterators."
    ]
    
    print(f"Generating {n_examples} synthetic C/C++ examples...")
    # In a real implementation, you would call an external LLM API here
    # For simplicity, we'll create a simple template
    
    synthetic_data = []
    
    # Simplified synthetic data generation (in real implementation, use an LLM API)
    for i in range(n_examples):
        prompt = random.choice(prompts)
        
        # Simple template for memory management bug
        if "memory management" in prompt or "malloc" in prompt:
            buggy_code = """
#include <stdlib.h>

void* allocate_memory() {
    int* array = (int*)malloc(10 * sizeof(int));
    array[0] = 42;
    // Missing free(array) before return
    return array;
}
            """
            
            fixed_code = """
#include <stdlib.h>

void* allocate_memory() {
    int* array = (int*)malloc(10 * sizeof(int));
    array[0] = 42;
    return array;
    // Caller is responsible for freeing the memory
}

void cleanup_memory(void* ptr) {
    free(ptr);
}
            """
            
            synthetic_data.append({
                "task": "Bug Detection and Fix",
                "input_code": buggy_code,
                "instruction": "Identify and fix the memory management issue in this C function.",
                "output_code": fixed_code,
                "language": "c",
                "source": "synthetic"
            })
        
        # Template for buffer overflow
        elif "buffer overflow" in prompt:
            buggy_code = """
#include <stdio.h>
#include <string.h>

void copy_string(char* dest, const char* src) {
    // No bounds checking - potential buffer overflow
    strcpy(dest, src);
}

int main() {
    char small_buffer[5];
    char* long_string = "This string is too long for the buffer";
    copy_string(small_buffer, long_string);
    printf("%s\\n", small_buffer);
    return 0;
}
            """
            
            fixed_code = """
#include <stdio.h>
#include <string.h>

void copy_string(char* dest, const char* src, size_t dest_size) {
    // Safe string copy with bounds checking
    strncpy(dest, src, dest_size - 1);
    dest[dest_size - 1] = '\\0';  // Ensure null termination
}

int main() {
    char small_buffer[5];
    char* long_string = "This string is too long for the buffer";
    copy_string(small_buffer, long_string, sizeof(small_buffer));
    printf("%s\\n", small_buffer);
    return 0;
}
            """
            
            synthetic_data.append({
                "task": "Bug Detection and Fix",
                "input_code": buggy_code,
                "instruction": "Fix the buffer overflow vulnerability in this C code.",
                "output_code": fixed_code,
                "language": "c",
                "source": "synthetic"
            })
        
        # Template for C++ class with resource management
        elif "class" in prompt and "destructor" in prompt:
            buggy_code = """
#include <iostream>

class ResourceManager {
private:
    int* data;
    
public:
    ResourceManager(int size) {
        data = new int[size];
        std::cout << "Resource allocated\\n";
    }
    
    // Missing destructor - memory leak
    
    void use_resource() {
        std::cout << "Using resource\\n";
    }
};

int main() {
    ResourceManager* rm = new ResourceManager(100);
    rm->use_resource();
    // Missing delete rm - another memory leak
    return 0;
}
            """
            
            fixed_code = """
#include <iostream>

class ResourceManager {
private:
    int* data;
    
public:
    ResourceManager(int size) {
        data = new int[size];
        std::cout << "Resource allocated\\n";
    }
    
    ~ResourceManager() {
        delete[] data;
        std::cout << "Resource freed\\n";
    }
    
    void use_resource() {
        std::cout << "Using resource\\n";
    }
};

int main() {
    ResourceManager* rm = new ResourceManager(100);
    rm->use_resource();
    delete rm;  // Properly clean up
    return 0;
}
            """
            
            synthetic_data.append({
                "task": "Bug Detection and Fix",
                "input_code": buggy_code,
                "instruction": "Fix the memory leaks in this C++ class by implementing proper resource management.",
                "output_code": fixed_code,
                "language": "cpp",
                "source": "synthetic"
            })
    
    return synthetic_data


# Generate synthetic examples specific to each language
python_synthetic_examples = []
cpp_synthetic_examples = []

# In a real implementation, read/generate actual synthetic examples
# For this demo, we'll generate some C/C++ specific examples
cpp_synthetic_examples = generate_cpp_synthetic_examples(300)

# Read general synthetic examples generated earlier
try:
    with open("code_data/processed/synthetic_examples.json", "r") as f:
        python_synthetic_examples = json.load(f)
except FileNotFoundError:
    print("Warning: synthetic_examples.json not found. Using only C/C++ synthetic data.")

# Combine all synthetic examples
synthetic_examples = python_synthetic_examples + cpp_synthetic_examples
training_examples.extend(synthetic_examples)

print(f"Created {len(training_examples)} training examples")
print(f"  - Real examples: {len(training_examples) - len(synthetic_examples)}")
print(f"  - Synthetic examples: {len(synthetic_examples)}")
print(f"    - Python: {len(python_synthetic_examples)}")
print(f"    - C/C++: {len(cpp_synthetic_examples)}")

# Balance the dataset by language
language_counts = {}
for example in training_examples:
    lang = example.get('language', 'unknown')
    if lang not in language_counts:
        language_counts[lang] = 0
    language_counts[lang] += 1

print("Language distribution in training data:")
for lang, count in language_counts.items():
    print(f"  {lang}: {count} examples ({count/len(training_examples)*100:.1f}%)")

# Split into train/val/test sets
random.shuffle(training_examples)
train_size = int(len(training_examples) * 0.8)
val_size = int(len(training_examples) * 0.1)

train_data = training_examples[:train_size]
val_data = training_examples[train_size:train_size + val_size]
test_data = training_examples[train_size + val_size:]

print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")
```

### 3.5. Format Data for Fine-Tuning

```python
def format_prompt(example):
    """Format the example into a prompt for training."""
    language = example.get('language', 'python')
    code_tag = language if language != 'other' else 'code'
    
    if "template" in example:
        prompt = f"""### Task: {example['task']}
### Template:
```{code_tag}
{example['template']}
```
### Instruction:
{example['instruction']}
### Output Code:
```{code_tag}
{example['output_code']}
```"""
    else:
        prompt = f"""### Task: {example['task']}
### Input Code:
```{code_tag}
{example['input_code']}
```
### Instruction:
{example['instruction']}
### Output Code:
```{code_tag}
{example['output_code']}
```"""
    return prompt

# Prepare the datasets in the format expected by the trainer
def prepare_dataset(examples, output_file):
    formatted_data = []
    for example in examples:
        formatted_data.append({
            "text": format_prompt(example)
        })
    
    # Save formatted data
    with open(output_file, "w") as f:
        json.dump(formatted_data, f, indent=2)
    
    return len(formatted_data)

# Prepare all datasets
train_count = prepare_dataset(train_data, "code_data/processed/train.json")
val_count = prepare_dataset(val_data, "code_data/processed/validation.json")
test_count = prepare_dataset(test_data, "code_data/processed/test.json")

print(f"Saved {train_count} training examples, {val_count} validation examples, and {test_count} test examples")
```

## 4. AWS Setup and Fine-Tuning Implementation

### 4.1. AWS Setup and Permissions

1. **Required IAM Permissions**

   Before starting, ensure your IAM role has the following permissions:
   ```json
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Action": [
                   "sagemaker:*",
                   "s3:*",
                   "logs:*",
                   "iam:PassRole",
                   "ecr:GetAuthorizationToken",
                   "ecr:BatchCheckLayerAvailability",
                   "ecr:GetDownloadUrlForLayer",
                   "ecr:BatchGetImage"
               ],
               "Resource": "*"
           }
       ]
   }
   ```

2. **Create a SageMaker Notebook Instance**
   - Navigate to AWS SageMaker in the console
   - Select "Notebook instances" and click "Create notebook instance"
   - Choose `ml.t3.medium` for the notebook (sufficient for setup)
   - Select the IAM role with appropriate permissions
   - Set Volume Size to at least 50GB
   - Create and launch the instance

3. **Access to LLaMA 4 Models**
   - LLaMA 4 models require explicit permissions
   - Register at [Meta AI's website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
   - After approval, follow Meta's instructions to create a Hugging Face token
   - Store this token securely for use in your training scripts
   - Add the token to SageMaker as a secret:

   ```python
   import boto3
   from botocore.exceptions import ClientError
   
   def create_secret(secret_name, secret_value, region_name="us-east-1"):
       """Create a secret in AWS Secrets Manager."""
       client = boto3.client('secretsmanager', region_name=region_name)
       
       try:
           response = client.create_secret(
               Name=secret_name,
               SecretString=secret_value
           )
           print(f"Secret created: {response['ARN']}")
           return response['ARN']
       except ClientError as e:
           if e.response['Error']['Code'] == 'ResourceExistsException':
               print(f"Secret {secret_name} already exists. Updating...")
               response = client.update_secret(
                   SecretId=secret_name,
                   SecretString=secret_value
               )
               print(f"Secret updated: {response['ARN']}")
               return response['ARN']
           else:
               print(f"Error creating secret: {e}")
               raise
   
   # Create or update the HF token secret
   create_secret('hf-access-token', 'your_hugging_face_token_here')
   ```

### 4.2. SageMaker Setup and Training

1. **Upload Data to S3**
   ```python
   import boto3
   import sagemaker
   from sagemaker.pytorch import PyTorch
   import time
   
   # Initialize SageMaker session
   sagemaker_session = sagemaker.Session()
   role = sagemaker.get_execution_role()
   
   # Upload data to S3
   bucket = sagemaker_session.default_bucket()
   prefix = 'code-model-fine-tuning'
   
   def upload_directory_to_s3(local_dir, s3_prefix):
       """Upload a directory to S3."""
       s3_uri = sagemaker_session.upload_data(
           path=local_dir,
           bucket=bucket,
           key_prefix=s3_prefix
       )
       return s3_uri
   
   # Upload processed data
   training_data_uri = upload_directory_to_s3('code_data/processed', f"{prefix}/data")
   print(f"Data uploaded to: {training_data_uri}")
   
   # Create a requirements.txt file
   with open("requirements.txt", "w") as f:
       f.write("\n".join([
           "transformers>=4.31.0",
           "peft>=0.4.0",
           "datasets>=2.13.0",
           "accelerate>=0.20.3",
           "bitsandbytes>=0.40.0",
           "flash-attn>=2.0.0",  # For Flash Attention 2 optimization
           "tensorboardX>=2.6.0"
       ]))
       
   # Upload requirements
   requirements_uri = upload_directory_to_s3('requirements.txt', f"{prefix}/requirements")
   ```

2. **Create Training Script (train.py)**
   ```python
   # Save this to a file named train.py
   import os
   import json
   import torch
   import argparse
   import logging
   from datasets import load_dataset
   from transformers import (
       AutoModelForCausalLM,
       AutoTokenizer,
       BitsAndBytesConfig,
       TrainingArguments,
       Trainer,
       DataCollatorForLanguageModeling,
       EarlyStoppingCallback,
   )
   from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
   
   # Set up logging
   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger(__name__)
   
   def parse_args():
       parser = argparse.ArgumentParser()
       parser.add_argument("--model-id", type=str, default="meta-llama/Llama-4-Scout-17B-16E-Instruct")
       parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
       parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
       parser.add_argument("--epochs", type=int, default=3)
       parser.add_argument("--batch-size", type=int, default=4)
       parser.add_argument("--learning-rate", type=float, default=1e-4)
       parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
       parser.add_argument("--lora-r", type=int, default=16)
       parser.add_argument("--lora-alpha", type=int, default=32)
       parser.add_argument("--lora-dropout", type=float, default=0.05)
       parser.add_argument("--max-seq-length", type=int, default=2048)
       parser.add_argument("--warmup-ratio", type=float, default=0.1)
       parser.add_argument("--weight-decay", type=float, default=0.01)
       parser.add_argument("--use-flash-attn", type=bool, default=True)
       parser.add_argument("--add-special-tokens", type=bool, default=True)
       parser.add_argument("--target-languages", type=str, default="cpp,python")
       parser.add_argument("--packing", type=bool, default=True)
       return parser.parse_args()
   
   def main():
       args = parse_args()
       logger.info(f"Starting fine-tuning with arguments: {args}")
       
       # Set Hugging Face token from environment if available
       hf_token = os.environ.get("HF_TOKEN", None)
       
       # Load model and tokenizer
       logger.info(f"Loading model: {args.model_id}")
       tokenizer = AutoTokenizer.from_pretrained(
           args.model_id,
           token=hf_token,
           trust_remote_code=True
       )
       tokenizer.pad_token = tokenizer.eos_token
       
       # Configure 4-bit quantization
       bnb_config = BitsAndBytesConfig(
           load_in_4bit=True,
           bnb_4bit_use_double_quant=True,
           bnb_4bit_quant_type="nf4",
           bnb_4bit_compute_dtype=torch.bfloat16
       )
       
       # Load model with quantization for memory efficiency
       logger.info("Loading model with 4-bit quantization")
       model = AutoModelForCausalLM.from_pretrained(
           args.model_id,
           device_map="auto",
           quantization_config=bnb_config,
           token=hf_token,
           trust_remote_code=True,
           use_flash_attention_2=args.use_flash_attn,  # Enable Flash Attention 2 if available
       )
       
       # Prepare model for QLoRA fine-tuning
       logger.info("Preparing model for QLoRA fine-tuning")
       model = prepare_model_for_kbit_training(model)
       
       # LoRA configuration
       logger.info(f"Configuring LoRA with r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
       lora_config = LoraConfig(
           r=args.lora_r,
           lora_alpha=args.lora_alpha,
           lora_dropout=args.lora_dropout,
           bias="none",
           task_type="CAUSAL_LM",
           target_modules=[
               "q_proj", "k_proj", "v_proj", "o_proj",  # Attention modules
               "gate_proj", "up_proj", "down_proj"  # MLP modules
           ]
       )
       
       # Apply LoRA to model
       logger.info("Applying LoRA to model")
       model = get_peft_model(model, lora_config)
       model.print_trainable_parameters()
       
       # Load datasets
       logger.info(f"Loading datasets from {args.data_dir}")
       train_file = os.path.join(args.data_dir, "train.json")
       val_file = os.path.join(args.data_dir, "validation.json")
       
       train_dataset = load_dataset("json", data_files=train_file)["train"]
       val_dataset = load_dataset("json", data_files=val_file)["train"]
       
       logger.info(f"Train dataset size: {len(train_dataset)}")
       logger.info(f"Validation dataset size: {len(val_dataset)}")
       
       # Function to tokenize inputs
       def tokenize_function(examples):
           return tokenizer(
               examples["text"], 
               padding="max_length",
               truncation=True,
               max_length=args.max_seq_length,
           )
       
       logger.info("Tokenizing datasets")
       tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
       tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
       
       # Configure training arguments
       logger.info("Configuring training arguments")
       training_args = TrainingArguments(
           output_dir=args.output_dir,
           num_train_epochs=args.epochs,
           per_device_train_batch_size=args.batch_size,
           per_device_eval_batch_size=args.batch_size,
           gradient_accumulation_steps=args.gradient_accumulation_steps,
           learning_rate=args.learning_rate,
           weight_decay=args.weight_decay,
           warmup_ratio=args.warmup_ratio,
           save_strategy="epoch",
           evaluation_strategy="epoch",
           load_best_model_at_end=True,
           fp16=True,
           logging_steps=10,
           report_to="tensorboard",
           save_total_limit=3,  # Keep only the last 3 checkpoints
       )
       
       # Initialize trainer
       logger.info("Initializing Trainer")
       trainer = Trainer(
           model=model,
           args=training_args,
           train_dataset=tokenized_train,
           eval_dataset=tokenized_val,
           tokenizer=tokenizer,
           data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
           callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
       )
       
       # Start training
       logger.info("Starting training")
       trainer.train()
       
       # Save the fine-tuned model
       logger.info(f"Saving model to {args.output_dir}")
       trainer.save_model(args.output_dir)
       tokenizer.save_pretrained(args.output_dir)
       
       # Save model configuration
       with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
           json.dump({
               "model_id": args.model_id,
               "epochs": args.epochs,
               "batch_size": args.batch_size,
               "learning_rate": args.learning_rate,
               "gradient_accumulation_steps": args.gradient_accumulation_steps,
               "lora_r": args.lora_r,
               "lora_alpha": args.lora_alpha,
               "lora_dropout": args.lora_dropout,
               "max_seq_length": args.max_seq_length,
               "use_flash_attn": args.use_flash_attn,
               "training_time": trainer.state.log_history[-1]["train_runtime"]
           }, f, indent=2)
       
       logger.info("Training completed successfully!")
   
   if __name__ == "__main__":
       main()
   ```

3. **Configure and Launch Training Job**
   ```python
   # First retrieve the Hugging Face token from AWS Secrets Manager
   import boto3
   from botocore.exceptions import ClientError
   
   def get_secret(secret_name, region_name="us-east-1"):
       """Get a secret from AWS Secrets Manager."""
       client = boto3.client('secretsmanager', region_name=region_name)
       
       try:
           get_secret_value_response = client.get_secret_value(SecretId=secret_name)
           return get_secret_value_response['SecretString']
       except ClientError as e:
           print(f"Error retrieving secret: {e}")
           return None
           
   # Get HF token
   hf_token = get_secret("hf-access-token")
   
   # Define the PyTorch estimator
   estimator = PyTorch(
       entry_point='train.py',
       source_dir='.',
       role=role,
       framework_version='2.0.1',
       py_version='py310',
       instance_count=1,
       instance_type='ml.g5.2xlarge',  # As recommended for LLaMA 4 with QLoRA
       max_run=86400,  # 24 hours max runtime
       keep_alive_period_in_seconds=1800,  # 30 minutes
       container_log_level=20,  # INFO level
       environment={
           'HF_TOKEN': hf_token,
       },
       hyperparameters={
           'model-id': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
           'epochs': 3,
           'batch-size': 2,
           'learning-rate': 1e-4,
           'lora-r': 16,
           'lora-alpha': 32,
           'lora-dropout': 0.05,
           'gradient-accumulation-steps': 16,  # Increased for stability
           'max-seq-length': 2048,
           'warmup-ratio': 0.1,
           'weight-decay': 0.01,
           'use-flash-attn': True
       },
   )
   
   # Start training job
   job_name = f"llama4-code-finetune-{int(time.time())}"
   estimator.fit(
       {'training': training_data_uri},
       job_name=job_name
   )
   
   # Get the model artifacts
   model_data = estimator.model_data
   print(f"Model artifacts saved to: {model_data}")
   ```

4. **Monitor Training Progress**

   ```python
   # Monitor training progress
   import time
   import boto3
   
   client = boto3.client('sagemaker')
   
   def get_job_status(job_name):
       """Get the status of the training job."""
       response = client.describe_training_job(TrainingJobName=job_name)
       return response['TrainingJobStatus'], response.get('SecondaryStatus', '')
   
   def get_training_metrics(job_name):
       """Get metrics from CloudWatch Logs."""
       logs_client = boto3.client('logs')
       response = client.describe_training_job(TrainingJobName=job_name)
       log_stream_name = response['TrainingJobStatus']
       
       try:
           response = logs_client.get_log_events(
               logGroupName=f"/aws/sagemaker/TrainingJobs/{job_name}",
               logStreamName=log_stream_name,
               limit=100
           )
           
           # Process and extract metrics
           metrics = []
           for event in response['events']:
               if 'loss' in event['message'] or 'eval_loss' in event['message']:
                   metrics.append(event['message'])
           
           return metrics
       except Exception as e:
           print(f"Error getting logs: {e}")
           return []
   
   def print_training_metrics(job_name):
       """Print the training and validation metrics."""
       response = client.describe_training_job(TrainingJobName=job_name)
       metrics = response.get('FinalMetricDataList', [])
       
       print("Training Metrics:")
       for metric in metrics:
           print(f"{metric['MetricName']}: {metric['Value']}")
   
   # Check status every 5 minutes
   while True:
       status, secondary_status = get_job_status(job_name)
       print(f"Job status: {status} - {secondary_status}")
       
       # Get some training metrics if available
       metrics = get_training_metrics(job_name)
       if metrics:
           print("Recent training metrics:")
           for metric in metrics[-5:]:  # Show last 5 metrics
               print(f"  {metric}")
       
       if status in ['Completed', 'Failed', 'Stopped']:
           print("Training job finished.")
           if status == 'Completed':
               print_training_metrics(job_name)
           break
           
       time.sleep(300)  # Wait 5 minutes
   ```

## 5. Model Evaluation and Deployment

### 5.1. Create Evaluation Script

```python
# Save this to a file named evaluate.py
import os
import json
import torch
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-id", type=str, default="meta-llama/Llama-4-Scout-17B-16E-Instruct")
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, default="evaluation_results.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()

def extract_generated_code(text):
    """Extract code from the generated text."""
    if "```" in text:
        # Extract code between triple backticks
        code_blocks = text.split("```")
        if len(code_blocks) >= 3:
            # Skip the language identifier if present
            code_block = code_blocks[1]
            if code_block.startswith(("python", "java", "cpp", "js")):
                code_block = code_block[code_block.find("\n")+1:]
            return code_block.strip()
    
    # Try to find code after "Output Code:" marker
    if "### Output Code:" in text:
        code_part = text.split("### Output Code:")[1].strip()
        if "```" in code_part:
            return code_part.split("```")[1].strip()
        return code_part
    
    # Fall back to returning everything after the prompt
    return text

def calculate_metrics(predictions, references, example_tasks):
    """Calculate code-specific metrics."""
    # Exact match
    exact_match_count = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    exact_match = exact_match_count / len(predictions) if predictions else 0
    
    # Token-level accuracy
    token_accuracies = []
    for pred, ref in zip(predictions, references):
        pred_tokens = set(pred.split())
        ref_tokens = set(ref.split())
        
        if ref_tokens:
            overlap = len(pred_tokens.intersection(ref_tokens))
            token_acc = overlap / len(ref_tokens)
            token_accuracies.append(token_acc)
    
    token_accuracy = np.mean(token_accuracies) if token_accuracies else 0
    
    # Bug detection accuracy (for bug detection examples)
    bug_fixed_count = 0
    bug_examples_count = 0
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        if "Bug Detection" in example_tasks[i]:
            bug_examples_count += 1
            # Check if the bug was fixed (using a simple string comparison)
            if pred.strip() == ref.strip():
                bug_fixed_count += 1
    
    bug_detection_acc = bug_fixed_count / bug_examples_count if bug_examples_count else 0
    
    # Template filling accuracy (for template examples)
    template_filled_count = 0
    template_examples_count = 0
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        if "Fill Template" in example_tasks[i]:
            template_examples_count += 1
            # Check if template was correctly filled
            if pred.strip() == ref.strip():
                template_filled_count += 1
    
    template_filling_acc = template_filled_count / template_examples_count if template_examples_count else 0
    
    return {
        "exact_match": exact_match,
        "token_accuracy": token_accuracy,
        "bug_detection_accuracy": bug_detection_acc,
        "template_filling_accuracy": template_filling_acc,
        "total_examples": len(predictions),
        "bug_examples_count": bug_examples_count,
        "bug_fixed_count": bug_fixed_count,
        "template_examples_count": template_examples_count,
        "template_filled_count": template_filled_count
    }

def main():
    args = parse_args()
    
    # Load test data
    with open(args.test_file, "r") as f:
        test_data = json.load(f)
    
    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN", None)
    
    # Load base model
    print(f"Loading base model: {args.base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_id,
        token=hf_token,
        trust_remote_code=True
    )
    
    # Configure quantization for efficient evaluation
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print("Loading fine-tuned model")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        device_map="auto",
        quantization_config=bnb_config,
        token=hf_token,
        trust_remote_code=True
    )
    
    # Load the fine-tuned adapter
    print(f"Loading adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.eval()
    
    # Generate predictions
    predictions = []
    references = []
    example_tasks = []
    
    print(f"Generating predictions for {len(test_data)} examples")
    for i, item in enumerate(tqdm(test_data)):
        example = json.loads(item["text"]) if isinstance(item["text"], str) else item
        
        # Extract task and content
        task = example.get("task", "")
        if not task and "### Task:" in item["text"]:
            task = item["text"].split("### Task:")[1].split("\n")[0].strip()
        
        # Extract reference code
        if "output_code" in example:
            reference = example["output_code"]
        else:
            # Try to extract from formatted text
            text_parts = item["text"].split("### Output Code:")
            if len(text_parts) > 1:
                code_part = text_parts[1]
                if "```" in code_part:
                    reference = code_part.split("```")[1].strip()
                else:
                    reference = code_part.strip()
            else:
                reference = ""
        
        # Format prompt
        if isinstance(item["text"], str) and "### Task:" in item["text"]:
            # If text is already formatted, use only the input part
            prompt_parts = item["text"].split("### Output Code:")
            prompt = prompt_parts[0] + "### Output Code:\n```"
        else:
            instruction = example.get("instruction", "")
            if "template" in example:
                code_input = example["template"]
                type_label = "Template"
            else:
                code_input = example.get("input_code", "")
                type_label = "Input Code"
            
            language = example.get("language", "python")
            code_tag = language if language != "other" else "code"
            
            prompt = f"""### Task: {task}
### {type_label}:
```{code_tag}
{code_input}
```
### Instruction:
{instruction}
### Output Code:
```{code_tag}
"""
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.1,
                do_sample=False,
                num_return_sequences=1,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated code
        code_output = extract_generated_code(generated_text)
        
        predictions.append(code_output)
        references.append(reference)
        example_tasks.append(task)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, references, example_tasks)
    
    print(f"Evaluation results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
    
    # Save results
    result_data = {
        "metrics": metrics,
        "examples": [
            {
                "task": task,
                "prediction": pred,
                "reference": ref,
                "correct": pred.strip() == ref.strip()
            }
            for task, pred, ref in zip(example_tasks, predictions, references)
        ]
    }
    
    with open(args.output_file, "w") as f:
        json.dump(result_data, f, indent=2)
    
    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()
```

### 5.2. Run Model Evaluation

```python
# Upload and run evaluation script
evaluation_script_uri = upload_directory_to_s3('evaluate.py', f"{prefix}/eval")

# Deploy test data
test_data_uri = sagemaker_session.upload_data(
    path="code_data/processed/test.json",
    bucket=bucket,
    key_prefix=f"{prefix}/eval_data"
)

# Run evaluation in a SageMaker Processing job
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch.processing import PyTorchProcessor

processor = PyTorchProcessor(
    framework_version="2.0.1",
    role=role,
    instance_count=1,
    instance_type="ml.g5.xlarge",
    base_job_name=f"code-model-eval",
    sagemaker_session=sagemaker_session
)

processor.run(
    code="evaluate.py",
    inputs=[
        ProcessingInput(
            source=model_data,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=test_data_uri,
            destination="/opt/ml/processing/test"
        ),
    ],
    outputs=[
        ProcessingOutput(
            output_name="evaluation",
            source="/opt/ml/processing/evaluation"
        )
    ],
    arguments=[
        "--base-model-id", "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "--adapter-path", "/opt/ml/processing/model",
        "--test-file", "/opt/ml/processing/test/test.json",
        "--output-file", "/opt/ml/processing/evaluation/results.json"
    ]
)

# Download and analyze evaluation results
eval_results_path = sagemaker_session.download_data(
    path=f"s3://{bucket}/{prefix}/eval_results/results.json",
    local_path="."
)

with open("results.json", "r") as f:
    results = json.load(f)
    
print("Evaluation Results:")
for metric, value in results["metrics"].items():
    print(f"  {metric}: {value}")
```

### 5.3. Deploy and Test the Model

```python
from sagemaker.huggingface import HuggingFaceModel

# Configure custom inference script for PEFT model
with open("inference.py", "w") as f:
    f.write("""
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Global variables
model = None
tokenizer = None

def model_fn(model_dir):
    global model, tokenizer
    
    # Get environment variables
    hf_model_id = os.environ.get("HF_MODEL_ID", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
    
    # Load tokenizer from the model directory
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # Configure 4-bit quantization for efficient inference
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )
    
    # Load adapter weights (LoRA)
    model = PeftModel.from_pretrained(model, model_dir)
    
    return model

def predict_fn(data, model_and_tokenizer):
    global model, tokenizer
    
    # Parse input data
    inputs = data.pop("inputs", data)
    parameters = data.pop("parameters", {})
    
    # Set default parameters if not provided
    max_new_tokens = parameters.get("max_new_tokens", 512)
    temperature = parameters.get("temperature", 0.1)
    do_sample = parameters.get("do_sample", False)
    top_p = parameters.get("top_p", 0.95)
    
    # Tokenize inputs
    input_tokens = tokenizer(inputs, return_tensors="pt").to(model.device)
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **input_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract code from generated text
    if "### Output Code:" in generated_text:
        parts = generated_text.split("### Output Code:")
        if len(parts) > 1:
            code_part = parts[1].strip()
            if "```" in code_part:
                response = code_part.split("```")[1].strip()
            else:
                response = code_part
        else:
            response = generated_text
    else:
        response = generated_text
    
    return {"generated_text": response}
""")

# Upload inference script to S3
inference_script_uri = upload_directory_to_s3('inference.py', f"{prefix}/inference")

# Create model object for deployment
huggingface_model = HuggingFaceModel(
    model_data=model_data,
    role=role,
    transformers_version="4.31.0",
    pytorch_version="2.0.1",
    py_version="py310",
    entry_point="inference.py",
    env={
        'HF_MODEL_ID': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
        'HF_TASK': 'text-generation',
        'HF_TOKEN': hf_token,
    }
)

# Deploy the model to an endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.inf1.xlarge",  # Using AWS Inferentia for cost-effective inference
    endpoint_name=f"code-model-endpoint-{int(time.time())}",
)

# Test the model with example prompts
test_prompt = """### Task: Bug Detection and Fix
### Input Code:
```python
def calculate_average(numbers)
    total = sum(numbers)
    return total / len(numbers)
```
### Instruction:
Identify and fix any syntax or logical errors in the above code.
### Output Code:
```python
"""

response = predictor.predict({
    "inputs": test_prompt,
    "parameters": {
        "max_new_tokens": 512,
        "temperature": 0.1,
        "do_sample": False
    }
})

print("Model response:")
print(response["generated_text"])
```

### 5.4. Performance Monitoring and Logging

```python
# Set up CloudWatch monitoring for the endpoint
import boto3

cloudwatch = boto3.client('cloudwatch')

# Create a custom dashboard for the endpoint
dashboard_body = {
    "widgets": [
        {
            "type": "metric",
            "x": 0,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["AWS/SageMaker", "Invocations", "EndpointName", predictor.endpoint_name],
                    [".", "InvocationsPerInstance", ".", "."]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": boto3.Session().region_name,
                "title": "Endpoint Invocations",
                "period": 60
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 0,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["AWS/SageMaker", "ModelLatency", "EndpointName", predictor.endpoint_name],
                    [".", "OverheadLatency", ".", "."]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": boto3.Session().region_name,
                "title": "Endpoint Latency",
                "period": 60
            }
        },
        {
            "type": "metric",
            "x": 0,
            "y": 6,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["AWS/SageMaker", "CPUUtilization", "EndpointName", predictor.endpoint_name]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": boto3.Session().region_name,
                "title": "CPU Utilization",
                "period": 60
            }
        },
        {
            "type": "metric",
            "x": 12,
            "y": 6,
            "width": 12,
            "height": 6,
            "properties": {
                "metrics": [
                    ["AWS/SageMaker", "MemoryUtilization", "EndpointName", predictor.endpoint_name]
                ],
                "view": "timeSeries",
                "stacked": False,
                "region": boto3.Session().region_name,
                "title": "Memory Utilization",
                "period": 60
            }
        }
    ]
}

# Create a CloudWatch dashboard
cloudwatch.put_dashboard(
    DashboardName=f"CodeModel-{predictor.endpoint_name}",
    DashboardBody=json.dumps(dashboard_body)
)

print(f"Created CloudWatch dashboard: CodeModel-{predictor.endpoint_name}")
```

## 6. Expected Outputs and Fine-Tuning Results

After fine-tuning, you should expect:

1. **PEFT Adapter Model**: A lightweight adapter (typically <100MB) that can be applied to the base LLaMA 4 model
2. **Performance Metrics**:
   - Bug detection accuracy: 85-95% on the test set
   - Code completion accuracy: 70-85% token accuracy
   - Exact matches: 40-60% depending on task complexity
   - **With Synthetic Data**: +5-10% improvement in overall accuracy metrics

3. **Deployment Artifacts**:
   - Fine-tuned model in S3 
   - SageMaker endpoint for real-time inference
   - CloudWatch dashboard for monitoring performance

4. **Documentation**:
   - Training metrics and logs
   - Evaluation results on the test set
   - Example code for inference

## 7. Troubleshooting Common Issues

### 7.1. Memory Issues
- **Symptoms**: OOM (Out of Memory) errors during training
- **Solutions**:
  - Reduce batch size (try 1 or 2 instead of 4)
  - Increase gradient accumulation steps (16 or 32)
  - Use a larger instance type (upgrade to ml.g5.4xlarge)
  - Reduce sequence length (1024 instead of 2048)
  - Enable more aggressive 4-bit quantization options
  - Disable Flash Attention if it's causing issues

### 7.2. Training Instability
- **Symptoms**: Loss spikes or NaN values
- **Solutions**:
  - Lower the learning rate (5e-5 instead of 1e-4)
  - Increase warmup ratio (0.2 instead of 0.1)
  - Add gradient clipping (`max_grad_norm=1.0`)
  - Try a different optimizer (e.g., AdamW with weight decay)
  - Start with a smaller LoRA rank (r=8) and increase gradually

### 7.3. Deployment Issues
- **Symptoms**: Endpoint creation failure
- **Solutions**:
  - Check IAM permissions
  - Verify model artifacts format
  - Try a different instance type (ml.g4dn.xlarge may be more stable than inferentia for some models)
  - Increase endpoint timeout settings
  - Verify that the custom inference script handles all edge cases

### 7.4. Synthetic Data Quality Issues
- **Symptoms**: Model performs worse with synthetic data
- **Solutions**:
  - Use a more powerful model for synthetic data generation
  - Implement filtering to remove low-quality synthetic examples
  - Reduce the proportion of synthetic data in the training set
  - Use human validation for a subset of synthetic examples

## 8. Cost Analysis and Recommendations

### 8.1. Detailed Cost Breakdown for 1GB C/C++ Codebase Fine-Tuning

| Component | Details | Calculation | Cost |
|-----------|---------|-------------|------|
| SageMaker Notebook | ml.t3.medium (8 hours) | 8 hours × $0.05/hour | $0.40 |
| Data Processing | ml.t3.medium (4 hours) | 4 hours × $0.05/hour | $0.20 |
| Language Detection & Semantic Analysis | ml.m5.xlarge (6 hours) | 6 hours × $0.25/hour | $1.50 |
| Synthetic Data Generation | API calls to GPT-4 (5K examples) | 5,000 examples × $0.03/example | $150.00 |
| Advanced C/C++ Synthetic Data | API calls to GPT-4 (2K specialized examples) | 2,000 examples × $0.05/example | $100.00 |
| Model Fine-Tuning | ml.g5.2xlarge (24 hours) | 24 hours × $1.444/hour | $34.66 |
| Hyperparameter Tuning | ml.g5.2xlarge (3 jobs × 8 hours) | 24 hours × $1.444/hour | $34.66 |
| Evaluation | ml.g5.xlarge (5 hours) | 5 hours × $1.24/hour | $6.20 |
| S3 Storage | 8GB for 1 month | 8GB × $0.023/GB-month | $0.18 |
| Model Deployment | ml.inf1.xlarge (72 hours) | 72 hours × $0.585/hour | $42.12 |
| **Total Estimated Cost** | | | **$369.92** |

**Note**: The cost increase when adding language detection, semantic analysis, and specialized C/C++ synthetic data generation is substantial ($369.92 vs $260.10 for basic synthetic data or $120.97 for real data only). However, research indicates that language-specific fine-tuning with semantic understanding can improve accuracy by 15-25% on specialized code tasks, particularly for memory safety bugs in C/C++ which are critical security vulnerabilities.

### 8.2. Inference Cost (After Deployment)

| Instance Type | Hourly Cost | Monthly Cost (24/7) | Notes |
|---------------|-------------|---------------------|-------|
| ml.inf1.xlarge | $0.585/hour | ~$421/month | Cost-efficient for production |
| ml.g4dn.xlarge | $0.736/hour | ~$530/month | Good for testing and lower volume |
| ml.c5.xlarge | $0.20/hour | ~$144/month | For CPU-only deployment (limited performance) |

### 8.3. Cost-Benefit Analysis of Language-Aware Approaches

| Metric | Real Data Only | Generic Synthetic Data | Language-Aware with Semantic Analysis | 
|--------|---------------|------------------------|------------------------------------|
| Bug Detection Accuracy | 85-88% | 90-95% | 95-98% for C/C++ memory bugs |
| Code Completion Accuracy | 70-75% | 78-85% | 82-90% for C/C++ code |
| Edge Case Handling | Limited | Good | Excellent for language-specific patterns |
| Memory Safety Bug Detection | <60% | ~75% | 90-95% |
| Training Cost | ~$120 | ~$260 | ~$370 |
| Development Time | 4-5 days | 6-7 days | 8-10 days |
| ROI (1-year projection) | Baseline | 2-3x better | 3-5x better for systems with C/C++ |

The advanced approach with language detection and semantic analysis requires a significant additional investment (+$110 beyond generic synthetic data and +$250 beyond real data only). However, the returns are substantial for organizations working with C/C++ codebases:

1. **Critical bug detection**: A 15-20% improvement in memory safety bug detection translates directly to reduced security vulnerabilities and potential exploits
2. **Language-specific optimizations**: The model learns C/C++ specific patterns that wouldn't be captured in a language-agnostic approach
3. **Higher business value**: For organizations where memory bugs cause significant business impact (security, reliability, compliance), the ROI can be 3-5x better

This approach is particularly valuable when fine-tuning for security-critical applications or when working with legacy C/C++ codebases that may contain numerous latent bugs.

### 8.4. Cost Optimization Strategies

1. **Use SageMaker Serverless Inference**:
   - Zero cost when not in use
   - Pay only for the duration of the actual requests
   - Good for sporadic usage patterns

2. **Batch Transform for Bulk Processing**:
   - More cost-effective for large batches of code processing
   - No need to maintain a persistent endpoint

3. **Export to ONNX and Deploy on CPU**:
   - Quantize the model to INT8 for CPU inference
   - Up to 70% cost reduction compared to GPU inference
   - Slight latency increase but still acceptable for many use cases

4. **Selective Synthetic Data Generation**:
   - Generate synthetic data only for underrepresented cases
   - Use lower-cost models for simple synthetic examples
   - Implement a tiered approach with different quality levels

5. **Model Distillation**:
   - Distill the fine-tuned model to a smaller, more efficient version
   - Can reduce deployment costs by 40-60%
   - Maintain 90-95% of the performance of the larger model

## 9. Model Distillation for Smaller Parameter Size

Currently, LLaMA 4 models are only available in 17B active parameter versions (Scout with 109B total parameters and Maverick with 400B total parameters). For teams requiring a smaller model with performance approaching the larger models, model distillation offers a viable approach.

### 9.1. Model Distillation Overview

Model distillation is a process where knowledge from a larger "teacher" model is transferred to a smaller "student" model. The student model learns to mimic the teacher's behavior, often achieving comparable performance with significantly fewer parameters. This approach has been successfully applied to other large models, such as DeepSeek-R1.

For LLaMA 4, we can create a distilled 7B parameter model by:

1. **Teacher-Student Knowledge Transfer**: Using the larger LLaMA 4 Scout/Maverick model to generate high-quality labeled data
2. **Architectural Adaptation**: Designing a smaller architecture while maintaining critical components
3. **Response Matching**: Training the smaller model to match the logits and hidden representations of the larger model

### 9.2. Distillation Process for LLaMA 4

```python
# Example code for setting up a model distillation pipeline
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader

# 1. Load the teacher model (LLaMA 4 Scout)
teacher_model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_id)
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
teacher_model.eval()  # Set to evaluation mode

# 2. Initialize the student model architecture (7B size)
# This would be a smaller architecture initialized from scratch
# or from an existing 7B model like LLaMA 3 8B
student_model_id = "meta-llama/Meta-Llama-3-8B"  # Starting point
student_tokenizer = AutoTokenizer.from_pretrained(student_model_id)
student_model = AutoModelForCausalLM.from_pretrained(
    student_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# 3. Generate synthetic data using the teacher model
def generate_synthetic_data(prompts, teacher_model, teacher_tokenizer, batch_size=8):
    """Generate responses using the teacher model for distillation."""
    synthetic_data = []
    
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = teacher_tokenizer(batch, return_tensors="pt", padding=True).to(teacher_model.device)
        
        with torch.no_grad():
            outputs = teacher_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
        generated_texts = teacher_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for prompt, response in zip(batch, generated_texts):
            synthetic_data.append({
                "prompt": prompt,
                "response": response
            })
    
    return synthetic_data

# 4. Implement distillation loss function
def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """Compute the distillation loss between student and teacher logits."""
    scaled_student_logits = student_logits / temperature
    scaled_teacher_logits = teacher_logits / temperature
    
    soft_targets = torch.nn.functional.softmax(scaled_teacher_logits, dim=-1)
    loss = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(scaled_student_logits, dim=-1),
        soft_targets,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    return loss

# 5. Training loop (simplified)
def train_distilled_model(student_model, teacher_model, train_dataloader, optimizer, epochs=3):
    """Train the student model to match the teacher's outputs."""
    for epoch in range(epochs):
        for batch in train_dataloader:
            # Forward pass through both models
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)
                
            student_outputs = student_model(**batch)
            
            # Compute distillation loss
            loss = distillation_loss(student_outputs.logits, teacher_outputs.logits)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch}, Loss: {loss.item()}")
            
    return student_model
```

### 9.3. AWS SageMaker Implementation

To implement distillation on SageMaker, we use a multi-step pipeline:

1. **Generate Synthetic Data**: Use the LLaMA 4 Scout/Maverick model to generate high-quality responses for a diverse set of prompts
2. **Distillation Training**: Train a smaller 7B parameter model to match the responses of the larger model
3. **Evaluation**: Compare the performance of the distilled model against the original across benchmarks

```python
# SageMaker pipeline for model distillation
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Step 1: Generate synthetic data using the teacher model
data_generation_processor = PyTorch(
    entry_point='generate_data.py',
    role=role,
    framework_version='2.0.1',
    py_version='py310',
    instance_count=1,
    instance_type='ml.g5.4xlarge',  # Use a larger instance for the teacher model
    hyperparameters={
        'teacher_model_id': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
        'num_examples': 100000,
        'output_path': '/opt/ml/processing/output/'
    }
)

data_generation_step = ProcessingStep(
    name='GenerateSyntheticData',
    processor=data_generation_processor,
    outputs=[
        sagemaker.processing.ProcessingOutput(
            output_name='training',
            source='/opt/ml/processing/output/train',
            destination=f's3://{sagemaker_session.default_bucket()}/distillation/data/train'
        ),
        sagemaker.processing.ProcessingOutput(
            output_name='validation',
            source='/opt/ml/processing/output/validation',
            destination=f's3://{sagemaker_session.default_bucket()}/distillation/data/validation'
        )
    ]
)

# Step 2: Distillation training
distillation_estimator = PyTorch(
    entry_point='distill.py',
    role=role,
    framework_version='2.0.1',
    py_version='py310',
    instance_count=1,
    instance_type='ml.g5.12xlarge',  # Multiple GPUs for faster training
    hyperparameters={
        'teacher_model_id': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
        'student_model_id': 'meta-llama/Meta-Llama-3-8B',
        'epochs': 3,
        'learning_rate': 1e-5,
        'temperature': 2.0
    },
    distribution={'torch_distributed': {'enabled': True}}
)

training_step = TrainingStep(
    name='DistillationTraining',
    estimator=distillation_estimator,
    inputs={
        'training': sagemaker.inputs.TrainingInput(
            s3_data=f's3://{sagemaker_session.default_bucket()}/distillation/data/train',
            content_type='application/json'
        ),
        'validation': sagemaker.inputs.TrainingInput(
            s3_data=f's3://{sagemaker_session.default_bucket()}/distillation/data/validation',
            content_type='application/json'
        )
    }
)

# Define and execute the pipeline
pipeline = Pipeline(
    name='LLaMA4-Distillation-Pipeline',
    steps=[data_generation_step, training_step],
    sagemaker_session=sagemaker_session
)

pipeline.upsert(role_arn=role)
execution = pipeline.start()
```

### 9.4. Estimated Costs for Distillation

| Component | Details | Calculation | Cost |
|-----------|---------|-------------|------|
| Synthetic Data Generation | ml.g5.4xlarge (24 hours) | 24 hours × $1.866/hour | $44.78 |
| Distillation Training | ml.g5.12xlarge (72 hours) | 72 hours × $4.31/hour | $310.32 |
| Model Evaluation | ml.g5.2xlarge (6 hours) | 6 hours × $1.444/hour | $8.66 |
| S3 Storage | 20GB for 1 month | 20GB × $0.023/GB-month | $0.46 |
| **Total Estimated Cost** | | | **$364.22** |

### 9.5. Performance Comparison

Based on similar distillation approaches with other models, we can expect the following performance metrics for a distilled 7B parameter LLaMA 4 model:

| Metric | LLaMA 4 Scout 17B | Distilled 7B Model | Performance Retention |
|--------|------------------|-------------------|----------------------|
| General Knowledge | 100% | 80-85% | 80-85% |
| Reasoning | 100% | 75-80% | 75-80% |
| Code Generation | 100% | 80-85% | 80-85% |
| Instruction Following | 100% | 85-90% | 85-90% |
| Memory Requirements | ~25-30GB | ~14GB | ~50% |
| Inference Speed | 1x | 2.5-3x | 250-300% |
| Deployment Cost | 1x | 0.4-0.5x | 40-50% |

The distilled model offers a compelling alternative for teams that need to balance performance with computational constraints, especially for edge deployments or high-throughput inference scenarios.
        
        #
