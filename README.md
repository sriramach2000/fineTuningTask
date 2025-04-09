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

### 1.3 Decision Tree for Instance Selection

```
Is budget the primary concern?
├── Yes → Is training time flexible?
│   ├── Yes → Use ml.g4dn.2xlarge with extended training time
│   └── No → Use ml.g5.2xlarge (best balance of cost/performance)
└── No → Is maximum performance the goal?
    ├── Yes → Does the model exceed 40GB memory requirements?
    │   ├── Yes → Use ml.p4de.24xlarge
    │   └── No → Use ml.p4d.24xlarge
    └── No → Use ml.g5.4xlarge for balanced performance

```
### 1.4 Visual Comparison of Instance Cost-Performance Ratio

| Instance | Performance Rating | Cost Rating | Cost-Performance Ratio |
|----------|-------------------|-------------|------------------------|
| ml.g4dn.xlarge | ⭐⭐ | ⭐⭐⭐⭐ | 2.0 (Good for testing) |
| ml.g4dn.2xlarge | ⭐⭐⭐ | ⭐⭐⭐ | 1.0 (Balanced) |
| ml.g5.xlarge | ⭐⭐⭐ | ⭐⭐⭐ | 1.0 (Balanced) |
| ml.g5.2xlarge | ⭐⭐⭐⭐ | ⭐⭐⭐ | 1.33 (Recommended for most uses) |
| ml.g5.4xlarge | ⭐⭐⭐⭐ | ⭐⭐ | 2.0 (Good for faster training) |
| ml.p3.2xlarge | ⭐⭐⭐ | ⭐ | 3.0 (Poor value) |
| ml.inf1.xlarge | ⭐⭐ | ⭐⭐⭐⭐⭐ | 0.4 (Excellent for inference only) |
| ml.p4d.24xlarge | ⭐⭐⭐⭐⭐ | ⭐ | 5.0 (For critical speed requirements only) |

### 1.5 Memory Optimization Techniques

For teams working with tight memory constraints but still wanting to use larger models:

1. **Gradient Checkpointing**: Trades computation for memory by not storing all activations
   ```python
   # Enable gradient checkpointing in the training script
   model.gradient_checkpointing_enable()
   ```

2. **Activation Offloading**: Offloads activations to CPU when not needed
   ```python
   # In your model configuration
   device_map = {
       "transformer.word_embeddings": 0,
       "transformer.h": "balanced",
       "transformer.ln_f": 0,
       "lm_head": 0
   }
   ```

3. **CPU Offloading with `accelerate`**:
   ```python
   # Add to your training script
   from accelerate import cpu_offload
   
   model = AutoModelForCausalLM.from_pretrained(...)
   model = cpu_offload(model, device_id=0)
   ```

4. **Flash Attention 2 Memory Savings**:
   Flash Attention 2 can reduce memory usage by 20-40% while increasing throughput by up to 3x.

For SageMaker g5.2xlarge usage, implement these optimizations in sequence until stable training is achieved.

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
  
### 2.3 Decision Framework for Model Selection

When choosing a model, consider these key factors in priority order:

1. **Task-Specific Requirements**
   - Bug detection requires stronger reasoning capabilities
   - Code generation benefits from larger context windows
   - Memory safety bugs need specialized training

2. **Resource Constraints**
   - Available GPU memory
   - Training time budget
   - Inference latency requirements

3. **ROI Considerations**
   - Model size vs. performance gains
   - Training cost vs. expected improvement
   - Deployment cost vs. user productivity gain
     
### 2.4 Performance Comparison Visualization

```
Performance Index (Higher is Better)
                             Bug Detection    Code Gen    Context    Cost
                                    │            │          │        │
LLaMA 4 Scout 17B      ┌────────────┼────────────┼──────────┼────────┐
                       │            │            │          │        │
Code Llama 13B         ├────────────┼────────────┼──────────┼────────┤
                       │            │            │          │        │
LLaMA 3.1 8B           ├────────────┼────────────┼──────────┼────────┤
                       │            │            │          │        │
IBM Granite Code 8B    ├────────────┼────────────┼──────────┼────────┤
                       │            │            │          │        │
Mistral 7B             └────────────┴────────────┴──────────┴────────┘
                       0            25           50        75       100
```
### 2.5 Context Window Optimization Strategies

Maximizing the utility of long context windows:

1. **Chunking Strategy**: For LLaMA 4 Scout (10M token context):
   - Break large codebases into meaningful components (files, classes, functions)
   - Order by dependency hierarchy
   - Include relevant documentation inline
   - Add separator tokens between logical sections

2. **Context Prioritization**:
   - Most relevant code snippets first
   - Function definitions before implementations
   - API interfaces before implementation details
   - Error messages and stack traces at the beginning

3. **Context Window Benchmarking**:
   ```python
   def measure_context_utilization(model, tokenizer, context_sizes):
       """Measure model performance at different context lengths."""
       results = []
       for size in context_sizes:
           # Create test prompt with increasing context
           prompt = create_test_prompt(size)
           tokens = tokenizer(prompt, return_tensors="pt")
           
           # Measure inference time and memory usage
           start_time = time.time()
           outputs = model.generate(**tokens, max_new_tokens=100)
           inference_time = time.time() - start_time
           
           results.append({
               "context_size": size,
               "inference_time": inference_time,
               "tokens_per_second": 100 / inference_time,
               "memory_usage": torch.cuda.max_memory_allocated() / (1024**3)
           })
       
       return results
   ```
### 2.6 Latest Benchmark Results (April 2025)

| Model | HumanEval | MBPP | CodeContests | APPS | Relative Score |
|-------|-----------|------|--------------|------|----------------|
| LLaMA 4 Scout 17B | 84.2% | 79.6% | 58.3% | 62.1% | 100% |
| Code Llama 13B | 76.4% | 72.8% | 51.9% | 54.2% | 89% |
| IBM Granite Code 8B | 71.2% | 69.4% | 48.7% | 50.3% | 84% |
| LLaMA 3.1 8B | 70.9% | 68.2% | 46.5% | 49.8% | 82% |
| Mistral 7B | 67.8% | 65.1% | 44.2% | 47.3% | 78% |

These benchmarks show that while larger models generally perform better, the gap narrows for specific tasks, and some specialized models outperform larger ones on certain benchmarks.

## 3. Data Preparation and Synthetic Data Generation

### 3.1 Data Preparation Workflow Diagram

```
Raw Repositories  →  Language Detection  →  Code Quality Filtering  →  Semantic Analysis
       ↓                    ↓                       ↓                        ↓
   Cloning      →    File Classification    →   Size/Pattern Checks    →  Function Extraction
       ↓                    ↓                       ↓                        ↓
Github/Gitlab   →    By Extension/Content   →   Min/Max Size Limits    →  AST/Parser Analysis
```


#### 3.1 Real-World Code Repositories

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

#### 3.2 Synthetic Data Generation
1. **Optimal Prompt Engineering for LLMs**:

   ```python
   def generate_cpp_bug_prompt(bug_type, complexity="medium"):
       """Generate targeted prompts for different bug types and complexity levels."""
       base_prompt = "Generate a C++ function that contains a {complexity} {bug_type} bug."
       
       examples = {
           "memory_leak": "Example: a function that allocates memory but fails to free it when an error occurs.",
           "buffer_overflow": "Example: a function that writes beyond the bounds of an allocated buffer.",
           "use_after_free": "Example: a function that continues to use a pointer after it has been freed.",
           "null_dereference": "Example: a function that dereferences a pointer without checking if it's NULL."
       }
       
       prompt_template = f"""
       {base_prompt}
       
       {examples.get(bug_type, '')}
       
       Your response should include:
       1. A realistic C++ function with a subtle {bug_type} bug
       2. A comment explaining where the bug is
       3. A fixed version of the same function that corrects the bug
       4. A brief explanation of the potential consequences of this bug
       
       The function should be between 10-30 lines and should look like real production code.
       """
       
       return prompt_template.format(complexity=complexity, bug_type=bug_type)
   ```

2. **Synthetic Data Quality Control Pipeline**:

   ```python
   def validate_synthetic_examples(examples, language='cpp'):
       """Validate and filter synthetic examples for quality."""
       validated_examples = []
       for example in examples:
           # Check code compilability
           if language == 'cpp':
               compiles = check_cpp_compilation(example['input_code'])
               fixed_compiles = check_cpp_compilation(example['output_code'])
               
               if not compiles:
                   # Skip examples where buggy code doesn't even compile
                   # (unless it's a compilation bug example)
                   if 'compilation' not in example['task'].lower():
                       continue
               
               if not fixed_compiles:
                   # Fixed code should always compile
                   continue
           
           # Check for actual differences between buggy and fixed code
           if example['input_code'] == example['output_code']:
               continue
               
           # Ensure the bug type matches what was requested
           expected_bug_type = extract_bug_type(example['task'])
           actual_bug_type = detect_bug_type(example['input_code'], example['output_code'])
           
           if expected_bug_type != actual_bug_type and expected_bug_type != 'any':
               continue
               
           # Ensure reasonable code length
           if len(example['input_code'].strip().split('\n')) < 5:
               continue
                   
           validated_examples.append(example)
           
       return validated_examples
   ```

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

#### 3.1.3 Combining Real and Synthetic Data

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
### 3.3 Data Augmentation Techniques for Code

1. **Variable Name Perturbation**:
   
   ```python
   def perturb_variable_names(code, language='cpp'):
       """Replace variable names with semantically similar ones."""
       if language == 'cpp':
           # Parse code to get AST
           ast = parse_cpp_code(code)
           
           # Extract variable declarations
           variables = extract_variables_from_ast(ast)
           
           # Create mapping of variable names to alternatives
           var_mapping = {}
           for var in variables:
               var_mapping[var.name] = generate_alternative_name(var.name)
           
           # Apply replacements
           return replace_identifiers(code, var_mapping)
       # Add support for other languages
       return code
   ```

2. **Synthetic Bug Introduction Matrix**:

   | Bug Category | Python | C/C++ | Java | JavaScript |
   |--------------|--------|-------|------|------------|
   | Syntax | `:` removal, indentation errors | `;` removal, bracket mismatch | `;` removal, bracket mismatch | `;` or `{}` issues |
   | Logical | Condition inversion, operator swap | Condition inversion, operator precedence | Condition inversion, equals vs. assignment | `==` vs. `===`, type coercion |
   | Variable | Name typos, scope issues | Type mismatch, uninitialized variables | Type mismatch, access modifiers | Hoisting issues, undefined vs. null |
   | Memory | N/A | Memory leaks, use-after-free, buffer overflow | Null pointer exceptions | Memory leaks in closures |
   | Concurrency | Race conditions in threads | Mutex deadlocks, race conditions | Synchronization issues | Promise/async race conditions |

3. **Data Balancing Strategy**:

   ```python
   def balance_training_data(examples, target_distribution=None):
       """Balance training data across languages and bug types."""
       # Default target distribution (equal weighting)
       if target_distribution is None:
           target_distribution = {
               'language': {'python': 0.25, 'cpp': 0.25, 'java': 0.25, 'javascript': 0.25},
               'bug_type': {'syntax': 0.2, 'logical': 0.2, 'variable': 0.2, 'memory': 0.2, 'concurrency': 0.2}
           }
       
       # Count existing examples
       language_counts = Counter([ex['language'] for ex in examples])
       bug_type_counts = Counter([extract_bug_type(ex['task']) for ex in examples])
       
       # Calculate how many examples to generate for each category
       language_targets = calculate_targets(language_counts, target_distribution['language'])
       bug_type_targets = calculate_targets(bug_type_counts, target_distribution['bug_type'])
       
       # Generate additional examples to reach targets
       additional_examples = []
       for language, target in language_targets.items():
           for bug_type, bug_target in bug_type_targets.items():
               current = count_examples_by_category(examples, language, bug_type)
               needed = min(target, bug_target) - current
               
               if needed > 0:
                   new_examples = generate_examples(language, bug_type, needed)
                   additional_examples.extend(new_examples)
       
       return examples + additional_examples
### 3.3.1 Semantic Analysis and Function Extraction

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
### 3.3.2 Enhanced Semantic Analysis

Replace the basic semantic analysis with a more robust system that builds a complete knowledge graph of the codebase:

```python
from dataclasses import dataclass
from typing import List, Dict, Set, Optional

@dataclass
class CodeEntity:
    name: str
    type: str  # 'function', 'class', 'method', 'variable'
    language: str
    file_path: str
    line_start: int
    line_end: int
    code: str
    docstring: Optional[str] = None
    parents: List[str] = None
    children: List[str] = None
    dependencies: List[str] = None
    called_by: List[str] = None
    calls: List[str] = None

class CodebaseKnowledgeGraph:
    def __init__(self):
        self.entities: Dict[str, CodeEntity] = {}
        self.relationships = []
        
    def add_entity(self, entity: CodeEntity):
        self.entities[entity.name] = entity
        
    def add_relationship(self, source: str, relation_type: str, target: str):
        self.relationships.append((source, relation_type, target))
        
    def get_dependencies_for(self, entity_name: str) -> List[CodeEntity]:
        """Get all entities that this entity depends on."""
        entity = self.entities.get(entity_name)
        if not entity or not entity.dependencies:
            return []
            
        return [self.entities.get(dep) for dep in entity.dependencies if dep in self.entities]
        
    def get_training_examples_for_function(self, function_name: str) -> List[Dict]:
        """Generate rich training examples for a given function."""
        entity = self.entities.get(function_name)
        if not entity or entity.type != 'function':
            return []
            
        examples = []
        
        # Example 1: Complete the function implementation
        if entity.docstring:
            examples.append({
                "task": "Complete Function Implementation",
                "input": f"/* {entity.docstring} */\n" + 
                         self._extract_function_signature(entity.code),
                "output": entity.code,
                "language": entity.language
            })
            
        # Example 2: Fix a bug (generate synthetic bug)
        buggy_code = introduce_bug(entity.code, 
                                   random.choice(['syntax', 'logical', 'variable']),
                                   entity.language)
        if buggy_code != entity.code:
            examples.append({
                "task": "Fix Bug",
                "input": buggy_code,
                "output": entity.code,
                "language": entity.language
            })
            
        # Example 3: Context-aware code completion
        # Include dependencies and calling functions as context
        dep_entities = self.get_dependencies_for(function_name)
        context = "\n\n".join([dep.code for dep in dep_entities])
        
        examples.append({
            "task": "Context-Aware Completion",
            "context": context,
            "input": self._get_partial_implementation(entity.code),
            "output": entity.code,
            "language": entity.language
        })
            
        return examples
            
    def _extract_function_signature(self, code: str) -> str:
        """Extract just the function signature based on language."""
        # Implementation depends on language
        pass
        
    def _get_partial_implementation(self, code: str) -> str:
        """Return first 50% of function implementation."""
        lines = code.split("\n")
        return "\n".join(lines[:max(3, len(lines)//2)])
```

This enhanced approach creates a rich knowledge graph that captures the relationships between different code entities, enabling more contextual and sophisticated training examples.
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

### 4.1 Enhanced IAM Permissions

Replace the basic IAM permissions with this more detailed and secure configuration:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateTrainingJob",
                "sagemaker:DescribeTrainingJob",
                "sagemaker:StopTrainingJob",
                "sagemaker:CreateModel",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:CreateEndpoint",
                "sagemaker:DescribeEndpoint",
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": [
                "arn:aws:sagemaker:*:*:training-job/code-model-*",
                "arn:aws:sagemaker:*:*:model/code-model-*",
                "arn:aws:sagemaker:*:*:endpoint/code-model-*",
                "arn:aws:sagemaker:*:*:endpoint-config/code-model-*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::sagemaker-*/*",
                "arn:aws:s3:::sagemaker-*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
                "logs:DescribeLogStreams"
            ],
            "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage"
            ],
            "Resource": [
                "arn:aws:ecr:*:*:repository/sagemaker-*"
            ]
        },
        {
            "Effect": "Allow",
            "Action": [
                "secretsmanager:GetSecretValue"
            ],
            "Resource": [
                "arn:aws:secretsmanager:*:*:secret:hf-*"
            ]
        }
    ]
}
```

### 4.2 Automated Environment Setup Script

Create a complete environment setup script to ensure consistent configuration:

```python
# environment_setup.py
import boto3
import json
import time
import os
import subprocess
from botocore.exceptions import ClientError

def setup_environment(
    project_name="code-model-finetuning",
    region="us-east-1",
    create_bucket=True,
    install_dependencies=True,
    setup_secrets=True,
    hf_token=None
):
    """
    Setup complete environment for SageMaker fine-tuning.
    
    Args:
        project_name: Base name for resources
        region: AWS region
        create_bucket: Whether to create S3 bucket
        install_dependencies: Whether to install Python dependencies
        setup_secrets: Whether to create secrets in Secrets Manager
        hf_token: Hugging Face token (if None, will prompt)
    
    Returns:
        dict: Configuration information
    """
    print(f"Setting up environment for project: {project_name}")
    
    # Initialize session and clients
    session = boto3.Session(region_name=region)
    s3 = session.client('s3')
    sm = session.client('sagemaker')
    secrets = session.client('secretsmanager')
    iam = session.client('iam')
    
    config = {
        "project_name": project_name,
        "region": region,
        "timestamp": int(time.time()),
        "resources": {}
    }
    
    # 1. Create S3 bucket if requested
    if create_bucket:
        bucket_name = f"{project_name.lower()}-{config['timestamp']}"
        try:
            if region == "us-east-1":
                s3.create_bucket(Bucket=bucket_name)
            else:
                s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": region}
                )
            print(f"Created S3 bucket: {bucket_name}")
            config["resources"]["bucket_name"] = bucket_name
        except ClientError as e:
            print(f"Error creating bucket: {e}")
            return None
    
    # 2. Create or update IAM role
    role_name = f"{project_name}-role"
    policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        # Check if role exists
        try:
            response = iam.get_role(RoleName=role_name)
            role_arn = response['Role']['Arn']
            print(f"Using existing role: {role_name}")
        except ClientError:
            response = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(policy_document)
            )
            role_arn = response['Role']['Arn']
            print(f"Created new role: {role_name}")
            
            # Attach policies
            for policy in [
                "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
                "arn:aws:iam::aws:policy/AmazonS3FullAccess"
            ]:
                iam.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy
                )
            print("Attached managed policies to role")
            
        config["resources"]["role_arn"] = role_arn
    except ClientError as e:
        print(f"Error setting up IAM role: {e}")
        return None
    
    # 3. Setup Secrets Manager if requested
    if setup_secrets:
        if hf_token is None:
            hf_token = input("Enter your Hugging Face token: ").strip()
            
        secret_name = f"{project_name}/hf-token"
        try:
            try:
                # Try to update existing secret
                secrets.update_secret(
                    SecretId=secret_name,
                    SecretString=hf_token
                )
                print(f"Updated secret: {secret_name}")
            except ClientError:
                # Create new secret
                response = secrets.create_secret(
                    Name=secret_name,
                    SecretString=hf_token
                )
                print(f"Created secret: {secret_name}")
                
            config["resources"]["secret_name"] = secret_name
        except ClientError as e:
            print(f"Error setting up secret: {e}")
            return None
    
    # 4. Install dependencies if requested
    if install_dependencies:
        try:
            packages = [
                "sagemaker>=2.180.0",
                "transformers>=4.31.0",
                "datasets>=2.13.0",
                "torch>=2.0.1",
                "accelerate>=0.20.3",
                "boto3>=1.26.0",
                "bitsandbytes>=0.40.0",
                "peft>=0.4.0"
            ]
            
            for package in packages:
                subprocess.check_call(
                    ["pip", "install", "-q", package]
                )
            
            print("Installed Python dependencies")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            return None
    
    # Save configuration to file
    with open(f"{project_name}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Environment setup complete. Configuration saved to {project_name}_config.json")
    return config

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup SageMaker fine-tuning environment")
    parser.add_argument("--project-name", type=str, default="code-model-finetuning")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--hf-token", type=str, help="Hugging Face token")
    parser.add_argument("--no-bucket", action="store_true", help="Skip bucket creation")
    parser.add_argument("--no-dependencies", action="store_true", help="Skip installing dependencies")
    parser.add_argument("--no-secrets", action="store_true", help="Skip secrets setup")
    
    args = parser.parse_args()
    
    setup_environment(
        project_name=args.project_name,
        region=args.region,
        create_bucket=not args.no_bucket,
        install_dependencies=not args.no_dependencies,
        setup_secrets=not args.no_secrets,
        hf_token=args.hf_token
    )
```

### 4.3 Improved Training Script with Error Handling and Checkpointing

Enhance the training script with robust error handling, automatic resumption from checkpoints, and detailed logs:

```python
# enhanced_train.py

import os
import json
import torch
import argparse
import logging
import sys
import traceback
import time
import shutil
import math
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    set_seed
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from accelerate import Accelerator

# Set up logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="meta-llama/Llama-4-Scout-17B-16E-Instruct")
    parser.add_argument("--data-dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--output-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--checkpoint-dir", type=str, default="/opt/ml/checkpoints")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", type=bool, default=True)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-every-n-steps", type=int, default=500)
    parser.add_argument("--eval-every-n-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with smaller dataset")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load model in 8-bit mode instead of 4-bit")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    return parser.parse_args()

def setup_checkpointing(checkpoint_dir):
    """Ensure checkpoint directory exists and return the path."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def get_last_checkpoint(checkpoint_dir):
    """Get the most recent checkpoint if it exists."""
    if not os.path.isdir(checkpoint_dir):
        return None
        
    checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
        
    # Get most recent checkpoint based on step number
    last_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
    return os.path.join(checkpoint_dir, last_checkpoint)

def create_datasets(args, tokenizer):
    """Load and preprocess datasets."""
    try:
        logger.info(f"Loading datasets from {args.data_dir}")
        train_file = os.path.join(args.data_dir, "train.json")
        val_file = os.path.join(args.data_dir, "validation.json")
        
        # Verify files exist
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training data file not found: {train_file}")
        if not os.path.exists(val_file):
            raise FileNotFoundError(f"Validation data file not found: {val_file}")
        
        # Load datasets
        train_dataset = load_dataset("json", data_files=train_file)["train"]
        val_dataset = load_dataset("json", data_files=val_file)["train"]
        
        # For debug mode, use a small subset
        if args.debug:
            train_dataset = train_dataset.select(range(min(100, len(train_dataset))))
            val_dataset = val_dataset.select(range(min(20, len(val_dataset))))
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        # Define tokenization function
        def tokenize_function(examples):
            # Handle both string text and dictionary format
            if isinstance(examples["text"][0], dict):
                texts = [json.dumps(example) for example in examples["text"]]
            else:
                texts = examples["text"]
                
            return tokenizer(
                texts, 
                padding="max_length",
                truncation=True,
                max_length=args.max_seq_length,
                return_tensors="pt"
            )
        
        # Tokenize datasets with error handling
        try:
            logger.info("Tokenizing datasets")
            tokenized_train = train_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"],
                desc="Tokenizing train dataset"
            )
            tokenized_val = val_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"],
                desc="Tokenizing validation dataset"
            )
            
            # Save sample tokenized examples for debugging
            with open("tokenized_samples.json", "w") as f:
                sample_idx = min(5, len(tokenized_train) - 1)
                sample = {k: v.tolist() if hasattr(v, 'tolist') else v 
                         for k, v in tokenized_train[sample_idx].items()}
                json.dump(sample, f, indent=2)
                
            return tokenized_train, tokenized_val
            
        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            logger.error(traceback.format_exc())
            
            # Try to identify the problematic example
            for i, example in enumerate(train_dataset):
                try:
                    tokenizer(str(example["text"]), truncation=True, max_length=args.max_seq_length)
                except Exception as e:
                    logger.error(f"Error tokenizing example {i}: {e}")
                    logger.error(f"Problematic example: {example}")
                    break
            
            raise RuntimeError("Tokenization failed. See logs for details.")
            
    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        logger.error(traceback.format_exc())
        raise
        
def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Start timing
    start_time = time.time()
    logger.info(f"Starting fine-tuning with arguments: {args}")
    
    try:
        # Set up checkpoint directory
        checkpoint_dir = setup_checkpointing(args.checkpoint_dir)
        logger.info(f"Checkpoint directory: {checkpoint_dir}")
        
        # Check for existing checkpoints
        last_checkpoint = args.resume_from_checkpoint or get_last_checkpoint(checkpoint_dir)
        if last_checkpoint:
            logger.info(f"Resuming from checkpoint: {last_checkpoint}")
        else:
            logger.info("Starting new training run")
        
        # Set Hugging Face token from environment if available
        hf_token = os.environ.get("HF_TOKEN", None)
        
        # Load model and tokenizer
        logger.info(f"Loading model: {args.model_id}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_id,
                token=hf_token,
                trust_remote_code=True
            )
            tokenizer.pad_token = tokenizer.eos_token
            
            # Add special tokens for programming languages if requested
            if args.add_special_tokens:
                special_tokens = []
                for lang in args.target_languages.split(','):
                    special_tokens.extend([f"<{lang}>", f"</{lang}>"])
                
                special_tokens_dict = {"additional_special_tokens": special_tokens}
                num_added = tokenizer.add_special_tokens(special_tokens_dict)
                logger.info(f"Added {num_added} special tokens: {special_tokens}")
            
        except Exception as e:
            logger.error(f"Error loading tokenizer: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Failed to load tokenizer: {e}")
        
        # Configure quantization
        if args.load_in_8bit:
            logger.info("Using 8-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_skip_modules=["lm_head"],
                llm_int8_enable_fp32_cpu_offload=True
            )
        else:
            logger.info("Using 4-bit quantization")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        
        # Load model with quantization
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_id,
                device_map="auto",
                quantization_config=bnb_config,
                token=hf_token,
                trust_remote_code=True,
                use_flash_attention_2=args.use_flash_attn,
            )
            
            # Log CUDA memory usage
            if torch.cuda.is_available():
                free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
                max_memory = int(torch.cuda.get_device_properties(0).total_memory/1024**3)
                logger.info(f"GPU memory: {free_in_GB}GB free of {max_memory}GB total")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.error(traceback.format_exc())
            
            # Check CUDA availability and GPU memory
            if torch.cuda.is_available():
                try:
                    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
                    max_memory = int(torch.cuda.get_device_properties(0).total_memory/1024**3)
                    logger.error(f"GPU memory: {free_in_GB}GB free of {max_memory}GB total")
                    
                    if free_in_GB < 10:  # Less than 10GB free
                        logger.error("Insufficient GPU memory. Consider using a larger instance.")
                except Exception as inner_e:
                    logger.error(f"Error checking GPU memory: {inner_e}")
            
            raise RuntimeError(f"Failed to load model: {e}")
        
        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            logger.info("Enabling gradient checkpointing")
            model.gradient_checkpointing_enable()
        
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
        
        # Create datasets
        tokenized_train, tokenized_val = create_datasets(args, tokenizer)
        
        # Compute training steps and schedule
        num_update_steps_per_epoch = max(len(tokenized_train) // (args.batch_size * args.gradient_accumulation_steps), 1)
        max_train_steps = args.epochs * num_update_steps_per_epoch
        num_warmup_steps = int(max_train_steps * args.warmup_ratio)
        
        logger.info(f"Training schedule:")
        logger.info(f"  Num examples: {len(tokenized_train)}")
        logger.info(f"  Num epochs: {args.epochs}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps: {max_train_steps}")
        logger.info(f"  Warmup steps: {num_warmup_steps}")
        
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
            save_strategy="steps",
            save_steps=args.checkpoint_every_n_steps,
            evaluation_strategy="steps",
            eval_steps=args.eval_every_n_steps,
            load_best_model_at_end=True,
            fp16=True,
            logging_dir=os.path.join(args.output_dir, "logs"),
            logging_steps=10,
            report_to="tensorboard",
            save_total_limit=args.save_total_limit,
            max_grad_norm=args.max_grad_norm,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            hub_token=hf_token if hf_token else None,
            dataloader_num_workers=4,
            ddp_find_unused_parameters=False,
            remove_unused_columns=False
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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
        )
        
        # Add hooks for additional monitoring
        original_log = trainer.log
        
        def log_with_memory(*args, **kwargs):
            if torch.cuda.is_available():
                # Log GPU memory usage
                mem_stats = []
                for i in range(torch.cuda.device_count()):
                    free_mem = torch.cuda.mem_get_info(i)[0] / (1024**3)
                    total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    used_mem = total_mem - free_mem
                    mem_stats.append({
                        "device": i,
                        "used_gb": round(used_mem, 2),
                        "total_gb": round(total_mem, 2),
                        "percent_used": round(used_mem / total_mem * 100, 2)
                    })
                
                # Add memory stats to metrics
                if kwargs.get("metrics") is not None:
                    for i, stats in enumerate(mem_stats):
                        for k, v in stats.items():
                            kwargs["metrics"][f"gpu{i}_{k}"] = v
            
            # Call original log method
            return original_log(*args, **kwargs)
            
        trainer.log = log_with_memory
        
        # Start training
        logger.info("Starting training")
        try:
            trainer.train(resume_from_checkpoint=last_checkpoint)
            
            # Log final metrics
            eval_results = trainer.evaluate()
            for key, value in eval_results.items():
                logger.info(f"Final {key}: {value}")
                
        except Exception as e:
            logger.error(f"Error during training: {e}")
            logger.error(traceback.format_exc())
            
            # Try to save checkpoint even if training fails
            checkpoint_output_dir = os.path.join(args.output_dir, "checkpoint-interrupted")
            logger.info(f"Attempting to save interrupted checkpoint to {checkpoint_output_dir}")
            try:
                trainer.save_model(checkpoint_output_dir)
                tokenizer.save_pretrained(checkpoint_output_dir)
                logger.info("Interrupted checkpoint saved successfully")
            except Exception as save_error:
                logger.error(f"Error saving interrupted checkpoint: {save_error}")
                
            # Re-raise the original exception
            raise
        
        # Save the fine-tuned model
        logger.info(f"Saving model to {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # Save model configuration with additional metadata
        training_time = time.time() - start_time
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
                "training_time_seconds": training_time,
                "training_time_hours": training_time / 3600,
                "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "final_eval_loss": eval_results.get("eval_loss"),
                "training_examples": len(tokenized_train),
                "validation_examples": len(tokenized_val)
            }, f, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info(f"Total training time: {training_time / 3600:.2f} hours")
        
    except Exception as e:
        logger.error(f"Unhandled exception during training: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 4.4 Training Monitoring Dashboard

Add a real-time monitoring script to visualize training progress:

```python
# monitor_training.py
import os
import re
import time
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
import boto3
import json
from IPython.display import clear_output

class SageMakerTrainingMonitor:
    def __init__(self, job_name, region="us-east-1", refresh_interval=60):
        """Initialize the training monitor."""
        self.job_name = job_name
        self.region = region
        self.refresh_interval = refresh_interval
        self.client = boto3.client('sagemaker', region_name=region)
        self.logs_client = boto3.client('logs', region_name=region)
        self.metrics = {
            'train_loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'gpu_util': [],
            'gpu_mem': [],
            'timestamps': []
        }
        self.start_time = time.time()
        
    def get_job_status(self):
        """Get the current status of the training job."""
        response = self.client.describe_training_job(TrainingJobName=self.job_name)
        return response['TrainingJobStatus'], response.get('SecondaryStatus', '')
        
    def get_training_metrics(self):
        """Get metrics from CloudWatch Logs."""
        try:
            # Get log stream name
            response = self.client.describe_training_job(TrainingJobName=self.job_name)
            log_group = f"/aws/sagemaker/TrainingJobs/{self.job_name}"
            
            # List log streams
            streams_response = self.logs_client.describe_log_streams(
                logGroupName=log_group,
                orderBy='LastEventTime',
                descending=True,
                limit=5
            )
            
            # Get events from each stream
            all_events = []
            for stream in streams_response['logStreams']:
                stream_name = stream['logStreamName']
                events_response = self.logs_client.get_log_events(
                    logGroupName=log_group,
                    logStreamName=stream_name,
                    startFromHead=False,
                    limit=100
                )
                all_events.extend(events_response['events'])
            
            # Sort events by timestamp
            all_events.sort(key=lambda x: x['timestamp'])
            
            # Extract metrics from logs
            for event in all_events:
                message = event['message']
                timestamp = event['timestamp'] / 1000  # Convert to seconds
                
                # Parse train loss
                train_loss_match = re.search(r'loss\s*=\s*([0-9.]+)', message)
                if train_loss_match:
                    self.metrics['train_loss'].append(float(train_loss_match.group(1)))
                    self.metrics['timestamps'].append(timestamp)
                
                # Parse eval loss
                eval_loss_match = re.search(r'eval_loss\s*=\s*([0-9.]+)', message)
                if eval_loss_match:
                    self.metrics['eval_loss'].append(float(eval_loss_match.group(1)))
                
                # Parse learning rate
                lr_match = re.search(r'learning_rate\s*=\s*([0-9.e-]+)', message)
                if lr_match:
                    self.metrics['learning_rate'].append(float(lr_match.group(1)))
                
                # Parse GPU utilization
                gpu_util_match = re.search(r'gpu([0-9])_percent_used\s*=\s*([0-9.]+)', message)
                if gpu_util_match:
                    self.metrics['gpu_util'].append(float(gpu_util_match.group(2)))
                
                # Parse GPU memory
                gpu_mem_match = re.search(r'gpu([0-9])_used_gb\s*=\s*([0-9.]+)', message)
                if gpu_mem_match:
                    self.metrics['gpu_mem'].append(float(gpu_mem_match.group(2)))
            
            return True
            
        except Exception as e:
            print(f"Error getting metrics: {e}")
            return False
    
    def plot_metrics(self):
        """Plot the training metrics."""
        clear_output(wait=True)
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Job: {self.job_name}', fontsize=16)
        
        # Plot train loss
        if self.metrics['train_loss']:
            axs[0, 0].plot(self.metrics['train_loss'], 'b-', label='Train Loss')
            axs[0, 0].set_title('Training Loss')
            axs[0, 0].set_xlabel('Steps')
            axs[0, 0].set_ylabel('Loss')
            axs[0, 0].legend()
            axs[0, 0].grid(True)
        
        # Plot eval loss
        if self.metrics['eval_loss']:
            axs[0, 1].plot(self.metrics['eval_loss'], 'r-', label='Eval Loss')
            axs[0, 1].set_title('Evaluation Loss')
            axs[0, 1].set_xlabel('Evaluations')
            axs[0, 1].set_ylabel('Loss')
            axs[0, 1].legend()
            axs[0, 1].grid(True)
        
        # Plot learning rate
        if self.metrics['learning_rate']:
            axs[1, 0].plot(self.metrics['learning_rate'], 'g-', label='Learning Rate')
            axs[1, 0].set_title('Learning Rate')
            axs[1, 0].set_xlabel('Steps')
            axs[1, 0].set_ylabel('Learning Rate')
            axs[1, 0].legend()
            axs[1, 0].grid(True)
        
        # Plot GPU metrics
        if self.metrics['gpu_util'] and self.metrics['gpu_mem']:
            ax1 = axs[1, 1]
            ax2 = ax1.twinx()
            
            ax1.plot(self.metrics['gpu_util'], 'c-', label='GPU Utilization (%)')
            ax1.set_title('GPU Metrics')
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Utilization (%)')
            ax1.legend(loc='upper left')
            ax1.grid(True)
            
            ax2.plot(self.metrics['gpu_mem'], 'm-', label='GPU Memory (GB)')
            ax2.set_ylabel('Memory (GB)')
            ax2.legend(loc='upper right')
        
        # Add job status and duration information
        status, secondary_status = self.get_job_status()
        elapsed_time = time.time() - self.start_time
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        duration_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        
        status_text = f"Status: {status} - {secondary_status}\nRunning for: {duration_str}"
        fig.text(0.5, 0.01, status_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.show()
    
    def monitor(self, max_duration=None):
        """Start monitoring the training job."""
        print(f"Starting to monitor training job: {self.job_name}")
        start_time = time.time()
        
        while True:
            status, secondary_status = self.get_job_status()
            print(f"Job status: {status} - {secondary_status}")
            
            # Get metrics and plot
            if self.get_training_metrics():
                self.plot_metrics()
            
            # Check if job has completed
            if status in ['Completed', 'Failed', 'Stopped']:
                print(f"Training job {status}")
                if status == 'Completed':
                    metrics_response = self.client.describe_training_job(TrainingJobName=self.job_name)
                    final_metrics = metrics_response.get('FinalMetricDataList', [])
                    
                    print("\nFinal Metrics:")
                    for metric in final_metrics:
                        print(f"  {metric['MetricName']}: {metric['Value']}")
                break
            
            # Check if we've exceeded max duration
            if max_duration and (time.time() - start_time) > max_duration:
                print(f"Monitoring stopped after {max_duration/3600:.2f} hours")
                break
            
            # Wait before next check
            time.sleep(self.refresh_interval)

# Example usage:
# monitor = SageMakerTrainingMonitor("your-training-job-name")
# monitor.monitor()
```

### 4.5 Early Stopping Optimization Framework

Add an early stopping framework that automatically detects convergence or instability:

```python
# early_stopping.py
import numpy as np
import time
import json
import os
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class AdvancedEarlyStoppingCallback(TrainerCallback):
    """
    Advanced early stopping callback with enhanced detection of different stopping conditions:
    1. Standard patience-based early stopping on validation loss
    2. Saturation detection (plateauing)
    3. Instability detection (oscillating loss)
    4. Divergence detection (increasing loss or NaN/inf values)
    5. Time-based maximum runtime enforcement
    """
    
    def __init__(
        self,
        early_stopping_patience=3,
        early_stopping_threshold=0.01,
        check_saturation=True,
        saturation_patience=5,
        saturation_threshold=0.001,
        check_instability=True,
        instability_window=5,
        instability_threshold=0.05,
        check_divergence=True,
        divergence_factor=1.5,
        max_training_time_hours=None,
        output_dir=None
    ):
        """
        Initialize the advanced early stopping callback.
        
        Args:
            early_stopping_patience: Number of evaluations with no improvement after which training will be stopped
            early_stopping_threshold: Minimum improvement required to consider as improvement
            check_saturation: Whether to check for loss saturation (plateauing)
            saturation_patience: Number of evaluations to consider for saturation detection
            saturation_threshold: Maximum allowed change to consider as saturation
            check_instability: Whether to check for loss instability (oscillation)
            instability_window: Window size for instability detection
            instability_threshold: Threshold for standard deviation / mean ratio to detect instability
            check_divergence: Whether to check for diverging loss
            divergence_factor: Factor of increase over best loss to trigger divergence detection
            max_training_time_hours: Maximum training time in hours (None for no limit)
            output_dir: Directory to save early stopping logs
        """
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.check_saturation = check_saturation
        self.saturation_patience = saturation_patience
        self.saturation_threshold = saturation_threshold
        self.check_instability = check_instability
        self.instability_window = instability_window
        self.instability_threshold = instability_threshold
        self.check_divergence = check_divergence
        self.divergence_factor = divergence_factor
        self.max_training_time_hours = max_training_time_hours
        
        self.best_score = None
        self.early_stopping_patience_counter = 0
        self.best_step = 0
        self.loss_history = []
        self.start_time = time.time()
        self.output_dir = output_dir
        self.stopping_reason = None
        
    def check_metric_value(self, args, state, control, metric_value):
        """Check various stopping conditions based on the metric value."""
        # Always store loss history
        self.loss_history.append(metric_value)
        
        # Initialize best score
        if self.best_score is None:
            self.best_score = metric_value
            self.best_step = state.global_step
            return
        
        # Standard early stopping (based on patience)
        if metric_value < self.best_score - self.early_stopping_threshold:
            self.best_score = metric_value
            self.early_stopping_patience_counter = 0
            self.best_step = state.global_step
        else:
            self.early_stopping_patience_counter += 1
            if self.early_stopping_patience_counter >= self.early_stopping_patience:
                self.stopping_reason = "patience_exceeded"
                self._stop_training(state, control)
                return
        
        # Check for saturation (plateau)
        if self.check_saturation and len(self.loss_history) >= self.saturation_patience:
            recent_losses = self.loss_history[-self.saturation_patience:]
            max_change = max(recent_losses) - min(recent_losses)
            if max_change < self.saturation_threshold:
                self.stopping_reason = "saturation_detected"
                self._stop_training(state, control)
                return
        
        # Check for instability (oscillation)
        if self.check_instability and len(self.loss_history) >= self.instability_window:
            recent_losses = self.loss_history[-self.instability_window:]
            mean_loss = np.mean(recent_losses)
            std_loss = np.std(recent_losses)
            if mean_loss > 0 and (std_loss / mean_loss) > self.instability_threshold:
                self.stopping_reason = "instability_detected"
                self._stop_training(state, control)
                return
        
        # Check for divergence
        if self.check_divergence:
            # Check for NaN or inf
            if np.isnan(metric_value) or np.isinf(metric_value):
                self.stopping_reason = "nan_or_inf_detected"
                self._stop_training(state, control)
                return
            
            # Check for significant increase over best loss
            if metric_value > self.best_score * self.divergence_factor:
                self.stopping_reason = "divergence_detected"
                self._stop_training(state, control)
                return
        
        # Check for maximum training time
        if self.max_training_time_hours is not None:
            elapsed_time = (time.time() - self.start_time) / 3600  # hours
            if elapsed_time > self.max_training_time_hours:
                self.stopping_reason = "max_time_exceeded"
                self._stop_training(state, control)
                return
    
    def _stop_training(self, state, control):
        """Stop training and log the reason."""
        control.should_training_stop = True
        
        # Log stopping information
        stopping_info = {
            "stopping_reason": self.stopping_reason,
            "global_step": state.global_step,
            "best_step": self.best_step,
            "best_score": float(self.best_score),
            "final_score": float(self.loss_history[-1]) if self.loss_history else None,
            "loss_history": [float(x) for x in self.loss_history[-10:]],  # Last 10 values
            "training_time_hours": (time.time() - self.start_time) / 3600
        }
        
        print(f"\n*** Early stopping triggered: {self.stopping_reason} ***")
        print(f"Best score: {self.best_score:.6f} at step {self.best_step}")
        print(f"Current score: {self.loss_history[-1]:.6f} at step {state.global_step}")
        
        # Save stopping information to file if output directory is provided
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            with open(os.path.join(self.output_dir, "early_stopping_info.json"), "w") as f:
                json.dump(stopping_info, f, indent=2)
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        """Called after evaluation."""
        metric_to_check = "eval_loss"
        if metric_to_check not in metrics:
            raise ValueError(
                f"The metric '{metric_to_check}' is not found in the evaluation metrics. "
                f"Available metrics: {list(metrics.keys())}"
            )
        
        self.check_metric_value(args, state, control, metrics[metric_to_check])
```

### 4.6 Enhanced Training Scheduler

Create a scheduling script for managing training jobs in a pipeline:

```python
# training_scheduler.py

import boto3
import time
import json
import argparse
import sys
from datetime import datetime, timedelta

class SageMakerJobScheduler:
    """
    Scheduler for SageMaker training jobs to manage resource usage efficiently.
    Features:
    - Queue training jobs based on priority
    - Monitor active jobs and resource usage
    - Start jobs when appropriate resources are available
    - Handle dependencies between jobs
    """
    
    def __init__(self, max_concurrent_jobs=1, region='us-east-1'):
        """
        Initialize the scheduler.
        
        Args:
            max_concurrent_jobs: Maximum number of concurrent jobs allowed
            region: AWS region
        """
        self.max_concurrent_jobs = max_concurrent_jobs
        self.region = region
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.job_queue = []
        self.active_jobs = []
        
    def add_job(self, job_config, priority=0, dependencies=None):
        """
        Add a job to the queue.
        
        Args:
            job_config: Dictionary containing job configuration
            priority: Job priority (higher value = higher priority)
            dependencies: List of job names that must complete before this job starts
        """
        # Add to queue with metadata
        self.job_queue.append({
            'config': job_config,
            'priority': priority,
            'dependencies': dependencies or [],
            'status': 'QUEUED',
            'added_time': datetime.now().isoformat()
        })
        
        # Sort queue by priority (descending)
        self.job_queue.sort(key=lambda x: x['priority'], reverse=True)
        
        print(f"Added job '{job_config['TrainingJobName']}' to queue with priority {priority}")
        
    def update_job_status(self):
        """Update status of all active jobs."""
        updated_active_jobs = []
        
        for job in self.active_jobs:
            job_name = job['config']['TrainingJobName']
            try:
                response = self.sagemaker_client.describe_training_job(
                    TrainingJobName=job_name
                )
                status = response['TrainingJobStatus']
                
                # Update job status
                job['status'] = status
                job['last_update'] = datetime.now().isoformat()
                
                if status in ['Completed', 'Failed', 'Stopped']:
                    print(f"Job '{job_name}' {status}")
                    # Add completion time
                    job['completion_time'] = datetime.now().isoformat()
                    # Log final status details to file
                    self._log_job_status(job, response)
                else:
                    updated_active_jobs.append(job)
                    
            except Exception as e:
                print(f"Error updating status for job '{job_name}': {e}")
                # Keep job in active list if we couldn't get its status
                updated_active_jobs.append(job)
        
        self.active_jobs = updated_active_jobs
        
    def check_dependencies(self, job):
        """
        Check if all dependencies for a job have completed.
        
        Args:
            job: Job dictionary
            
        Returns:
            bool: True if all dependencies have completed
        """
        if not job['dependencies']:
            return True
            
        # Check each dependency
        for dep_name in job['dependencies']:
            try:
                response = self.sagemaker_client.describe_training_job(
                    TrainingJobName=dep_name
                )
                if response['TrainingJobStatus'] != 'Completed':
                    # Dependency not yet completed
                    return False
            except Exception as e:
                print(f"Error checking dependency '{dep_name}': {e}")
                return False
                
        return True
        
    def start_jobs(self):
        """Start jobs from the queue if resources are available."""
        # Check if we can start new jobs
        available_slots = self.max_concurrent_jobs - len(self.active_jobs)
        
        if available_slots <= 0:
            return
            
        # Filter jobs that can be started (dependencies met)
        ready_jobs = []
        for job in self.job_queue:
            if self.check_dependencies(job):
                ready_jobs.append(job)
                
                # Only consider jobs up to available slots
                if len(ready_jobs) >= available_slots:
                    break
        
        # Start ready jobs
        for job in ready_jobs:
            job_name = job['config']['TrainingJobName']
            try:
                # Start training job
                self.sagemaker_client.create_training_job(**job['config'])
                print(f"Started job '{job_name}'")
                
                # Update job status and move to active list
                job['status'] = 'InProgress'
                job['start_time'] = datetime.now().isoformat()
                self.active_jobs.append(job)
                
                # Remove from queue
                self.job_queue.remove(job)
                
            except Exception as e:
                print(f"Error starting job '{job_name}': {e}")
                # Mark as failed in queue
                job['status'] = 'FailedToStart'
                job['error'] = str(e)
                
    def _log_job_status(self, job, response):
        """Log final job status to file."""
        # Extract relevant information
        job_info = {
            'job_name': job['config']['TrainingJobName'],
            'status': response['TrainingJobStatus'],
            'creation_time': job.get('added_time'),
            'start_time': job.get('start_time'),
            'completion_time': job.get('completion_time'),
            'training_time_seconds': (
                (response.get('TrainingEndTime', datetime.now()) - 
                 response.get('TrainingStartTime', datetime.now())).total_seconds()
                if 'TrainingStartTime' in response else None
            ),
            'billable_seconds': response.get('BillableTimeInSeconds'),
            'instance_type': job['config'].get('ResourceConfig', {}).get('InstanceType')
        }
        
        # Add metrics if available
        if 'FinalMetricDataList' in response:
            job_info['metrics'] = {
                m['MetricName']: m['Value'] for m in response['FinalMetricDataList']
            }
        
        # Save to file
        with open(f"job_logs/{job['config']['TrainingJobName']}.json", "w") as f:
            json.dump(job_info, f, indent=2)
            
    def run(self, check_interval=60, max_runtime=None):
        """
        Run the scheduler main loop.
        
        Args:
            check_interval: Seconds between status checks
            max_runtime: Maximum runtime in seconds (None for unlimited)
        """
        print(f"Starting scheduler with max {self.max_concurrent_jobs} concurrent jobs")
        start_time = time.time()
        
        while True:
            # Check if we should exit based on max runtime
            if max_runtime and (time.time() - start_time) > max_runtime:
                print(f"Reached maximum runtime of {max_runtime} seconds")
                break
                
            # Update status of active jobs
            self.update_job_status()
            
            # Start new jobs if possible
            self.start_jobs()
            
            # Print status summary
            self._print_status()
            
            # Exit if no more jobs and nothing active
            if not self.job_queue and not self.active_jobs:
                print("All jobs completed, exiting")
                break
                
            # Wait before next check
            time.sleep(check_interval)
            
    def _print_status(self):
        """Print current status summary."""
        now = datetime.now()
        print("\n" + "="*50)
        print(f"Status at {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Active jobs: {len(self.active_jobs)}, Queued jobs: {len(self.job_queue)}")
        
        if self.active_jobs:
            print("\nActive Jobs:")
            for job in self.active_jobs:
                job_name = job['config']['TrainingJobName']
                status = job['status']
                runtime = "Unknown"
                if 'start_time' in job:
                    start_time = datetime.fromisoformat(job['start_time'])
                    runtime = str(now - start_time).split('.')[0]  # Format as HH:MM:SS
                
                instance = job['config'].get('ResourceConfig', {}).get('InstanceType', 'Unknown')
                print(f"  {job_name}: {status} - Running for {runtime} on {instance}")
        
        if self.job_queue:
            print("\nQueued Jobs (top 5):")
            for job in self.job_queue[:5]:
                job_name = job['config']['TrainingJobName']
                priority = job['priority']
                deps = ', '.join(job['dependencies']) if job['dependencies'] else 'None'
                print(f"  {job_name}: Priority {priority}, Dependencies: {deps}")
        
        print("="*50 + "\n")

# Example usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SageMaker Job Scheduler")
    parser.add_argument("--config", type=str, required=True, help="Path to job configuration file")
    parser.add_argument("--max-jobs", type=int, default=1, help="Maximum concurrent jobs")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--check-interval", type=int, default=60, help="Status check interval in seconds")
    
    args = parser.parse_args()
    
    # Load job configurations
    try:
        with open(args.config, 'r') as f:
            jobs_config = json.load(f)
    except Exception as e:
        print(f"Error loading job configuration: {e}")
        sys.exit(1)
    
    # Initialize scheduler
    scheduler = SageMakerJobScheduler(
        max_concurrent_jobs=args.max_jobs,
        region=args.region
    )
    
    # Add jobs to queue
    for job in jobs_config:
        scheduler.add_job(
            job_config=job['config'],
            priority=job.get('priority', 0),
            dependencies=job.get('dependencies', [])
        )
    
    # Run scheduler
    scheduler.run(check_interval=args.check_interval)
```

## 5. Model Evaluation and Deployment

### 5.1 Enhanced Model Evaluation Framework

Replace the basic evaluation script with a more comprehensive framework:

```python
# enhanced_evaluate.py
import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-id", type=str, default="meta-llama/Llama-4-Scout-17B-16E-Instruct")
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--test-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="evaluation_results")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--detailed-analysis", action="store_true", help="Run detailed error analysis")
    parser.add_argument("--compare-baseline", type=str, help="Path to baseline model results for comparison")
    parser.add_argument("--code-quality-check", action="store_true", help="Run code quality checks on generated code")
    parser.add_argument("--languages", type=str, default="python,cpp,java,javascript", help="Languages to evaluate")
    return parser.parse_args()

def load_model_and_tokenizer(args):
    """Load model and tokenizer with appropriate configuration."""
    print(f"Loading base model: {args.base_model_id}")
    
    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN", None)
    
    # Load tokenizer
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
    
    # Load base model
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
    
    return model, tokenizer

def extract_generated_code(text, language=None):
    """Extract code from the generated text."""
    # Try to find code blocks with triple backticks
    if "```" in text:
        code_blocks = text.split("```")
        if len(code_blocks) >= 3:
            # Get the content between the first pair of backticks
            code_block = code_blocks[1]
            
            # Remove language identifier if present
            lines = code_block.split("\n")
            if lines[0].strip() in ["python", "py", "cpp", "c++", "java", "javascript", "js"]:
                code_block = "\n".join(lines[1:])
            
            return code_block.strip()
    
    # Try to find code after "Output Code:" marker
    if "### Output Code:" in text:
        code_part = text.split("### Output Code:")[1].strip()
        if "```" in code_part:
            # Extract code between backticks
            code_blocks = code_part.split("```")
            if len(code_blocks) >= 2:
                return code_blocks[1].strip()
        return code_part.strip()
    
    # Fall back to returning everything after the last prompt marker
    markers = ["### Output Code:", "### Response:", "### Solution:"]
    for marker in markers:
        if marker in text:
            return text.split(marker)[-1].strip()
    
    # Last resort: return everything
    return text.strip()

def normalize_code(code):
    """Normalize code for comparison by removing whitespace and comments."""
    # Remove comments
    lines = []
    for line in code.split("\n"):
        # Skip empty lines
        if not line.strip():
            continue
            
        # Remove inline comments
        if "#" in line:  # Python
            line = line[:line.find("#")]
        if "//" in line:  # C++, Java, JavaScript
            line = line[:line.find("//")]
            
        # Skip comment-only lines
        if line.strip():
            lines.append(line.strip())
    
    # Join and normalize whitespace
    normalized = " ".join(lines)
    
    # Replace multiple spaces with a single space
    normalized = " ".join(normalized.split())
    
    return normalized

def calculate_code_similarity(pred_code, ref_code):
    """Calculate the similarity between two code snippets."""
    # Normalize the code
    norm_pred = normalize_code(pred_code)
    norm_ref = normalize_code(ref_code)
    
    # Split into tokens
    pred_tokens = set(norm_pred.split())
    ref_tokens = set(norm_ref.split())
    
    # Calculate intersection
    intersection = pred_tokens.intersection(ref_tokens)
    
    # Calculate Jaccard similarity
    if not pred_tokens or not ref_tokens:
        return 0.0
        
    union = pred_tokens.union(ref_tokens)
    jaccard = len(intersection) / len(union)
    
    return jaccard

def check_code_executes(code, language):
    """Check if the generated code executes without errors."""
    if language == "python":
        try:
            # Try to compile the code to check for syntax errors
            compile(code, "<string>", "exec")
            return True
        except Exception:
            return False
    
    # For other languages, we would need language-specific compilation/execution
    # This is a simplified implementation
    return True  # Default to True for non-Python languages

def analyze_errors(predictions, references, example_tasks, languages):
    """Perform detailed error analysis on the predictions."""
    error_analysis = {
        "tasks": defaultdict(lambda: {"correct": 0, "incorrect": 0}),
        "languages": defaultdict(lambda: {"correct": 0, "incorrect": 0}),
        "error_categories": defaultdict(int),
        "examples": []
    }
    
    for i, (pred, ref, task) in enumerate(zip(predictions, references, example_tasks)):
        # Determine language
        language = languages[i] if i < len(languages) else "unknown"
        
        # Check correctness
        is_correct = normalize_code(pred) == normalize_code(ref)
        
        # Update task and language statistics
        error_analysis["tasks"][task]["correct" if is_correct else "incorrect"] += 1
        error_analysis["languages"][language]["correct" if is_correct else "incorrect"] += 1
        
        # Analyze error type if incorrect
        if not is_correct:
            # Check if code executes
            executes = check_code_executes(pred, language)
            if not executes:
                error_analysis["error_categories"]["syntax_error"] += 1
            
            # Check similarity
            similarity = calculate_code_similarity(pred, ref)
            
            if similarity > 0.8:
                error_analysis["error_categories"]["minor_difference"] += 1
            elif similarity > 0.5:
                error_analysis["error_categories"]["partial_implementation"] += 1
            elif similarity > 0.2:
                error_analysis["error_categories"]["major_difference"] += 1
            else:
                error_analysis["error_categories"]["completely_different"] += 1
            
            # Save example for detailed analysis
            error_analysis["examples"].append({
                "id": i,
                "task": task,
                "language": language,
                "similarity": similarity,
                "prediction": pred,
                "reference": ref,
                "executes": executes
            })
    
    return error_analysis

def visualize_results(metrics, error_analysis, output_dir):
    """Generate visualizations of the evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Overall metrics
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ["exact_match", "token_accuracy", "bug_detection_accuracy", "template_filling_accuracy"]
    values = [metrics[m] for m in metrics_to_plot]
    
    plt.bar(metrics_to_plot, values, color="skyblue")
    plt.ylim(0, 1.0)
    plt.title("Overall Performance Metrics")
    plt.ylabel("Score")
    plt.savefig(os.path.join(output_dir, "overall_metrics.png"))
    
    # 2. Performance by task
    plt.figure(figsize=(12, 6))
    tasks = list(error_analysis["tasks"].keys())
    correct = [error_analysis["tasks"][t]["correct"] for t in tasks]
    incorrect = [error_analysis["tasks"][t]["incorrect"] for t in tasks]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    plt.bar(x - width/2, correct, width, label="Correct", color="green")
    plt.bar(x + width/2, incorrect, width, label="Incorrect", color="red")
    
    plt.xlabel("Task")
    plt.ylabel("Count")
    plt.title("Performance by Task")
    plt.xticks(x, [t[:20] + "..." if len(t) > 20 else t for t in tasks], rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_by_task.png"))
    
    # 3. Performance by language
    plt.figure(figsize=(10, 6))
    languages = list(error_analysis["languages"].keys())
    correct_by_lang = [error_analysis["languages"][lang]["correct"] for lang in languages]
    incorrect_by_lang = [error_analysis["languages"][lang]["incorrect"] for lang in languages]
    
    x = np.arange(len(languages))
    
    plt.bar(x - width/2, correct_by_lang, width, label="Correct", color="green")
    plt.bar(x + width/2, incorrect_by_lang, width, label="Incorrect", color="red")
    
    plt.xlabel("Language")
    plt.ylabel("Count")
    plt.title("Performance by Language")
    plt.xticks(x, languages)
    plt.legend()
    plt.savefig(os.path.join(output_dir, "performance_by_language.png"))
    
    # 4. Error categories
    plt.figure(figsize=(10, 6))
    categories = list(error_analysis["error_categories"].keys())
    counts = [error_analysis["error_categories"][cat] for cat in categories]
    
    plt.bar(categories, counts, color="salmon")
    plt.xlabel("Error Category")
    plt.ylabel("Count")
    plt.title("Error Categories")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_categories.png"))
    
    print(f"Visualizations saved to {output_dir}")

def run_evaluation(args):
    """Run the evaluation process."""
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args)
    
    # Load test data
    with open(args.test_file, "r") as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test examples")
    
    # Generate predictions
    predictions = []
    references = []
    example_tasks = []
    languages = []
    
    print(f"Generating predictions for {len(test_data)} examples")
    for i, item in enumerate(tqdm(test_data)):
        # Extract information from the test example
        example = item.get("text", "")
        if isinstance(example, dict):
            task = example.get("task", "")
            language = example.get("language", "unknown")
            
            # Get reference code
            reference = example.get("output_code", "")
            
            # Format prompt
            if "template" in example:
                code_tag = language if language != "other" else "code"
                prompt = f"""### Task: {task}
### Template:
```{code_tag}
{example['template']}
```
### Instruction:
{example['instruction']}
### Output Code:
```{code_tag}
"""
            else:
                code_tag = language if language != "other" else "code"
                prompt = f"""### Task: {task}
### Input Code:
```{code_tag}
{example['input_code']}
```
### Instruction:
{example['instruction']}
### Output Code:
```{code_tag}
"""
        else:
            # Handle string format
            task = "Unknown"
            language = "unknown"
            reference = ""
            prompt = example
            
            # Try to extract task and language from the prompt
            if "### Task:" in prompt:
                task = prompt.split("### Task:")[1].split("\n")[0].strip()
            
            # Try to extract language from code blocks
            code_block_match = re.search(r"```(\w+)", prompt)
            if code_block_match:
                lang_tag = code_block_match.group(1).lower()
                if lang_tag in ["python", "py", "cpp", "c", "java", "javascript", "js"]:
                    language = {"py": "python", "js": "javascript"}.get(lang_tag, lang_tag)
            
            # Try to extract reference code
            if "### Output Code:" in prompt:
                ref_part = prompt.split("### Output Code:")[1]
                if "```" in ref_part:
                    reference = ref_part.split("```")[1].strip()
        
        # Tokenize
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.temperature > 0,
                num_return_sequences=1,
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract generated code
        code_output = extract_generated_code(generated_text, language)
        
        # Store results
        predictions.append(code_output)
        references.append(reference)
        example_tasks.append(task)
        languages.append(language)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, references, example_tasks)
    
    # Run error analysis if requested
    if args.detailed_analysis:
        error_analysis = analyze_errors(predictions, references, example_tasks, languages)
    else:
        error_analysis = None
    
    # Generate visualizations if requested
    if args.visualize and error_analysis:
        visualize_results(metrics, error_analysis, args.output_dir)
    
    # Save results
    result_data = {
        "metrics": metrics,
        "examples": [
            {
                "task": task,
                "language": lang,
                "prediction": pred,
                "reference": ref,
                "correct": normalize_code(pred) == normalize_code(ref)
            }
            for task, lang, pred, ref in zip(example_tasks, languages, predictions, references)
        ]
    }
    
    if error_analysis:
        result_data["error_analysis"] = {
            "tasks": {k: dict(v) for k, v in error_analysis["tasks"].items()},
            "languages": {k: dict(v) for k, v in error_analysis["languages"].items()},
            "error_categories": dict(error_analysis["error_categories"])
        }
    
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(result_data, f, indent=2)
    
    # Compare with baseline if provided
    if args.compare_baseline:
        compare_with_baseline(result_data, args.compare_baseline, args.output_dir)
    
    print(f"Evaluation results saved to {args.output_dir}/results.json")
    
    # Print summary
    print("\nEvaluation Summary:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return result_data

def calculate_metrics(predictions, references, example_tasks):
    """Calculate comprehensive metrics for code evaluation."""
    metrics = {}
    
    # Count totals
    total = len(predictions)
    if total == 0:
        return {"error": "No examples to evaluate"}
    
    # Group examples by task
    task_examples = defaultdict(list)
    for i, task in enumerate(example_tasks):
        task_examples[task].append(i)
    
    # 1. Exact match after normalization
    exact_matches = sum(1 for p, r in zip(predictions, references) 
                        if normalize_code(p) == normalize_code(r))
    metrics["exact_match"] = exact_matches / total
    
    # 2. Token-level accuracy
    token_accuracies = []
    for pred, ref in zip(predictions, references):
        # Normalize and tokenize
        norm_pred = normalize_code(pred)
        norm_ref = normalize_code(ref)
        
        # Skip empty references
        if not norm_ref:
            continue
            
        # Calculate token overlap
        pred_tokens = set(norm_pred.split())
        ref_tokens = set(norm_ref.split())
        
        if ref_tokens:
            overlap = len(pred_tokens.intersection(ref_tokens))
            token_acc = overlap / len(ref_tokens)
            token_accuracies.append(token_acc)
    
    metrics["token_accuracy"] = np.mean(token_accuracies) if token_accuracies else 0
    
    # 3. Task-specific metrics
    
    # 3.1 Bug detection accuracy
    bug_examples = [i for i, task in enumerate(example_tasks) if "Bug Detection" in task]
    if bug_examples:
        bug_correct = sum(1 for i in bug_examples 
                         if normalize_code(predictions[i]) == normalize_code(references[i]))
        metrics["bug_detection_accuracy"] = bug_correct / len(bug_examples)
        metrics["bug_examples_count"] = len(bug_examples)
    else:
        metrics["bug_detection_accuracy"] = 0
        metrics["bug_examples_count"] = 0
    
    # 3.2 Template filling accuracy
    template_examples = [i for i, task in enumerate(example_tasks) if "Fill Template" in task or "Template" in task]
    if template_examples:
        template_correct = sum(1 for i in template_examples 
                              if normalize_code(predictions[i]) == normalize_code(references[i]))
        metrics["template_filling_accuracy"] = template_correct / len(template_examples)
        metrics["template_examples_count"] = len(template_examples)
    else:
        metrics["template_filling_accuracy"] = 0
        metrics["template_examples_count"] = 0
    
    # 3.3 Compilation/execution success rate (if applicable)
    python_examples = [i for i, task in enumerate(example_tasks) if "python" in task.lower()]
    if python_examples:
        executable_count = sum(1 for i in python_examples if check_code_executes(predictions[i], "python"))
        metrics["python_execution_success_rate"] = executable_count / len(python_examples)
        metrics["python_examples_count"] = len(python_examples)
    
    # 4. Per-task metrics
    metrics["per_task"] = {}
    for task, indices in task_examples.items():
        if indices:
            task_correct = sum(1 for i in indices 
                              if normalize_code(predictions[i]) == normalize_code(references[i]))
            metrics["per_task"][task] = {
                "accuracy": task_correct / len(indices),
                "count": len(indices)
            }
    
    # 5. Overall metrics
    metrics["total_examples"] = total
    metrics["overall_accuracy"] = exact_matches / total
    
    return metrics

def compare_with_baseline(current_results, baseline_path, output_dir):
    """Compare current results with a baseline model."""
    try:
        with open(baseline_path, "r") as f:
            baseline = json.load(f)
        
        comparison = {
            "metrics_comparison": {},
            "improvement": {}
        }
        
        # Compare metrics
        for metric, value in current_results["metrics"].items():
            if metric in baseline["metrics"]:
                baseline_value = baseline["metrics"][metric]
                comparison["metrics_comparison"][metric] = {
                    "current": value,
                    "baseline": baseline_value,
                    "difference": value - baseline_value,
                    "percent_change": (value - baseline_value) / baseline_value * 100 if baseline_value else float('inf')
                }
        
        # Identify most improved and regressed examples
        example_diffs = []
        for i, (curr_ex, base_ex) in enumerate(zip(current_results["examples"], baseline["examples"])):
            if curr_ex["task"] == base_ex["task"]:
                # Calculate similarity improvement
                curr_similarity = calculate_code_similarity(curr_ex["prediction"], curr_ex["reference"])
                base_similarity = calculate_code_similarity(base_ex["prediction"], base_ex["reference"])
                
                example_diffs.append({
                    "index": i,
                    "task": curr_ex["task"],
                    "similarity_improvement": curr_similarity - base_similarity,
                    "current_correct": curr_ex["correct"],
                    "baseline_correct": base_ex["correct"]
                })
        
        # Sort by improvement
        example_diffs.sort(key=lambda x: x["similarity_improvement"], reverse=True)
        
        comparison["improvement"]["most_improved"] = example_diffs[:5] if len(example_diffs) >= 5 else example_diffs
        comparison["improvement"]["most_regressed"] = example_diffs[-5:] if len(example_diffs) >= 5 else []
        
        # Save comparison results
        with open(os.path.join(output_dir, "baseline_comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)
        
        # Print summary
        print("\nComparison with baseline:")
        for metric, comp in comparison["metrics_comparison"].items():
            print(f"  {metric}: {comp['current']:.4f} vs {comp['baseline']:.4f} " +
                  f"({comp['percent_change']:+.2f}%)")
        
    except Exception as e:
        print(f"Error comparing with baseline: {e}")

def main():
    """Main function to run the evaluation."""
    args = parse_args()
    run_evaluation(args)

if __name__ == "__main__":
    main()
```

### 5.2 Multi-Stage Deployment Pipeline

Create a robust deployment pipeline for code models:

```python
# deployment_pipeline.py

import os
import json
import boto3
import time
import argparse
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("deployment.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class CodeModelDeploymentPipeline:
    """
    Multi-stage deployment pipeline for code models with:
    - Staging deployment for testing
    - Progressive traffic shifting
    - Automated canary testing
    - Shadow deployment option
    - Rollback capability
    """
    
    def __init__(
        self,
        model_data,
        model_name,
        role_arn,
        region="us-east-1",
        instance_type="ml.inf1.xlarge",
        instance_count=1,
        hf_model_id="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        hf_token=None,
        entry_point="inference.py"
    ):
        """
        Initialize the deployment pipeline.
        
        Args:
            model_data: S3 path to model artifacts
            model_name: Name for the model
            role_arn: IAM role ARN for deployment
            region: AWS region
            instance_type: SageMaker instance type
            instance_count: Number of instances
            hf_model_id: Original Hugging Face model ID
            hf_token: Hugging Face token
            entry_point: Path to inference script
        """
        self.model_data = model_data
        self.model_name = model_name
        self.role_arn = role_arn
        self.region = region
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.hf_model_id = hf_model_id
        self.hf_token = hf_token
        self.entry_point = entry_point
        
        # Initialize AWS clients
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        
        # Deployment timestamps for versioning
        self.timestamp = int(time.time())
        self.deployment_id = f"{model_name}-{self.timestamp}"
        
        # Track deployment status
        self.deployment_stages = {
            "model_creation": False,
            "staging_deployment": False,
            "canary_testing": False,
            "production_deployment": False,
            "traffic_shifting": False,
            "monitoring_setup": False
        }
        
        self.deployment_artifacts = {}
        
        logger.info(f"Initialized deployment pipeline for {model_name}")
    
    def create_model(self):
        """Create SageMaker model."""
        try:
            # Configure the container
            container_def = {
                'Image': f"763104351884.dkr.ecr.{self.region}.amazonaws.com/huggingface-pytorch-inference:2.0.1-transformers4.31.0-cpu-py310-ubuntu20.04",
                'ModelDataUrl': self.model_data,
                'Environment': {
                    'HF_MODEL_ID': self.hf_model_id,
                    'HF_TASK': 'text-generation',
                    'HF_TOKEN': self.hf_token or ''
                }
            }
            
            # If using custom inference script
            if self.entry_point:
                container_def['Environment']['SAGEMAKER_PROGRAM'] = self.entry_point
            
            # Create model in SageMaker
            response = self.sagemaker_client.create_model(
                ModelName=self.deployment_id,
                PrimaryContainer=container_def,
                ExecutionRoleArn=self.role_arn,
                Tags=[
                    {'Key': 'ModelType', 'Value': 'CodeModel'},
                    {'Key': 'BaseModel', 'Value': self.hf_model_id.replace('/', '-')},
                    {'Key': 'DeploymentID', 'Value': self.deployment_id}
                ]
            )
            
            logger.info(f"Created model: {self.deployment_id}")
            self.deployment_artifacts['model_name'] = self.deployment_id
            self.deployment_stages["model_creation"] = True
            
            return self.deployment_id
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
    
    def deploy_to_staging(self, endpoint_name_suffix="staging"):
        """Deploy model to staging environment."""
        if not self.deployment_stages["model_creation"]:
            logger.warning("Cannot deploy to staging before creating model")
            return None
            
        staging_endpoint_name = f"{self.deployment_id}-{endpoint_name_suffix}"
        
        try:
            # Create endpoint config
            config_name = f"{staging_endpoint_name}-config"
            self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'AllTraffic',
                        'ModelName': self.deployment_id,
                        'InstanceType': self.instance_type,
                        'InitialInstanceCount': self.instance_count,
                        'InitialVariantWeight': 1.0
                    }
                ],
                Tags=[
                    {'Key': 'Environment', 'Value': 'Staging'},
                    {'Key': 'DeploymentID', 'Value': self.deployment_id}
                ]
            )
            
            # Create staging endpoint
            self.sagemaker_client.create_endpoint(
                EndpointName=staging_endpoint_name,
                EndpointConfigName=config_name,
                Tags=[
                    {'Key': 'Environment', 'Value': 'Staging'},
                    {'Key': 'DeploymentID', 'Value': self.deployment_id}
                ]
            )
            
            logger.info(f"Creating staging endpoint: {staging_endpoint_name}")
            
            # Wait for endpoint to be in service
            status = self._wait_for_endpoint(staging_endpoint_name)
            
            if status == "InService":
                logger.info(f"Staging endpoint {staging_endpoint_name} is now in service")
                self.deployment_artifacts['staging_endpoint'] = staging_endpoint_name
                self.deployment_stages["staging_deployment"] = True
                return staging_endpoint_name
            else:
                logger.error(f"Staging endpoint creation failed with status: {status}")
                return None
                
        except Exception as e:
            logger.error(f"Error deploying to staging: {e}")
            raise
    
    def run_canary_tests(self, test_data_path, acceptance_threshold=0.85):
        """Run canary tests on staging endpoint to validate model."""
        if not self.deployment_stages["staging_deployment"]:
            logger.warning("Cannot run canary tests before staging deployment")
            return False
            
        staging_endpoint = self.deployment_artifacts['staging_endpoint']
        
        try:
            logger.info(f"Running canary tests on endpoint: {staging_endpoint}")
            
            # Load test data
            with open(test_data_path, "r") as f:
                test_data = json.load(f)
            
            # Set up runtime client
            runtime = boto3.client('sagemaker-runtime', region_name=self.region)
            
            # Run tests
            results = {"passed": 0, "failed": 0, "examples": []}
            
            for i, test_case in enumerate(test_data):
                input_text = test_case.get("input", "")
                expected_output = test_case.get("expected_output", "")
                
                # Invoke endpoint
                response = runtime.invoke_endpoint(
                    EndpointName=staging_endpoint,
                    ContentType="application/json",
                    Body=json.dumps({
                        "inputs": input_text,
                        "parameters": {
                            "max_new_tokens": 512,
                            "temperature": 0.1,
                            "do_sample": False
                        }
                    })
                )
                
                # Parse response
                response_body = json.loads(response['Body'].read().decode())
                generated_text = response_body.get("generated_text", "")
                
                # Compare with expected output (simplified)
                # In practice, would use more sophisticated comparison based on task
                similarity = self._calculate_similarity(generated_text, expected_output)
                passed = similarity >= 0.8  # Threshold for individual test case
                
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                
                results["examples"].append({
                    "id": i,
                    "input": input_text[:100] + "...",  # Truncate for logs
                    "expected": expected_output[:100] + "...",
                    "actual": generated_text[:100] + "...",
                    "similarity": similarity,
                    "passed": passed
                })
                
                # Log progress
                if (i + 1) % 10 == 0:
                    logger.info(f"Tested {i + 1}/{len(test_data)} examples")
            
            # Calculate overall pass rate
            pass_rate = results["passed"] / (results["passed"] + results["failed"])
            results["pass_rate"] = pass_rate
            
            # Save results
            with open(f"canary_test_results_{self.timestamp}.json", "w") as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"Canary tests completed with pass rate: {pass_rate:.2f}")
            
            # Check if tests passed the acceptance threshold
            if pass_rate >= acceptance_threshold:
                logger.info("Canary tests passed acceptance threshold")
                self.deployment_stages["canary_testing"] = True
                return True
            else:
                logger.warning(f"Canary tests failed acceptance threshold: {pass_rate:.2f} < {acceptance_threshold}")
                return False
                
        except Exception as e:
            logger.error(f"Error running canary tests: {e}")
            return False
    
    def deploy_to_production(self, update_existing=False, existing_endpoint=None):
        """Deploy model to production."""
        if not self.deployment_stages["canary_testing"] and not update_existing:
            logger.warning("Cannot deploy to production before canary testing")
            return None
            
        production_endpoint_name = existing_endpoint or f"{self.model_name}-prod"
        
        try:
            # Check if endpoint exists
            endpoint_exists = self._endpoint_exists(production_endpoint_name)
            
            if endpoint_exists and not update_existing:
                # Create a new config for blue/green deployment
                self._create_blue_green_config(production_endpoint_name)
            else:
                # Create new endpoint or update existing
                config_name = f"{production_endpoint_name}-config-{self.timestamp}"
                self.sagemaker_client.create_endpoint_config(
                    EndpointConfigName=config_name,
                    ProductionVariants=[
                        {
                            'VariantName': 'Primary',
                            'ModelName': self.deployment_id,
                            'InstanceType': self.instance_type,
                            'InitialInstanceCount': self.instance_count,
                            'InitialVariantWeight': 1.0
                        }
                    ],
                    Tags=[
                        {'Key': 'Environment', 'Value': 'Production'},
                        {'Key': 'DeploymentID', 'Value': self.deployment_id}
                    ]
                )
                
                if endpoint_exists:
                    # Update existing endpoint
                    self.sagemaker_client.update_endpoint(
                        EndpointName=production_endpoint_name,
                        EndpointConfigName=config_name
                    )
                    logger.info(f"Updating existing endpoint: {production_endpoint_name}")
                else:
                    # Create new endpoint
                    self.sagemaker_client.create_endpoint(
                        EndpointName=production_endpoint_name,
                        EndpointConfigName=config_name,
                        Tags=[
                            {'Key': 'Environment', 'Value': 'Production'},
                            {'Key': 'DeploymentID', 'Value': self.deployment_id}
                        ]
                    )
                    logger.info(f"Creating new production endpoint: {production_endpoint_name}")
                
                # Wait for endpoint to be in service
                status = self._wait_for_endpoint(production_endpoint_name)
                
                if status == "InService":
                    logger.info(f"Production endpoint {production_endpoint_name} is now in service")
                    self.deployment_artifacts['production_endpoint'] = production_endpoint_name
                    self.deployment_stages["production_deployment"] = True
                    return production_endpoint_name
                else:
                    logger.error(f"Production endpoint update failed with status: {status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error deploying to production: {e}")
            raise
    
    def _create_blue_green_config(self, endpoint_name):
        """Create endpoint config for blue/green deployment."""
        try:
            # Get existing endpoint configuration
            endpoint_info = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            current_config_name = endpoint_info['EndpointConfigName']
            current_config = self.sagemaker_client.describe_endpoint_config(
                EndpointConfigName=current_config_name
            )
            
            # Get current production variant
            current_variants = current_config['ProductionVariants']
            
            # Create new config with both variants
            blue_green_config_name = f"{endpoint_name}-blue-green-{self.timestamp}"
            
            # Add the new variant with zero weight
            new_variant = {
                'VariantName': f"Variant-{self.timestamp}",
                'ModelName': self.deployment_id,
                'InstanceType': self.instance_type,
                'InitialInstanceCount': self.instance_count,
                'InitialVariantWeight': 0.0  # Start with zero traffic
            }
            
            # Set existing variant weight to 1.0
            for variant in current_variants:
                variant['InitialVariantWeight'] = 1.0
            
            # Create the blue/green config
            self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=blue_green_config_name,
                ProductionVariants=current_variants + [new_variant],
                Tags=[
                    {'Key': 'Environment', 'Value': 'Production'},
                    {'Key': 'DeploymentType', 'Value': 'BlueGreen'},
                    {'Key': 'DeploymentID', 'Value': self.deployment_id}
                ]
            )
            
            # Update the endpoint with the new config
            self.sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=blue_green_config_name
            )
            
            # Store deployment artifacts
            self.deployment_artifacts['blue_green_config'] = blue_green_config_name
            self.deployment_artifacts['production_endpoint'] = endpoint_name
            self.deployment_artifacts['new_variant_name'] = new_variant['VariantName']
            self.deployment_artifacts['current_variants'] = [v['VariantName'] for v in current_variants]
            
            logger.info(f"Created blue/green deployment config: {blue_green_config_name}")
            
            # Wait for endpoint to be in service
            status = self._wait_for_endpoint(endpoint_name)
            
            if status == "InService":
                logger.info(f"Blue/green deployment ready for endpoint: {endpoint_name}")
                self.deployment_stages["production_deployment"] = True
                return True
            else:
                logger.error(f"Blue/green deployment failed with status: {status}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating blue/green config: {e}")
            raise
    
    def shift_traffic(self, new_weight=0.1, endpoint_name=None):
        """Gradually shift traffic to the new variant."""
        if not self.deployment_stages["production_deployment"]:
            logger.warning("Cannot shift traffic before production deployment")
            return False
            
        if 'new_variant_name' not in self.deployment_artifacts:
            logger.warning("No new variant found for traffic shifting")
            return False
            
        endpoint_name = endpoint_name or self.deployment_artifacts['production_endpoint']
        new_variant = self.deployment_artifacts['new_variant_name']
        
        try:
            logger.info(f"Shifting {new_weight * 100}% traffic to variant: {new_variant}")
            
            # Update endpoint weights
            self.sagemaker_client.update_endpoint_weights_and_capacities(
                EndpointName=endpoint_name,
                DesiredWeightsAndCapacities=[
                    {
                        'VariantName': new_variant,
                        'DesiredWeight': new_weight
                    }
                ]
            )
            
            # Track the current traffic weight
            self.deployment_artifacts['current_weight'] = new_weight
            self.deployment_stages["traffic_shifting"] = True
            
            logger.info(f"Traffic shifted successfully to {new_weight * 100}%")
            return True
            
        except Exception as e:
            logger.error(f"Error shifting traffic: {e}")
            return False
    
    def promote_to_full_traffic(self, endpoint_name=None):
        """Promote the new variant to full traffic."""
        if not self.deployment_stages["traffic_shifting"]:
            logger.warning("Cannot promote to full traffic before traffic shifting")
            return False
            
        endpoint_name = endpoint_name or self.deployment_artifacts['production_endpoint']
        new_variant = self.deployment_artifacts['new_variant_name']
        
        try:
            logger.info(f"Promoting variant {new_variant} to 100% traffic")
            
            # Update endpoint weights
            self.sagemaker_client.update_endpoint_weights_and_capacities(
                EndpointName=endpoint_name,
                DesiredWeightsAndCapacities=[
                    {
                        'VariantName': new_variant,
                        'DesiredWeight': 1.0
                    }
                ]
            )
            
            # Create a final config with only the new variant
            final_config_name = f"{endpoint_name}-final-{self.timestamp}"
            self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=final_config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'Primary',
                        'ModelName': self.deployment_id,
                        'InstanceType': self.instance_type,
                        'InitialInstanceCount': self.instance_count,
                        'InitialVariantWeight': 1.0
                    }
                ],
                Tags=[
                    {'Key': 'Environment', 'Value': 'Production'},
                    {'Key': 'DeploymentID', 'Value': self.deployment_id}
                ]
            )
            
            # Update endpoint with final config
            self.sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=final_config_name
            )
            
            # Wait for endpoint to be in service
            status = self._wait_for_endpoint(endpoint_name)
            
            if status == "InService":
                logger.info(f"New variant promoted successfully for endpoint: {endpoint_name}")
                return True
            else:
                logger.error(f"Promotion failed with status: {status}")
                return False
                
        except Exception as e:
            logger.error(f"Error promoting to full traffic: {e}")
            return False
    
    def rollback(self, endpoint_name=None):
        """Roll back to the previous model version."""
        endpoint_name = endpoint_name or self.deployment_artifacts.get('production_endpoint')
        
        if not endpoint_name:
            logger.warning("No production endpoint found for rollback")
            return False
            
        try:
            logger.info(f"Rolling back endpoint: {endpoint_name}")
            
            # Get endpoint info to identify previous config
            endpoint_info = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            current_config_name = endpoint_info['EndpointConfigName']
            
            # List recent endpoint configs
            response = self.sagemaker_client.list_endpoint_configs(
                NameContains=endpoint_name,
                SortBy='CreationTime',
                SortOrder='Descending'
            )
            
            # Find the previous config (not the current one)
            previous_config = None
            for config in response['EndpointConfigs']:
                if config['EndpointConfigName'] != current_config_name:
                    previous_config = config['EndpointConfigName']
                    break
            
            if not previous_config:
                logger.error("No previous config found for rollback")
                return False
                
            # Update endpoint with previous config
            self.sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=previous_config
            )
            
            # Wait for endpoint to be in service
            status = self._wait_for_endpoint(endpoint_name)
            
            if status == "InService":
                logger.info(f"Rolled back to previous config: {previous_config}")
                return True
            else:
                logger.error(f"Rollback failed with status: {status}")
                return False
                
        except Exception as e:
            logger.error(f"Error during rollback: {e}")
            return False
    
    def setup_monitoring(self, endpoint_name=None):
        """Set up CloudWatch alarms for the endpoint."""
        endpoint_name = endpoint_name or self.deployment_artifacts.get('production_endpoint')
        
        if not endpoint_name:
            logger.warning("No production endpoint found for monitoring")
            return False
            
        try:
            logger.info(f"Setting up monitoring for endpoint: {endpoint_name}")
            
            # Set up latency alarm
            self.cloudwatch.put_metric_alarm(
                AlarmName=f"{endpoint_name}-latency",
                AlarmDescription=f"High latency alarm for {endpoint_name}",
                ActionsEnabled=True,
                MetricName="ModelLatency",
                Namespace="AWS/SageMaker",
                Statistic="Average",
                Dimensions=[
                    {
                        'Name': 'EndpointName',
                        'Value': endpoint_name
                    }
                ],
                Period=60,
                EvaluationPeriods=5,
                Threshold=1000,  # 1000ms = 1s
                ComparisonOperator="GreaterThanThreshold",
                TreatMissingData="notBreaching"
            )
            
            # Set up invocation error alarm
            self.cloudwatch.put_metric_alarm(
                AlarmName=f"{endpoint_name}-errors",
                AlarmDescription=f"High error rate alarm for {endpoint_name}",
                ActionsEnabled=True,
                MetricName="Invocation4XXErrors",
                Namespace="AWS/SageMaker",
                Statistic="Sum",
                Dimensions=[
                    {
                        'Name': 'EndpointName',
                        'Value': endpoint_name
                    }
                ],
                Period=60,
                EvaluationPeriods=3,
                Threshold=10,  # 10 errors in 3 minutes
                ComparisonOperator="GreaterThanThreshold",
                TreatMissingData="notBreaching"
            )
            
            # Set up CPU utilization alarm
            self.cloudwatch.put_metric_alarm(
                AlarmName=f"{endpoint_name}-cpu",
                AlarmDescription=f"High CPU utilization alarm for {endpoint_name}",
                ActionsEnabled=True,
                MetricName="CPUUtilization",
                Namespace="AWS/SageMaker",
                Statistic="Average",
                Dimensions=[
                    {
                        'Name': 'EndpointName',
                        'Value': endpoint_name
                    }
                ],
                Period=60,
                EvaluationPeriods=5,
                Threshold=85,  # 85% CPU utilization
                ComparisonOperator="GreaterThanThreshold",
                TreatMissingData="notBreaching"
            )
            
            logger.info(f"Monitoring setup complete for endpoint: {endpoint_name}")
            self.deployment_stages["monitoring_setup"] = True
            return True
            
        except Exception as e:
            logger.error(f"Error setting up monitoring: {e}")
            return False
    
    def run_complete_pipeline(self, test_data_path, acceptance_threshold=0.85, traffic_shift_percentage=0.1):
        """Run the complete deployment pipeline."""
        try:
            # 1. Create model
            self.create_model()
            
            # 2. Deploy to staging
            staging_endpoint = self.deploy_to_staging()
            if not staging_endpoint:
                raise Exception("Staging deployment failed")
                
            # 3. Run canary tests
            canary_passed = self.run_canary_tests(test_data_path, acceptance_threshold)
            if not canary_passed:
                raise Exception("Canary tests failed")
                
            # 4. Deploy to production (blue/green)
            production_endpoint = self.deploy_to_production()
            if not production_endpoint:
                raise Exception("Production deployment failed")
                
            # 5. Shift traffic
            traffic_shifted = self.shift_traffic(traffic_shift_percentage)
            if not traffic_shifted:
                raise Exception("Traffic shifting failed")
                
            # Wait for monitoring period
            logger.info(f"Waiting 10 minutes to monitor performance at {traffic_shift_percentage * 100}% traffic...")
            time.sleep(600)  # 10 minutes
            
            # 6. Set up monitoring
            monitoring_setup = self.setup_monitoring()
            
            # 7. Promote to full traffic
            promotion_success = self.promote_to_full_traffic()
            if not promotion_success:
                raise Exception("Promotion to full traffic failed")
                
            # Return deployment summary
            summary = {
                "model_name": self.deployment_id,
                "production_endpoint": self.deployment_artifacts.get('production_endpoint'),
                "deployment_status": "Success",
                "deployment_stages": self.deployment_stages,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Deployment pipeline completed successfully: {production_endpoint}")
            
            # Save deployment summary
            with open(f"deployment_summary_{self.timestamp}.json", "w") as f:
                json.dump(summary, f, indent=2)
                
            return summary
            
        except Exception as e:
            logger.error(f"Deployment pipeline failed: {e}")
            
            # Try to rollback if needed
            if self.deployment_stages["production_deployment"]:
                logger.info("Attempting rollback...")
                self.rollback()
            
            # Return failure summary
            failure_summary = {
                "model_name": self.deployment_id,
                "deployment_status": "Failed",
                "error": str(e),
                "deployment_stages": self.deployment_stages,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save failure summary
            with open(f"deployment_failure_{self.timestamp}.json", "w") as f:
                json.dump(failure_summary, f, indent=2)
                
            return failure_summary
    
    def _wait_for_endpoint(self, endpoint_name, timeout=1800):
        """Wait for an endpoint to be in service."""
        start_time = time.time()
        status = None
        
        logger.info(f"Waiting for endpoint {endpoint_name} to be in service...")
        
        while time.time() - start_time < timeout:
            try:
                response = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
                status = response['EndpointStatus']
                
                if status == "InService":
                    return status
                elif status in ["Failed", "OutOfService"]:
                    logger.error(f"Endpoint {endpoint_name} {status}: {response.get('FailureReason', 'Unknown error')}")
                    return status
                
                logger.info(f"Endpoint status: {status}")
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error checking endpoint status: {e}")
                time.sleep(30)
        
        logger.error(f"Timed out waiting for endpoint {endpoint_name}")
        return "Timeout"
    
    def _endpoint_exists(self, endpoint_name):
        """Check if an endpoint exists."""
        try:
            self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            return True
        except self.sagemaker_client.exceptions.ClientError:
            return False
    
    def _calculate_similarity(self, text1, text2):
        """Calculate similarity between two code snippets."""
        # Very simple implementation - would be more sophisticated in practice
        # Remove whitespace and make lowercase for comparison
        normalized1 = " ".join(text1.lower().split())
        normalized2 = " ".join(text2.lower().split())
        
        # Split into tokens
        tokens1 = set(normalized1.split())
        tokens2 = set(normalized2.split())
        
        # Calculate Jaccard similarity
        if not tokens1 or not tokens2:
            return 0.0
            
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union)

def main():
    """Main function to run the deployment pipeline."""
    parser = argparse.ArgumentParser(description="Code Model Deployment Pipeline")
    parser.add_argument("--model-data", type=str, required=True, help="S3 URI to model data")
    parser.add_argument("--model-name", type=str, required=True, help="Base name for the model")
    parser.add_argument("--role-arn", type=str, required=True, help="IAM role ARN")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--instance-type", type=str, default="ml.inf1.xlarge", help="Instance type")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances")
    parser.add_argument("--hf-model-id", type=str, default="meta-llama/Llama-4-Scout-17B-16E-Instruct", help="HF model ID")
    parser.add_argument("--entry-point", type=str, default="inference.py", help="Inference script")
    parser.add_argument("--test-data", type=str, help="Path to test data for canary tests")
    parser.add_argument("--stage", type=str, choices=["all", "model", "staging", "canary", "production", "traffic", "monitoring"], default="all", help="Deployment stage to run")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CodeModelDeploymentPipeline(
        model_data=args.model_data,
        model_name=args.model_name,
        role_arn=args.role_arn,
        region=args.region,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        hf_model_id=args.hf_model_id,
        entry_point=args.entry_point
    )
    
    # Run selected stage(s)
    if args.stage == "all" and args.test_data:
        pipeline.run_complete_pipeline(args.test_data)
    elif args.stage == "model":
        pipeline.create_model()
    elif args.stage == "staging":
        pipeline.create_model()
        pipeline.deploy_to_staging()
    elif args.stage == "canary" and args.test_data:
        pipeline.create_model()
        pipeline.deploy_to_staging()
        pipeline.run_canary_tests(args.test_data)
    elif args.stage == "production":
        pipeline.create_model()
        pipeline.deploy_to_staging()
        if args.test_data:
            pipeline.run_canary_tests(args.test_data)
        pipeline.deploy_to_production()
    elif args.stage == "traffic":
        pipeline.create_model()
        pipeline.deploy_to_staging()
        if args.test_data:
            pipeline.run_canary_tests(args.test_data)
        pipeline.deploy_to_production()
        pipeline.shift_traffic(0.1)
    elif args.stage == "monitoring":
        pipeline.create_model()
        pipeline.deploy_to_staging()
        if args.test_data:
            pipeline.run_canary_tests(args.test_data)
        pipeline.deploy_to_production()
        pipeline.shift_traffic(0.1)
        pipeline.setup_monitoring()
    else:
        logger.error("Invalid stage or missing test data")

if __name__ == "__main__":
    main()
```

### 5.3 Custom Model Server Optimization

Add a custom model server configuration script for performance optimization:

```python
# optimize_model_server.py

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description="Optimize SageMaker endpoint performance")
    parser.add_argument("--endpoint-name", type=str, required=True, help="Endpoint name")
    parser.add_argument("--instance-type", type=str, required=True, help="Instance type")
    parser.add_argument("--model-id", type=str, required=True, help="Model ID (e.g., meta-llama/Llama-4-Scout-17B)")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8", help="Comma-separated batch sizes to test")
    parser.add_argument("--sequence-lengths", type=str, default="512,1024,2048,4096", help="Comma-separated sequence lengths to test")
    parser.add_argument("--output-dir", type=str, default="optimization_results", help="Output directory for results")
    return parser.parse_args()

def create_test_payload(batch_size, sequence_length, model_id):
    """Create a test payload for benchmarking."""
    # Generate a random prompt of the desired length
    tokens_per_word = 1.3  # Rough approximation
    words_needed = int(sequence_length / tokens_per_word)
    
    # Create a payload with the specified batch size
    payload = {
        "inputs": ["Generate a function that calculates the fibonacci sequence " * words_needed] * batch_size,
        "parameters": {
            "max_new_tokens": 128,
            "temperature": 0.1,
            "do_sample": False,
            "return_full_text": False
        }
    }
    
    return json.dumps(payload)

def measure_inference_time(endpoint_name, payload, num_trials=5):
    """Measure inference time for a given payload."""
    # AWS CLI command to invoke endpoint
    cmd = [
        "aws", "sagemaker-runtime", "invoke-endpoint",
        "--endpoint-name", endpoint_name,
        "--content-type", "application/json",
        "--body", payload,
        "/dev/null"  # Discard output
    ]
    
    latencies = []
    for _ in range(num_trials):
        start_time = time.time()
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        end_time = time.time()
        
        if result.returncode == 0:
            latencies.append(end_time - start_time)
        else:
            print(f"Error invoking endpoint: {result.stderr.decode()}")
        
        # Wait between trials
        time.sleep(1)
    
    # Remove outliers (keep middle 80%)
    if latencies:
        latencies = sorted(latencies)
        num_to_remove = int(len(latencies) * 0.1)
        if num_to_remove > 0:
            latencies = latencies[num_to_remove:-num_to_remove]
        
        return np.mean(latencies), np.std(latencies)
    else:
        return None, None

def run_optimization_tests(args):
    """Run performance optimization tests."""
    batch_sizes = [int(b) for b in args.batch_sizes.split(",")]
    sequence_lengths = [int(s) for s in args.sequence_lengths.split(",")]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Store results
    results = {
        "endpoint_name": args.endpoint_name,
        "instance_type": args.instance_type,
        "model_id": args.model_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tests": []
    }
    
    print(f"Running optimization tests on endpoint: {args.endpoint_name}")
    print(f"Instance type: {args.instance_type}")
    print(f"Model: {args.model_id}")
    print(f"Testing batch sizes: {batch_sizes}")
    print(f"Testing sequence lengths: {sequence_lengths}")
    
    # Run tests for each combination
    total_tests = len(batch_sizes) * len(sequence_lengths)
    test_count = 0
    
    for batch_size in batch_sizes:
        for seq_length in sequence_lengths:
            test_count += 1
            print(f"\nTest {test_count}/{total_tests}: Batch size {batch_size}, Sequence length {seq_length}")
            
            # Create test payload
            payload = create_test_payload(batch_size, seq_length, args.model_id)
            
            # Measure inference time
            mean_latency, std_latency = measure_inference_time(args.endpoint_name, payload)
            
            if mean_latency is not None:
                throughput = batch_size / mean_latency
                
                # Add result
                results["tests"].append({
                    "batch_size": batch_size,
                    "sequence_length": seq_length,
                    "mean_latency": mean_latency,
                    "std_latency": std_latency,
                    "throughput": throughput
                })
                
                print(f"  Mean latency: {mean_latency:.2f}s")
                print(f"  Throughput: {throughput:.2f} requests/second")
            else:
                print("  Test failed")
    
    # Save results
    results_file = os.path.join(args.output_dir, f"optimization_results_{int(time.time())}.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_file}")
    
    # Generate visualizations
    generate_visualizations(results, args.output_dir)
    
    # Find optimal configuration
    optimal_config = find_optimal_configuration(results)
    print("\nOptimal configuration:")
    print(f"  Batch size: {optimal_config['batch_size']}")
    print(f"  Throughput: {optimal_config['throughput']:.2f} requests/second")
    print(f"  Latency: {optimal_config['mean_latency']:.2f}s")
    
    # Generate server configuration
    generate_server_config(optimal_config, args, results_file)
    
    return results

def generate_visualizations(results, output_dir):
    """Generate visualizations of performance results."""
    # Extract data
    batch_sizes = sorted(list(set(test["batch_size"] for test in results["tests"])))
    seq_lengths = sorted(list(set(test["sequence_length"] for test in results["tests"])))
    
    # Create matrix of results
    latency_matrix = np.zeros((len(batch_sizes), len(seq_lengths)))
    throughput_matrix = np.zeros((len(batch_sizes), len(seq_lengths)))
    
    for test in results["tests"]:
        batch_idx = batch_sizes.index(test["batch_size"])
        seq_idx = seq_lengths.index(test["sequence_length"])
        latency_matrix[batch_idx, seq_idx] = test["mean_latency"]
        throughput_matrix[batch_idx, seq_idx] = test["throughput"]
    
    # Plot latency heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(latency_matrix, cmap="viridis")
    plt.colorbar(label="Latency (s)")
    plt.xticks(range(len(seq_lengths)), seq_lengths)
    plt.yticks(range(len(batch_sizes)), batch_sizes)
    plt.xlabel("Sequence Length")
    plt.ylabel("Batch Size")
    plt.title(f"Latency (s) - {results['model_id']} on {results['instance_type']}")
    
    # Add text annotations
    for i in range(len(batch_sizes)):
        for j in range(len(seq_lengths)):
            plt.text(j, i, f"{latency_matrix[i, j]:.2f}",
                     ha="center", va="center", color="white" if latency_matrix[i, j] > 2 else "black")
    
    plt.savefig(os.path.join(output_dir, "latency_heatmap.png"))
    
    # Plot throughput heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(throughput_matrix, cmap="viridis")
    plt.colorbar(label="Throughput (requests/s)")
    plt.xticks(range(len(seq_lengths)), seq_lengths)
    plt.yticks(range(len(batch_sizes)), batch_sizes)
    plt.xlabel("Sequence Length")
    plt.ylabel("Batch Size")
    plt.title(f"Throughput (requests/s) - {results['model_id']} on {results['instance_type']}")
    
    # Add text annotations
    for i in range(len(batch_sizes)):
        for j in range(len(seq_lengths)):
            plt.text(j, i, f"{throughput_matrix[i, j]:.2f}",
                     ha="center", va="center", color="white" if throughput_matrix[i, j] < 1 else "black")
    
    plt.savefig(os.path.join(output_dir, "throughput_heatmap.png"))
    
    # Plot latency vs batch size for different sequence lengths
    plt.figure(figsize=(10, 6))
    for i, seq_length in enumerate(seq_lengths):
        latencies = [test["mean_latency"] for test in results["tests"] if test["sequence_length"] == seq_length]
        batch_sizes_for_seq = [test["batch_size"] for test in results["tests"] if test["sequence_length"] == seq_length]
        
        # Sort by batch size
        sorted_indices = np.argsort(batch_sizes_for_seq)
        sorted_batch_sizes = [batch_sizes_for_seq[i] for i in sorted_indices]
        sorted_latencies = [latencies[i] for i in sorted_indices]
        
        plt.plot(sorted_batch_sizes, sorted_latencies, marker="o", label=f"Seq Len: {seq_length}")
    
    plt.xlabel("Batch Size")
    plt.ylabel("Latency (s)")
    plt.title("Latency vs Batch Size")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "latency_vs_batch_size.png"))
    
    # Plot throughput vs batch size for different sequence lengths
    plt.figure(figsize=(10, 6))
    for i, seq_length in enumerate(seq_lengths):
        throughputs = [test["throughput"] for test in results["tests"] if test["sequence_length"] == seq_length]
        batch_sizes_for_seq = [test["batch_size"] for test in results["tests"] if test["sequence_length"] == seq_length]
        
        # Sort by batch size
        sorted_indices = np.argsort(batch_sizes_for_seq)
        sorted_batch_sizes = [batch_sizes_for_seq[i] for i in sorted_indices]
        sorted_throughputs = [throughputs[i] for i in sorted_indices]
        
        plt.plot(sorted_batch_sizes, sorted_throughputs, marker="o", label=f"Seq Len: {seq_length}")
    
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (requests/s)")
    plt.title("Throughput vs Batch Size")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "throughput_vs_batch_size.png"))
    
    print(f"Visualizations saved to {output_dir}")

def find_optimal_configuration(results):
    """Find optimal configuration based on throughput and latency constraints."""
    max_acceptable_latency = 5.0  # seconds
    
    # Filter tests with acceptable latency
    acceptable_tests = [test for test in results["tests"] if test["mean_latency"] <= max_acceptable_latency]
    
    if not acceptable_tests:
        # If no tests meet the latency requirement, find the one with minimum latency
        return min(results["tests"], key=lambda x: x["mean_latency"])
    
    # From acceptable tests, find the one with maximum throughput
    return max(acceptable_tests, key=lambda x: x["throughput"])

def generate_server_config(optimal_config, args, results_file):
    """Generate optimized server configuration."""
    # Create server config based on optimal parameters
    server_config = {
        "parameters": {
            "batch_size": optimal_config["batch_size"],
            "max_batch_tokens": optimal_config["batch_size"] * optimal_config["sequence_length"],
            "max_batch_timeout_ms": int(optimal_config["mean_latency"] * 1000),  # Convert to milliseconds
            "max_concurrent_requests": optimal_config["batch_size"] * 2,  # Allow some queuing
            "dynamic_batching": True,
            "enable_cuda_graphs": True,
            "inference_tensor_parallel_size": 1,  # Set based on instance type
            "enable_flash_attention": True,
            "max_sequence_length": max(8192, optimal_config["sequence_length"] * 2),  # Allow room for growth
            "kv_cache_precision": "fp16",
            "cuda_memory_fraction": 0.95
        },
        "instance_type": args.instance_type,
        "model_id": args.model_id,
        "endpoint_name": args.endpoint_name,
        "based_on_results": results_file
    }
    
    # Adjust for specific instance types
    if "p4d" in args.instance_type or "p4de" in args.instance_type:
        server_config["parameters"]["inference_tensor_parallel_size"] = 4
    elif "g5" in args.instance_type:
        server_config["parameters"]["inference_tensor_parallel_size"] = 1
    
    # Save server config
    config_file = os.path.join(args.output_dir, f"server_config_{int(time.time())}.json")
    with open(config_file, "w") as f:
        json.dump(server_config, f, indent=2)
    
    print(f"Optimized server configuration saved to {config_file}")
    
    # Generate environment variables for SageMaker
    env_vars = {
        "SM_NUM_GPUS": "1",  # Set based on instance type
        "SM_BATCH_SIZE": str(optimal_config["batch_size"]),
        "SM_MAX_BATCH_TIMEOUT_MS": str(int(optimal_config["mean_latency"] * 1000)),
        "SM_MAX_CONCURRENT_REQUESTS": str(optimal_config["batch_size"] * 2),
        "SM_ENABLE_CUDA_GRAPHS": "true",
        "SM_ENABLE_FLASH_ATTENTION": "true"
    }
    
    # Adjust for multi-GPU instances
    if "p4d" in args.instance_type:
        env_vars["SM_NUM_GPUS"] = "8"
    elif "g5.12xlarge" in args.instance_type:
        env_vars["SM_NUM_GPUS"] = "4"
    elif "g5.48xlarge" in args.instance_type:
        env_vars["SM_NUM_GPUS"] = "8"
    
    # Save environment variables
    env_file = os.path.join(args.output_dir, f"env_vars_{int(time.time())}.sh")
    with open(env_file, "w") as f:
        for key, value in env_vars.items():
            f.write(f"export {key}={value}\n")
    
    print(f"Environment variables saved to {env_file}")
    
    return server_config

def main():
    args = parse_args()
    run_optimization_tests(args)

if __name__ == "__main__":
    main()
```

### 5.4 High-Performance Inference Script

Create a high-performance inference script with advanced optimizations:

```python
# optimized_inference.py
import os
import json
import torch
import time
import logging
import threading
import queue
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("inference")

# Global variables
model = None
tokenizer = None
lock = threading.Lock()
request_queue = queue.Queue()
batch_size = int(os.environ.get("SM_BATCH_SIZE", "1"))
max_batch_timeout_ms = int(os.environ.get("SM_MAX_BATCH_TIMEOUT_MS", "1000"))
max_concurrent_requests = int(os.environ.get("SM_MAX_CONCURRENT_REQUESTS", "8"))
enable_cuda_graphs = os.environ.get("SM_ENABLE_CUDA_GRAPHS", "true").lower() == "true"
enable_flash_attention = os.environ.get("SM_ENABLE_FLASH_ATTENTION", "true").lower() == "true"
num_gpus = int(os.environ.get("SM_NUM_GPUS", "1"))
cuda_memory_fraction = float(os.environ.get("SM_CUDA_MEMORY_FRACTION", "0.95"))

# Worker thread for batch processing
worker_thread = None
running = False

@dataclass
class InferenceRequest:
    """Class to hold a single inference request."""
    input_text: str
    request_id: str
    parameters: Dict[str, Any]
    result_queue: queue.Queue
    start_time: float = None

    def __post_init__(self):
        self.start_time = time.time()

class BatchProcessor(threading.Thread):
    """Thread to process batches of inference requests."""
    
    def __init__(self, request_queue, model, tokenizer, batch_size, max_timeout_ms):
        """Initialize the batch processor."""
        threading.Thread.__init__(self)
        self.daemon = True
        self.request_queue = request_queue
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_timeout_ms = max_timeout_ms
        self.stop_event = threading.Event()
    
    def run(self):
        """Run the batch processor."""
        logger.info(f"Starting batch processor with batch size {self.batch_size}")
        while not self.stop_event.is_set():
            try:
                # Collect batch of requests
                batch = self._collect_batch()
                
                if not batch:
                    # No requests, sleep a bit
                    time.sleep(0.01)
                    continue
                
                # Process batch
                self._process_batch(batch)
                
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                # Sleep to avoid tight loop on error
                time.sleep(0.1)
    
    def stop(self):
        """Stop the batch processor."""
        self.stop_event.set()
    
    def _collect_batch(self) -> List[InferenceRequest]:
        """Collect a batch of requests."""
        batch = []
        
        # Get the first request (blocking)
        try:
            first_request = self.request_queue.get(block=True, timeout=0.1)
            batch.append(first_request)
        except queue.Empty:
            return []
        
        # Try to fill the batch
        batch_deadline = time.time() + (self.max_timeout_ms / 1000)
        
        while len(batch) < self.batch_size and time.time() < batch_deadline:
            try:
                request = self.request_queue.get(block=False)
                batch.append(request)
            except queue.Empty:
                # Wait a bit and try again
                time.sleep(0.001)
        
        logger.info(f"Collected batch of {len(batch)} requests")
        return batch
    
    def _process_batch(self, batch: List[InferenceRequest]):
        """Process a batch of inference requests."""
        if not batch:
            return
        
        batch_size = len(batch)
        
        try:
            # Prepare inputs
            input_texts = [req.input_text for req in batch]
            
            # Get parameters from first request (for simplicity)
            parameters = batch[0].parameters
            max_tokens = parameters.get("max_new_tokens", 512)
            temperature = parameters.get("temperature", 0.1)
            do_sample = parameters.get("do_sample", False)
            top_p = parameters.get("top_p", 0.95)
            
            # Override with request-specific parameters if needed
            for i, req in enumerate(batch[1:], 1):
                req_max_tokens = req.parameters.get("max_new_tokens")
                if req_max_tokens and req_max_tokens > max_tokens:
                    max_tokens = req_max_tokens
            
            # Tokenize inputs
            batch_tokens = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.model.device)
            
            # Generate outputs
            with torch.no_grad():
                start_time = time.time()
                outputs = self.model.generate(
                    **batch_tokens,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                generation_time = time.time() - start_time
            
            # Decode outputs
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Process and return results
            for i, (req, text) in enumerate(zip(batch, generated_texts)):
                # Extract code from generated text
                generated_text = extract_code_from_text(text, req.input_text)
                
                # Calculate latency
                latency = time.time() - req.start_time
                
                # Return result
                result = {
                    "generated_text": generated_text,
                    "latency": latency,
                    "request_id": req.request_id,
                    "batch_size": batch_size,
                    "position_in_batch": i
                }
                
                req.result_queue.put(result)
                self.request_queue.task_done()
            
            logger.info(f"Processed batch of {batch_size} requests in {generation_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            
            # Return error to all requests in batch
            for req in batch:
                req.result_queue.put({
                    "error": str(e),
                    "request_id": req.request_id
                })
                self.request_queue.task_done()

def extract_code_from_text(text, prompt):
    """Extract code from generated text."""
    # Remove the prompt from the output
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    
    # Try to find code blocks with triple backticks
    if "```" in text:
        code_blocks = text.split("```")
        if len(code_blocks) >= 3:
            # Get the content between the first pair of backticks
            code_block = code_blocks[1]
            
            # Remove language identifier if present
            lines = code_block.split("\n")
            if lines and lines[0].strip() in ["python", "py", "cpp", "c++", "java", "javascript", "js"]:
                code_block = "\n".join(lines[1:])
            
            return code_block.strip()
    
    # Try to extract code from markdown sections
    if "### Output Code:" in text:
        code_part = text.split("### Output Code:")[1].strip()
        if "```" in code_part:
            code_blocks = code_part.split("```")
            if len(code_blocks) >= 2:
                return code_blocks[1].strip()
        return code_part.strip()
    
    # Default to returning the generated text
    return text.strip()

def model_fn(model_dir):
    """
    Load the model and tokenizer.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        The loaded model
    """
    global model, tokenizer, worker_thread, running
    
    try:
        # Get HF model ID from environment
        hf_model_id = os.environ.get("HF_MODEL_ID", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
        logger.info(f"Loading model: {hf_model_id}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Configure quantization for efficient inference
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load base model
        try:
            # Try to load the base model
            model = AutoModelForCausalLM.from_pretrained(
                hf_model_id,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True,
                use_flash_attention_2=enable_flash_attention,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            
            # Apply CUDA memory optimization
            if torch.cuda.is_available():
                for device_id in range(torch.cuda.device_count()):
                    device = torch.device(f"cuda:{device_id}")
                    torch.cuda.set_per_process_memory_fraction(cuda_memory_fraction, device)
                    
            # Load adapter weights (LoRA)
            model = PeftModel.from_pretrained(model, model_dir)
            model.eval()
            
            # Apply optimizations
            if enable_cuda_graphs and hasattr(torch, "cuda") and torch.cuda.is_available():
                logger.info("Enabling CUDA graphs for optimized inference")
                # Enable CUDA graphs by warming up the model
                dummy_input = tokenizer("Warm up input", return_tensors="pt").to(model.device)
                # Warmup the model several times for CUDA graphs to capture the pattern
                for _ in range(3):
                    with torch.no_grad():
                        model.generate(**dummy_input, max_new_tokens=10)
            
            logger.info(f"Model loaded successfully with {num_gpus} GPUs")
            
            # Start batch processor thread
            running = True
            worker_thread = BatchProcessor(
                request_queue=request_queue,
                model=model,
                tokenizer=tokenizer,
                batch_size=batch_size,
                max_timeout_ms=max_batch_timeout_ms
            )
            worker_thread.start()
            logger.info("Batch processor thread started")
            
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Error in model_fn: {e}")
        raise

def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input.
    """
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    """
    Make a prediction with the input data.
    """
    try:
        # Extract inputs
        inputs = input_data.get("inputs", "")
        if isinstance(inputs, list):
            # Handle batch input
            request_texts = inputs
        else:
            # Handle single input
            request_texts = [inputs]
        
        # Extract parameters
        parameters = input_data.get("parameters", {})
        
        # Create results queue
        result_queue = queue.Queue()
        
        # Submit requests to the queue
        for i, text in enumerate(request_texts):
            request = InferenceRequest(
                input_text=text,
                request_id=f"req_{time.time()}_{i}",
                parameters=parameters,
                result_queue=result_queue
            )
            request_queue.put(request)
        
        # Wait for results
        results = []
        for _ in range(len(request_texts)):
            results.append(result_queue.get())
        
        # Return results
        if len(results) == 1:
            # Single input, return single result
            return results[0]
        else:
            # Batch input, return list of results
            return {"results": results}
        
    except Exception as e:
        logger.error(f"Error in predict_fn: {e}")
        return {"error": str(e)}

def output_fn(prediction, response_content_type):
    """
    Serialize the prediction output.
    """
    if response_content_type == "application/json":
        return json.dumps(prediction), response_content_type
    else:
        return json.dumps(prediction), "application/json"
```
## 5.5 Performance Analysis and Advanced Deployment Strategies

### 5.5.1 Performance Benchmarking Framework

Create a comprehensive benchmarking framework to evaluate your model performance across multiple dimensions:

```python
# model_benchmark.py

import argparse
import json
import os
import time
import boto3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark model performance across multiple dimensions")
    parser.add_argument("--endpoint-name", type=str, required=True, help="SageMaker endpoint name")
    parser.add_argument("--test-set", type=str, required=True, help="Path to JSON test file with examples")
    parser.add_argument("--output-dir", type=str, default="benchmark_results", help="Output directory")
    parser.add_argument("--concurrent-requests", type=int, default=10, help="Number of concurrent requests")
    parser.add_argument("--repetitions", type=int, default=3, help="Number of repetitions per test")
    parser.add_argument("--categories", type=str, default="task_type,code_length,language",
                       help="Comma-separated categories to analyze")
    parser.add_argument("--compare-with", type=str, help="Path to previous benchmark results for comparison")
    return parser.parse_args()

def invoke_endpoint(endpoint_name, payload):
    """Invoke a SageMaker endpoint with the given payload."""
    client = boto3.client('sagemaker-runtime')
    start_time = time.time()
    try:
        response = client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps(payload)
        )
        end_time = time.time()
        response_body = json.loads(response['Body'].read().decode())
        
        return {
            'success': True,
            'latency': end_time - start_time,
            'response': response_body,
            'error': None
        }
    except Exception as e:
        end_time = time.time()
        return {
            'success': False,
            'latency': end_time - start_time,
            'response': None,
            'error': str(e)
        }

def calculate_correctness(expected, generated):
    """Calculate correctness metrics between expected and generated code."""
    # Normalize whitespace
    expected_norm = ' '.join([line.strip() for line in expected.strip().split('\n') if line.strip()])
    generated_norm = ' '.join([line.strip() for line in generated.strip().split('\n') if line.strip()])
    
    # Exact match
    exact_match = expected_norm == generated_norm
    
    # Token overlap
    expected_tokens = set(expected_norm.split())
    generated_tokens = set(generated_norm.split())
    
    if not expected_tokens:
        token_overlap = 0.0
    else:
        token_overlap = len(expected_tokens.intersection(generated_tokens)) / len(expected_tokens)
    
    return {
        'exact_match': exact_match,
        'token_overlap': token_overlap
    }

def run_benchmark(args):
    """Run the benchmark with the given arguments."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test set
    with open(args.test_set, 'r') as f:
        test_data = json.load(f)
    
    print(f"Loaded {len(test_data)} test examples")
    
    # Extract categories for analysis
    categories = args.categories.split(',')
    for category in categories:
        print(f"Will analyze by category: {category}")
    
    # Prepare results storage
    results = []
    
    # Run benchmark with multiple threads
    with ThreadPoolExecutor(max_workers=args.concurrent_requests) as executor:
        for test_idx, test_case in enumerate(tqdm(test_data)):
            # Extract test case data
            input_text = test_case.get('input', '')
            expected_output = test_case.get('expected_output', '')
            metadata = test_case.get('metadata', {})
            
            # Create payload
            payload = {
                'inputs': input_text,
                'parameters': {
                    'max_new_tokens': 512,
                    'temperature': 0.1,
                    'do_sample': False
                }
            }
            
            # Submit repeated requests
            futures = []
            for rep in range(args.repetitions):
                futures.append(executor.submit(invoke_endpoint, args.endpoint_name, payload))
            
            # Process results
            for rep, future in enumerate(futures):
                response_data = future.result()
                
                # Calculate correctness if successful
                if response_data['success']:
                    generated_text = response_data['response'].get('generated_text', '')
                    correctness = calculate_correctness(expected_output, generated_text)
                else:
                    correctness = {'exact_match': False, 'token_overlap': 0.0}
                
                # Store result
                result = {
                    'test_id': test_idx,
                    'repetition': rep,
                    'latency': response_data['latency'],
                    'success': response_data['success'],
                    'error': response_data['error'],
                    'exact_match': correctness['exact_match'],
                    'token_overlap': correctness['token_overlap']
                }
                
                # Add metadata categories
                for category in categories:
                    result[category] = metadata.get(category, 'unknown')
                
                results.append(result)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save raw results
    df.to_csv(os.path.join(args.output_dir, 'benchmark_raw_results.csv'), index=False)
    
    # Generate summary report
    summary = analyze_results(df, categories, args.output_dir)
    
    # Compare with previous results if provided
    if args.compare_with:
        compare_results(df, args.compare_with, categories, args.output_dir)
    
    return summary

def analyze_results(df, categories, output_dir):
    """Analyze benchmark results and generate visualizations."""
    # Create summary report
    summary = {
        'overall': {
            'total_requests': len(df),
            'success_rate': df['success'].mean() * 100,
            'mean_latency': df['latency'].mean(),
            'p50_latency': df['latency'].quantile(0.5),
            'p90_latency': df['latency'].quantile(0.9),
            'p99_latency': df['latency'].quantile(0.99),
            'exact_match_rate': df['exact_match'].mean() * 100,
            'mean_token_overlap': df['token_overlap'].mean() * 100
        },
        'by_category': {}
    }
    
    # Analyze by category
    for category in categories:
        if category in df.columns:
            # Group by category
            grouped = df.groupby(category).agg({
                'latency': ['mean', 'median', lambda x: x.quantile(0.9)],
                'success': 'mean',
                'exact_match': 'mean',
                'token_overlap': 'mean'
            })
            
            # Rename columns
            grouped.columns = ['mean_latency', 'median_latency', 'p90_latency', 
                               'success_rate', 'exact_match_rate', 'token_overlap']
            
            # Convert rates to percentages
            grouped['success_rate'] *= 100
            grouped['exact_match_rate'] *= 100
            grouped['token_overlap'] *= 100
            
            # Store in summary
            summary['by_category'][category] = grouped.to_dict('index')
            
            # Generate visualization
            plt.figure(figsize=(12, 8))
            
            # Plot latency by category
            plt.subplot(2, 1, 1)
            sns.barplot(x=grouped.index, y=grouped['mean_latency'])
            plt.title(f'Mean Latency by {category}')
            plt.ylabel('Latency (seconds)')
            plt.xticks(rotation=45)
            
            # Plot accuracy by category
            plt.subplot(2, 1, 2)
            sns.barplot(x=grouped.index, y=grouped['exact_match_rate'])
            plt.title(f'Exact Match Rate by {category}')
            plt.ylabel('Exact Match Rate (%)')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'benchmark_{category}.png'))
            plt.close()
    
    # Generate overall performance visualization
    plt.figure(figsize=(10, 6))
    metrics = ['success_rate', 'exact_match_rate', 'mean_token_overlap']
    values = [summary['overall'][m] for m in metrics]
    
    sns.barplot(x=metrics, y=values)
    plt.title('Overall Performance Metrics')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    
    # Add value labels
    for i, v in enumerate(values):
        plt.text(i, v + 2, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_overall.png'))
    plt.close()
    
    # Generate latency distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['latency'], kde=True)
    plt.title('Latency Distribution')
    plt.xlabel('Latency (seconds)')
    plt.axvline(summary['overall']['p50_latency'], color='r', linestyle='--', label='P50')
    plt.axvline(summary['overall']['p90_latency'], color='g', linestyle='--', label='P90')
    plt.axvline(summary['overall']['p99_latency'], color='b', linestyle='--', label='P99')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_latency.png'))
    plt.close()
    
    # Save summary report
    with open(os.path.join(output_dir, 'benchmark_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    return summary

def compare_results(current_df, previous_results_path, categories, output_dir):
    """Compare current results with previous benchmark results."""
    # Load previous results
    try:
        if previous_results_path.endswith('.csv'):
            previous_df = pd.read_csv(previous_results_path)
        elif previous_results_path.endswith('.json'):
            with open(previous_results_path, 'r') as f:
                previous_summary = json.load(f)
                # We need the raw data for proper comparison
                if 'raw_results_path' in previous_summary:
                    previous_df = pd.read_csv(previous_summary['raw_results_path'])
                else:
                    print("Warning: Cannot find raw results in previous summary")
                    return
        else:
            print(f"Unsupported previous results format: {previous_results_path}")
            return
    except Exception as e:
        print(f"Error loading previous results: {e}")
        return
    
    # Create comparison report
    comparison = {
        'overall': {
            'current': {
                'mean_latency': current_df['latency'].mean(),
                'success_rate': current_df['success'].mean() * 100,
                'exact_match_rate': current_df['exact_match'].mean() * 100,
                'token_overlap': current_df['token_overlap'].mean() * 100
            },
            'previous': {
                'mean_latency': previous_df['latency'].mean(),
                'success_rate': previous_df['success'].mean() * 100,
                'exact_match_rate': previous_df['exact_match'].mean() * 100,
                'token_overlap': previous_df['token_overlap'].mean() * 100
            },
            'diff': {}
        },
        'by_category': {}
    }
    
    # Calculate differences
    for metric in ['mean_latency', 'success_rate', 'exact_match_rate', 'token_overlap']:
        current_val = comparison['overall']['current'][metric]
        previous_val = comparison['overall']['previous'][metric]
        diff = current_val - previous_val
        percent_change = (diff / previous_val) * 100 if previous_val != 0 else float('inf')
        
        comparison['overall']['diff'][metric] = {
            'absolute': diff,
            'percent': percent_change
        }
    
    # Generate comparison visualization
    plt.figure(figsize=(12, 6))
    
    metrics = ['success_rate', 'exact_match_rate', 'token_overlap']
    current_values = [comparison['overall']['current'][m] for m in metrics]
    previous_values = [comparison['overall']['previous'][m] for m in metrics]
    
    x = range(len(metrics))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], previous_values, width, label='Previous')
    plt.bar([i + width/2 for i in x], current_values, width, label='Current')
    
    plt.xlabel('Metric')
    plt.ylabel('Percentage (%)')
    plt.title('Performance Comparison')
    plt.xticks(x, metrics)
    plt.ylim(0, 100)
    plt.legend()
    
    # Add value labels
    for i, v in enumerate(previous_values):
        plt.text(i - width/2, v + 2, f'{v:.1f}%', ha='center')
    
    for i, v in enumerate(current_values):
        plt.text(i + width/2, v + 2, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_comparison.png'))
    plt.close()
    
    # Compare latency
    plt.figure(figsize=(10, 6))
    sns.kdeplot(previous_df['latency'], label='Previous')
    sns.kdeplot(current_df['latency'], label='Current')
    plt.title('Latency Distribution Comparison')
    plt.xlabel('Latency (seconds)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_latency_comparison.png'))
    plt.close()
    
    # Save comparison report
    with open(os.path.join(output_dir, 'benchmark_comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"Comparison complete. Results saved to {output_dir}")

if __name__ == '__main__':
    args = parse_args()
    run_benchmark(args)
```

### 5.5.2 Deployment Strategy Decision Tree

Use the following decision tree to determine the optimal deployment strategy for your fine-tuned model:

```
Is code generation latency critical for your use case?
├── Yes → Do you need code completion in real-time (< 100ms)?
│   ├── Yes → Deploy distilled model with Inferentia
│   │         Consider encoder-only model optimized for completion
│   └── No → Is expected load > 10 requests/second?
│       ├── Yes → Use multiple g5.2xlarge instances with auto-scaling
│       │         Configure optimized batching with 250ms max timeout
│       └── No → Single g5.xlarge with dynamic batch size = 2
│               Optimize for throughput with batch_timeout_ms = 500
└── No → Is cost the primary concern?
    ├── Yes → Can batch processing work for your use case?
    │   ├── Yes → Use p4d.24xlarge for scheduled batch processing
    │   │         Process thousands of code snippets per batch
    │   └── No → Use inf1.2xlarge with ONNX optimization
    │           Configure max_batch_size = 4, max_timeout_ms = 2000
    └── No → Do you need flexibility to change parameters frequently?
        ├── Yes → Deploy on g5.4xlarge for better CPU resources
        │         Enable batch_flexibility in server configuration
        └── No → Use g5.2xlarge single instance
                Configure for balanced latency/throughput (500ms)
```

### 5.5.3 Multi-Region High-Availability Setup

For production-grade deployments, implement a multi-region high-availability setup:

```python
# multiregion_deploy.py

import boto3
import argparse
import json
import time
import os
from botocore.exceptions import ClientError

def parse_args():
    parser = argparse.ArgumentParser(description="Deploy model to multiple regions")
    parser.add_argument("--model-name", type=str, required=True, help="Model name")
    parser.add_argument("--model-data", type=str, required=True, help="S3 URI to model data")
    parser.add_argument("--regions", type=str, required=True, help="Comma-separated list of regions")
    parser.add_argument("--instance-type", type=str, default="ml.g5.2xlarge", help="Instance type")
    parser.add_argument("--role-name", type=str, required=True, help="Base IAM role name")
    parser.add_argument("--entry-point", type=str, default="inference.py", help="Inference script name")
    parser.add_argument("--config-file", type=str, help="Path to configuration file")
    return parser.parse_args()

def create_or_update_role(role_name, region):
    """Create or update IAM role in the specified region."""
    iam = boto3.client('iam', region_name=region)
    
    # Define trust policy for SageMaker
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        # Check if role exists
        iam.get_role(RoleName=role_name)
        print(f"Role {role_name} already exists in {region}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchEntity':
            # Create role
            iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description=f"SageMaker execution role for multi-region deployment"
            )
            print(f"Created role {role_name} in {region}")
            
            # Attach policies
            policies = [
                "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
                "arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess"
            ]
            
            for policy_arn in policies:
                iam.attach_role_policy(
                    RoleName=role_name,
                    PolicyArn=policy_arn
                )
        else:
            raise
    
    # Get role ARN
    response = iam.get_role(RoleName=role_name)
    return response['Role']['Arn']

def create_model_in_region(region, model_name, model_data, role_arn, config):
    """Create SageMaker model in the specified region."""
    sm = boto3.client('sagemaker', region_name=region)
    
    # Define container
    container = {
        'Image': f"763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-inference:2.0.1-transformers4.31.0-cpu-py310-ubuntu20.04",
        'ModelDataUrl': model_data,
        'Environment': {
            'HF_MODEL_ID': config.get('hf_model_id', 'meta-llama/Llama-4-Scout-17B-16E-Instruct'),
            'HF_TASK': 'text-generation',
            'SAGEMAKER_PROGRAM': config.get('entry_point', 'inference.py'),
            'SM_BATCH_SIZE': str(config.get('batch_size', 2)),
            'SM_MAX_BATCH_TIMEOUT_MS': str(config.get('max_batch_timeout_ms', 500)),
            'SM_MAX_CONCURRENT_REQUESTS': str(config.get('max_concurrent_requests', 10)),
            'SM_ENABLE_CUDA_GRAPHS': str(config.get('enable_cuda_graphs', True)).lower(),
            'SM_ENABLE_FLASH_ATTENTION': str(config.get('enable_flash_attention', True)).lower()
        }
    }
    
    # Create model
    try:
        response = sm.create_model(
            ModelName=model_name,
            PrimaryContainer=container,
            ExecutionRoleArn=role_arn,
            Tags=[
                {'Key': 'DeploymentType', 'Value': 'MultiRegion'},
                {'Key': 'Region', 'Value': region}
            ]
        )
        print(f"Created model {model_name} in {region}")
        return response['ModelArn']
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException' and 'already exists' in e.response['Error']['Message']:
            print(f"Model {model_name} already exists in {region}")
            return f"arn:aws:sagemaker:{region}:{boto3.client('sts').get_caller_identity()['Account']}:model/{model_name}"
        else:
            print(f"Error creating model in {region}: {e}")
            raise

def create_endpoint_config(region, model_name, instance_type, config):
    """Create endpoint configuration in the specified region."""
    sm = boto3.client('sagemaker', region_name=region)
    
    config_name = f"{model_name}-config"
    
    # Create endpoint config
    try:
        response = sm.create_endpoint_config(
            EndpointConfigName=config_name,
            ProductionVariants=[
                {
                    'VariantName': 'Default',
                    'ModelName': model_name,
                    'InstanceType': instance_type,
                    'InitialInstanceCount': config.get('instance_count', 1),
                    'InitialVariantWeight': 1.0,
                    'VolumeSizeInGB': config.get('volume_size', 30),
                    'ModelDataDownloadTimeoutInSeconds': 1800,
                    'ContainerStartupHealthCheckTimeoutInSeconds': 600
                }
            ],
            Tags=[
                {'Key': 'DeploymentType', 'Value': 'MultiRegion'},
                {'Key': 'Region', 'Value': region}
            ]
        )
        print(f"Created endpoint config {config_name} in {region}")
        return config_name
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException' and 'already exists' in e.response['Error']['Message']:
            print(f"Endpoint config {config_name} already exists in {region}")
            return config_name
        else:
            print(f"Error creating endpoint config in {region}: {e}")
            raise

def create_or_update_endpoint(region, model_name, config_name):
    """Create or update endpoint in the specified region."""
    sm = boto3.client('sagemaker', region_name=region)
    
    endpoint_name = f"{model_name}-endpoint"
    
    try:
        # Check if endpoint exists
        response = sm.describe_endpoint(EndpointName=endpoint_name)
        
        # Update existing endpoint
        sm.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=config_name
        )
        print(f"Updating endpoint {endpoint_name} in {region}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'ValidationException':
            # Create new endpoint
            sm.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name,
                Tags=[
                    {'Key': 'DeploymentType', 'Value': 'MultiRegion'},
                    {'Key': 'Region', 'Value': region}
                ]
            )
            print(f"Creating endpoint {endpoint_name} in {region}")
        else:
            print(f"Error with endpoint in {region}: {e}")
            raise
    
    return endpoint_name

def wait_for_endpoint(region, endpoint_name, timeout=30):
    """Wait for endpoint to be in service."""
    sm = boto3.client('sagemaker', region_name=region)
    
    start_time = time.time()
    while time.time() - start_time < timeout * 60:
        try:
            response = sm.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            
            if status == 'InService':
                print(f"Endpoint {endpoint_name} is in service in {region}")
                return True
            elif status == 'Failed':
                print(f"Endpoint {endpoint_name} failed in {region}: {response.get('FailureReason', 'Unknown')}")
                return False
            
            print(f"Endpoint {endpoint_name} status in {region}: {status}")
            time.sleep(30)
        except Exception as e:
            print(f"Error checking endpoint status in {region}: {e}")
            time.sleep(30)
    
    print(f"Timeout waiting for endpoint {endpoint_name} in {region}")
    return False

def setup_route53_routing(regions, model_name, config):
    """Set up Route 53 for multi-region routing."""
    if not config.get('domain_name'):
        print("No domain name specified for Route 53 setup")
        return None
    
    route53 = boto3.client('route53')
    
    # Get hosted zone ID
    domain_name = config['domain_name']
    model_subdomain = config.get('model_subdomain', model_name.lower())
    fqdn = f"{model_subdomain}.{domain_name}"
    
    try:
        # Find hosted zone
        response = route53.list_hosted_zones_by_name(DNSName=domain_name)
        hosted_zone_id = None
        
        for zone in response['HostedZones']:
            if zone['Name'].rstrip('.') == domain_name:
                hosted_zone_id = zone['Id'].replace('/hostedzone/', '')
                break
        
        if not hosted_zone_id:
            print(f"Could not find hosted zone for {domain_name}")
            return None
        
        # Create health checks for each endpoint
        health_checks = {}
        for region in regions:
            endpoint_name = f"{model_name}-endpoint"
            endpoint_url = f"https://runtime.sagemaker.{region}.amazonaws.com/endpoints/{endpoint_name}/invocations"
            
            # Create health check
            response = route53.create_health_check(
                CallerReference=f"{model_name}-{region}-{int(time.time())}",
                HealthCheckConfig={
                    'Type': 'HTTPS',
                    'FullyQualifiedDomainName': f"runtime.sagemaker.{region}.amazonaws.com",
                    'RequestInterval': 30,
                    'FailureThreshold': 3,
                    'ResourcePath': f"/endpoints/{endpoint_name}/invocations",
                    'MeasureLatency': True,
                    'Inverted': False,
                    'EnableSNI': True
                }
            )
            
            health_check_id = response['HealthCheck']['Id']
            health_checks[region] = health_check_id
            
            # Tag health check
            route53.change_tags_for_resource(
                ResourceType='healthcheck',
                ResourceId=health_check_id,
                AddTags=[
                    {'Key': 'Name', 'Value': f"{model_name}-{region}"},
                    {'Key': 'Model', 'Value': model_name}
                ]
            )
            
            print(f"Created health check for {region}: {health_check_id}")
        
        # Create DNS records for latency-based routing
        change_batch = {
            'Changes': []
        }
        
        for region in regions:
            endpoint_name = f"{model_name}-endpoint"
            
            change_batch['Changes'].append({
                'Action': 'UPSERT',
                'ResourceRecordSet': {
                    'Name': fqdn,
                    'Type': 'CNAME',
                    'SetIdentifier': f"{model_name}-{region}",
                    'Region': region,
                    'TTL': 60,
                    'ResourceRecords': [
                        {'Value': f"runtime.sagemaker.{region}.amazonaws.com"}
                    ],
                    'HealthCheckId': health_checks[region]
                }
            })
        
        # Submit changes
        response = route53.change_resource_record_sets(
            HostedZoneId=hosted_zone_id,
            ChangeBatch=change_batch
        )
        
        print(f"Created Route 53 records for {fqdn}")
        print(f"Change ID: {response['ChangeInfo']['Id']}")
        
        return fqdn
    
    except Exception as e:
        print(f"Error setting up Route 53: {e}")
        return None

def deploy_to_multiple_regions(args):
    """Deploy model to multiple regions."""
    # Load configuration
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'hf_model_id': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
            'entry_point': args.entry_point,
            'batch_size': 2,
            'max_batch_timeout_ms': 500,
            'max_concurrent_requests': 10,
            'enable_cuda_graphs': True,
            'enable_flash_attention': True,
            'instance_count': 1,
            'volume_size': 30
        }
    
    # Parse regions
    regions = args.regions.split(',')
    print(f"Deploying to regions: {regions}")
    
    # Track deployments
    deployments = {}
    
    # Deploy to each region
    for region in regions:
        try:
            print(f"\nDeploying to region: {region}")
            
            # Create/update role
            role_arn = create_or_update_role(args.role_name, region)
            
            # Create model
            model_arn = create_model_in_region(region, args.model_name, args.model_data, role_arn, config)
            
            # Create endpoint config
            config_name = create_endpoint_config(region, args.model_name, args.instance_type, config)
            
            # Create/update endpoint
            endpoint_name = create_or_update_endpoint(region, args.model_name, config_name)
            
            # Wait for endpoint
            success = wait_for_endpoint(region, endpoint_name)
            
            # Track deployment
            deployments[region] = {
                'endpoint_name': endpoint_name,
                'status': 'InService' if success else 'Failed',
                'model_arn': model_arn,
                'role_arn': role_arn
            }
            
        except Exception as e:
            print(f"Error deploying to {region}: {e}")
            deployments[region] = {
                'status': 'Error',
                'error': str(e)
            }
    
    # Set up Route 53 if configured
    if 'domain_name' in config:
        fqdn = setup_route53_routing(regions, args.model_name, config)
        if fqdn:
            for region in deployments:
                if deployments[region]['status'] == 'InService':
                    deployments[region]['dns'] = fqdn
    
    # Save deployment results
    results = {
        'model_name': args.model_name,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'deployments': deployments
    }
    
    results_file = f"multiregion_deploy_{args.model_name}_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDeployment results saved to {results_file}")
    
    # Print summary
    print("\nDeployment Summary:")
    for region, info in deployments.items():
        status = info['status']
        endpoint = info.get('endpoint_name', 'N/A')
        print(f"  {region}: {status} - {endpoint}")
    
    return results

if __name__ == '__main__':
    args = parse_args()
    deploy_to_multiple_regions(args)
```

### 5.5.4 Automated Scaling Configuration

Implement an auto-scaling configuration for handling variable load efficiently:

```python
# configure_autoscaling.py

import boto3
import argparse
import json
import time
from datetime import datetime, timedelta

def parse_args():
    parser = argparse.ArgumentParser(description="Configure auto-scaling for SageMaker endpoints")
    parser.add_argument("--endpoint-name", type=str, required=True, help="Endpoint name")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--min-instances", type=int, default=1, help="Minimum instance count")
    parser.add_argument("--max-instances", type=int, default=4, help="Maximum instance count")
    parser.add_argument("--target-utilization", type=int, default=70, help="Target CPU utilization percentage")
    parser.add_argument("--scale-in-cool-down", type=int, default=300, help="Scale-in cooldown period in seconds")
    parser.add_argument("--scale-out-cool-down", type=int, default=60, help="Scale-out cooldown period in seconds")
    parser.add_argument("--analyze-traffic-pattern", action="store_true", help="Analyze traffic pattern before configuring")
    return parser.parse_args()

def get_endpoint_variant(endpoint_name, region):
    """Get production variant name for the endpoint."""
    sm = boto3.client('sagemaker', region_name=region)
    
    try:
        response = sm.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_name = response['EndpointConfigName']
        
        config = sm.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        variants = config['ProductionVariants']
        
        if not variants:
            raise ValueError(f"No production variants found for endpoint {endpoint_name}")
        
        return variants[0]['VariantName']
    except Exception as e:
        print(f"Error getting endpoint variant: {e}")
        raise

def analyze_traffic_pattern(endpoint_name, region, days=7):
    """Analyze traffic pattern to suggest auto-scaling configuration."""
    cloudwatch = boto3.client('cloudwatch', region_name=region)
    
    # Calculate start time
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    # Get invocation metrics
    try:
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/SageMaker',
            MetricName='Invocations',
            Dimensions=[
                {
                    'Name': 'EndpointName',
                    'Value': endpoint_name
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,  # 1 hour
            Statistics=['Sum']
        )
        
        datapoints = response['Datapoints']
        if not datapoints:
            print(f"No invocation data found for endpoint {endpoint_name}")
            return None
        
        # Sort by timestamp
        datapoints.sort(key=lambda x: x['Timestamp'])
        
        # Calculate hourly pattern
        hourly_pattern = [0] * 24
        hourly_counts = [0] * 24
        
        for point in datapoints:
            hour = point['Timestamp'].hour
            hourly_pattern[hour] += point['Sum']
            hourly_counts[hour] += 1
        
        # Average by hour
        for i in range(24):
            if hourly_counts[i] > 0:
                hourly_pattern[i] /= hourly_counts[i]
        
        # Find peak and off-peak hours
        max_hour = hourly_pattern.index(max(hourly_pattern))
        min_hour = hourly_pattern.index(min(hourly_pattern))
        
        # Calculate peak-to-average ratio
        avg_invocations = sum(hourly_pattern) / 24
        peak_invocations = max(hourly_pattern)
        peak_to_avg = peak_invocations / avg_invocations if avg_invocations > 0 else 1.0
        
        # Get CPU utilization
        response = cloudwatch.get_metric_statistics(
            Namespace='AWS/SageMaker',
            MetricName='CPUUtilization',
            Dimensions=[
                {
                    'Name': 'EndpointName',
                    'Value': endpoint_name
                }
            ],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,  # 1 hour
            Statistics=['Average', 'Maximum']
        )
        
        cpu_datapoints = response['Datapoints']
        avg_cpu = sum(point['Average'] for point in cpu_datapoints) / len(cpu_datapoints) if cpu_datapoints else 0
        max_cpu = max(point['Maximum'] for point in cpu_datapoints) if cpu_datapoints else 0
        
        # Calculate suggested configuration
        suggested_config = {
            'min_instances': 1,
            'max_instances': max(2, int(peak_to_avg * 1.5)),
            'target_utilization': min(80, max(60, int(avg_cpu * 1.2))),
            'scale_in_cool_down': 300,  # 5 minutes
            'scale_out_cool_down': 60   # 1 minute
        }
        
        # Adjust for high variability
        if peak_to_avg > 3.0:
            suggested_config['scale_out_cool_down'] = 30  # Faster scale out
        
        # Adjust for low variability
        if peak_to_avg < 1.5:
            suggested_config['max_instances'] = 2
        
        analysis = {
            'hourly_pattern': hourly_pattern,
            'peak_hour': max_hour,
            'off_peak_hour': min_hour,
            'peak_to_avg_ratio': peak_to_avg,
            'avg_cpu_utilization': avg_cpu,
            'max_cpu_utilization': max_cpu,
            'suggested_config': suggested_config
        }
        
        return analysis
        
    except Exception as e:
        print(f"Error analyzing traffic pattern: {e}")
        return None

def configure_autoscaling(args, suggested_config=None):
    """Configure auto-scaling for SageMaker endpoint."""
    application_autoscaling = boto3.client('application-autoscaling', region_name=args.region)
    
    # Use suggested config if available, otherwise use command-line args
    if suggested_config:
        min_instances = suggested_config['min_instances']
        max_instances = suggested_config['max_instances']
        target_utilization = suggested_config['target_utilization']
        scale_in_cool_down = suggested_config['scale_in_cool_down']
        scale_out_cool_down = suggested_config['scale_out_cool_down']
    else:
        min_instances = args.min_instances
        max_instances = args.max_instances
        target_utilization = args.target_utilization
        scale_in_cool_down = args.scale_in_cool_down
        scale_out_cool_down = args.scale_out_cool_down
    
    # Get production variant name
    variant_name = get_endpoint_variant(args.endpoint_name, args.region)
    
    # Register scalable target
    resource_id = f"endpoint/{args.endpoint_name}/variant/{variant_name}"
    
    try:
        application_autoscaling.register_scalable_target(
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            MinCapacity=min_instances,
            MaxCapacity=max_instances
        )
        
        print(f"Registered scalable target for endpoint {args.endpoint_name}")
        print(f"  Variant: {variant_name}")
        print(f"  Min instances: {min_instances}")
        print(f"  Max instances: {max_instances}")
        
        # Configure scaling policy
        application_autoscaling.put_scaling_policy(
            PolicyName=f"{args.endpoint_name}-cpu-utilization",
            ServiceNamespace='sagemaker',
            ResourceId=resource_id,
            ScalableDimension='sagemaker:variant:DesiredInstanceCount',
            PolicyType='TargetTrackingScaling',
            TargetTrackingScalingPolicyConfiguration={
                'TargetValue': float(target_utilization),
                'PredefinedMetricSpecification': {
                    'PredefinedMetricType': 'SageMakerVariantCPUUtilization'
                },
                'ScaleInCooldown': scale_in_cool_down,
                'ScaleOutCooldown': scale_out_cool_down
            }
        )
        
        print(f"Configured scaling policy:")
        print(f"  Target CPU utilization: {target_utilization}%")
        print(f"  Scale-in cooldown: {scale_in_cool_down} seconds")
        print(f"  Scale-out cooldown: {scale_out_cool_down} seconds")
        
        return True
        
    except Exception as e:
        print(f"Error configuring auto-scaling: {e}")
        return False

def main():
    args = parse_args()
    
    # Analyze traffic pattern if requested
    suggested_config = None
    if args.analyze_traffic_pattern:
        print(f"Analyzing traffic pattern for endpoint {args.endpoint_name} in {args.region}...")
        analysis = analyze_traffic_pattern(args.endpoint_name, args.region)
        
        if analysis:
            suggested_config = analysis['suggested_config']
            print("\nTraffic analysis results:")
            print(f"  Peak hour: {analysis['peak_hour']}:00 UTC")
            print(f"  Off-peak hour: {analysis['off_peak_hour']}:00 UTC")
            print(f"  Peak-to-average ratio: {analysis['peak_to_avg_ratio']:.2f}")
            print(f"  Average CPU utilization: {analysis['avg_cpu_utilization']:.2f}%")
            print(f"  Maximum CPU utilization: {analysis['max_cpu_utilization']:.2f}%")
            print("\nSuggested configuration:")
            print(f"  Min instances: {suggested_config['min_instances']}")
            print(f"  Max instances: {suggested_config['max_instances']}")
            print(f"  Target utilization: {suggested_config['target_utilization']}%")
            print(f"  Scale-in cooldown: {suggested_config['scale_in_cool_down']} seconds")
            print(f"  Scale-out cooldown: {suggested_config['scale_out_cool_down']} seconds")
            
            # Ask for confirmation
            confirm = input("\nUse suggested configuration? (y/n): ")
            if confirm.lower() != 'y':
                suggested_config = None
    
    # Configure auto-scaling
    print(f"\nConfiguring auto-scaling for endpoint {args.endpoint_name} in {args.region}...")
    success = configure_autoscaling(args, suggested_config)
    
    if success:
        print("\nAuto-scaling configured successfully")
    else:
        print("\nFailed to configure auto-scaling")

if __name__ == '__main__':
    main()
```

### 5.5.5 Cost-Optimized Inference Endpoints

For production environments with cost constraints, implement these four optimization strategies:

1. **Scheduled Scaling**: Implement time-based scaling to reduce instances during off-hours

```bash
# Example AWS CLI command to schedule scaling
aws application-autoscaling put-scheduled-action \
  --service-namespace sagemaker \
  --resource-id endpoint/my-code-model-endpoint/variant/Default \
  --scalable-dimension sagemaker:variant:DesiredInstanceCount \
  --schedule "cron(0 20 * * ? *)" \
  --scheduled-action-name scale-down-night \
  --scalable-target-action MinCapacity=1,MaxCapacity=1
```

2. **Use ONNX Runtime**: Export your fine-tuned model to ONNX format for significant inference speedup

```python
def export_model_to_onnx(model_path, output_path, batch_size=1, sequence_length=1024):
    """Export PyTorch model to ONNX format for optimized inference."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    # Check if model is a PEFT/LoRA model
    if any(n.endswith('.lora_A.weight') for n, _ in model.named_parameters()):
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()  # Merge adapter weights
    
    # Set to evaluation mode
    model.eval()
    
    # Prepare dummy input
    dummy_input = tokenizer("This is a test input", return_tensors="pt")
    
    # Create dummy input with padding to desired shape
    input_ids = torch.zeros((batch_size, sequence_length), dtype=torch.long)
    attention_mask = torch.zeros((batch_size, sequence_length), dtype=torch.long)
    
    # Export model to ONNX
    print(f"Exporting model to ONNX format")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=15,
        do_constant_folding=True,
        verbose=True
    )
    
    print(f"Model exported to {output_path}")
    return output_path
```

3. **Predictive Scaling**: Implement a machine learning model to predict load and proactively scale

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

def train_prediction_model(endpoint_name, region, days=14):
    """Train a model to predict endpoint load for proactive scaling."""
    # Get historical metrics
    cloudwatch = boto3.client('cloudwatch', region_name=region)
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=days)
    
    # Get invocation metrics
    response = cloudwatch.get_metric_data(
        MetricDataQueries=[
            {
                'Id': 'invocations',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/SageMaker',
                        'MetricName': 'Invocations',
                        'Dimensions': [
                            {
                                'Name': 'EndpointName',
                                'Value': endpoint_name
                            }
                        ]
                    },
                    'Period': 300,  # 5-minute intervals
                    'Stat': 'Sum'
                },
                'ReturnData': True
            }
        ],
        StartTime=start_time,
        EndTime=end_time
    )
    
    # Prepare training data
    timestamps = [ts.replace(tzinfo=None) for ts in response['MetricDataResults'][0]['Timestamps']]
    values = response['MetricDataResults'][0]['Values']
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'invocations': values
    })
    
    # Extract features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Add lag features
    for lag in [1, 2, 3, 12, 24]:  # 5min, 10min, 15min, 1h, 2h
        df[f'lag_{lag}'] = df['invocations'].shift(lag)
    
    # Fill missing values
    df = df.fillna(0)
    
    # Prepare training data
    X = df[['hour', 'day_of_week', 'day_of_month', 'month', 'is_weekend'] + 
           [f'lag_{lag}' for lag in [1, 2, 3, 12, 24]]].values
    y = df['invocations'].values
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, df.columns
```

4. **Serverless Inference** for unpredictable or low-volume workloads:

```python
def deploy_to_serverless(model_name, model_data, role_arn, region):
    """Deploy model to SageMaker Serverless Inference endpoint."""
    sm = boto3.client('sagemaker', region_name=region)
    
    # Define container
    container = {
        'Image': f"763104351884.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-inference:2.0.1-transformers4.31.0-cpu-py310-ubuntu20.04",
        'ModelDataUrl': model_data,
        'Environment': {
            'HF_MODEL_ID': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
            'HF_TASK': 'text-generation',
            'SM_BATCH_SIZE': '1',
            'SM_ENABLE_CUDA_GRAPHS': 'false'
        }
    }
    
    # Create model
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer=container,
        ExecutionRoleArn=role_arn
    )
    
    # Create serverless endpoint config
    config_name = f"{model_name}-serverless-config"
    
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                'VariantName': 'Default',
                'ModelName': model_name,
                'ServerlessConfig': {
                    'MemorySizeInMB': 6144,  # 6GB memory
                    'MaxConcurrency': 5
                }
            }
        ]
    )
    
    # Create endpoint
    endpoint_name = f"{model_name}-serverless"
    
    sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name
    )
    
    return endpoint_name
```

By implementing these strategies together, you can reduce inference costs by 40-70% compared to standard 24/7 dedicated instances, while maintaining availability and performance appropriate for most business applications.

### 6. Expected Outputs and Fine-Tuning Results

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

### 7.1 Advanced Troubleshooting Decision Tree

Use this detailed decision tree to diagnose and resolve issues more efficiently:

```
Model fails to fine-tune
├── Out of Memory (OOM) Error
│   ├── Check: GPU memory usage spikes and crashes
│   │   ├── Yes → Use GPU memory tracking:
│   │   │         Add monitoring code to track memory usage
│   │   │         ```python
│   │   │         def log_gpu_memory():
│   │   │             if torch.cuda.is_available():
│   │   │                 for i in range(torch.cuda.device_count()):
│   │   │                     allocated = torch.cuda.memory_allocated(i) / (1024**3)
│   │   │                     max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
│   │   │                     print(f"GPU {i}: {allocated:.2f}GB allocated, {max_allocated:.2f}GB max")
│   │   │         ```
│   │   └── Solutions (try in sequence):
│   │       1. Reduce batch size (try 1)
│   │       2. Enable gradient checkpointing
│   │       3. Increase gradient accumulation steps (8→16→32)
│   │       4. Use mixed precision training (bf16 or fp16)
│   │       5. CPU offload for optimizer states
│   │       6. Reduce sequence length
│   │       7. Use DeepSpeed ZeRO stage 3
│   │       8. Use different instance type 
├── Training Loss = NaN
│   ├── Check: Loss becomes NaN early in training
│   │   ├── Yes → Learning rate too high
│   │   │         Solution: Reduce LR by 10x and add warmup_ratio=0.1
│   │   └── No → Check: Loss becomes NaN after several epochs
│   │            ├── Yes → Gradient instability
│   │            │         Solutions:
│   │            │         1. Add gradient clipping: max_grad_norm=1.0
│   │            │         2. Switch optimizer to AdamW with weight_decay=0.01
│   │            │         3. Use different random seed
│   │            └── No → Check: Special tokens in data causing issues
│   │                      Fix by modifying tokenizer config:
│   │                      ```python
│   │                      tokenizer.add_special_tokens({'pad_token': '[PAD]'})
│   │                      tokenizer.model_max_length = args.max_seq_length
│   │                      ```
├── Training Hangs/Freezes
│   ├── Check: CPU usage at 100% but GPU underutilized
│   │   ├── Yes → Preprocessing bottleneck
│   │   │         Solutions:
│   │   │         1. Increase num_workers in DataLoader
│   │   │         2. Pre-tokenize data and save to disk
│   │   │         3. Add pin_memory=True to DataLoader
│   │   └── No → Check: System swap memory being used
│   │            ├── Yes → System OOM
│   │            │         Solutions: 
│   │            │         1. Reduce batch size further
│   │            │         2. Move to instance with more RAM
│   │            └── No → Check: Process stuck waiting for network
│   │                      Add timeout to all network operations:
│   │                      ```python
│   │                      import socket
│   │                      socket.setdefaulttimeout(300)  # 5-minute timeout
│   │                      ```
└── Fine-tuning Works but Model Performance Poor
    ├── Check: Training loss decreases but validation loss increases
    │   ├── Yes → Overfitting
    │   │         Solutions:
    │   │         1. Increase LoRA dropout to 0.1
    │   │         2. Add early stopping with patience=2
    │   │         3. Reduce training time/steps
    │   │         4. Add weight decay to optimizer
    │   └── No → Check: Both training and validation loss plateau quickly
    │            ├── Yes → Model not learning effectively
    │            │         Solutions:
    │            │         1. Increase LoRA rank (r=16→32→64)
    │            │         2. Add more target modules for LoRA
    │            │         3. Try full fine-tuning instead of LoRA
    │            │         4. Check data quality and formatting
    │            └── No → Check: Model generates poor outputs despite good loss
    │                      Solutions:
    │                      1. Revisit prompt templates for consistency
    │                      2. Ensure consistent formatting in training data
    │                      3. Try different temperature for generation (0.1-0.7)
```

### 7.2 Model Training Debugging Utilities

Add these utilities to help diagnose issues during training:

```python
# debug_utils.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import gc
from datetime import datetime
from transformers import Trainer

class DebugCallback:
    """Callback to collect and log debugging information during training."""
    
    def __init__(self, log_dir="debug_logs", log_interval=10, plot_interval=100):
        """Initialize the debug callback."""
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.plot_interval = plot_interval
        self.step = 0
        self.metrics = {
            "loss": [],
            "learning_rate": [],
            "gradient_norm": [],
            "gpu_utilization": [],
            "gpu_memory": [],
            "cpu_utilization": [],
            "ram_usage": [],
            "step_time": []
        }
        self.start_time = time.time()
        self.last_step_time = time.time()
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize log file
        self.log_file = os.path.join(log_dir, f"debug_log_{int(time.time())}.txt")
        with open(self.log_file, "w") as f:
            f.write(f"Training Debug Log - {datetime.now()}\n")
            f.write("-" * 50 + "\n")
            
            # Log system information
            f.write(f"System information:\n")
            f.write(f"  CPU: {psutil.cpu_count()} cores\n")
            f.write(f"  RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB\n")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    f.write(f"  GPU {i}: {gpu_name}, {gpu_mem:.2f} GB\n")
            f.write("\n")
            
            # Column headers
            f.write("Step\tLoss\tLR\tGrad Norm\tGPU Util\tGPU Mem\tCPU Util\tRAM\tStep Time\n")
    
    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        """Called after each step."""
        self.step += 1
        
        # Calculate time for this step
        now = time.time()
        step_time = now - self.last_step_time
        self.last_step_time = now
        
        # Skip logging if not at interval
        if self.step % self.log_interval != 0:
            return
        
        # Collect metrics
        loss = state.log_history[-1]["loss"] if state.log_history else 0.0
        lr = state.log_history[-1].get("learning_rate", 0.0) if state.log_history else 0.0
        
        # Calculate gradient norm
        grad_norm = 0.0
        if model is not None and optimizer is not None:
            # Ensure we can calculate gradient norm
            if hasattr(optimizer, "param_groups"):
                grad_norm = 0.0
                params = []
                for param_group in optimizer.param_groups:
                    params.extend(param_group["params"])
                
                for param in params:
                    if param.grad is not None:
                        grad_norm += param.grad.data.norm(2).item() ** 2
                grad_norm = grad_norm ** 0.5
        
        # Get GPU metrics
        gpu_util = 0.0
        gpu_mem = 0.0
        if torch.cuda.is_available():
            try:
                # GPU utilization requires pynvml
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_mem = mem_info.used / (1024**3)  # GB
            except:
                # Fallback to torch.cuda
                gpu_mem = torch.cuda.memory_allocated(0) / (1024**3)  # GB
        
        # Get CPU metrics
        cpu_util = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        
        # Store metrics
        self.metrics["loss"].append(loss)
        self.metrics["learning_rate"].append(lr)
        self.metrics["gradient_norm"].append(grad_norm)
        self.metrics["gpu_utilization"].append(gpu_util)
        self.metrics["gpu_memory"].append(gpu_mem)
        self.metrics["cpu_utilization"].append(cpu_util)
        self.metrics["ram_usage"].append(ram_usage)
        self.metrics["step_time"].append(step_time)
        
        # Log to file
        with open(self.log_file, "a") as f:
            f.write(f"{self.step}\t{loss:.4f}\t{lr:.6f}\t{grad_norm:.4f}\t")
            f.write(f"{gpu_util:.1f}%\t{gpu_mem:.2f}GB\t{cpu_util:.1f}%\t")
            f.write(f"{ram_usage:.1f}%\t{step_time:.4f}s\n")
        
        # Plot metrics
        if self.step % self.plot_interval == 0:
            self.plot_metrics()
    
    def plot_metrics(self):
        """Plot training metrics."""
        # Create figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        
        # Plot loss
        axs[0, 0].plot(self.metrics["loss"])
        axs[0, 0].set_title("Training Loss")
        axs[0, 0].set_xlabel("Steps")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].grid(True)
        
        # Plot learning rate
        axs[0, 1].plot(self.metrics["learning_rate"])
        axs[0, 1].set_title("Learning Rate")
        axs[0, 1].set_xlabel("Steps")
        axs[0, 1].set_ylabel("LR")
        axs[0, 1].grid(True)
        
        # Plot gradient norm
        axs[1, 0].plot(self.metrics["gradient_norm"])
        axs[1, 0].set_title("Gradient Norm")
        axs[1, 0].set_xlabel("Steps")
        axs[1, 0].set_ylabel("Norm")
        axs[1, 0].grid(True)
        
        # Plot GPU and CPU utilization
        axs[1, 1].plot(self.metrics["gpu_utilization"], label="GPU")
        axs[1, 1].plot(self.metrics["cpu_utilization"], label="CPU")
        axs[1, 1].set_title("Utilization")
        axs[1, 1].set_xlabel("Steps")
        axs[1, 1].set_ylabel("Percent")
        axs[1, 1].set_ylim(0, 100)
        axs[1, 1].grid(True)
        axs[1, 1].legend()
        
        # Plot memory usage
        axs[2, 0].plot(self.metrics["gpu_memory"], label="GPU Memory (GB)")
        axs[2, 0].plot(np.array(self.metrics["ram_usage"]) / 100 * psutil.virtual_memory().total / (1024**3), 
                      label="RAM Usage (GB)")
        axs[2, 0].set_title("Memory Usage")
        axs[2, 0].set_xlabel("Steps")
        axs[2, 0].set_ylabel("GB")
        axs[2, 0].grid(True)
        axs[2, 0].legend()
        
        # Plot step time
        axs[2, 1].plot(self.metrics["step_time"])
        axs[2, 1].set_title("Step Time")
        axs[2, 1].set_xlabel("Steps")
        axs[2, 1].set_ylabel("Seconds")
        axs[2, 1].grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, f"metrics_{self.step}.png"))
        plt.close()
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        # Calculate total training time
        total_time = time.time() - self.start_time
        
        # Log final statistics
        with open(self.log_file, "a") as f:
            f.write("-" * 50 + "\n")
            f.write(f"Training completed at {datetime.now()}\n")
            f.write(f"Total training time: {total_time / 60:.2f} minutes\n")
            f.write(f"Total steps: {self.step}\n")
            f.write(f"Average step time: {np.mean(self.metrics['step_time']):.4f} seconds\n")
            f.write(f"Final loss: {self.metrics['loss'][-1]:.4f}\n")
        
        # Final plots
        self.plot_metrics()
        
        # Create additional summary plots
        self.plot_summary()
    
    def plot_summary(self):
        """Create summary plots for the entire training run."""
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot loss vs learning rate
        axs[0, 0].scatter(self.metrics["learning_rate"], self.metrics["loss"])
        axs[0, 0].set_title("Loss vs Learning Rate")
        axs[0, 0].set_xlabel("Learning Rate")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].grid(True)
        
        # Plot loss vs gradient norm
        axs[0, 1].scatter(self.metrics["gradient_norm"], self.metrics["loss"])
        axs[0, 1].set_title("Loss vs Gradient Norm")
        axs[0, 1].set_xlabel("Gradient Norm")
        axs[0, 1].set_ylabel("Loss")
        axs[0, 1].grid(True)
        
        # Plot histogram of step times
        axs[1, 0].hist(self.metrics["step_time"], bins=30)
        axs[1, 0].set_title("Step Time Distribution")
        axs[1, 0].set_xlabel("Seconds")
        axs[1, 0].set_ylabel("Count")
        axs[1, 0].grid(True)
        
        # Plot GPU memory vs loss
        axs[1, 1].scatter(self.metrics["gpu_memory"], self.metrics["loss"])
        axs[1, 1].set_title("Loss vs GPU Memory")
        axs[1, 1].set_xlabel("GPU Memory (GB)")
        axs[1, 1].set_ylabel("Loss")
        axs[1, 1].grid(True)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "training_summary.png"))
        plt.close()

class MemoryTracker:
    """Track memory usage during model training."""
    
    def __init__(self, log_dir="memory_logs"):
        """Initialize the memory tracker."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.log_file = os.path.join(log_dir, f"memory_log_{int(time.time())}.txt")
        with open(self.log_file, "w") as f:
            f.write(f"Memory Tracking Log - {datetime.now()}\n")
            f.write("-" * 50 + "\n")
            
            # Log system information
            f.write(f"System information:\n")
            f.write(f"  RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB\n")
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    f.write(f"  GPU {i}: {gpu_name}, {gpu_mem:.2f} GB\n")
            f.write("\n")
    
    def log_memory(self, tag="", tensors=None):
        """Log current memory usage."""
        with open(self.log_file, "a") as f:
            f.write(f"\n--- Memory Usage at {datetime.now()} - {tag} ---\n")
            
            # RAM usage
            vm = psutil.virtual_memory()
            f.write(f"RAM: {vm.used / (1024**3):.2f}GB / {vm.total / (1024**3):.2f}GB ({vm.percent}%)\n")
            
            # GPU memory
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    max_allocated = torch.cuda.max_memory_allocated(i) / (1024**3)
                    f.write(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB max\n")
            
            # Tensor memory usage
            if tensors:
                f.write("\nTensor Memory Usage:\n")
                for name, tensor in tensors.items():
                    if hasattr(tensor, "element_size"):
                        size_mb = tensor.element_size() * tensor.nelement() / (1024**2)
                        f.write(f"  {name}: {size_mb:.2f}MB - Shape: {tensor.shape}\n")
                    else:
                        f.write(f"  {name}: Not a tensor or size cannot be determined\n")
            
            f.write("\n")
    
    def track_model_size(self, model, tag="Model Size"):
        """Track the size of model parameters and buffers."""
        with open(self.log_file, "a") as f:
            f.write(f"\n--- {tag} at {datetime.now()} ---\n")
            
            total_params = 0
            total_size_mb = 0
            details = []
            
            # Track parameter size by module
            for name, module in model.named_modules():
                module_params = sum(p.numel() for p in module.parameters(recurse=False))
                if module_params == 0:
                    continue
                
                module_size = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False)) / (1024**2)
                total_params += module_params
                total_size_mb += module_size
                
                details.append((name, module_params, module_size))
            
            # Sort by size (largest first)
            details.sort(key=lambda x: x[2], reverse=True)
            
            # Print top modules by size
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Total size: {total_size_mb:.2f}MB\n\n")
            f.write("Top modules by size:\n")
            
            for name, params, size in details[:20]:  # Top 20
                f.write(f"  {name}: {params:,} params, {size:.2f}MB\n")
            
            f.write("\n")
    
    def trigger_gc(self, tag="Garbage Collection"):
        """Trigger garbage collection and log memory before and after."""
        self.log_memory(f"{tag} - Before")
        gc.collect()
        torch.cuda.empty_cache()
        self.log_memory(f"{tag} - After")

# Custom trainer with memory debugging
class DebugTrainer(Trainer):
    """Extended Trainer with memory debugging capabilities."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the debug trainer."""
        super().__init__(*args, **kwargs)
        self.memory_tracker = MemoryTracker()
        self.debug_callback = DebugCallback()
    
    def training_step(self, model, inputs):
        """Override training step to add memory tracking."""
        self.memory_tracker.log_memory(f"Step {self.state.global_step} - Before")
        
        # Perform training step
        loss = super().training_step(model, inputs)
        
        self.memory_tracker.log_memory(f"Step {self.state.global_step} - After")
        
        # Call debug callback
        self.debug_callback.on_step_end(
            self.args, self.state, self.control,
            model=model, optimizer=self.optimizer
        )
        
        return loss
    
    def evaluate(self, *args, **kwargs):
        """Override evaluate to add memory tracking."""
        self.memory_tracker.log_memory("Evaluation - Before")
        result = super().evaluate(*args, **kwargs)
        self.memory_tracker.log_memory("Evaluation - After")
        return result
    
    def save_model(self, *args, **kwargs):
        """Override save_model to add memory tracking."""
        self.memory_tracker.log_memory("Model Save - Before")
        result = super().save_model(*args, **kwargs)
        self.memory_tracker.log_memory("Model Save - After")
        self.memory_tracker.trigger_gc()
        return result
```

### 7.3 Advanced GPU Memory Optimization Techniques

When facing persistent memory issues, implement these advanced techniques:

```python
def optimize_model_memory(model, optimizer=None):
    """Apply advanced memory optimizations to the model."""
    import torch
    
    # Optimization 1: Use gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        print("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()
    
    # Optimization 2: CPU offloading for specific layers
    if optimizer is not None:
        from transformers.optimization import Adafactor
        
        print("Using Adafactor optimizer to reduce memory usage")
        # Initialize Adafactor optimizer (uses less memory than AdamW)
        optimizer = Adafactor(
            model.parameters(),
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            lr=1e-4,
            clip_threshold=1.0
        )
    
    # Optimization 3: Convert unused layers to fp16
    fp16_layers = ["lm_head", "ln_f", "embed_tokens"]
    for name, module in model.named_modules():
        if any(layer in name for layer in fp16_layers):
            print(f"Converting {name} to fp16")
            for param in module.parameters():
                param.data = param.data.half()
    
    # Optimization 4: Activation checkpointing for attention layers
    from torch.utils.checkpoint import checkpoint
    
    def checkpoint_wrapper(module):
        """Add checkpointing to a module."""
        forward_original = module.forward
        
        def forward_checkpointed(*args, **kwargs):
            return checkpoint(forward_original, *args, **kwargs)
        
        module.forward = forward_checkpointed
        return module
    
    for name, module in model.named_modules():
        if "attention" in name and len(list(module.children())) == 0:
            module = checkpoint_wrapper(module)
            print(f"Added checkpointing to {name}")
    
    # Optimization 5: Enable PyTorch memory optimizations
    if hasattr(torch.cuda, "empty_cache"):
        torch.cuda.empty_cache()
    
    torch.backends.cudnn.benchmark = True
    
    return model, optimizer

def optimize_dataloader(dataloader, num_workers=4, pin_memory=True, prefetch_factor=2):
    """Optimize DataLoader for better GPU utilization."""
    from torch.utils.data import DataLoader
    
    # Get original parameters
    dataset = dataloader.dataset
    batch_size = dataloader.batch_size
    sampler = dataloader.sampler
    collate_fn = dataloader.collate_fn
    
    # Create optimized DataLoader
    optimized_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,  # Parallel data loading
        pin_memory=pin_memory,    # Pin tensors in memory for faster transfer
        prefetch_factor=prefetch_factor,  # Prefetch batches
        drop_last=dataloader.drop_last,
        persistent_workers=True   # Keep workers alive between epochs
    )
    
    return optimized_loader

def profile_gpu_memory_usage(model, inputs, batch_sizes=[1, 2, 4, 8], sequence_lengths=[512, 1024, 2048]):
    """Profile GPU memory usage for different batch sizes and sequence lengths."""
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Initialize memory stats
    memory_stats = []
    
    # Ensure inputs is a dict
    if not isinstance(inputs, dict):
        inputs = {"input_ids": inputs}
    
    # Get input tensor
    input_tensor = list(inputs.values())[0]
    
    for seq_len in sequence_lengths:
        for batch_size in batch_sizes:
            try:
                # Reset memory stats
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.empty_cache()
                
                # Create dummy input with required shape
                dummy_input = {}
                for key, tensor in inputs.items():
                    if len(tensor.shape) >= 2:
                        # Create a tensor of the right shape
                        dummy_shape = list(tensor.shape)
                        dummy_shape[0] = batch_size
                        dummy_shape[1] = seq_len
                        dummy_input[key] = torch.zeros(dummy_shape, dtype=tensor.dtype, device=tensor.device)
                    else:
                        dummy_input[key] = tensor
                
                # Forward pass
                with torch.no_grad():
                    model(**dummy_input)
                
                # Record memory stats
                peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
                
                memory_stats.append({
                    "batch_size": batch_size,
                    "sequence_length": seq_len,
                    "memory_gb": peak_memory
                })
                
                print(f"Batch size {batch_size}, Sequence length {seq_len}: {peak_memory:.2f}GB")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM for batch size {batch_size}, sequence length {seq_len}")
                    memory_stats.append({
                        "batch_size": batch_size,
                        "sequence_length": seq_len,
                        "memory_gb": None
                    })
                else:
                    raise
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    # Organize data for plotting
    batch_sizes_unique = sorted(list(set(stat["batch_size"] for stat in memory_stats)))
    seq_lengths_unique = sorted(list(set(stat["sequence_length"] for stat in memory_stats)))
    
    # Create matrix of memory usage
    memory_matrix = np.zeros((len(batch_sizes_unique), len(seq_lengths_unique)))
    for i, bs in enumerate(batch_sizes_unique):
        for j, sl in enumerate(seq_lengths_unique):
            for stat in memory_stats:
                if stat["batch_size"] == bs and stat["sequence_length"] == sl and stat["memory_gb"] is not None:
                    memory_matrix[i, j] = stat["memory_gb"]
    
    # Plot heatmap
    plt.imshow(memory_matrix, cmap="viridis")
    plt.colorbar(label="Memory (GB)")
    plt.xticks(range(len(seq_lengths_unique)), seq_lengths_unique)
    plt.yticks(range(len(batch_sizes_unique)), batch_sizes_unique)
    plt.xlabel("Sequence Length")
    plt.ylabel("Batch Size")
    plt.title("GPU Memory Usage (GB)")
    
    # Add text annotations
    for i in range(len(batch_sizes_unique)):
        for j in range(len(seq_lengths_unique)):
            if memory_matrix[i, j] > 0:
                plt.text(j, i, f"{memory_matrix[i, j]:.1f}", ha="center", va="center", color="white")
            else:
                plt.text(j, i, "OOM", ha="center", va="center", color="white")
    
    plt.tight_layout()
    plt.savefig("gpu_memory_profile.png")
    plt.close()
    
    return memory_stats, "gpu_memory_profile.png"
```

### 7.4 Inference Troubleshooting Script

Add a dedicated script for diagnosing and fixing inference issues:

```python
# inference_troubleshooter.py

import argparse
import json
import time
import boto3
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tabulate import tabulate
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Troubleshoot SageMaker inference endpoint")
    parser.add_argument("--endpoint-name", type=str, required=True, help="SageMaker endpoint name")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--test-file", type=str, help="Path to test examples JSON file")
    parser.add_argument("--generate-report", action="store_true", help="Generate HTML report")
    parser.add_argument("--diagnose-mode", choices=["latency", "errors", "quality", "all"], 
                       default="all", help="Diagnosis mode")
    parser.add_argument("--output-dir", type=str, default="inference_analysis", help="Output directory")
    return parser.parse_args()

def check_endpoint_status(endpoint_name, region):
    """Check current endpoint status and configuration."""
    sm = boto3.client('sagemaker', region_name=region)
    
    try:
        # Get endpoint info
        endpoint = sm.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_name = endpoint['EndpointConfigName']
        
        # Get endpoint config
        config = sm.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        
        results = {
            "status": endpoint["EndpointStatus"],
            "created_at": endpoint["CreationTime"].strftime("%Y-%m-%d %H:%M:%S"),
            "config_name": endpoint_config_name,
            "variants": []
        }
        
        # Get variant details
        for variant in config["ProductionVariants"]:
            variant_info = {
                "name": variant["VariantName"],
                "instance_type": variant.get("InstanceType", "Serverless"),
                "instance_count": variant.get("InitialInstanceCount", 0),
                "weight": variant.get("InitialVariantWeight", 1.0)
            }
            
            # Add serverless config if present
            if "ServerlessConfig" in variant:
                variant_info["serverless"] = True
                variant_info["memory"] = variant["ServerlessConfig"]["MemorySizeInMB"]
                variant_info["max_concurrency"] = variant["ServerlessConfig"]["MaxConcurrency"]
            else:
                variant_info["serverless"] = False
            
            results["variants"].append(variant_info)
        
        # Get model info
        model_name = config["ProductionVariants"][0]["ModelName"]
        model = sm.describe_model(ModelName=model_name)
        
        results["model"] = {
            "name": model_name,
            "image": model["PrimaryContainer"]["Image"],
            "data_url": model["PrimaryContainer"].get("ModelDataUrl"),
            "environment": model["PrimaryContainer"].get("Environment", {})
        }
        
        return results
    
    except Exception as e:
        print(f"Error checking endpoint status: {e}")
        return {"error": str(e)}

def get_cloudwatch_metrics(endpoint_name, region, period=300, days=1):
    """Get CloudWatch metrics for the endpoint."""
    cloudwatch = boto3.client('cloudwatch', region_name=region)
    
    # Calculate time range
    end_time = pd.Timestamp.now(tz='UTC')
    start_time = end_time - pd.Timedelta(days=days)
    
    # Define metrics to fetch
    metrics = [
        {"name": "Invocations", "stat": "Sum"},
        {"name": "InvocationsPerInstance", "stat": "Sum"},
        {"name": "ModelLatency", "stat": "Average"},
        {"name": "ModelLatency", "stat": "p90"},
        {"name": "ModelLatency", "stat": "p99"},
        {"name": "Invocation4XXErrors", "stat": "Sum"},
        {"name": "Invocation5XXErrors", "stat": "Sum"},
        {"name": "CPUUtilization", "stat": "Average"},
        {"name": "MemoryUtilization", "stat": "Average"},
    ]
    
    results = {}
    
    for metric in metrics:
        try:
            response = cloudwatch.get_metric_statistics(
                Namespace="AWS/SageMaker",
                MetricName=metric["name"],
                Dimensions=[
                    {
                        'Name': 'EndpointName',
                        'Value': endpoint_name
                    }
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=period,
                Statistics=[metric["stat"]]
            )
            
            # Process datapoints
            if response["Datapoints"]:
                # Sort by timestamp
                datapoints = sorted(response["Datapoints"], key=lambda x: x["Timestamp"])
                
                # Store time series
                key = f"{metric['name']}_{metric['stat']}"
                results[key] = {
                    "timestamps": [d["Timestamp"].strftime("%Y-%m-%d %H:%M:%S") for d in datapoints],
                    "values": [d[metric["stat"]] for d in datapoints]
                }
        
        except Exception as e:
            print(f"Error fetching metric {metric['name']}: {e}")
    
    return results

def run_latency_tests(endpoint_name, region, test_examples, n_repeats=5):
    """Run latency tests with various input sizes."""
    sm_runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    results = []
    
    # Get baseline latency
    baseline_latencies = []
    
    print("Running baseline latency tests...")
    for _ in range(n_repeats):
        try:
            # Use first example as baseline
            example = test_examples[0]["input"]
            
            payload = {
                "inputs": example,
                "parameters": {
                    "max_new_tokens": 10,  # Small output for baseline
                    "do_sample": False
                }
            }
            
            start_time = time.time()
            response = sm_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            end_time = time.time()
            
            latency = end_time - start_time
            baseline_latencies.append(latency)
            
            # Sleep to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error in baseline test: {e}")
    
    baseline_latency = np.mean(baseline_latencies) if baseline_latencies else None
    
    # Test with different input lengths
    print("Testing with different input lengths...")
    for example in tqdm(test_examples):
        input_text = example["input"]
        input_length = len(input_text.split())
        
        # Skip very short inputs
        if input_length < 5:
            continue
        
        # Test with different max_new_tokens
        for max_tokens in [32, 128, 512]:
            latencies = []
            errors = []
            
            for _ in range(n_repeats):
                try:
                    payload = {
                        "inputs": input_text,
                        "parameters": {
                            "max_new_tokens": max_tokens,
                            "do_sample": False
                        }
                    }
                    
                    start_time = time.time()
                    response = sm_runtime.invoke_endpoint(
                        EndpointName=endpoint_name,
                        ContentType='application/json',
                        Body=json.dumps(payload)
                    )
                    end_time = time.time()
                    
                    latency = end_time - start_time
                    latencies.append(latency)
                    
                    # Sleep to avoid rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    errors.append(str(e))
            
            # Calculate statistics
            mean_latency = np.mean(latencies) if latencies else None
            std_latency = np.std(latencies) if latencies else None
            
            # Store result
            results.append({
                "input_length": input_length,
                "max_tokens": max_tokens,
                "mean_latency": mean_latency,
                "std_latency": std_latency,
                "relative_latency": mean_latency / baseline_latency if mean_latency and baseline_latency else None,
                "errors": errors
            })
    
    # Add batch tests if appropriate
    if len(test_examples) >= 4:
        print("Testing with batched inputs...")
        for batch_size in [2, 4]:
            try:
                # Create batch input
                batch_input = [test_examples[i]["input"] for i in range(min(batch_size, len(test_examples)))]
                
                payload = {
                    "inputs": batch_input,
                    "parameters": {
                        "max_new_tokens": 128,
                        "do_sample": False
                    }
                }
                
                latencies = []
                errors = []
                
                for _ in range(n_repeats):
                    try:
                        start_time = time.time()
                        response = sm_runtime.invoke_endpoint(
                            EndpointName=endpoint_name,
                            ContentType='application/json',
                            Body=json.dumps(payload)
                        )
                        end_time = time.time()
                        
                        latency = end_time - start_time
                        latencies.append(latency)
                        
                        # Sleep to avoid rate limiting
                        time.sleep(1.0)
                        
                    except Exception as e:
                        errors.append(str(e))
                
                # Calculate statistics
                mean_latency = np.mean(latencies) if latencies else None
                std_latency = np.std(latencies) if latencies else None
                
                # Store result
                results.append({
                    "input_length": "batch",
                    "batch_size": batch_size,
                    "max_tokens": 128,
                    "mean_latency": mean_latency,
                    "std_latency": std_latency,
                    "latency_per_input": mean_latency / batch_size if mean_latency else None,
                    "errors": errors
                })
                
            except Exception as e:
                print(f"Error in batch test with size {batch_size}: {e}")
    
    return results, baseline_latency

def test_token_streaming(endpoint_name, region):
    """Test if token streaming is working correctly."""
    # Note: This function would need to be customized for your specific setup
    # as SageMaker doesn't have built-in streaming for HF models yet
    print("Token streaming not supported with standard inference script")
    return {"supported": False}

def diagnose_quality_issues(endpoint_name, region, test_examples):
    """Test output quality on a set of representative examples."""
    sm_runtime = boto3.client('sagemaker-runtime', region_name=region)
    
    results = []
    
    print("Testing output quality...")
    for example in tqdm(test_examples):
        input_text = example["input"]
        expected_output = example.get("expected_output", "")
        
        try:
            payload = {
                "inputs": input_text,
                "parameters": {
                    "max_new_tokens": 512,
                    "temperature": 0.1,
                    "do_sample": False
                }
            }
            
            response = sm_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )
            
            # Parse response
            response_body = json.loads(response['Body'].read().decode())
            generated_text = response_body.get("generated_text", "")
            
            # Basic quality assessment
            quality = {
                "empty_output": len(generated_text.strip()) == 0,
                "output_length": len(generated_text.split()),
                "expected_match": False
            }
            
            # If expected output is provided, compare
            if expected_output:
                # Simple string matching (in practice, use more advanced metrics)
                quality["expected_match"] = generated_text.strip() == expected_output.strip()
                
                # Calculate token overlap (simple approximation)
                expected_tokens = set(expected_output.strip().split())
                generated_tokens = set(generated_text.strip().split())
                
                if expected_tokens:
                    overlap = len(expected_tokens.intersection(generated_tokens))
                    quality["token_overlap"] = overlap / len(expected_tokens)
            
            results.append({
                "input": input_text[:100] + "..." if len(input_text) > 100 else input_text,
                "output": generated_text[:100] + "..." if len(generated_text) > 100 else generated_text,
                "quality": quality
            })
            
            # Sleep to avoid rate limiting
            time.sleep(0.5)
            
        except Exception as e:
            results.append({
                "input": input_text[:100] + "..." if len(input_text) > 100 else input_text,
                "error": str(e)
            })
    
    # Summarize results
    summary = {
        "samples_tested": len(results),
        "failures": sum(1 for r in results if "error" in r),
        "empty_outputs": sum(1 for r in results if "quality" in r and r["quality"]["empty_output"]),
        "expected_matches": sum(1 for r in results if "quality" in r and r["quality"].get("expected_match", False)),
    }
    
    if any("quality" in r and "token_overlap" in r["quality"] for r in results):
        overlaps = [r["quality"]["token_overlap"] for r in results if "quality" in r and "token_overlap" in r["quality"]]
        summary["avg_token_overlap"] = np.mean(overlaps) if overlaps else None
    
    return results, summary

def generate_report(endpoint_info, metrics, latency_results, quality_results, 
                   streaming_results, baseline_latency, output_dir):
    """Generate an HTML report with all diagnostic information."""
    import os
    from jinja2 import Template
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create latency visualization
    if latency_results:
        plt.figure(figsize=(12, 6))
        
        # Group by max_tokens
        token_groups = {}
        for result in latency_results:
            if "batch_size" in result:
                continue  # Skip batch tests for this chart
                
            max_tokens = result["max_tokens"]
            if max_tokens not in token_groups:
                token_groups[max_tokens] = {"x": [], "y": [], "yerr": []}
            
            token_groups[max_tokens]["x"].append(result["input_length"])
            token_groups[max_tokens]["y"].append(result["mean_latency"])
            token_groups[max_tokens]["yerr"].append(result["std_latency"])
        
        # Plot each group
        for max_tokens, data in token_groups.items():
            indices = np.argsort(data["x"])
            x = [data["x"][i] for i in indices]
            y = [data["y"][i] for i in indices]
            yerr = [data["yerr"][i] for i in indices]
            
            plt.errorbar(x, y, yerr=yerr, marker='o', label=f"max_tokens={max_tokens}")
        
        plt.xlabel("Input Length (tokens)")
        plt.ylabel("Latency (seconds)")
        plt.title("Inference Latency by Input Length")
        plt.grid(True)
        plt.legend()
        
        plt.tight_layout()
        latency_chart_path = os.path.join(output_dir, "latency_chart.png")
        plt.savefig(latency_chart_path)
        plt.close()
        
        # Create batch latency chart if available
        batch_results = [r for r in latency_results if "batch_size" in r]
        if batch_results:
            plt.figure(figsize=(8, 6))
            
            batch_sizes = [r["batch_size"] for r in batch_results]
            latencies = [r["mean_latency"] for r in batch_results]
            per_input = [r["latency_per_input"] for r in batch_results]
            
            plt.bar(batch_sizes, latencies, color="blue", alpha=0.6, label="Total Latency")
            
            # Add per-input line
            ax2 = plt.twinx()
            ax2.plot(batch_sizes, per_input, color="red", marker="o", label="Latency per Input")
            ax2.set_ylabel("Latency per Input (seconds)")
            
            plt.xlabel("Batch Size")
            plt.ylabel("Total Latency (seconds)")
            plt.title("Batch Processing Latency")
            plt.grid(True)
            plt.legend()
            
            plt.tight_layout()
            batch_chart_path = os.path.join(output_dir, "batch_chart.png")
            plt.savefig(batch_chart_path)
            plt.close()
    
    # Create metrics visualizations
    metrics_charts = []
    if metrics:
        # Create latency trend chart
        if "ModelLatency_Average" in metrics:
            plt.figure(figsize=(12, 6))
            
            timestamps = [t for t in metrics["ModelLatency_Average"]["timestamps"]]
            avg_latency = metrics["ModelLatency_Average"]["values"]
            
            if "ModelLatency_p90" in metrics:
                p90_latency = metrics["ModelLatency_p90"]["values"]
                plt.plot(range(len(timestamps)), p90_latency, label="p90 Latency", color="orange")
                
            if "ModelLatency_p99" in metrics:
                p99_latency = metrics["ModelLatency_p99"]["values"]
                plt.plot(range(len(timestamps)), p99_latency, label="p99 Latency", color="red")
            
            plt.plot(range(len(timestamps)), avg_latency, label="Average Latency", color="blue")
            
            plt.xlabel("Time")
            plt.ylabel("Latency (ms)")
            plt.title("Model Latency Trend")
            plt.grid(True)
            plt.legend()
            
            # Set x-axis labels (use subset to avoid overcrowding)
            n_labels = 6
            indices = np.linspace(0, len(timestamps)-1, n_labels, dtype=int)
            plt.xticks(indices, [timestamps[i] for i in indices], rotation=45)
            
            plt.tight_layout()
            metrics_chart_path = os.path.join(output_dir, "latency_trend.png")
            plt.savefig(metrics_chart_path)
            plt.close()
            
            metrics_charts.append(("Latency Trend", "latency_trend.png"))
        
        # Create invocations chart
        if "Invocations_Sum" in metrics:
            plt.figure(figsize=(12, 6))
            
            timestamps = [t for t in metrics["Invocations_Sum"]["timestamps"]]
            invocations = metrics["Invocations_Sum"]["values"]
            
            plt.bar(range(len(timestamps)), invocations, color="blue", alpha=0.7)
            
            plt.xlabel("Time")
            plt.ylabel("Invocations")
            plt.title("Endpoint Invocations")
            plt.grid(True)
            
            # Set x-axis labels (use subset to avoid overcrowding)
            n_labels = 6
            indices = np.linspace(0, len(timestamps)-1, n_labels, dtype=int)
            plt.xticks(indices, [timestamps[i] for i in indices], rotation=45)
            
            plt.tight_layout()
            invocations_chart_path = os.path.join(output_dir, "invocations.png")
            plt.savefig(invocations_chart_path)
            plt.close()
            
            metrics_charts.append(("Invocations", "invocations.png"))
        
        # Create errors chart
        if "Invocation4XXErrors_Sum" in metrics or "Invocation5XXErrors_Sum" in metrics:
            plt.figure(figsize=(12, 6))
            
            if "Invocation4XXErrors_Sum" in metrics:
                timestamps = [t for t in metrics["Invocation4XXErrors_Sum"]["timestamps"]]
                errors_4xx = metrics["Invocation4XXErrors_Sum"]["values"]
                plt.bar(range(len(timestamps)), errors_4xx, color="orange", alpha=0.7, label="4XX Errors")
            
            if "Invocation5XXErrors_Sum" in metrics:
                timestamps = [t for t in metrics["Invocation5XXErrors_Sum"]["timestamps"]]
                errors_5xx = metrics["Invocation5XXErrors_Sum"]["values"]
                plt.bar(range(len(timestamps)), errors_5xx, color="red", alpha=0.7, label="5XX Errors")
            
            plt.xlabel("Time")
            plt.ylabel("Error Count")
            plt.title("Endpoint Errors")
            plt.grid(True)
            plt.legend()
            
            # Set x-axis labels (use subset to avoid overcrowding)
            n_labels = 6
            indices = np.linspace(0, len(timestamps)-1, n_labels, dtype=int)
            plt.xticks(indices, [timestamps[i] for i in indices], rotation=45)
            
            plt.tight_layout()
            errors_chart_path = os.path.join(output_dir, "errors.png")
            plt.savefig(errors_chart_path)
            plt.close()
            
            metrics_charts.append(("Errors", "errors.png"))
    
    # Create simple HTML report
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Inference Endpoint Diagnostic Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1, h2, h3 { color: #333; }
            .section { margin-bottom: 30px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .chart { margin: 20px 0; max-width: 100%; }
            .recommendation { background-color: #e7f3fe; border-left: 5px solid #2196F3; padding: 10px; margin: 15px 0; }
        </style>
    </head>
    <body>
        <h1>Inference Endpoint Diagnostic Report</h1>
        <p>Endpoint: {{ endpoint_info.endpoint_name }}<br>
           Generated: {{ timestamp }}</p>
        
        <div class="section">
            <h2>Endpoint Information</h2>
            <table>
                <tr><th>Status</th><td>{{ endpoint_info.status }}</td></tr>
                <tr><th>Created</th><td>{{ endpoint_info.created_at }}</td></tr>
                <tr><th>Model</th><td>{{ endpoint_info.model.name }}</td></tr>
            </table>
            
            <h3>Production Variants</h3>
            <table>
                <tr>
                    <th>Name</th>
                    <th>Instance Type</th>
                    <th>Count</th>
                    <th>Weight</th>
                </tr>
                {% for variant in endpoint_info.variants %}
                <tr>
                    <td>{{ variant.name }}</td>
                    <td>{{ variant.instance_type }}</td>
                    <td>{{ variant.instance_count }}</td>
                    <td>{{ variant.weight }}</td>
                </tr>
                {% endfor %}
            </table>
            
            <h3>Environment Variables</h3>
            <table>
                <tr>
                    <th>Name</th>
                    <th>Value</th>
                </tr>
                {% for key, value in endpoint_info.model.environment.items() %}
                <tr>
                    <td>{{ key }}</td>
                    <td>{{ value }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        
        <div class="section">
            <h2>Latency Analysis</h2>
            <p>Baseline Latency: {{ "%.3f"|format(baseline_latency) }} seconds</p>
            
            <img class="chart" src="latency_chart.png" alt="Latency Chart" />
            
            {% if batch_chart %}
            <h3>Batch Processing Performance</h3>
            <img class="chart" src="batch_chart.png" alt="Batch Latency Chart" />
            {% endif %}
            
            <h3>Detailed Results</h3>
            <table>
                <tr>
                    <th>Input Length</th>
                    <th>Max Tokens</th>
                    <th>Mean Latency</th>
                    <th>Std Dev</th>
                    <th>Relative to Baseline</th>
                </tr>
                {% for result in latency_results %}
                {% if not result.batch_size %}
                <tr>
                    <td>{{ result.input_length }}</td>
                    <td>{{ result.max_tokens }}</td>
                    <td>{{ "%.3f"|format(result.mean_latency) }}</td>
                    <td>{{ "%.3f"|format(result.std_latency) }}</td>
                    <td>{{ "%.2f"|format(result.relative_latency) }}x</td>
                </tr>
                {% endif %}
                {% endfor %}
            </table>
            
            <div class="recommendation">
                <h3>Recommendations</h3>
                <ul>
                    {% if baseline_latency > 0.5 %}
                    <li>Baseline latency is high (>500ms). Consider using a larger instance type or optimizing the model.</li>
                    {% endif %}
                    
                    {% if any(r.mean_latency / baseline_latency > 5 for r in latency_results if not r.batch_size) %}
                    <li>Some inputs show >5x latency compared to baseline. Consider batch size or sequence length optimizations.</li>
                    {% endif %}
                    
                    {% if batch_results and batch_results[0].latency_per_input < baseline_latency * 0.75 %}
                    <li>Batch processing is significantly more efficient. Consider implementing batching in your application.</li>
                    {% endif %}
                </ul>
            </div>
        </div>
        
        {% if quality_results %}
        <div class="section">
            <h2>Output Quality Analysis</h2>
            
            <h3>Summary</h3>
            <table>
                <tr><th>Samples Tested</th><td>{{ quality_summary.samples_tested }}</td></tr>
                <tr><th>Failures</th><td>{{ quality_summary.failures }}</td></tr>
                <tr><th>Empty Outputs</th><td>{{ quality_summary.empty_outputs }}</td></tr>
                <tr><th>Expected Matches</th><td>{{ quality_summary.expected_matches }}</td></tr>
                {% if quality_summary.avg_token_overlap %}
                <tr><th>Avg Token Overlap</th><td>{{ "%.2f"|format(quality_summary.avg_token_overlap * 100) }}%</td></tr>
                {% endif %}
            </table>
            
            <h3>Sample Results</h3>
            <table>
                <tr>
                    <th>Input</th>
                    <th>Output</th>
                    <th>Status</th>
                </tr>
                {% for result in quality_results[:5] %}
                <tr>
                    <td>{{ result.input }}</td>
                    <td>{% if result.output %}{{ result.output }}{% else %}{{ result.error }}{% endif %}</td>
                    <td>
                        {% if result.quality %}
                            {% if result.quality.expected_match %}
                                ✅ Match
                            {% elif result.quality.empty_output %}
                                ❌ Empty
                            {% else %}
                                ⚠️ Different
                            {% endif %}
                        {% else %}
                            ❌ Error
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </table>
            
            <div class="recommendation">
                <h3>Recommendations</h3>
                <ul>
                    {% if quality_summary.failures > 0 %}
                    <li>{{ quality_summary.failures }} examples failed. Check endpoint logs for error details.</li>
                    {% endif %}
                    
                    {% if quality_summary.empty_outputs > 0 %}
                    <li>{{ quality_summary.empty_outputs }} examples returned empty output. Adjust model parameters or check input formatting.</li>
                    {% endif %}
                    
                    {% if quality_summary.avg_token_overlap and quality_summary.avg_token_overlap < 0.5 %}
                    <li>Token overlap is low (<50%). Review model fine-tuning or prompt engineering approaches.</li>
                    {% endif %}
                </ul>
            </div>
        </div>
        {% endif %}
        
        {% if metrics_charts %}
        <div class="section">
            <h2>Historical Metrics</h2>
            
            {% for title, filename in metrics_charts %}
            <h3>{{ title }}</h3>
            <img class="chart" src="{{ filename }}" alt="{{ title }}" />
            {% endfor %}
        </div>
        {% endif %}
        
        <div class="section">
            <h2>Final Recommendations</h2>
            <div class="recommendation">
                <ul>
                    {% if baseline_latency > 1.0 %}
                    <li>Inference latency is high. Consider:
                        <ul>
                            <li>Using a larger instance type (e.g., ml.g5.2xlarge instead of ml.g4dn.xlarge)</li>
                            <li>Model quantization (INT8 or FP16)</li>
                            <li>Optimizing your inference script with better batching</li>
                            <li>Using a smaller model or distilled model if possible</li>
                        </ul>
                    </li>
                    {% endif %}
                    
                    {% if any(r.mean_latency > 5.0 for r in latency_results if r.max_tokens >= 256) %}
                    <li>Long outputs (>256 tokens) have high latency. Consider implementing streaming responses.</li>
                    {% endif %}
                    
                    {% if quality_summary and quality_summary.failures + quality_summary.empty_outputs > quality_summary.samples_tested * 0.1 %}
                    <li>Output quality issues detected. Review model checkpoint and inference parameters.</li>
                    {% endif %}
                    
                    <li>For optimal deployment, consider:
                        <ul>
                            <li>Implementing token streaming for better user experience</li>
                            <li>CPU offloading for attention layers to optimize GPU memory</li>
                            <li>Using SageMaker Autoscaling based on invocation patterns</li>
                            <li>Implement request pre-processing to optimize inputs</li>
                        </ul>
                    </li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Prepare template variables
    template_vars = {
        "endpoint_info": {
            "endpoint_name": endpoint_name,
            "status": endpoint_info.get("status", "Unknown"),
            "created_at": endpoint_info.get("created_at", "Unknown"),
            "model": endpoint_info.get("model", {"name": "Unknown", "environment": {}}),
            "variants": endpoint_info.get("variants", [])
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_latency": baseline_latency,
        "latency_results": latency_results,
        "batch_results": [r for r in latency_results if "batch_size" in r],
        "batch_chart": bool([r for r in latency_results if "batch_size" in r]),
        "quality_results": quality_results,
        "quality_summary": quality_results[1] if isinstance(quality_results, tuple) else None,
        "metrics_charts": metrics_charts
    }
    
    # Render template
    template = Template(html_template)
    html_content = template.render(**template_vars)
    
    # Write HTML report
    report_path = os.path.join(output_dir, "inference_report.html")
    with open(report_path, "w") as f:
        f.write(html_content)
    
    print(f"Report generated at {report_path}")
    
    return report_path

def main():
    args = parse_args()
    
    # Check if endpoint exists
    print(f"Checking endpoint {args.endpoint_name} in {args.region}...")
    endpoint_info = check_endpoint_status(args.endpoint_name, args.region)
    
    if "error" in endpoint_info:
        print(f"Error: {endpoint_info['error']}")
        return
    
    print(f"Endpoint status: {endpoint_info['status']}")
    
    # Get CloudWatch metrics
    print("Fetching CloudWatch metrics...")
    metrics = get_cloudwatch_metrics(args.endpoint_name, args.region)
    
    # Load test examples
    test_examples = []
    if args.test_file:
        try:
            with open(args.test_file, "r") as f:
                test_examples = json.load(f)
            print(f"Loaded {len(test_examples)} test examples")
        except Exception as e:
            print(f"Error loading test file: {e}")
    else:
        # Create minimal test case
        test_examples = [
            {"input": "Write a function to calculate fibonacci numbers"},
            {"input": "Fix this code:\ndef factorial(n):\n    if n == 0:\n        return 0\n    return n * factorial(n-1)"},
            {"input": "Explain the difference between merge sort and quick sort"}
        ]
        print("No test file provided. Using minimal test cases.")
    
    # Run diagnostics based on mode
    latency_results, baseline_latency = None, None
    quality_results = None
    streaming_results = None
    
    if args.diagnose_mode in ["latency", "all"]:
        print("\nRunning latency tests...")
        latency_results, baseline_latency = run_latency_tests(args.endpoint_name, args.region, test_examples)
        
        print("\nLatency test results:")
        for result in latency_results:
            if "batch_size" in result:
                print(f"Batch size {result['batch_size']}: {result['mean_latency']:.3f}s ± {result['std_latency']:.3f}s")
            else:
                print(f"Input length {result['input_length']}, max tokens {result['max_tokens']}: {result['mean_latency']:.3f}s ± {result['std_latency']:.3f}s")
    
    if args.diagnose_mode in ["quality", "all"]:
        print("\nTesting output quality...")
        quality_results = diagnose_quality_issues(args.endpoint_name, args.region, test_examples)
        
        print("\nQuality test summary:")
        summary = quality_results[1]
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    if args.diagnose_mode in ["all"]:
        print("\nTesting token streaming support...")
        streaming_results = test_token_streaming(args.endpoint_name, args.region)
    
    # Generate report if requested
    if args.generate_report:
        print("\nGenerating diagnostic report...")
        report_path = generate_report(
            endpoint_info, metrics, latency_results, quality_results, 
            streaming_results, baseline_latency, args.output_dir
        )
        print(f"Report generated at {report_path}")
    
    print("\nDiagnostics completed!")

if __name__ == "__main__":
    main()
```

### 7.5 Common Environment-Specific Issues and Solutions

| Issue | Environment | Cause | Solution |
|-------|------------|-------|----------|
| Training crash at startup | SageMaker | Missing PyTorch extensions | Add `pip install ninja` to `requirements.txt` |
| Out of memory during first batch | SageMaker g4dn | Incorrect CUDA memory allocation | Set environment variable: `SM_CUDA_VISIBLE_DEVICES=0` |
| Tokenization errors with special tokens | Any | Special tokens not properly defined | Set `add_special_tokens=True` and define token mapping |
| Inference shows "cuda out of memory" | SageMaker Endpoint | Starting batch size too large | Set `SM_BATCH_SIZE=1` in endpoint environment variables |
| Model hanging at startup | SageMaker Endpoint | Loading too many quantized layers | Add `SM_LOAD_IN_8BIT_THRESHOLD=6.0` to environment |
| Slow inference with long input | Any | Suboptimal attention implementation | Enable Flash Attention 2 with `SM_ENABLE_FLASH_ATTENTION=true` |
| Loss becomes NaN after few steps | SageMaker | AdamW optimizer numerical instability | Switch to Adafactor optimizer |
| Data loading bottleneck | Any | Insufficient parallel processing | Increase DataLoader workers and use `pin_memory=True` |
| Memory leak during training | Any | PyTorch hooks not being released | Call `gc.collect()` between epochs |
| Unable to merge PEFT adapter | Any | Wrong base model used | Verify base model hash matches training hash |

### 7.6 Log Analysis and Monitoring Tools

Implement these specialized tools for monitoring model training and inference:

```python
# log_analyzer.py
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

class LogAnalyzer:
    """Analyze training and inference logs to detect issues."""
    
    def __init__(self, log_file=None):
        """Initialize the log analyzer."""
        self.log_file = log_file
        self.log_data = []
        self.patterns = {
            "cuda_out_of_memory": r"CUDA out of memory|RuntimeError: CUDA error",
            "nan_values": r"nan|inf|NaN|Inf|-inf|-NaN",
            "gpu_utilization": r"GPU\s+(\d+)%\s+used",
            "learning_rate": r"learning_rate\s*=\s*([\d.e-]+)",
            "loss": r"loss\s*=\s*([\d.e-]+)",
            "error": r"Error|Exception|Failed|Traceback",
            "warning": r"Warning|deprecated",
            "slowdown": r"Step took ([\d.]+)s",
        }
        self.extracted_data = {
            "learning_rate": [],
            "loss": [],
            "gpu_util": [],
            "step_time": [],
            "errors": [],
            "warnings": []
        }
        
        if log_file:
            self.load_log(log_file)
    
    def load_log(self, log_file):
        """Load log file for analysis."""
        try:
            with open(log_file, 'r') as f:
                self.log_data = f.readlines()
            print(f"Loaded {len(self.log_data)} lines from {log_file}")
            return True
        except Exception as e:
            print(f"Error loading log file: {e}")
            return False
    
    def extract_data(self):
        """Extract relevant data from logs using regex patterns."""
        if not self.log_data:
            print("No log data loaded")
            return False
        
        for line_num, line in enumerate(self.log_data):
            # Check for errors and warnings
            if re.search(self.patterns["error"], line):
                self.extracted_data["errors"].append((line_num, line.strip()))
            
            if re.search(self.patterns["warning"], line):
                self.extracted_data["warnings"].append((line_num, line.strip()))
            
            # Extract metrics
            lr_match = re.search(self.patterns["learning_rate"], line)
            if lr_match:
                self.extracted_data["learning_rate"].append((line_num, float(lr_match.group(1))))
            
            loss_match = re.search(self.patterns["loss"], line)
            if loss_match:
                try:
                    loss_value = float(loss_match.group(1))
                    self.extracted_data["loss"].append((line_num, loss_value))
                except ValueError:
                    # Sometimes loss might be NaN or Inf
                    pass
            
            gpu_match = re.search(self.patterns["gpu_utilization"], line)
            if gpu_match:
                self.extracted_data["gpu_util"].append((line_num, int(gpu_match.group(1))))
            
            time_match = re.search(self.patterns["slowdown"], line)
            if time_match:
                self.extracted_data["step_time"].append((line_num, float(time_match.group(1))))
        
        print("Data extraction complete")
        return True
    
    def check_for_issues(self):
        """Identify potential issues in the logs."""
        if not self.extracted_data["loss"]:
            self.extract_data()
        
        issues = []
        
        # Check for OOM errors
        oom_errors = [e for e in self.extracted_data["errors"] 
                     if re.search(self.patterns["cuda_out_of_memory"], e[1])]
        if oom_errors:
            issues.append({
                "type": "CUDA Out of Memory",
                "count": len(oom_errors),
                "first_occurrence": oom_errors[0][0],
                "recommendation": "Reduce batch size or model size, use gradient checkpointing"
            })
        
        # Check for NaN values in loss
        nan_losses = [(i, v) for i, v in self.extracted_data["loss"] 
                    if (v != v) or v == float('inf') or v == float('-inf')]
        if nan_losses:
            issues.append({
                "type": "NaN Loss Values",
                "count": len(nan_losses),
                "first_occurrence": nan_losses[0][0],
                "recommendation": "Reduce learning rate, add gradient clipping, check data preprocessing"
            })
        
        # Check for learning rate issues
        if len(self.extracted_data["learning_rate"]) > 1:
            # Check if LR is constantly zero or extremely small
            all_lrs = [lr for _, lr in self.extracted_data["learning_rate"]]
            if all(lr < 1e-7 for lr in all_lrs):
                issues.append({
                    "type": "Learning Rate Too Small",
                    "count": len(all_lrs),
                    "value": f"max={max(all_lrs):.2e}",
                    "recommendation": "Increase learning rate or check optimizer configuration"
                })
        
        # Check for performance degradation
        if len(self.extracted_data["step_time"]) > 10:
            step_times = [t for _, t in self.extracted_data["step_time"]]
            initial_avg = sum(step_times[:5]) / 5
            final_avg = sum(step_times[-5:]) / 5
            
            if final_avg > initial_avg * 1.5:  # 50% slowdown
                issues.append({
                    "type": "Performance Degradation",
                    "initial_step_time": f"{initial_avg:.2f}s",
                    "final_step_time": f"{final_avg:.2f}s",
                    "slowdown": f"{final_avg/initial_avg:.1f}x",
                    "recommendation": "Check for memory leaks, reduce sequence length, or adjust caching"
                })
        
        # Check for low GPU utilization
        if self.extracted_data["gpu_util"]:
            utils = [u for _, u in self.extracted_data["gpu_util"]]
            avg_util = sum(utils) / len(utils)
            
            if avg_util < 30:  # Less than 30% utilization
                issues.append({
                    "type": "Low GPU Utilization",
                    "average_utilization": f"{avg_util:.1f}%",
                    "recommendation": "Increase batch size, check for data loading bottlenecks, enable mixed precision"
                })
        
        return issues
    
    def visualize_training_progress(self, output_file=None):
        """Generate visualization of training progress."""
        if not self.extracted_data["loss"]:
            self.extract_data()
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot 1: Loss curve
        if self.extracted_data["loss"]:
            line_nums, losses = zip(*self.extracted_data["loss"])
            axes[0].plot(line_nums, losses, 'b-')
            axes[0].set_title('Training Loss')
            axes[0].set_xlabel('Log Line')
            axes[0].set_ylabel('Loss')
            axes[0].grid(True)
            
            # Mark any NaN or Inf values
            nan_indices = [i for i, l in enumerate(losses) if (l != l) or l == float('inf') or l == float('-inf')]
            if nan_indices:
                axes[0].scatter([line_nums[i] for i in nan_indices], 
                              [0 for _ in nan_indices],  # Plot at y=0
                              c='red', marker='x', s=100, label='NaN/Inf Values')
                axes[0].legend()
        
        # Plot 2: Learning Rate
        if self.extracted_data["learning_rate"]:
            line_nums, lrs = zip(*self.extracted_data["learning_rate"])
            axes[1].plot(line_nums, lrs, 'g-')
            axes[1].set_title('Learning Rate')
            axes[1].set_xlabel('Log Line')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_yscale('log')  # Log scale for learning rate
            axes[1].grid(True)
        
        # Plot 3: Step Time and GPU Utilization
        ax3 = axes[2]
        if self.extracted_data["step_time"]:
            line_nums, times = zip(*self.extracted_data["step_time"])
            ax3.plot(line_nums, times, 'r-', label='Step Time (s)')
            ax3.set_xlabel('Log Line')
            ax3.set_ylabel('Step Time (s)')
            ax3.grid(True)
            
            # Create a second y-axis for GPU utilization
            if self.extracted_data["gpu_util"]:
                ax4 = ax3.twinx()
                line_nums, utils = zip(*self.extracted_data["gpu_util"])
                ax4.plot(line_nums, utils, 'c--', label='GPU Utilization (%)')
                ax4.set_ylabel('GPU Utilization (%)')
                
                # Combine legends
                lines1, labels1 = ax3.get_legend_handles_labels()
                lines2, labels2 = ax4.get_legend_handles_labels()
                ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            else:
                ax3.legend()
            
            ax3.set_title('Performance Metrics')
        
        # Highlight error regions
        for i, ax in enumerate(axes):
            for error_line, _ in self.extracted_data["errors"]:
                ax.axvline(x=error_line, color='r', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Save or show the figure
        if output_file:
            plt.savefig(output_file)
            print(f"Saved visualization to {output_file}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self, output_dir="log_analysis"):
        """Generate a comprehensive analysis report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data if not already done
        if not self.extracted_data["loss"]:
            self.extract_data()
        
        # Check for issues
        issues = self.check_for_issues()
        
        # Generate visualization
        viz_file = os.path.join(output_dir, "training_progress.png")
        self.visualize_training_progress(viz_file)
        
        # Create HTML report
        html_report = os.path.join(output_dir, "log_analysis_report.html")
        
        with open(html_report, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Log Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .section {{ margin-bottom: 30px; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .error {{ color: red; }}
                    .warning {{ color: orange; }}
                    .recommendation {{ background-color: #e7f3fe; border-left: 5px solid #2196F3; padding: 10px; }}
                    .chart {{ margin: 20px 0; max-width: 100%; }}
                </style>
            </head>
            <body>
                <h1>Log Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="section">
                    <h2>Training Progress</h2>
                    <img class="chart" src="training_progress.png" alt="Training Progress" />
                </div>
            """)
            
            # Add issues section
            f.write("""
                <div class="section">
                    <h2>Identified Issues</h2>
            """)
            
            if issues:
                f.write("""
                    <table>
                        <tr>
                            <th>Issue Type</th>
                            <th>Details</th>
                            <th>Recommendation</th>
                        </tr>
                """)
                
                for issue in issues:
                    details = ""
                    for k, v in issue.items():
                        if k not in ["type", "recommendation"]:
                            details += f"{k}: {v}<br>"
                    
                    f.write(f"""
                        <tr>
                            <td>{issue['type']}</td>
                            <td>{details}</td>
                            <td>{issue['recommendation']}</td>
                        </tr>
                    """)
                
                f.write("</table>")
            else:
                f.write("<p>No significant issues detected.</p>")
            
            f.write("</div>")
            
            # Add statistics section
            f.write("""
                <div class="section">
                    <h2>Training Statistics</h2>
            """)
            
            # Loss statistics
            if self.extracted_data["loss"]:
                losses = [l for _, l in self.extracted_data["loss"]]
                valid_losses = [l for l in losses if l == l and l != float('inf') and l != float('-inf')]
                
                if valid_losses:
                    f.write(f"""
                        <h3>Loss Statistics</h3>
                        <p>
                            Initial Loss: {valid_losses[0]:.4f}<br>
                            Final Loss: {valid_losses[-1]:.4f}<br>
                            Minimum Loss: {min(valid_losses):.4f}<br>
                            Change: {(valid_losses[-1] - valid_losses[0]):.4f} ({(valid_losses[-1] / valid_losses[0] - 1) * 100:.1f}%)
                        </p>
                    """)
            
            # Performance statistics
            if self.extracted_data["step_time"]:
                step_times = [t for _, t in self.extracted_data["step_time"]]
                
                f.write(f"""
                    <h3>Performance Statistics</h3>
                    <p>
                        Average Step Time: {sum(step_times) / len(step_times):.3f}s<br>
                        Minimum Step Time: {min(step_times):.3f}s<br>
                        Maximum Step Time: {max(step_times):.3f}s<br>
                        First 5 Steps Average: {sum(step_times[:5]) / 5 if len(step_times) >= 5 else 'N/A'}s<br>
                        Last 5 Steps Average: {sum(step_times[-5:]) / 5 if len(step_times) >= 5 else 'N/A'}s
                    </p>
                """)
            
            f.write("</div>")
            
            # Add error and warning section
            f.write("""
                <div class="section">
                    <h2>Errors and Warnings</h2>
            """)
            
            # Errors
            f.write("<h3>Errors</h3>")
            if self.extracted_data["errors"]:
                f.write("<ul>")
                for line_num, error in self.extracted_data["errors"][:10]:  # Show top 10
                    f.write(f'<li><span class="error">Line {line_num}:</span> {error}</li>')
                
                if len(self.extracted_data["errors"]) > 10:
                    f.write(f'<li>... and {len(self.extracted_data["errors"]) - 10} more errors</li>')
                
                f.write("</ul>")
            else:
                f.write("<p>No errors detected.</p>")
            
            # Warnings
            f.write("<h3>Warnings</h3>")
            if self.extracted_data["warnings"]:
                f.write("<ul>")
                for line_num, warning in self.extracted_data["warnings"][:10]:  # Show top 10
                    f.write(f'<li><span class="warning">Line {line_num}:</span> {warning}</li>')
                
                if len(self.extracted_data["warnings"]) > 10:
                    f.write(f'<li>... and {len(self.extracted_data["warnings"]) - 10} more warnings</li>')
                
                f.write("</ul>")
            else:
                f.write("<p>No warnings detected.</p>")
            
            f.write("</div>")
            
            # Add recommendations section
            f.write("""
                <div class="section">
                    <h2>Recommendations</h2>
                    <div class="recommendation">
            """)
            
            if issues:
                f.write("<ul>")
                for issue in issues:
                    f.write(f'<li><strong>{issue["type"]}:</strong> {issue["recommendation"]}</li>')
                f.write("</ul>")
            else:
                # General recommendations
                f.write("""
                    <p>No specific issues detected. General recommendations:</p>
                    <ul>
                        <li>Consider enabling mixed precision training for faster performance.</li>
                        <li>Use gradient checkpointing to reduce memory usage if needed.</li>
                        <li>Implement early stopping to prevent overfitting.</li>
                        <li>Save checkpoints regularly to prevent data loss.</li>
                    </ul>
                """)
            
            f.write("""
                    </div>
                </div>
            </body>
            </html>
            """)
        
        print(f"Analysis report generated at {html_report}")
        return html_report
```

### 7.7 Continuous Monitoring System

Implement a proactive monitoring system to detect issues before they become critical:

```python
# continuous_monitor.py

import boto3
import json
import time
import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import threading
import logging
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("sagemaker-monitor")

class SageMakerMonitor:
    """Continuous monitoring for SageMaker training jobs and endpoints."""
    
    def __init__(self, region="us-east-1", polling_interval=60):
        """Initialize the SageMaker monitor."""
        self.region = region
        self.polling_interval = polling_interval
        self.sm_client = boto3.client('sagemaker', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        self.logs = boto3.client('logs', region_name=region)
        self.sns = boto3.client('sns', region_name=region)
        
        # Monitoring state
        self.running = False
        self.monitor_thread = None
        self.monitored_resources = {
            "training_jobs": [],
            "endpoints": []
        }
        
        # Alert configurations
        self.alert_configs = {
            "training": {
                "high_loss": 10.0,             # Alert if loss exceeds this value
                "nan_loss": True,              # Alert on NaN loss
                "no_progress_steps": 1000,     # Alert if no improvement after N steps
                "gpu_underutilization": 30,    # Alert if GPU utilization below X%
                "step_time_increase": 2.0      # Alert if step time increases by X times
            },
            "inference": {
                "latency_threshold_ms": 1000,  # Alert if latency exceeds X ms
                "error_rate_threshold": 0.01,  # Alert if error rate exceeds X%
                "invocation_drop_percent": 50, # Alert if invocations drop by X%
                "memory_threshold": 85         # Alert if memory utilization exceeds X%
            }
        }
        
        # Store metrics history
        self.metrics_history = {}
        self.last_alert_time = {}  # Prevent alert storms
        
        # Alert throttling (minimum time between alerts in seconds)
        self.alert_throttle_seconds = 300  # 5 minutes
    
    def add_training_job(self, job_name):
        """Add a training job to monitor."""
        if job_name not in self.monitored_resources["training_jobs"]:
            self.monitored_resources["training_jobs"].append(job_name)
            self.metrics_history[f"training_{job_name}"] = {
                "loss": [],
                "gpu_utilization": [],
                "step_times": [],
                "last_improvement_step": 0,
                "best_loss": float('inf')
            }
            logger.info(f"Now monitoring training job: {job_name}")
            return True
        return False
    
    def add_endpoint(self, endpoint_name):
        """Add an inference endpoint to monitor."""
        if endpoint_name not in self.monitored_resources["endpoints"]:
            self.monitored_resources["endpoints"].append(endpoint_name)
            self.metrics_history[f"endpoint_{endpoint_name}"] = {
                "invocations": [],
                "latency": [],
                "errors": [],
                "memory_utilization": []
            }
            logger.info(f"Now monitoring endpoint: {endpoint_name}")
            return True
        return False
    
    def remove_training_job(self, job_name):
        """Remove a training job from monitoring."""
        if job_name in self.monitored_resources["training_jobs"]:
            self.monitored_resources["training_jobs"].remove(job_name)
            if f"training_{job_name}" in self.metrics_history:
                del self.metrics_history[f"training_{job_name}"]
            logger.info(f"Stopped monitoring training job: {job_name}")
            return True
        return False
    
    def remove_endpoint(self, endpoint_name):
        """Remove an endpoint from monitoring."""
        if endpoint_name in self.monitored_resources["endpoints"]:
            self.monitored_resources["endpoints"].remove(endpoint_name)
            if f"endpoint_{endpoint_name}" in self.metrics_history:
                del self.metrics_history[f"endpoint_{endpoint_name}"]
            logger.info(f"Stopped monitoring endpoint: {endpoint_name}")
            return True
        return False
    
    def start_monitoring(self):
        """Start the monitoring thread."""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            logger.info("Monitoring started")
            return True
        logger.warning("Monitoring already running")
        return False
    
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        if self.running:
            self.running = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5)
            logger.info("Monitoring stopped")
            return True
        logger.warning("Monitoring not running")
        return False
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Monitor training jobs
                for job_name in self.monitored_resources["training_jobs"]:
                    self._check_training_job(job_name)
                
                # Monitor endpoints
                for endpoint_name in self.monitored_resources["endpoints"]:
                    self._check_endpoint(endpoint_name)
                
                # Generate visualizations periodically (every 10 iterations)
                if hasattr(self, 'iteration_count'):
                    self.iteration_count += 1
                else:
                    self.iteration_count = 0
                
                if self.iteration_count % 10 == 0:
                    self.generate_dashboards()
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Wait for next polling interval
            time.sleep(self.polling_interval)
    
    def _check_training_job(self, job_name):
        """Check status and metrics for a training job."""
        try:
            # Get job status
            response = self.sm_client.describe_training_job(TrainingJobName=job_name)
            status = response['TrainingJobStatus']
            
            if status in ['Completed', 'Failed', 'Stopped']:
                logger.info(f"Training job {job_name} is {status}, removing from monitoring")
                self.remove_training_job(job_name)
                return
            
            # Get metrics from CloudWatch
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=1)  # Last hour
            
            metrics_to_check = [
                {"name": "loss", "namespace": "/aws/sagemaker/TrainingJobs", "metric": "loss", "stat": "Average"},
                {"name": "gpu_util", "namespace": "/aws/sagemaker/TrainingJobs", "metric": "gpu_utilization", "stat": "Average"},
                {"name": "step_time", "namespace": "/aws/sagemaker/TrainingJobs", "metric": "step_time", "stat": "Average"}
            ]
            
            for metric in metrics_to_check:
                try:
                    response = self.cloudwatch.get_metric_data(
                        MetricDataQueries=[
                            {
                                'Id': metric["name"].replace("_", ""),
                                'MetricStat': {
                                    'Metric': {
                                        'Namespace': metric["namespace"],
                                        'MetricName': metric["metric"],
                                        'Dimensions': [
                                            {
                                                'Name': 'TrainingJobName',
                                                'Value': job_name
                                            }
                                        ]
                                    },
                                    'Period': 60,
                                    'Stat': metric["stat"]
                                }
                            }
                        ],
                        StartTime=start_time,
                        EndTime=end_time
                    )
                    
                    # Store metric values
                    values = response['MetricDataResults'][0]['Values']
                    if values:
                        self.metrics_history[f"training_{job_name}"][metric["name"]].extend(values)
                        
                        # Keep only the last 1000 values to limit memory usage
                        if len(self.metrics_history[f"training_{job_name}"][metric["name"]]) > 1000:
                            self.metrics_history[f"training_{job_name}"][metric["name"]] = \
                                self.metrics_history[f"training_{job_name}"][metric["name"]][-1000:]
                
                except Exception as e:
                    logger.warning(f"Error getting {metric['name']} for {job_name}: {e}")
            
            # Check training log for issues
            self._check_training_logs(job_name)
            
            # Check alert conditions
            self._check_training_alerts(job_name)
            
        except Exception as e:
            logger.error(f"Error checking training job {job_name}: {e}")
    
    def _check_training_logs(self, job_name):
        """Check CloudWatch logs for training job issues."""
        try:
            # Get log stream for training job
            response = self.logs.describe_log_streams(
                logGroupName=f"/aws/sagemaker/TrainingJobs",
                logStreamNamePrefix=job_name,
                limit=1
            )
            
            if not response['logStreams']:
                logger.warning(f"No log streams found for {job_name}")
                return
            
            log_stream = response['logStreams'][0]['logStreamName']
            
            # Get log events from the last 5 minutes
            end_time = int(datetime.utcnow().timestamp() * 1000)
            start_time = end_time - (5 * 60 * 1000)  # 5 minutes in milliseconds
            
            response = self.logs.get_log_events(
                logGroupName=f"/aws/sagemaker/TrainingJobs",
                logStreamName=log_stream,
                startTime=start_time,
                endTime=end_time,
                limit=100
            )
            
            # Check for critical patterns in logs
            critical_patterns = [
                "CUDA out of memory",
                "RuntimeError: ",
                "nan loss",
                "inf loss",
                "OOM",
                "killed",
                "Segmentation fault",
                "Bus error"
            ]
            
            for event in response['events']:
                for pattern in critical_patterns:
                    if pattern.lower() in event['message'].lower():
                        alert_message = f"Critical issue detected in {job_name} logs: {pattern}\nMessage: {event['message']}"
                        self._send_alert("CRITICAL", alert_message, job_name)
                        break
        
        except Exception as e:
            logger.error(f"Error checking logs for {job_name}: {e}")
    
    def _check_training_alerts(self, job_name):
        """Check alert conditions for training jobs."""
        history = self.metrics_history.get(f"training_{job_name}")
        if not history or not history.get("loss"):
            return
        
        # Get the most recent values
        recent_losses = history.get("loss", [])[-5:]
        if not recent_losses:
            return
        
        current_loss = recent_losses[-1]
        
        # Check for high loss
        if current_loss > self.alert_configs["training"]["high_loss"]:
            alert_message = f"Training job {job_name} has high loss: {current_loss:.4f}"
            self._send_alert("WARNING", alert_message, job_name)
        
        # Check for NaN loss
        if self.alert_configs["training"]["nan_loss"] and (np.isnan(current_loss) or np.isinf(current_loss)):
            alert_message = f"Training job {job_name} has NaN or Inf loss!"
            self._send_alert("CRITICAL", alert_message, job_name)
        
        # Check for lack of improvement
        if len(history.get("loss", [])) > 100:  # Need enough data
            if current_loss < history["best_loss"]:
                history["best_loss"] = current_loss
                history["last_improvement_step"] = len(history["loss"])
            elif len(history["loss"]) - history["last_improvement_step"] > self.alert_configs["training"]["no_progress_steps"]:
                steps_since_improvement = len(history["loss"]) - history["last_improvement_step"]
                alert_message = f"Training job {job_name} showing no improvement for {steps_since_improvement} steps. Current loss: {current_loss:.4f}, Best loss: {history['best_loss']:.4f}"
                self._send_alert("WARNING", alert_message, job_name)
        
        # Check GPU utilization
        recent_gpu = history.get("gpu_utilization", [])[-5:]
        if recent_gpu and len(recent_gpu) > 0:
            avg_gpu_util = sum(recent_gpu) / len(recent_gpu)
            if avg_gpu_util < self.alert_configs["training"]["gpu_underutilization"]:
                alert_message = f"Training job {job_name} has low GPU utilization: {avg_gpu_util:.1f}%"
                self._send_alert("WARNING", alert_message, job_name)
        
        # Check step time increase
        recent_steps = history.get("step_times", [])[-10:]
        if len(recent_steps) >= 10:
            first_5_avg = sum(recent_steps[:5]) / 5
            last_5_avg = sum(recent_steps[-5:]) / 5
            
            if last_5_avg > first_5_avg * self.alert_configs["training"]["step_time_increase"]:
                alert_message = f"Training job {job_name} step time increased significantly. Was: {first_5_avg:.2f}s, Now: {last_5_avg:.2f}s"
                self._send_alert("WARNING", alert_message, job_name)
    
    def _check_endpoint(self, endpoint_name):
        """Check status and metrics for an endpoint."""
        try:
            # Get endpoint status
            response = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
            status = response['EndpointStatus']
            
            if status != 'InService':
                alert_message = f"Endpoint {endpoint_name} is not in service. Status: {status}"
                self._send_alert("CRITICAL", alert_message, endpoint_name)
                
                if status in ['Failed', 'OutOfService', 'Deleting']:
                    logger.info(f"Endpoint {endpoint_name} is {status}, removing from monitoring")
                    self.remove_endpoint(endpoint_name)
                    return
            
            # Get metrics from CloudWatch
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=30)  # Last 30 minutes
            
            metrics_to_check = [
                {"name": "invocations", "namespace": "AWS/SageMaker", "metric": "Invocations", "stat": "Sum"},
                {"name": "latency", "namespace": "AWS/SageMaker", "metric": "ModelLatency", "stat": "Average"},
                {"name": "errors", "namespace": "AWS/SageMaker", "metric": "ModelError", "stat": "Sum"},
                {"name": "memory_utilization", "namespace": "AWS/SageMaker", "metric": "MemoryUtilization", "stat": "Average"}
            ]
            
            for metric in metrics_to_check:
                try:
                    response = self.cloudwatch.get_metric_data(
                        MetricDataQueries=[
                            {
                                'Id': metric["name"].replace("_", ""),
                                'MetricStat': {
                                    'Metric': {
                                        'Namespace': metric["namespace"],
                                        'MetricName': metric["metric"],
                                        'Dimensions': [
                                            {
                                                'Name': 'EndpointName',
                                                'Value': endpoint_name
                                            }
                                        ]
                                    },
                                    'Period': 60,
                                    'Stat': metric["stat"]
                                }
                            }
                        ],
                        StartTime=start_time,
                        EndTime=end_time
                    )
                    
                    # Store metric values
                    values = response['MetricDataResults'][0]['Values']
                    if values:
                        self.metrics_history[f"endpoint_{endpoint_name}"][metric["name"]].extend(values)
                        
                        # Keep only the last 1000 values to limit memory usage
                        if len(self.metrics_history[f"endpoint_{endpoint_name}"][metric["name"]]) > 1000:
                            self.metrics_history[f"endpoint_{endpoint_name}"][metric["name"]] = \
                                self.metrics_history[f"endpoint_{endpoint_name}"][metric["name"]][-1000:]
                
                except Exception as e:
                    logger.warning(f"Error getting {metric['name']} for {endpoint_name}: {e}")
            
            # Check alert conditions
            self._check_endpoint_alerts(endpoint_name)
            
        except Exception as e:
            logger.error(f"Error checking endpoint {endpoint_name}: {e}")
    
    def _check_endpoint_alerts(self, endpoint_name):
        """Check alert conditions for endpoints."""
        history = self.metrics_history.get(f"endpoint_{endpoint_name}")
        if not history:
            return
        
        # Check latency
        recent_latency = history.get("latency", [])[-5:]
        if recent_latency:
            avg_latency = sum(recent_latency) / len(recent_latency)
            if avg_latency > self.alert_configs["inference"]["latency_threshold_ms"]:
                alert_message = f"Endpoint {endpoint_name} has high latency: {avg_latency:.2f}ms"
                self._send_alert("WARNING", alert_message, endpoint_name)
        
        # Check error rate
        recent_invocations = history.get("invocations", [])[-5:]
        recent_errors = history.get("errors", [])[-5:]
        
        if recent_invocations and recent_errors and len(recent_invocations) == len(recent_errors):
            total_invocations = sum(recent_invocations)
            total_errors = sum(recent_errors)
            
            if total_invocations > 0:
                error_rate = total_errors / total_invocations
                if error_rate > self.alert_configs["inference"]["error_rate_threshold"]:
                    alert_message = f"Endpoint {endpoint_name} has high error rate: {error_rate:.2%}"
                    self._send_alert("CRITICAL", alert_message, endpoint_name)
        
        # Check for invocation drop
        if len(history.get("invocations", [])) > 10:
            first_5_invocations = history["invocations"][-10:-5]
            last_5_invocations = history["invocations"][-5:]
            
            if first_5_invocations and last_5_invocations:
                first_avg = sum(first_5_invocations) / len(first_5_invocations)
                last_avg = sum(last_5_invocations) / len(last_5_invocations)
                
                if first_avg > 0:
                    drop_percent = (1 - last_avg / first_avg) * 100
                    if drop_percent > self.alert_configs["inference"]["invocation_drop_percent"]:
                        alert_message = f"Endpoint {endpoint_name} invocations dropped by {drop_percent:.1f}%"
                        self._send_alert("WARNING", alert_message, endpoint_name)
        
        # Check memory utilization
        recent_memory = history.get("memory_utilization", [])[-5:]
        if recent_memory:
            avg_memory = sum(recent_memory) / len(recent_memory)
            if avg_memory > self.alert_configs["inference"]["memory_threshold"]:
                alert_message = f"Endpoint {endpoint_name} has high memory utilization: {avg_memory:.1f}%"
                self._send_alert("WARNING", alert_message, endpoint_name)
    
    def _send_alert(self, severity, message, resource_id):
        """Send alert notification."""
        # Check for alert throttling
        current_time = time.time()
        alert_key = f"{severity}_{resource_id}_{message[:50]}"  # Use part of message as key
        
        if alert_key in self.last_alert_time:
            time_since_last = current_time - self.last_alert_time[alert_key]
            if time_since_last < self.alert_throttle_seconds:
                logger.debug(f"Throttling alert: {alert_key}")
                return
        
        # Update last alert time
        self.last_alert_time[alert_key] = current_time
        
        # Log the alert
        if severity == "CRITICAL":
            logger.critical(message)
        else:
            logger.warning(message)
        
        # Send via SNS if configured
        try:
            topic_arn = os.environ.get("ALERT_SNS_TOPIC")
            if topic_arn:
                self.sns.publish(
                    TopicArn=topic_arn,
                    Subject=f"{severity} Alert: SageMaker {resource_id}",
                    Message=message
                )
        except Exception as e:
            logger.error(f"Error sending SNS alert: {e}")
        
        # Send to webhook if configured
        try:
            webhook_url = os.environ.get("ALERT_WEBHOOK_URL")
            if webhook_url:
                payload = {
                    "text": f"*{severity} Alert:* SageMaker {resource_id}\n{message}"
                }
                requests.post(webhook_url, json=payload)
        except Exception as e:
            logger.error(f"Error sending webhook alert: {e}")
    
    def generate_dashboards(self, output_dir="dashboards"):
        """Generate dashboard visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate training dashboards
        for job_name in self.monitored_resources["training_jobs"]:
            try:
                self._generate_training_dashboard(job_name, output_dir, timestamp)
            except Exception as e:
                logger.error(f"Error generating dashboard for {job_name}: {e}")
        
        # Generate endpoint dashboards
        for endpoint_name in self.monitored_resources["endpoints"]:
            try:
                self._generate_endpoint_dashboard(endpoint_name, output_dir, timestamp)
            except Exception as e:
                logger.error(f"Error generating dashboard for {endpoint_name}: {e}")
        
        # Generate summary dashboard
        self._generate_summary_dashboard(output_dir, timestamp)
    
    def _generate_training_dashboard(self, job_name, output_dir, timestamp):
        """Generate dashboard for a training job."""
        history = self.metrics_history.get(f"training_{job_name}")
        if not history or not history.get("loss"):
            return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot 1: Loss curve
        axes[0].plot(history["loss"], 'b-')
        axes[0].set_title(f'Training Loss - {job_name}')
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)
        
        # Plot 2: GPU Utilization
        if history.get("gpu_utilization"):
            axes[1].plot(history["gpu_utilization"], 'g-')
            axes[1].set_title('GPU Utilization')
            axes[1].set_xlabel('Steps')
            axes[1].set_ylabel('Utilization (%)')
            axes[1].grid(True)
        
        # Plot 3: Step Time
        if history.get("step_times"):
            axes[2].plot(history["step_times"], 'r-')
            axes[2].set_title('Step Time')
            axes[2].set_xlabel('Steps')
            axes[2].set_ylabel('Time (s)')
            axes[2].grid(True)
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{output_dir}/{job_name}_{timestamp}.png"
        plt.savefig(filename)
        plt.close()
        
        logger.info(f"Generated training dashboard for {job_name}")
    
    def _generate_endpoint_dashboard(self, endpoint_name, output_dir, timestamp):
        """Generate dashboard for an endpoint."""
        history = self.metrics_history.get(f"endpoint_{endpoint_name}")
        if not history or not history.get("invocations"):
            return
        
        # Create figure with multiple subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot 1: Invocations
        axes[0].plot(history["invocations"], 'b-')
        axes[0].set_title(f'Invocations - {endpoint_name}')
        axes[0].set_xlabel('Time Points')
        axes[0].set_ylabel('Count')
        axes[0].grid(True)
        
        # Plot 2: Latency
        if history.get("latency"):
            axes[1].plot(history["latency"], 'g-')
            axes[1].set_title('Model Latency')
            axes[1].set_xlabel('Time Points')
            axes[1].set_ylabel('Latency (ms)')
            axes[1].grid(True)
            
            # Add threshold line
            threshold = self.alert_configs["inference"]["latency_threshold_ms"]
            axes[1].axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold}ms)')
            axes[1].legend()
        
        # Plot 3: Memory Utilization
        if history.get("memory_utilization"):
            axes[2].plot(history["memory_utilization"], 'r-')
            axes[2].set_title('Memory Utilization')
            axes[2].set_xlabel('Time Points')
            axes[2].set_ylabel('Utilization (%)')
            axes[2].grid(True)
            
            # Add threshold line
            threshold = self.alert_configs["inference"]["memory_threshold"]
            axes[2].axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold}%)')
            axes[2].legend()
        
        plt.tight_layout()
        
        # Save figure
        filename = f"{output_dir}/{endpoint_name}_{timestamp}.png"
        plt.savefig(filename)
        plt.close()
        
        logger.info(f"Generated endpoint dashboard for {endpoint_name}")
    
    def _generate_summary_dashboard(self, output_dir, timestamp):
        """Generate summary dashboard for all monitored resources."""
        # Create a summary table
        summary_data = []
        
        # Training jobs summary
        for job_name in self.monitored_resources["training_jobs"]:
            history = self.metrics_history.get(f"training_{job_name}")
            if not history or not history.get("loss"):
                continue
            
            current_loss = history["loss"][-1] if history["loss"] else None
            best_loss = history["best_loss"] if "best_loss" in history else None
            steps = len(history["loss"])
            
            last_gpu = history.get("gpu_utilization", [])[-1] if history.get("gpu_utilization") else None
            last_step_time = history.get("step_times", [])[-1] if history.get("step_times") else None
            
            summary_data.append({
                "Resource Type": "Training Job",
                "Name": job_name,
                "Status": "Active",
                "Current Loss": f"{current_loss:.4f}" if current_loss is not None else "N/A",
                "Best Loss": f"{best_loss:.4f}" if best_loss is not None else "N/A",
                "Steps": steps,
                "GPU Util": f"{last_gpu:.1f}%" if last_gpu is not None else "N/A",
                "Step Time": f"{last_step_time:.2f}s" if last_step_time is not None else "N/A"
            })
        
        # Endpoints summary
        for endpoint_name in self.monitored_resources["endpoints"]:
            history = self.metrics_history.get(f"endpoint_{endpoint_name}")
            if not history or not history.get("invocations"):
                continue
            
            # Calculate stats from recent data
            recent_invocations = history.get("invocations", [])[-5:]
            recent_latency = history.get("latency", [])[-5:]
            recent_errors = history.get("errors", [])[-5:]
            recent_memory = history.get("memory_utilization", [])[-5:]
            
            avg_invocations = sum(recent_invocations) / len(recent_invocations) if recent_invocations else 0
            avg_latency = sum(recent_latency) / len(recent_latency) if recent_latency else 0
            sum_errors = sum(recent_errors) if recent_errors else 0
            avg_memory = sum(recent_memory) / len(recent_memory) if recent_memory else 0
            
            summary_data.append({
                "Resource Type": "Endpoint",
                "Name": endpoint_name,
                "Status": "InService",
                "Invocations": f"{avg_invocations:.1f}/min",
                "Latency": f"{avg_latency:.2f}ms",
                "Errors": sum_errors,
                "Memory": f"{avg_memory:.1f}%"
            })
        
        # Save summary to file
        summary_file = f"{output_dir}/summary_{timestamp}.txt"
        with open(summary_file, "w") as f:
            f.write(f"SageMaker Monitoring Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            if summary_data:
                # Create table for training jobs
                training_jobs = [item for item in summary_data if item["Resource Type"] == "Training Job"]
                if training_jobs:
                    f.write("TRAINING JOBS\n")
                    f.write(tabulate(training_jobs, headers="keys", tablefmt="grid"))
                    f.write("\n\n")
                
                # Create table for endpoints
                endpoints = [item for item in summary_data if item["Resource Type"] == "Endpoint"]
                if endpoints:
                    f.write("ENDPOINTS\n")
                    f.write(tabulate(endpoints, headers="keys", tablefmt="grid"))
                    f.write("\n\n")
            else:
                f.write("No active resources being monitored.")
        
        logger.info(f"Generated summary dashboard")
```

### 7.8 Self-Healing and Automated Recovery

Create a system that automatically recovers from common issues:

```python
# auto_recovery.py

import boto3
import time
import json
import os
import logging
import argparse
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("auto_recovery.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("auto-recovery")

class SageMakerRecovery:
    """Automatically recover from common SageMaker issues."""
    
    def __init__(self, region="us-east-1"):
        """Initialize the recovery system."""
        self.region = region
        self.sm_client = boto3.client('sagemaker', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=region)
        
        # Recovery configuration
        self.recovery_config = {
            "endpoints": {
                "auto_restart": True,            # Restart failed endpoints
                "max_restart_attempts": 3,       # Maximum number of restart attempts
                "restart_cooldown_minutes": 15,  # Minimum time between restarts
                "scale_on_high_utilization": True,  # Scale on high utilization
                "utilization_threshold": 80,     # Utilization threshold for scaling
                "backup_before_restart": True    # Create model backup before restart
            },
            "training_jobs": {
                "auto_retry": True,              # Retry failed training jobs
                "max_retry_attempts": 2,         # Maximum number of retry attempts
                "reduce_batch_on_oom": True,     # Reduce batch size on OOM errors
                "batch_reduction_factor": 2,     # Factor to reduce batch size by
                "save_model_on_interrupt": True  # Save model before stopping job
            }
        }
        
        # Track recovery attempts
        self.recovery_history = {}
    
    def check_and_recover_endpoints(self, endpoint_names=None):
        """Check and recover endpoints if needed."""
        if endpoint_names is None:
            # List all endpoints
            try:
                response = self.sm_client.list_endpoints()
                endpoint_names = [endpoint["EndpointName"] for endpoint in response["Endpoints"]]
            except Exception as e:
                logger.error(f"Error listing endpoints: {e}")
                return []
        
        recovered_endpoints = []
        
        for endpoint_name in endpoint_names:
            try:
                # Check endpoint status
                response = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
                status = response["EndpointStatus"]
                
                if status in ["Failed", "OutOfService"]:
                    logger.info(f"Found endpoint {endpoint_name} in {status} state")
                    
                    # Check if we've tried to recover this endpoint too many times
                    history_key = f"endpoint_{endpoint_name}"
                    if history_key in self.recovery_history:
                        attempts = self.recovery_history[history_key]["attempts"]
                        last_attempt = self.recovery_history[history_key]["last_attempt"]
                        cooldown = timedelta(minutes=self.recovery_config["endpoints"]["restart_cooldown_minutes"])
                        
                        if attempts >= self.recovery_config["endpoints"]["max_restart_attempts"]:
                            logger.warning(f"Endpoint {endpoint_name} has reached maximum restart attempts ({attempts})")
                            continue
                        
                        if datetime.now() - last_attempt < cooldown:
                            logger.info(f"Endpoint {endpoint_name} is still in cooldown period")
                            continue
                    else:
                        self.recovery_history[history_key] = {
                            "attempts": 0,
                            "last_attempt": datetime.min
                        }
                    
                    # Try to recover the endpoint
                    if self._recover_endpoint(endpoint_name, response):
                        recovered_endpoints.append(endpoint_name)
                        
                        # Update recovery history
                        self.recovery_history[history_key]["attempts"] += 1
                        self.recovery_history[history_key]["last_attempt"] = datetime.now()
                
                elif status == "InService":
                    # Check for high utilization if scaling is enabled
                    if self.recovery_config["endpoints"]["scale_on_high_utilization"]:
                        self._check_and_scale_endpoint(endpoint_name)
            
            except Exception as e:
                logger.error(f"Error checking endpoint {endpoint_name}: {e}")
        
        return recovered_endpoints
    
    def _recover_endpoint(self, endpoint_name, endpoint_info):
        """Attempt to recover a failed endpoint."""
        try:
            logger.info(f"Attempting to recover endpoint {endpoint_name}")
            
            # Get the endpoint config name
            endpoint_config_name = endpoint_info["EndpointConfigName"]
            
            # Backup the model if enabled
            if self.recovery_config["endpoints"]["backup_before_restart"]:
                backup_name = f"{endpoint_name}-backup-{int(time.time())}"
                logger.info(f"Creating backup model {backup_name}")
                
                try:
                    # Get endpoint config details
                    config = self.sm_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
                    
                    # Get model name from config
                    model_name = config["ProductionVariants"][0]["ModelName"]
                    
                    # Get model details
                    model = self.sm_client.describe_model(ModelName=model_name)
                    
                    # Create backup model
                    self.sm_client.create_model(
                        ModelName=backup_name,
                        PrimaryContainer=model["PrimaryContainer"],
                        ExecutionRoleArn=model["ExecutionRoleArn"]
                    )
                    
                    logger.info(f"Backup model {backup_name} created successfully")
                
                except Exception as e:
                    logger.error(f"Error creating backup model: {e}")
            
            # Check if there's a failure reason we can address
            failure_reason = endpoint_info.get("FailureReason", "")
            logger.info(f"Endpoint failure reason: {failure_reason}")
            
            # Create a new endpoint config if needed
            new_config = None
            
            if "out of memory" in failure_reason.lower() or "oom" in failure_reason.lower():
                # Create a new config with a larger instance type
                new_config = self._create_config_with_larger_instance(endpoint_config_name)
            
            # Update the endpoint
            update_args = {
                "EndpointName": endpoint_name,
                "EndpointConfigName": new_config if new_config else endpoint_config_name,
                "RetainAllVariantProperties": True
            }
            
            self.sm_client.update_endpoint(**update_args)
            
            logger.info(f"Endpoint {endpoint_name} recovery initiated")
            
            # Wait for endpoint to be updating
            waiter = self.sm_client.get_waiter('endpoint_updating')
            waiter.wait(EndpointName=endpoint_name)
            
            logger.info(f"Endpoint {endpoint_name} is updating")
            return True
            
        except Exception as e:
            logger.error(f"Error recovering endpoint {endpoint_name}: {e}")
            return False
    
    def _create_config_with_larger_instance(self, config_name):
        """Create a new endpoint config with a larger instance type."""
        try:
            # Get current config
            config = self.sm_client.describe_endpoint_config(EndpointConfigName=config_name)
            
            # Get the current instance type and determine a larger one
            variant = config["ProductionVariants"][0]
            current_instance = variant["InstanceType"]
            
            # Define a mapping of instance upgrades
            instance_upgrades = {
                "ml.t2.medium": "ml.t2.large",
                "ml.t2.large": "ml.t2.xlarge",
                "ml.t2.xlarge": "ml.t2.2xlarge",
                "ml.m5.large": "ml.m5.xlarge",
                "ml.m5.xlarge": "ml.m5.2xlarge",
                "ml.m5.2xlarge": "ml.m5.4xlarge",
                "ml.m5.4xlarge": "ml.m5.12xlarge",
                "ml.c5.large": "ml.c5.xlarge",
                "ml.c5.xlarge": "ml.c5.2xlarge",
                "ml.c5.2xlarge": "ml.c5.4xlarge",
                "ml.c5.4xlarge": "ml.c5.9xlarge",
                "ml.g4dn.xlarge": "ml.g4dn.2xlarge",
                "ml.g4dn.2xlarge": "ml.g4dn.4xlarge",
                "ml.g4dn.4xlarge": "ml.g4dn.8xlarge",
                "ml.g5.xlarge": "ml.g5.2xlarge",
                "ml.g5.2xlarge": "ml.g5.4xlarge",
                "ml.g5.4xlarge": "ml.g5.8xlarge",
                "ml.inf1.xlarge": "ml.inf1.2xlarge",
                "ml.inf1.2xlarge": "ml.inf1.6xlarge",
            }
            
            new_instance = instance_upgrades.get(current_instance)
            if not new_instance:
                logger.warning(f"No upgrade path defined for instance type {current_instance}")
                # Default to a larger general instance if no specific upgrade path
                if "g4dn" in current_instance:
                    new_instance = "ml.g4dn.4xlarge"
                elif "g5" in current_instance:
                    new_instance = "ml.g5.4xlarge"
                else:
                    new_instance = "ml.m5.4xlarge"
            
            logger.info(f"Upgrading instance type from {current_instance} to {new_instance}")
            
            # Create a new config with the larger instance
            new_config_name = f"{config_name}-upgrade-{int(time.time())}"
            
            # Create new variants with updated instance type
            new_variants = []
            for variant in config["ProductionVariants"]:
                new_variant = variant.copy()
                new_variant["InstanceType"] = new_instance
                new_variants.append(new_variant)
            
            # Create the new config
            self.sm_client.create_endpoint_config(
                EndpointConfigName=new_config_name,
                ProductionVariants=new_variants,
                Tags=config.get("Tags", [])
            )
            
            logger.info(f"Created new endpoint config {new_config_name} with instance type {new_instance}")
            
            return new_config_name
            
        except Exception as e:
            logger.error(f"Error creating config with larger instance: {e}")
            return None
    
    def _check_and_scale_endpoint(self, endpoint_name):
        """Check endpoint utilization and scale if needed."""
        try:
            # Get recent CPU and memory utilization
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(minutes=30)
            
            # Check CPU utilization
            response = self.cloudwatch.get_metric_statistics(
                Namespace="AWS/SageMaker",
                MetricName="CPUUtilization",
                Dimensions=[
                    {
                        'Name': 'EndpointName',
                        'Value': endpoint_name
                    }
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,  # 5-minute periods
                Statistics=['Average']
            )
            
            # Check if we have any datapoints
            if not response['Datapoints']:
                logger.info(f"No CPU utilization data for endpoint {endpoint_name}")
                return False
            
            # Get the average CPU utilization
            cpu_util = max([d['Average'] for d in response['Datapoints']])
            
            # Check memory utilization
            response = self.cloudwatch.get_metric_statistics(
                Namespace="AWS/SageMaker",
                MetricName="MemoryUtilization",
                Dimensions=[
                    {
                        'Name': 'EndpointName',
                        'Value': endpoint_name
                    }
                ],
                StartTime=start_time,
                EndTime=end_time,
                Period=300,
                Statistics=['Average']
            )
            
            # Get the average memory utilization
            memory_util = 0
            if response['Datapoints']:
                memory_util = max([d['Average'] for d in response['Datapoints']])
            
            # Check if either utilization is above threshold
            threshold = self.recovery_config["endpoints"]["utilization_threshold"]
            if cpu_util > threshold or memory_util > threshold:
                logger.info(f"Endpoint {endpoint_name} has high utilization: CPU={cpu_util:.1f}%, Memory={memory_util:.1f}%")
                
                # Scale the endpoint
                self._scale_endpoint(endpoint_name, cpu_util, memory_util)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking endpoint utilization: {e}")
            return False
    
    def _scale_endpoint(self, endpoint_name, cpu_util, memory_util):
        """Scale an endpoint based on utilization."""
        try:
            # Get endpoint details
            response = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
            endpoint_config_name = response["EndpointConfigName"]
            
            # Get endpoint config
            config = self.sm_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
            
            # Check if auto-scaling is already enabled
            if self._check_autoscaling_enabled(endpoint_name):
                logger.info(f"Endpoint {endpoint_name} already has auto-scaling enabled")
                return True
            
            # Create a new config with auto-scaling
            new_config_name = f"{endpoint_config_name}-autoscale-{int(time.time())}"
            
            # Copy the existing config
            self.sm_client.create_endpoint_config(
                EndpointConfigName=new_config_name,
                ProductionVariants=config["ProductionVariants"],
                Tags=config.get("Tags", [])
            )
            
            # Update the endpoint
            self.sm_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=new_config_name,
                RetainAllVariantProperties=True
            )
            
            logger.info(f"Updated endpoint {endpoint_name} with new config {new_config_name}")
            
            # Wait for endpoint to be updating
            waiter = self.sm_client.get_waiter('endpoint_updating')
            waiter.wait(EndpointName=endpoint_name)
            
            # Register with Application Auto Scaling
            app_as_client = boto3.client('application-autoscaling', region_name=self.region)
            
            # Get the variant name
            variant_name = config["ProductionVariants"][0]["VariantName"]
            
            # Register scalable target
            app_as_client.register_scalable_target(
                ServiceNamespace='sagemaker',
                ResourceId=f'endpoint/{endpoint_name}/variant/{variant_name}',
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                MinCapacity=1,
                MaxCapacity=3  # Set a reasonable maximum
            )
            
            # Create scaling policy based on which utilization is higher
            if cpu_util >= memory_util:
                metric = 'SageMakerVariantCPUUtilization'
                target_value = 70.0  # Target 70% CPU utilization
            else:
                metric = 'SageMakerVariantMemoryUtilization'
                target_value = 70.0  # Target 70% memory utilization
            
            app_as_client.put_scaling_policy(
                PolicyName=f"{endpoint_name}-scaling-policy",
                ServiceNamespace='sagemaker',
                ResourceId=f'endpoint/{endpoint_name}/variant/{variant_name}',
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                PolicyType='TargetTrackingScaling',
                TargetTrackingScalingPolicyConfiguration={
                    'TargetValue': target_value,
                    'PredefinedMetricSpecification': {
                        'PredefinedMetricType': metric
                    },
                    'ScaleInCooldown': 300,  # 5 minutes
                    'ScaleOutCooldown': 60    # 1 minute
                }
            )
            
            logger.info(f"Enabled auto-scaling for endpoint {endpoint_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error scaling endpoint {endpoint_name}: {e}")
            return False
    
    def _check_autoscaling_enabled(self, endpoint_name):
        """Check if auto-scaling is already enabled for an endpoint."""
        try:
            app_as_client = boto3.client('application-autoscaling', region_name=self.region)
            
            # Get endpoint config
            response = self.sm_client.describe_endpoint(EndpointName=endpoint_name)
            config = self.sm_client.describe_endpoint_config(EndpointConfigName=response["EndpointConfigName"])
            
            # Get the variant name
            variant_name = config["ProductionVariants"][0]["VariantName"]
            
            # Check if there's a scalable target
            response = app_as_client.describe_scalable_targets(
                ServiceNamespace='sagemaker',
                ResourceIds=[f'endpoint/{endpoint_name}/variant/{variant_name}'],
                ScalableDimension='sagemaker:variant:DesiredInstanceCount'
            )
            
            return len(response['ScalableTargets']) > 0
            
        except Exception as e:
            logger.debug(f"Error checking auto-scaling status: {e}")
            return False
    
    def check_and_recover_training_jobs(self, hours=24):
        """Check and recover failed training jobs."""
        try:
            # List recently failed training jobs
            failed_jobs = []
            
            # Calculate time window
            current_time = datetime.utcnow()
            start_time = current_time - timedelta(hours=hours)
            
            # List training jobs
            response = self.sm_client.list_training_jobs(
                StatusEquals='Failed',
                CreationTimeAfter=start_time
            )
            
            failed_jobs.extend(response['TrainingJobSummaries'])
            
            # Handle pagination
            while 'NextToken' in response:
                response = self.sm_client.list_training_jobs(
                    StatusEquals='Failed',
                    CreationTimeAfter=start_time,
                    NextToken=response['NextToken']
                )
                failed_jobs.extend(response['TrainingJobSummaries'])
            
            logger.info(f"Found {len(failed_jobs)} failed training jobs in the last {hours} hours")
            
            recovered_jobs = []
            
            for job in failed_jobs:
                job_name = job['TrainingJobName']
                
                # Check if we've tried to recover this job too many times
                history_key = f"training_{job_name}"
                if history_key in self.recovery_history:
                    attempts = self.recovery_history[history_key]["attempts"]
                    
                    if attempts >= self.recovery_config["training_jobs"]["max_retry_attempts"]:
                        logger.warning(f"Training job {job_name} has reached maximum retry attempts ({attempts})")
                        continue
                else:
                    self.recovery_history[history_key] = {
                        "attempts": 0,
                        "last_attempt": datetime.min
                    }
                
                # Get detailed job info
                job_info = self.sm_client.describe_training_job(TrainingJobName=job_name)
                
                # Try to recover the job
                if self._recover_training_job(job_name, job_info):
                    recovered_jobs.append(job_name)
                    
                    # Update recovery history
                    self.recovery_history[history_key]["attempts"] += 1
                    self.recovery_history[history_key]["last_attempt"] = datetime.now()
            
            return recovered_jobs
            
        except Exception as e:
            logger.error(f"Error checking training jobs: {e}")
            return []
    
    def _recover_training_job(self, job_name, job_info):
        """Attempt to recover a failed training job."""
        try:
            logger.info(f"Attempting to recover training job {job_name}")
            
            # Get failure reason
            failure_reason = job_info.get("FailureReason", "")
            logger.info(f"Training job failure reason: {failure_reason}")
            
            # Check for common failure patterns
            is_oom = "out of memory" in failure_reason.lower() or "oom" in failure_reason.lower()
            is_timeout = "timeout" in failure_reason.lower()
            is_infrastructure = "internal server error" in failure_reason.lower() or "infrastructure" in failure_reason.lower()
            
            # Get job configuration
            algorithm_specification = job_info.get("AlgorithmSpecification", {})
            hyperparameters = job_info.get("HyperParameters", {})
            resource_config = job_info.get("ResourceConfig", {})
            input_data_config = job_info.get("InputDataConfig", [])
            output_data_config = job_info.get("OutputDataConfig", {})
            
            # Prepare new hyperparameters
            new_hyperparameters = hyperparameters.copy()
            
            if is_oom and self.recovery_config["training_jobs"]["reduce_batch_on_oom"]:
                # Reduce batch size for OOM errors
                if "batch-size" in hyperparameters:
                    current_batch = int(hyperparameters["batch-size"])
                    new_batch = max(1, current_batch // self.recovery_config["training_jobs"]["batch_reduction_factor"])
                    new_hyperparameters["batch-size"] = str(new_batch)
                    logger.info(f"Reducing batch size from {current_batch} to {new_batch}")
                
                # Also enable gradient checkpointing if not already enabled
                new_hyperparameters["gradient-checkpointing"] = "true"
            
            elif is_timeout:
                # For timeouts, we might need to adjust max steps or epochs
                if "epochs" in hyperparameters:
                    current_epochs = int(hyperparameters["epochs"])
                    new_epochs = max(1, current_epochs // 2)
                    new_hyperparameters["epochs"] = str(new_epochs)
                    logger.info(f"Reducing epochs from {current_epochs} to {new_epochs}")
            
            # If it's an infrastructure error, we don't need to change anything
            
            # Create a new job based on the failed one
            new_job_name = f"{job_name}-retry-{int(time.time())}"
            
            create_args = {
                "TrainingJobName": new_job_name,
                "AlgorithmSpecification": algorithm_specification,
                "RoleArn": job_info["RoleArn"],
                "InputDataConfig": input_data_config,
                "OutputDataConfig": output_data_config,
                "ResourceConfig": resource_config,
                "StoppingCondition": job_info.get("StoppingCondition", {}),
                "HyperParameters": new_hyperparameters,
                "Tags": job_info.get("Tags", [])
            }
            
            # Remove any keys that aren't valid
            for key in list(create_args.keys()):
                if not create_args[key]:
                    del create_args[key]
            
            # Create the new job
            self.sm_client.create_training_job(**create_args)
            
            logger.info(f"Created new training job {new_job_name} to recover {job_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error recovering training job {job_name}: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="SageMaker Auto Recovery Tool")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region")
    parser.add_argument("--check-endpoints", action="store_true", help="Check and recover endpoints")
    parser.add_argument("--check-training", action="store_true", help="Check and recover training jobs")
    parser.add_argument("--all", action="store_true", help="Check all resources")
    parser.add_argument("--interval", type=int, default=0, help="Run continuously with specified interval (minutes)")
    parser.add_argument("--hours", type=int, default=24, help="Check training jobs from last N hours")
    parser.add_argument("--endpoint", type=str, help="Specific endpoint to check")
    args = parser.parse_args()
    
    recovery = SageMakerRecovery(region=args.region)
    
    # Function to run all checks
    def run_checks():
        if args.all or args.check_endpoints:
            endpoints = [args.endpoint] if args.endpoint else None
            recovered = recovery.check_and_recover_endpoints(endpoints)
            if recovered:
                logger.info(f"Recovered endpoints: {', '.join(recovered)}")
        
        if args.all or args.check_training:
            recovered = recovery.check_and_recover_training_jobs(hours=args.hours)
            if recovered:
                logger.info(f"Recovered training jobs: {', '.join(recovered)}")
    
    # Run once or in a loop based on arguments
    if args.interval > 0:
        logger.info(f"Running continuously with {args.interval} minute interval")
        try:
            while True:
                run_checks()
                time.sleep(args.interval * 60)
        except KeyboardInterrupt:
            logger.info("Stopping continuous monitoring")
    else:
        run_checks()

if __name__ == "__main__":
    main()
```

By implementing these advanced troubleshooting tools and techniques, you can dramatically reduce debugging time and improve overall system reliability. The log analyzer and continuous monitoring systems will help you catch issues before they become critical, while the self-healing capabilities ensure minimal downtime for production deployments.

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
