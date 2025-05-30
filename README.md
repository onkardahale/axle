# Axle

Axle is a command-line tool that helps you write better commit messages using AI. It analyzes your code changes and suggests a commit message in a structured format.

## Features

- 🤖 AI-powered commit message generation using the Qwen2.5-Coder-3B-Instruct model
- 📚 Language-agnostic knowledge base for improved context-aware commit messages
- 🔄 Interactive regeneration of commit messages with user context
- 🔗 Issue linking and breaking change declarations
- 💾 Automatic model caching for faster subsequent runs
- 🛠️ Configurable generation parameters
- 📚 Support for conventional commit types and scopes
- 🌳 Tree-sitter based code parsing for better context understanding
- 🌐 Multi-language support through tree-sitter grammars

## Installation

```bash
pip install git+https://github.com/onkardahale/axle.git
```

## Usage

1. Initialize the knowledge base (recommended for better commit messages):
```bash
axle init
```

2. Stage your changes:
```bash
git add .
```

3. Generate a commit message:
```bash
axle commit
```

4. The tool will:
   - Analyze your staged changes
   - Use the knowledge base for context (if available)
   - Generate a commit message
   - Open your default editor for review
   - Prompt for issue linking and breaking changes
   - Create the commit if you save and close the editor

## Knowledge Base

The `axle init` command creates a local knowledge base in the `.axle` directory of your project. This knowledge base:

- Analyzes your codebase using tree-sitter for accurate parsing
- Extracts structural information, docstrings, and imports
- Categorizes files based on their purpose and imports
- Provides rich context for commit message generation
- Supports multiple programming languages through tree-sitter grammars
- Automatically detects and processes supported file types
- Maintains a detailed log of analyzed and skipped files

The knowledge base is automatically used when generating commit messages, leading to more accurate and contextually relevant messages.

### Language Support

Axle is designed to be language-agnostic through its use of tree-sitter. Currently, we have dedicated analyzers for:

- Python (`.py`)
- JavaScript (`.js`, `.jsx`, `.mjs`, `.cjs`)

While the project uses `tree-sitter-language-pack` which provides grammar support for many languages, we need community contributions to add analyzers for additional languages. Languages that could be supported (but need analyzer implementation) include:

- TypeScript (`.ts`, `.tsx`)
- Java (`.java`)
- Go (`.go`)
- Rust (`.rs`)
- C/C++ (`.c`, `.cpp`, `.h`, `.hpp`)
- Ruby (`.rb`)
- PHP (`.php`)
- And many more through tree-sitter grammars

The knowledge base automatically:
- Detects file types based on extensions
- Uses appropriate language-specific analyzers
- Extracts language-specific constructs (classes, functions, imports, etc.)
- Categorizes files based on their content and structure
- Handles language-specific syntax and patterns

### Contributing Language Analyzers

We welcome contributions to add support for more languages! To add a new language analyzer:

1. Create a new analyzer class in `src/axle/treesitter/analyzers/` following the pattern of existing analyzers
2. Implement the required methods:
   - `analyze_file()`: Main analysis method
   - `_process_imports()`: Handle language-specific import statements
   - `_process_classes()`: Extract class definitions and methods
   - `_process_functions()`: Extract function definitions
   - `_process_variables()`: Extract variable declarations

3. Add the analyzer to the `TreeSitterParser` class in `src/axle/treesitter/parser.py`
4. Add tests in `tests/test_<language>_analyzer.py`
5. Update documentation

Example structure for a new analyzer:
```python
from .base import BaseAnalyzer

class NewLanguageAnalyzer(BaseAnalyzer):
    """Analyzer for NewLanguage."""
    
    def __init__(self):
        super().__init__("new_language")
        self.extension_map = {
            ".ext1": "new_language",
            ".ext2": "new_language"
        }
    
    def analyze_file(self, file_path: Path) -> FileAnalysis:
        # Implementation
        pass
    
    def _process_imports(self, tree: Tree) -> List[Import]:
        # Implementation
        pass
    
    # ... other required methods
```

Check out our existing analyzers for reference:
- `javascript_analyzer.py`: JavaScript/JSX support
- `python_analyzer.py`: Python support

### File Categorization

The knowledge base categorizes files based on their imports and path patterns. Here are the currently implemented categories:

#### Import-based Categories
Files are categorized based on their imported libraries and frameworks:

- `web_framework`: Files importing web frameworks
  - Python: Django, Flask
  - JavaScript: React, Angular, Vue, Express

- `test`: Files importing testing frameworks
  - Python: pytest, unittest
  - JavaScript: jest, mocha

- `database`: Files importing database libraries
  - Python: SQLAlchemy
  - JavaScript: mongoose, sequelize

- `data_processing`: Files importing data processing libraries
  - Python: pandas, numpy

- `ml`: Files importing machine learning libraries
  - Python: tensorflow, torch, sklearn

#### Path-based Categories
Files are categorized based on their path and filename patterns:

- `util`: Files in paths containing 'util', 'helper', or 'lib'
- `test`: Files in paths containing 'test' or 'spec'
- `config`: Files in paths containing 'config', 'conf', or 'setup'
- `model`: Files in paths containing 'model' or 'schema'
- `controller`: Files in paths containing 'controller', 'handler', 'router', or 'route'
- `service`: Files in paths containing 'service'
- `ui_component`: Files in paths containing 'component' or 'view'
- `package_init`: Files named `__init__.py`
- `entrypoint`: Files named:
  - Python: `main.py`, `cli.py`
  - JavaScript: `main.js`, `index.js`, `app.js`, `server.js`, `cli.js`

Files that don't match any of these patterns are categorized as `unknown`.

## Interactive Features

After generating a commit message, you can:

1. **Commit**: Save and commit the message
2. **Edit**: Modify the message in your editor
3. **Regenerate**: Generate a new message with additional context
4. **Abort**: Cancel the commit

You'll also be prompted to:
- Link related issues (e.g., "Fixes #123")
- Declare breaking changes with descriptions

## Project History

This project was developed using Axle itself! 
Here are some of the commit messages that were generated by Axle during development:

```
ca0e81ef60ca52f40fa6345bcdf74deefc6020e7
feat(tests): Add new test files for AI utils
```

``` 
2535d745da45049509b4690217fd494a0c2da32d
feat(axle): Add initial implementation of axle CLI tool
```

```
0618e91c437c540b64dea6394a79151d9faa2e4b
feat(src/axle): Add AI-based commit message generation functionality
```

Each commit message was automatically generated by Axle based on the actual changes made to the codebase.

## Configuration

The tool uses a configuration file at `~/.cache/axle/model_config.json` with the following structure:

```json
{
    "model_name": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "temperature": 0.7,
    "max_new_tokens": 200,
    "top_p": 0.95,
    "top_k": 50
}
```

You can modify these parameters to adjust the generation behavior.

## Commit Message Format

Commit messages are generated in JSON format with the following structure:

```json
{
    "type": "feat|fix|docs|style|refactor|test|chore",
    "scope": "optional-scope",
    "description": "concise description of changes"
}
```

The final commit message will include:
- Type and scope (if any)
- Description
- Issue references (if provided)
- Breaking change description (if declared)

## Requirements

- Python 3.9+
- Git
- A text editor (vim, nano, etc.)

## Dependencies

- click
- transformers
- torch
- accelerate
- tree-sitter (>=0.20.3,<0.23.0)
- tree-sitter-language-pack
- pydantic (>=2.0.0)

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 