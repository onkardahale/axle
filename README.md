# Axle

Axle is a command-line tool that helps you write better commit messages using AI. It analyzes your code changes and suggests a commit message in a structured format.

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Knowledge Base](#knowledge-base)
- [Language Support](#language-support)
- [Interactive Features](#interactive-features)
- [Configuration](#configuration)
- [Contributing](#contributing)

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

## Quick Start

```bash
# Install
pip install git+https://github.com/onkardahale/axle.git

# Initialize knowledge base (optional but recommended)
axle init

# Stage your changes and generate commit message
git add .
axle commit
```

## Installation

```bash
pip install git+https://github.com/onkardahale/axle.git
```

## Usage

1. **Initialize the knowledge base** (recommended for better commit messages):
```bash
axle init
```

2. **Stage your changes:**
```bash
git add .
```

3. **Generate a commit message:**
```bash
axle commit
```

4. **The tool will:**
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

### Ignoring Files and Directories

Create a `.axleignore` file in your project root to control which files are analyzed:

```
# Build directories
build/
dist/
target/

# Temporary files
*.tmp
*.log
temp*

# IDE files
.vscode/
.idea/

# Documentation
docs/
*.md
```

**Default ignored directories:** `.git`, `.hg`, `.svn`, `.vscode`, `node_modules`, `__pycache__`, `.axle`

## Language Support

Axle supports multiple programming languages through dedicated analyzers:

### **Currently Supported:**
- **Python** (`.py`)
- **JavaScript** (`.js`, `.jsx`, `.mjs`, `.cjs`)
- **C++** (`.cpp`, `.cc`, `.cxx`, `.hpp`, `.hxx`)
- **Julia** (`.jl`) ✨ *New!*

### **Planned Languages:**
- TypeScript (`.ts`, `.tsx`)
- Java (`.java`)
- Go (`.go`)
- Rust (`.rs`)
- Ruby (`.rb`)
- PHP (`.php`)

> **Note:** While tree-sitter-language-pack provides grammar support for many languages, each requires a dedicated analyzer implementation. [Contributions welcome!](#contributing-language-analyzers)

### File Categorization

Files are automatically categorized based on:

**Import-based categories:**
- `web_framework`: Django, Flask, React, Vue, etc.
- `test`: pytest, jest, mocha, etc.
- `database`: SQLAlchemy, mongoose, etc.
- `data_processing`: pandas, numpy
- `ml`: tensorflow, pytorch, sklearn

**Path-based categories:**
- `util`, `test`, `config`, `model`, `controller`, `service`, `ui_component`, `entrypoint`

### Contributing Language Analyzers

To add support for a new language:

1. **Create analyzer** in `src/axle/treesitter/analyzers/`
2. **Implement required methods** following existing patterns
3. **Add tests** in `tests/test_<language>_analyzer.py`
4. **Update documentation**

See `javascript_analyzer.py`, `cpp_analyzer.py`, and `julia_analyzer.py` for reference implementations.

## Interactive Features

After generating a commit message, you can:

1. **Commit**: Save and commit the message
2. **Edit**: Modify the message in your editor
3. **Regenerate**: Generate a new message with additional context
4. **Abort**: Cancel the commit

You'll also be prompted to:
- Link related issues (e.g., "Fixes #123")
- Declare breaking changes with descriptions

## Configuration

Customize generation behavior via `~/.cache/axle/model_config.json`:

```json
{
    "model_name": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "temperature": 0.2,
    "top_p": 0.95,
    "num_return_sequences": 1
}
```

## Commit Message Format

Generated commits follow conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Requirements

- **Python 3.12+**
- **Git**
- **Text editor** (vim, nano, etc.)

## Dependencies

- click, transformers, torch, accelerate
- tree-sitter, tree-sitter-language-pack
- pydantic (>=2.0.0)

## Project History

This project was developed using Axle itself! Here are some example generated commit messages:

```
feat(tests): Add setup and teardown methods to BaseAxleTestCase for cleaner test environments

Added `setUp` and `tearDown` methods to `BaseAxleTestCase` to handle common setup 
and teardown tasks for tests. This includes creating temporary directories for 
backups and test data, backing up original configuration files, and cleaning up 
resources after tests.

Fixes: #8
```

```
refactor(ai_utils): improve robustness and error handling in commit message generation

Added checks to validate and clamp generation parameters to prevent tensor issues:
- Ensured temperature is within [0.01, 0.6]
- Clamped top_p to [0.01, 0.95]
- Set num_return_sequences to 1 to avoid excessive output

Fixes: #6
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

**Areas where we need help:**
- New language analyzers
- Improved file categorization
- Documentation improvements
- Bug reports and fixes

## License

MIT 