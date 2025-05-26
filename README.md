# Axle

Axle is a command-line tool that helps you write better commit messages using AI. It analyzes your code changes and suggests a commit message in a structured format.

## Features

- 🤖 AI-powered commit message generation using the Qwen2.5-Coder-3B-Instruct model
- 🔄 Interactive regeneration of commit messages (in-development)
- 💾 Automatic model caching for faster subsequent runs
- 🛠️ Configurable generation parameters
- 📚 Support for conventional commit types and scopes

## Installation

```bash
pip install git+https://github.com/onkardahale/axle.git
```

## Usage

1. Stage your changes:
```bash
git add .
```

2. Generate a commit message:
```bash
axle 
```

3. The tool will:
   - Analyze your staged changes
   - Generate a commit message
   - Open your default editor for review
   - Create the commit if you save and close the editor

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

## Requirements

- Python 3.8+
- Git
- A text editor (vim, nano, etc.)

## Dependencies

- transformers
- torch
- accelerate
- click

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 