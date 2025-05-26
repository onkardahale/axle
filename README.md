# Axle

A CLI tool for generating Git commit messages.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/axle.git
cd axle

# Install in development mode
pip install -e .
```

## Usage

```bash
# Show version
axle --version

# Show help
axle --help

# Generate commit message (coming soon)
axle
```

## Development

```bash
# Run tests
python -m unittest discover tests
```

## License

MIT License

## AI Integration

The project now uses a Hugging Face model for generating commit messages. The model is loaded using the `transformers` library, and it generates commit messages based on the provided git diff.

### Setup

1. Ensure you have the required dependencies installed:
   ```bash
   pip install -e .
   ```

2. The model will be downloaded automatically when you first run the tool.

### Usage

- Run the tool as before, and it will use the Hugging Face model to generate commit messages.
- If you are not satisfied with the generated message, you can regenerate it by choosing the 'e' option in the interactive prompt.

### Error Handling

- If the model fails to generate a message, an error will be displayed, and the tool will exit with a non-zero status code.
- Ensure you have a stable internet connection for the initial model download. 