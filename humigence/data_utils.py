# humigence/data_utils.py
import json
from pathlib import Path

from rich.console import Console

console = Console()


def create_demo_dataset(out: Path, schema: str = "chat_messages", n: int = 1000) -> Path:
    """Create a realistic fine-tuning dataset with diverse, high-quality examples."""
    out = Path(out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    # Realistic conversation templates for fine-tuning
    conversation_templates = [
        {
            "system": "You are a helpful, knowledgeable, and friendly AI assistant.",
            "user": "What is machine learning?",
            "assistant": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns."
        },
        {
            "system": "You are an expert in data science and programming.",
            "user": "How do I implement a neural network in Python?",
            "assistant": "You can implement a neural network in Python using libraries like PyTorch or TensorFlow. Here's a basic example with PyTorch:\n\n```python\nimport torch\nimport torch.nn as nn\n\nclass SimpleNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(input_size, hidden_size)\n        self.fc2 = nn.Linear(hidden_size, output_size)\n```"
        },
        {
            "system": "You are a helpful coding assistant.",
            "user": "What's the difference between a list and a tuple in Python?",
            "assistant": "Lists and tuples are both sequence types in Python, but they have key differences:\n\n- Lists are mutable (can be changed after creation)\n- Tuples are immutable (cannot be changed after creation)\n- Lists use square brackets [1, 2, 3]\n- Tuples use parentheses (1, 2, 3)\n- Lists are typically used for collections that change\n- Tuples are used for collections that shouldn't change"
        },
        {
            "system": "You are a knowledgeable AI assistant.",
            "user": "Explain the concept of overfitting in machine learning.",
            "assistant": "Overfitting occurs when a machine learning model learns the training data too well, including noise and irrelevant patterns. This results in poor generalization to new, unseen data. Signs of overfitting include:\n\n- High training accuracy but low validation accuracy\n- Model memorizes training data instead of learning patterns\n- Poor performance on new data\n\nTo prevent overfitting, use techniques like regularization, cross-validation, early stopping, and data augmentation."
        },
        {
            "system": "You are a helpful technical assistant.",
            "user": "What is the difference between supervised and unsupervised learning?",
            "assistant": "Supervised learning uses labeled training data where the correct output is provided, while unsupervised learning works with unlabeled data to find hidden patterns:\n\nSupervised Learning:\n- Has labeled training data\n- Learns to map inputs to known outputs\n- Examples: classification, regression\n- Can measure performance directly\n\nUnsupervised Learning:\n- Works with unlabeled data\n- Discovers hidden patterns and structures\n- Examples: clustering, dimensionality reduction\n- Performance harder to evaluate"
        }
    ]

    rows = []
    if schema == "chat_messages":
        for i in range(n):
            # Use templates with variations to create diverse data
            template = conversation_templates[i % len(conversation_templates)]

            # Add variations to make each example unique
            variation = i % 10
            if variation == 0:
                user_msg = template["user"]
                assistant_msg = template["assistant"]
            elif variation == 1:
                user_msg = f"Can you explain: {template['user']}"
                assistant_msg = f"Certainly! {template['assistant']}"
            elif variation == 2:
                user_msg = f"I need help understanding: {template['user']}"
                assistant_msg = f"I'd be happy to help! {template['assistant']}"
            elif variation == 3:
                user_msg = f"Tell me about: {template['user']}"
                assistant_msg = f"Here's what you should know: {template['assistant']}"
            elif variation == 4:
                user_msg = f"What do you know about: {template['user']}"
                assistant_msg = f"Let me explain: {template['assistant']}"
            elif variation == 5:
                user_msg = f"Help me with: {template['user']}"
                assistant_msg = f"I can assist you with that! {template['assistant']}"
            elif variation == 6:
                user_msg = f"Can you clarify: {template['user']}"
                assistant_msg = f"Of course! {template['assistant']}"
            elif variation == 7:
                user_msg = f"I'm confused about: {template['user']}"
                assistant_msg = f"Let me break this down for you: {template['assistant']}"
            elif variation == 8:
                user_msg = f"Explain this concept: {template['user']}"
                assistant_msg = f"Here's a clear explanation: {template['assistant']}"
            else:
                user_msg = f"I want to learn about: {template['user']}"
                assistant_msg = f"Great question! {template['assistant']}"

            rows.append({
                "messages": [
                    {"role": "system", "content": template["system"]},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg}
                ]
            })
    else:
        # Generic fallback for other schemas
        for i in range(n):
            rows.append({"text": f"Sample text #{i} for training purposes."})

    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    console.print(f"[green]✔ Realistic fine-tuning dataset written:[/green] {out}  ({len(rows)} rows)")
    console.print(f"[yellow]Note: This dataset will enable proper training with {n} samples[/yellow]")
    return out


def doctor_dataset(path: Path) -> dict:
    path = Path(path).expanduser().resolve()
    info = {"exists": path.exists(), "lines": 0, "first_row": None}
    if not path.exists():
        return info
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                try:
                    import json

                    info["first_row"] = json.loads(line.strip())
                except Exception:
                    info["first_row"] = "INVALID_JSON"
            info["lines"] += 1
    return info
