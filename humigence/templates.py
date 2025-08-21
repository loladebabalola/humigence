"""
Chat templates for Humigence.
Provides prompt formatting for Qwen models.
"""


class ChatTemplate:
    """Chat template for Qwen models."""

    def __init__(self, model_name: str = "qwen"):
        self.model_name = model_name.lower()
        self._setup_template()

    def _setup_template(self):
        """Set up the appropriate template based on model name."""
        if "qwen" in self.model_name:
            self.user_prefix = "<|im_start|>user\n"
            self.user_suffix = "<|im_end|>\n"
            self.assistant_prefix = "<|im_start|>assistant\n"
            self.assistant_suffix = "<|im_end|>\n"
            self.system_prefix = "<|im_start|>system\n"
            self.system_suffix = "<|im_end|>\n"
        else:
            # Default to Qwen format
            self.user_prefix = "<|im_start|>user\n"
            self.user_suffix = "<|im_end|>\n"
            self.assistant_prefix = "<|im_start|>assistant\n"
            self.assistant_suffix = "<|im_end|>\n"
            self.system_prefix = "<|im_start|>system\n"
            self.system_suffix = "<|im_end|>\n"

    def format_user_message(self, message: str) -> str:
        """Format a user message."""
        return f"{self.user_prefix}{message}{self.user_suffix}"

    def format_assistant_message(self, message: str) -> str:
        """Format an assistant message."""
        return f"{self.assistant_prefix}{message}{self.assistant_suffix}"

    def format_system_message(self, message: str) -> str:
        """Format a system message."""
        return f"{self.system_prefix}{message}{self.system_suffix}"

    def format_chat(
        self,
        messages: list[dict[str, str]],
        system_message: str | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Format a chat conversation.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            system_message: Optional system message to prepend
            add_generation_prompt: Whether to add the assistant prefix for generation

        Returns:
            Formatted chat string
        """
        formatted_parts = []

        # Add system message if provided
        if system_message:
            formatted_parts.append(self.format_system_message(system_message))

        # Format each message
        for message in messages:
            role = message.get("role", "").lower()
            content = message.get("content", "")

            if role == "user":
                formatted_parts.append(self.format_user_message(content))
            elif role == "assistant":
                formatted_parts.append(self.format_assistant_message(content))
            elif role == "system":
                formatted_parts.append(self.format_system_message(content))
            else:
                # Unknown role, treat as user message
                formatted_parts.append(self.format_user_message(content))

        # Add generation prompt if requested
        if add_generation_prompt:
            formatted_parts.append(self.assistant_prefix.rstrip())

        return "".join(formatted_parts)

    def format_instruction(
        self,
        instruction: str,
        input_text: str | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Format an instruction-following prompt.

        Args:
            instruction: The instruction to follow
            input_text: Optional input text
            add_generation_prompt: Whether to add the assistant prefix for generation

        Returns:
            Formatted instruction string
        """
        if input_text:
            prompt = f"Instruction: {instruction}\n\nInput: {input_text}\n\nResponse:"
        else:
            prompt = f"Instruction: {instruction}\n\nResponse:"

        if add_generation_prompt:
            prompt += f"\n{self.assistant_prefix.rstrip()}"

        return prompt

    def format_qa(
        self,
        question: str,
        context: str | None = None,
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Format a question-answering prompt.

        Args:
            question: The question to answer
            context: Optional context information
            add_generation_prompt: Whether to add the assistant prefix for generation

        Returns:
            Formatted QA string
        """
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"

        if add_generation_prompt:
            prompt += f"\n{self.assistant_prefix.rstrip()}"

        return prompt

    def get_stop_tokens(self) -> list[str]:
        """Get stop tokens for the model."""
        if "qwen" in self.model_name:
            return ["<|im_end|>", "<|endoftext|>"]
        else:
            return ["<|im_end|>", "<|endoftext|>"]

    def get_eos_token(self) -> str:
        """Get the end-of-sequence token."""
        if "qwen" in self.model_name:
            return "<|im_end|>"
        else:
            return "<|im_end|>"


# Convenience functions
def format_chat_messages(
    messages: list[dict[str, str]],
    model_name: str = "qwen",
    system_message: str | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """Format chat messages using the default template."""
    template = ChatTemplate(model_name)
    return template.format_chat(messages, system_message, add_generation_prompt)


def format_instruction_prompt(
    instruction: str,
    input_text: str | None = None,
    model_name: str = "qwen",
    add_generation_prompt: bool = True,
) -> str:
    """Format an instruction prompt using the default template."""
    template = ChatTemplate(model_name)
    return template.format_instruction(instruction, input_text, add_generation_prompt)


def format_qa_prompt(
    question: str,
    context: str | None = None,
    model_name: str = "qwen",
    add_generation_prompt: bool = True,
) -> str:
    """Format a QA prompt using the default template."""
    template = ChatTemplate(model_name)
    return template.format_qa(question, context, add_generation_prompt)
