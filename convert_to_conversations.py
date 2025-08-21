#!/usr/bin/env python3
"""Convert OpenAssist single messages to conversation pairs for Humigence."""

import json
from pathlib import Path
from datasets import load_from_disk
from collections import defaultdict

def convert_to_conversations():
    """Convert OpenAssist dataset to conversation pairs."""
    
    # Load dataset
    dataset = load_from_disk("/home/joshua/fine_tuning_project/data/oasst1")
    output_path = Path("data/raw/oasst1_conversations.jsonl")
    
    print("Converting OpenAssist to conversation pairs...")
    
    # Group messages by conversation tree
    conversations = defaultdict(list)
    
    # Process train split
    print("Processing train split...")
    for i, example in enumerate(dataset['train']):
        if i % 10000 == 0:
            print(f"  Processed {i} examples...")
        
        tree_id = example['message_tree_id']
        role = example['role']
        text = example['text']
        
        # Map roles
        if role == 'prompter':
            role = 'user'
        elif role == 'assistant':
            role = 'assistant'
        else:
            role = 'user'
        
        conversations[tree_id].append({
            'role': role,
            'content': text,
            'rank': example.get('rank', 0)
        })
    
    # Process validation split
    print("Processing validation split...")
    for i, example in enumerate(dataset['validation']):
        if i % 1000 == 0:
            print(f"  Processed {i} examples...")
        
        tree_id = example['message_tree_id']
        role = example['role']
        text = example['text']
        
        # Map roles
        if role == 'prompter':
            role = 'user'
        elif role == 'assistant':
            role = 'assistant'
        else:
            role = 'user'
        
        conversations[tree_id].append({
            'role': role,
            'content': text,
            'rank': example.get('rank', 0)
        })
    
    # Create conversation pairs
    print("Creating conversation pairs...")
    conversation_pairs = []
    
    for tree_id, messages in conversations.items():
        if len(messages) < 2:
            continue  # Skip conversations with less than 2 messages
        
        # Sort by rank (assistant responses have rank, user prompts don't)
        user_messages = [m for m in messages if m['role'] == 'user']
        assistant_messages = [m for m in messages if m['role'] == 'assistant']
        
        # Create pairs: user -> assistant
        for user_msg in user_messages:
            # Find best assistant response (lowest rank, or first if no rank)
            best_assistant = None
            if assistant_messages:
                best_assistant = min(assistant_messages, key=lambda x: x.get('rank', 999))
            
            if best_assistant:
                conversation_pairs.append({
                    "messages": [
                        {"role": user_msg['role'], "content": user_msg['content']},
                        {"role": best_assistant['role'], "content": best_assistant['content']}
                    ]
                })
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pair in conversation_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    
    print(f"✅ Conversion complete!")
    print(f"Output file: {output_path}")
    print(f"Total conversation pairs: {len(conversation_pairs):,}")
    
    # Verify first few examples
    print("\nVerifying first 3 conversation pairs:")
    with open(output_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            data = json.loads(line.strip())
            print(f"Pair {i}: {len(data['messages'])} messages")
            for j, msg in enumerate(data['messages']):
                print(f"  Message {j}: {msg['role']} - {msg['content'][:50]}...")
    
    return output_path

if __name__ == "__main__":
    convert_to_conversations()
