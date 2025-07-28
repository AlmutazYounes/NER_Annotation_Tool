import json
import tempfile
import os
from typing import List, Dict, Any, Optional
from models.database import Database

try:
    from huggingface_hub import HfApi, create_repo
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

class ExportUtils:
    """Utility class for exporting annotations in various formats"""
    
    def __init__(self, database: Database):
        self.db = database
    
    def export_to_huggingface_format(self, project_id: int) -> Dict[str, Any]:
        """
        Export project annotations in Hugging Face datasets format
        Returns both IOB format and span-based format
        """
        project = self.db.get_project(project_id)
        if not project:
            raise ValueError(f"Project {project_id} not found")
        
        # Get all sentences for the project
        conn = self.db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT s.id, s.text, s.sentence_index, s.is_annotated
            FROM sentences s
            WHERE s.project_id = ?
            ORDER BY s.sentence_index
        ''', (project_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Process data for both formats
        iob_data = []
        span_data = []
        
        for row in rows:
            sentence_text = row['text']
            sentence_id = row['id']

            # Get annotations for this sentence directly
            annotations = self.db.get_annotations(sentence_id)
            
            # Generate IOB format
            iob_tokens, iob_labels = self._generate_iob_format(sentence_text, annotations)
            iob_data.append({
                'id': row['id'],
                'tokens': iob_tokens,
                'ner_tags': iob_labels
            })
            
            # Generate span format
            span_entities = []
            for ann in annotations:
                span_entities.append({
                    'start': ann['start_pos'],
                    'end': ann['end_pos'],
                    'label': ann['entity_label'],
                    'text': ann['annotated_text']
                })
            
            span_data.append({
                'id': row['id'],
                'text': sentence_text,
                'entities': span_entities
            })
        
        # Create the export package
        export_package = {
            'project_info': {
                'name': project['name'],
                'description': project['description'],
                'entity_tags': project['entity_tags'],
                'created_at': project['created_at']
            },
            'data_formats': {
                'iob_format': {
                    'description': 'IOB (Inside-Outside-Begin) tagging format',
                    'data': iob_data
                },
                'span_format': {
                    'description': 'Span-based entity format',
                    'data': span_data
                }
            },
            'statistics': self._generate_statistics(project_id, iob_data, span_data)
        }
        
        return export_package
    
    def _generate_iob_format(self, text: str, annotations: List[Dict]) -> tuple:
        """
        Generate IOB format tokens and labels for a sentence
        """
        # Simple tokenization (split by whitespace)
        tokens = text.split()
        labels = ['O'] * len(tokens)
        
        # Sort annotations by start position
        sorted_annotations = sorted(annotations, key=lambda x: x['start_pos'])
        
        # Map character positions to token positions
        char_to_token = {}
        current_pos = 0
        
        for i, token in enumerate(tokens):
            # Find the token in the original text
            token_start = text.find(token, current_pos)
            if token_start != -1:
                token_end = token_start + len(token)
                for char_pos in range(token_start, token_end):
                    char_to_token[char_pos] = i
                current_pos = token_end
        
        # Apply annotations
        for annotation in sorted_annotations:
            start_pos = annotation['start_pos']
            end_pos = annotation['end_pos']
            label = annotation['entity_label']
            
            # Find which tokens this annotation spans
            start_token = None
            end_token = None
            
            for char_pos in range(start_pos, end_pos):
                if char_pos in char_to_token:
                    token_idx = char_to_token[char_pos]
                    if start_token is None:
                        start_token = token_idx
                    end_token = token_idx
            
            # Apply IOB labels
            if start_token is not None and end_token is not None:
                for token_idx in range(start_token, end_token + 1):
                    if token_idx == start_token:
                        labels[token_idx] = f'B-{label}'
                    else:
                        labels[token_idx] = f'I-{label}'
        
        return tokens, labels
    
    def _generate_statistics(self, project_id: int, iob_data: List[Dict], span_data: List[Dict]) -> Dict:
        """Generate statistics about the exported data"""
        progress = self.db.get_project_progress(project_id)
        
        # Count entities by type
        entity_counts = {}
        total_entities = 0
        
        for item in span_data:
            for entity in item['entities']:
                label = entity['label']
                entity_counts[label] = entity_counts.get(label, 0) + 1
                total_entities += 1
        
        # Calculate token statistics
        total_tokens = sum(len(item['tokens']) for item in iob_data)
        
        return {
            'total_sentences': progress['total_sentences'],
            'annotated_sentences': progress['annotated_sentences'],
            'annotation_progress': progress['progress_percentage'],
            'total_entities': total_entities,
            'entity_distribution': entity_counts,
            'total_tokens': total_tokens,
            'avg_tokens_per_sentence': total_tokens / len(iob_data) if iob_data else 0
        }
    
    def export_to_jsonl(self, project_id: int, format_type: str = 'span') -> str:
        """
        Export data as JSONL string
        format_type: 'span' or 'iob'
        """
        export_data = self.export_to_huggingface_format(project_id)
        
        if format_type == 'span':
            data = export_data['data_formats']['span_format']['data']
        else:
            data = export_data['data_formats']['iob_format']['data']
        
        jsonl_lines = []
        for item in data:
            jsonl_lines.append(json.dumps(item))
        
        return '\n'.join(jsonl_lines)
    
    def export_to_conll(self, project_id: int) -> str:
        """
        Export data in CoNLL format (token per line, empty line between sentences)
        """
        export_data = self.export_to_huggingface_format(project_id)
        iob_data = export_data['data_formats']['iob_format']['data']
        
        conll_lines = []
        
        for item in iob_data:
            tokens = item['tokens']
            labels = item['ner_tags']
            
            for token, label in zip(tokens, labels):
                conll_lines.append(f'{token}\t{label}')
            
            # Empty line between sentences
            conll_lines.append('')
        
        return '\n'.join(conll_lines)

    def upload_to_huggingface(self, project_id: int, hf_token: str, dataset_name: str,
                             description: str = "", private: bool = False) -> Dict[str, Any]:
        """
        Upload project annotations to Hugging Face Hub
        """
        if not HF_AVAILABLE:
            return {'success': False, 'error': 'Hugging Face Hub library not available'}

        try:
            # Get project data
            project = self.db.get_project(project_id)
            if not project:
                return {'success': False, 'error': 'Project not found'}

            # Export data
            export_data = self.export_to_huggingface_format(project_id)

            # Initialize HF API
            api = HfApi()

            # Create repository
            repo_id = f"{api.whoami(token=hf_token)['name']}/{dataset_name}"

            try:
                create_repo(
                    repo_id=repo_id,
                    token=hf_token,
                    repo_type="dataset",
                    private=private
                )
            except Exception as e:
                if "already exists" not in str(e).lower():
                    return {'success': False, 'error': f'Failed to create repository: {str(e)}'}

            # Create temporary files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save dataset files
                train_file = os.path.join(temp_dir, "train.jsonl")
                with open(train_file, 'w', encoding='utf-8') as f:
                    for item in export_data['data_formats']['span_format']['data']:
                        f.write(json.dumps(item) + '\n')

                # Create dataset card
                card_content = f"""---
license: mit
task_categories:
- token-classification
language:
- en
tags:
- named-entity-recognition
- ner
- annotation
size_categories:
- n<1K
---

# {project['name']}

{description or project.get('description', 'Named Entity Recognition dataset')}

## Dataset Information

- **Created with**: DataLabel Pro
- **Task**: Named Entity Recognition
- **Entity Types**: {', '.join([tag.get('name', tag) if isinstance(tag, dict) else tag for tag in project['entity_tags']])}
- **Total Sentences**: {export_data['statistics']['total_sentences']}
- **Total Entities**: {export_data['statistics']['total_entities']}

## Entity Types and Colors

"""

                for tag in project['entity_tags']:
                    if isinstance(tag, dict):
                        card_content += f"- **{tag['name']}**: {tag.get('color', '#000000')}\n"
                    else:
                        card_content += f"- **{tag}**\n"

                card_content += f"""

## Statistics

- Annotation Progress: {export_data['statistics']['annotation_progress']:.1f}%
- Average Tokens per Sentence: {export_data['statistics']['avg_tokens_per_sentence']:.1f}

## Data Format

The dataset is provided in JSONL format with the following structure:

```json
{{
  "id": 1,
  "text": "Apple Inc. is based in Cupertino, California.",
  "entities": [
    {{"start": 0, "end": 10, "label": "ORG", "text": "Apple Inc."}},
    {{"start": 23, "end": 32, "label": "LOC", "text": "Cupertino"}},
    {{"start": 34, "end": 44, "label": "LOC", "text": "California"}}
  ]
}}
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("{repo_id}")
```
"""

                readme_file = os.path.join(temp_dir, "README.md")
                with open(readme_file, 'w', encoding='utf-8') as f:
                    f.write(card_content)

                # Upload files
                api.upload_folder(
                    folder_path=temp_dir,
                    repo_id=repo_id,
                    repo_type="dataset",
                    token=hf_token,
                    commit_message=f"Upload NER dataset: {project['name']}"
                )

            return {
                'success': True,
                'repo_id': repo_id,
                'url': f"https://huggingface.co/datasets/{repo_id}",
                'message': 'Dataset uploaded successfully to Hugging Face Hub'
            }

        except Exception as e:
            return {'success': False, 'error': f'Upload failed: {str(e)}'}
