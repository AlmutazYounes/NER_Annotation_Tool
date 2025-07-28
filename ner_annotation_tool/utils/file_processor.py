import pandas as pd
import re
from typing import List
import os

class FileProcessor:
    """Utility class for processing uploaded files and extracting sentences"""
    
    def __init__(self):
        self.sentence_endings = r'[.!?]+(?:\s|$)'
    
    def process_file(self, filepath: str) -> List[str]:
        """
        Process uploaded file and return list of sentences
        Supports .txt and .csv files
        """
        file_extension = os.path.splitext(filepath)[1].lower()
        
        if file_extension == '.txt':
            return self._process_txt_file(filepath)
        elif file_extension == '.csv':
            return self._process_csv_file(filepath)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _process_txt_file(self, filepath: str) -> List[str]:
        """Process plain text file and extract sentences"""
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Split into sentences
            sentences = self._split_into_sentences(content)
            return [s.strip() for s in sentences if s.strip()]
        
        except UnicodeDecodeError:
            # Try with different encoding
            with open(filepath, 'r', encoding='latin-1') as file:
                content = file.read()
            
            sentences = self._split_into_sentences(content)
            return [s.strip() for s in sentences if s.strip()]
    
    def _process_csv_file(self, filepath: str) -> List[str]:
        """
        Process CSV file and extract sentences
        Assumes sentences are in a column named 'text', 'sentence', or the first column
        """
        try:
            df = pd.read_csv(filepath)
            
            # Try to find the text column
            text_column = None
            possible_columns = ['text', 'sentence', 'content', 'data']
            
            for col in possible_columns:
                if col in df.columns:
                    text_column = col
                    break
            
            # If no standard column found, use the first column
            if text_column is None:
                text_column = df.columns[0]
            
            # Extract sentences from the column
            sentences = []
            for text in df[text_column].dropna():
                if isinstance(text, str):
                    # Split each cell into sentences if it contains multiple sentences
                    cell_sentences = self._split_into_sentences(text)
                    sentences.extend([s.strip() for s in cell_sentences if s.strip()])
                else:
                    # Convert to string if not already
                    text_str = str(text).strip()
                    if text_str:
                        cell_sentences = self._split_into_sentences(text_str)
                        sentences.extend([s.strip() for s in cell_sentences if s.strip()])
            
            return sentences
        
        except Exception as e:
            raise ValueError(f"Error processing CSV file: {str(e)}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex
        Handles common sentence endings and abbreviations
        """
        # Clean up the text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Simple sentence splitting - can be improved with more sophisticated NLP
        # This regex looks for sentence endings followed by whitespace and capital letter
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Additional cleanup
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Ensure sentence ends with punctuation
                if not re.search(r'[.!?]$', sentence):
                    sentence += '.'
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def validate_file(self, filepath: str) -> dict:
        """
        Validate uploaded file and return information about it
        """
        if not os.path.exists(filepath):
            return {'valid': False, 'error': 'File does not exist'}
        
        file_size = os.path.getsize(filepath)
        file_extension = os.path.splitext(filepath)[1].lower()
        
        # Check file size (max 16MB)
        if file_size > 16 * 1024 * 1024:
            return {'valid': False, 'error': 'File too large (max 16MB)'}
        
        # Check file extension
        if file_extension not in ['.txt', '.csv']:
            return {'valid': False, 'error': 'Unsupported file type'}
        
        try:
            sentences = self.process_file(filepath)
            sentence_count = len(sentences)
            
            if sentence_count == 0:
                return {'valid': False, 'error': 'No sentences found in file'}
            
            return {
                'valid': True,
                'file_size': file_size,
                'file_type': file_extension,
                'sentence_count': sentence_count,
                'preview': sentences[:3] if sentences else []  # First 3 sentences as preview
            }
        
        except Exception as e:
            return {'valid': False, 'error': f'Error processing file: {str(e)}'}
    
    def get_file_stats(self, sentences: List[str]) -> dict:
        """Get statistics about the processed sentences"""
        if not sentences:
            return {'total_sentences': 0, 'avg_length': 0, 'min_length': 0, 'max_length': 0}
        
        lengths = [len(sentence.split()) for sentence in sentences]
        
        return {
            'total_sentences': len(sentences),
            'avg_length': sum(lengths) / len(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'total_words': sum(lengths)
        }
