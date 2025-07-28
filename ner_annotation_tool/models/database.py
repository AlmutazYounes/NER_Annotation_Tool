import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import os

# OpenAI model pricing (per 1M tokens)
OPENAI_MODEL_PRICES = {
    # GPT-4.1 series
    "gpt-4.1": {
        "input": 2.00,
        "cached_input": 0.50,
        "output": 8.00
    },
    "gpt-4.1-mini": {
        "input": 0.40,
        "cached_input": 0.10,
        "output": 1.60
    },
    "gpt-4.1-nano": {
        "input": 0.10,
        "cached_input": 0.025,
        "output": 0.40
    },
    # GPT-4.5
    "gpt-4.5-preview": {
        "input": 75.00,
        "cached_input": 37.50,
        "output": 150.00
    },
    # GPT-4o series
    "gpt-4o": {
        "input": 2.50,
        "cached_input": 1.25,
        "output": 10.00
    },
    "gpt-4o-mini": {
        "input": 0.15,
        "cached_input": 0.075,
        "output": 0.60
    },
    # o-series
    "o1": {
        "input": 15.00,
        "cached_input": 7.50,
        "output": 60.00
    },
    "o1-pro": {
        "input": 150.00,
        "output": 600.00
    },
    "o1-mini": {
        "input": 1.10,
        "cached_input": 0.55,
        "output": 4.40
    },
    "o3": {
        "input": 2.00,
        "cached_input": 0.50,
        "output": 8.00
    },
    "o3-pro": {
        "input": 20.00,
        "output": 80.00
    },
    "o3-mini": {
        "input": 1.10,
        "cached_input": 0.55,
        "output": 4.40
    },
    "o4-mini": {
        "input": 1.10,
        "cached_input": 0.275,
        "output": 4.40
    },
}

class Database:
    def __init__(self, db_path: str = "ner_annotations.db"):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection with row factory for dict-like access"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        """Initialize database with required tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Projects table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                entity_tags TEXT NOT NULL,  -- JSON string of entity tags
                llm_enabled INTEGER DEFAULT 0,  -- Boolean: 1 if LLM is enabled, 0 if not
                llm_api_key TEXT,  -- OpenAI API key (should be encrypted in production)
                llm_model TEXT DEFAULT 'gpt-4.1-mini',  -- OpenAI model name
                llm_custom_instructions TEXT,  -- Custom instructions for the LLM
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Sentences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                text TEXT NOT NULL,
                sentence_index INTEGER NOT NULL,
                is_annotated BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
            )
        ''')
        
        # Annotations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence_id INTEGER NOT NULL,
                start_pos INTEGER NOT NULL,
                end_pos INTEGER NOT NULL,
                entity_label TEXT NOT NULL,
                annotated_text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (sentence_id) REFERENCES sentences (id) ON DELETE CASCADE
            )
        ''')

        # AI Usage tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cached_tokens INTEGER DEFAULT 0,
                input_cost REAL NOT NULL,
                output_cost REAL NOT NULL,
                cached_cost REAL DEFAULT 0,
                total_cost REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects (id) ON DELETE CASCADE
            )
        ''')

        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentences_project_id ON sentences(project_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_annotations_sentence_id ON annotations(sentence_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sentences_project_index ON sentences(project_id, sentence_index)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ai_usage_project_id ON ai_usage(project_id)')

        # Add LLM columns to existing projects table if they don't exist
        self._add_llm_columns_if_not_exist(cursor)

        conn.commit()
        conn.close()

    def _add_llm_columns_if_not_exist(self, cursor):
        """Add LLM configuration columns to projects table if they don't exist"""
        # Check if LLM columns exist
        cursor.execute("PRAGMA table_info(projects)")
        columns = [column[1] for column in cursor.fetchall()]

        if 'llm_enabled' not in columns:
            cursor.execute('ALTER TABLE projects ADD COLUMN llm_enabled INTEGER DEFAULT 0')
        if 'llm_api_key' not in columns:
            cursor.execute('ALTER TABLE projects ADD COLUMN llm_api_key TEXT')
        if 'llm_model' not in columns:
            cursor.execute('ALTER TABLE projects ADD COLUMN llm_model TEXT DEFAULT "gpt-4.1-mini"')
        if 'llm_custom_instructions' not in columns:
            cursor.execute('ALTER TABLE projects ADD COLUMN llm_custom_instructions TEXT')
        if 'auto_ai_enabled' not in columns:
            cursor.execute('ALTER TABLE projects ADD COLUMN auto_ai_enabled INTEGER DEFAULT 0')

    def create_project(self, name: str, description: str, entity_tags: List[Dict],
                      llm_config: Dict = None) -> int:
        """Create a new project and return its ID
        entity_tags should be a list of dicts with 'name' and 'color' keys
        llm_config should be a dict with 'enabled', 'api_key', 'model', 'custom_instructions' keys
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        entity_tags_json = json.dumps(entity_tags)

        # Handle LLM configuration
        llm_enabled = 0
        llm_api_key = None
        llm_model = 'gpt-4.1-mini'
        llm_custom_instructions = None

        if llm_config:
            llm_enabled = 1 if llm_config.get('enabled', False) else 0
            llm_api_key = llm_config.get('api_key')
            llm_model = llm_config.get('model', 'gpt-4.1-mini')
            llm_custom_instructions = llm_config.get('custom_instructions')

        cursor.execute('''
            INSERT INTO projects (name, description, entity_tags, llm_enabled,
                                llm_api_key, llm_model, llm_custom_instructions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (name, description, entity_tags_json, llm_enabled,
              llm_api_key, llm_model, llm_custom_instructions))

        project_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return project_id
    
    def get_project(self, project_id: int) -> Optional[Dict]:
        """Get project by ID"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM projects WHERE id = ?', (project_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            project = dict(row)
            project['entity_tags'] = json.loads(project['entity_tags'])
            return project
        return None
    
    def get_all_projects(self) -> List[Dict]:
        """Get all projects"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM projects ORDER BY created_at DESC')
        rows = cursor.fetchall()
        conn.close()
        
        projects = []
        for row in rows:
            project = dict(row)
            project['entity_tags'] = json.loads(project['entity_tags'])
            projects.append(project)
        return projects
    
    def add_sentences(self, project_id: int, sentences: List[str]) -> int:
        """Add sentences to a project and return count of added sentences"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get current max sentence index for this project
        cursor.execute('SELECT MAX(sentence_index) FROM sentences WHERE project_id = ?', (project_id,))
        max_index = cursor.fetchone()[0] or -1
        
        # Insert sentences
        sentence_data = [
            (project_id, sentence.strip(), max_index + i + 1)
            for i, sentence in enumerate(sentences) if sentence.strip()
        ]
        
        cursor.executemany('''
            INSERT INTO sentences (project_id, text, sentence_index)
            VALUES (?, ?, ?)
        ''', sentence_data)
        
        count = cursor.rowcount
        conn.commit()
        conn.close()
        return count
    
    def get_sentence(self, project_id: int, sentence_index: int) -> Optional[Dict]:
        """Get a specific sentence by project and index"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM sentences 
            WHERE project_id = ? AND sentence_index = ?
        ''', (project_id, sentence_index))
        
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def get_sentence_count(self, project_id: int) -> int:
        """Get total number of sentences in a project"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM sentences WHERE project_id = ?', (project_id,))
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_annotations(self, sentence_id: int) -> List[Dict]:
        """Get all annotations for a sentence"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM annotations 
            WHERE sentence_id = ? 
            ORDER BY start_pos
        ''', (sentence_id,))
        
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    
    def get_annotations_for_sentence(self, sentence_id: int) -> List[Dict]:
        """Get all annotations for a specific sentence"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, sentence_id, start_pos, end_pos, entity_label, annotated_text, created_at
            FROM annotations
            WHERE sentence_id = ?
            ORDER BY start_pos
        ''', (sentence_id,))

        annotations = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return annotations

    def save_annotation(self, sentence_id: int, start_pos: int, end_pos: int,
                       entity_label: str, annotated_text: str) -> int:
        """Save an annotation and return its ID"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO annotations (sentence_id, start_pos, end_pos, entity_label, annotated_text)
            VALUES (?, ?, ?, ?, ?)
        ''', (sentence_id, start_pos, end_pos, entity_label, annotated_text))

        annotation_id = cursor.lastrowid

        # Mark sentence as annotated
        cursor.execute('UPDATE sentences SET is_annotated = TRUE WHERE id = ?', (sentence_id,))

        conn.commit()
        conn.close()
        return annotation_id
    
    def delete_annotation(self, annotation_id: int) -> bool:
        """Delete an annotation"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM annotations WHERE id = ?', (annotation_id,))
        success = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        return success
    
    def get_project_progress(self, project_id: int) -> Dict:
        """Get annotation progress for a project"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_sentences,
                SUM(CASE WHEN is_annotated THEN 1 ELSE 0 END) as annotated_sentences
            FROM sentences 
            WHERE project_id = ?
        ''', (project_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        total = row[0] or 0
        annotated = row[1] or 0
        
        return {
            'total_sentences': total,
            'annotated_sentences': annotated,
            'progress_percentage': (annotated / total * 100) if total > 0 else 0
        }

    def get_project_analytics(self, project_id: int) -> Dict:
        """Get comprehensive analytics for a project"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Get basic project info
            project = self.get_project(project_id)
            if not project:
                return {}

            # Get entity type counts
            cursor.execute('''
                SELECT entity_label, COUNT(*) as count
                FROM annotations a
                JOIN sentences s ON a.sentence_id = s.id
                WHERE s.project_id = ?
                GROUP BY entity_label
                ORDER BY count DESC
            ''', (project_id,))
            entity_counts = [{'label': row[0], 'count': row[1]} for row in cursor.fetchall()]

            # Get progress metrics
            progress = self.get_project_progress(project_id)

            # Get annotation density (annotations per sentence)
            cursor.execute('''
                SELECT
                    COUNT(DISTINCT a.id) as total_annotations,
                    COUNT(DISTINCT s.id) as annotated_sentences,
                    CAST(COUNT(DISTINCT a.id) AS FLOAT) / NULLIF(COUNT(DISTINCT s.id), 0) as avg_annotations_per_sentence
                FROM sentences s
                LEFT JOIN annotations a ON s.id = a.sentence_id
                WHERE s.project_id = ?
                AND s.is_annotated = TRUE
            ''', (project_id,))
            density_result = cursor.fetchone()

            # Get annotation timeline (annotations by date)
            cursor.execute('''
                SELECT
                    DATE(a.created_at) as date,
                    COUNT(*) as count
                FROM annotations a
                JOIN sentences s ON a.sentence_id = s.id
                WHERE s.project_id = ?
                GROUP BY DATE(a.created_at)
                ORDER BY date
            ''', (project_id,))
            timeline_data = [{'date': row[0], 'count': row[1]} for row in cursor.fetchall()]

            # Calculate total annotations
            total_annotations = sum(item['count'] for item in entity_counts)

            # Calculate entity distribution percentages
            for item in entity_counts:
                item['percentage'] = (item['count'] / total_annotations * 100) if total_annotations > 0 else 0

            analytics = {
                'project': project,
                'entity_counts': entity_counts,
                'total_annotations': total_annotations,
                'progress': progress,
                'annotation_density': {
                    'total_annotations': density_result[0] if density_result else 0,
                    'annotated_sentences': density_result[1] if density_result else 0,
                    'avg_per_sentence': round(density_result[2], 2) if density_result and density_result[2] else 0
                },
                'timeline': timeline_data
            }

            return analytics

        finally:
            conn.close()

    def get_entity_statistics(self, project_id: int) -> Dict:
        """Get detailed entity statistics for charts"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Get entity counts with colors from project config
            project = self.get_project(project_id)
            if not project:
                return {}

            entity_tags = project.get('entity_tags', [])
            entity_colors = {tag['name']: tag['color'] for tag in entity_tags}

            cursor.execute('''
                SELECT entity_label, COUNT(*) as count
                FROM annotations a
                JOIN sentences s ON a.sentence_id = s.id
                WHERE s.project_id = ?
                GROUP BY entity_label
                ORDER BY count DESC
            ''', (project_id,))

            results = cursor.fetchall()
            total = sum(row[1] for row in results)

            statistics = {
                'labels': [row[0] for row in results],
                'counts': [row[1] for row in results],
                'colors': [entity_colors.get(row[0], '#6c757d') for row in results],
                'percentages': [round(row[1] / total * 100, 1) if total > 0 else 0 for row in results],
                'total': total
            }

            return statistics

        finally:
            conn.close()

    def update_project_entity_tags(self, project_id: int, entity_tags: List[Dict],
                                  tag_mappings: Dict[str, str] = None) -> bool:
        """
        Update entity tags for a project and optionally update existing annotations
        tag_mappings: dict mapping old tag names to new tag names for annotation updates
        """
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Update project entity tags
            entity_tags_json = json.dumps(entity_tags)
            cursor.execute('''
                UPDATE projects
                SET entity_tags = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (entity_tags_json, project_id))

            # Update existing annotations if tag mappings provided
            if tag_mappings:
                for old_tag, new_tag in tag_mappings.items():
                    if old_tag != new_tag:  # Only update if tag name actually changed
                        cursor.execute('''
                            UPDATE annotations
                            SET entity_label = ?
                            WHERE entity_label = ? AND sentence_id IN (
                                SELECT id FROM sentences WHERE project_id = ?
                            )
                        ''', (new_tag, old_tag, project_id))

            conn.commit()
            return True

        except Exception as e:
            conn.rollback()
            return False
        finally:
            conn.close()

    def delete_project(self, project_id: int) -> bool:
        """Delete a project and all associated data"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Delete annotations first (foreign key constraint)
            cursor.execute('''
                DELETE FROM annotations
                WHERE sentence_id IN (
                    SELECT id FROM sentences WHERE project_id = ?
                )
            ''', (project_id,))

            # Delete sentences
            cursor.execute('DELETE FROM sentences WHERE project_id = ?', (project_id,))

            # Delete project
            cursor.execute('DELETE FROM projects WHERE id = ?', (project_id,))

            conn.commit()
            return True

        except Exception as e:
            conn.rollback()
            return False
        finally:
            conn.close()

    def calculate_ai_cost(self, model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> Dict:
        """Calculate cost for AI usage based on model pricing"""
        if model not in OPENAI_MODEL_PRICES:
            # Default to gpt-4.1-mini pricing if model not found
            model = 'gpt-4.1-mini'

        pricing = OPENAI_MODEL_PRICES[model]

        # Calculate costs (pricing is per 1M tokens)
        input_cost = (input_tokens / 1_000_000) * pricing['input']
        output_cost = (output_tokens / 1_000_000) * pricing['output']

        # Handle cached tokens if supported by model
        cached_cost = 0
        if cached_tokens > 0 and 'cached_input' in pricing:
            cached_cost = (cached_tokens / 1_000_000) * pricing['cached_input']

        total_cost = input_cost + output_cost + cached_cost

        return {
            'input_cost': round(input_cost, 6),
            'output_cost': round(output_cost, 6),
            'cached_cost': round(cached_cost, 6),
            'total_cost': round(total_cost, 6)
        }

    def track_ai_usage(self, project_id: int, model: str, input_tokens: int,
                      output_tokens: int, cached_tokens: int = 0) -> int:
        """Track AI usage and calculate costs"""
        costs = self.calculate_ai_cost(model, input_tokens, output_tokens, cached_tokens)

        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO ai_usage (project_id, model, input_tokens, output_tokens, cached_tokens,
                                input_cost, output_cost, cached_cost, total_cost)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (project_id, model, input_tokens, output_tokens, cached_tokens,
              costs['input_cost'], costs['output_cost'], costs['cached_cost'], costs['total_cost']))

        usage_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return usage_id

    def get_project_ai_costs(self, project_id: int) -> Dict:
        """Get AI usage costs for a project"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                SUM(total_cost) as total_cost,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(cached_tokens) as total_cached_tokens,
                COUNT(*) as total_requests,
                model
            FROM ai_usage
            WHERE project_id = ?
            GROUP BY model
            ORDER BY total_cost DESC
        ''', (project_id,))

        model_costs = []
        total_cost = 0
        total_requests = 0

        for row in cursor.fetchall():
            model_data = {
                'model': row[5],
                'total_cost': round(row[0], 6),
                'input_tokens': row[1],
                'output_tokens': row[2],
                'cached_tokens': row[3],
                'requests': row[4]
            }
            model_costs.append(model_data)
            total_cost += row[0]
            total_requests += row[4]

        # Get overall totals
        cursor.execute('''
            SELECT
                SUM(total_cost) as total_cost,
                SUM(input_tokens) as total_input_tokens,
                SUM(output_tokens) as total_output_tokens,
                SUM(cached_tokens) as total_cached_tokens,
                COUNT(*) as total_requests
            FROM ai_usage
            WHERE project_id = ?
        ''', (project_id,))

        totals = cursor.fetchone()
        conn.close()

        return {
            'total_cost': round(totals[0], 6) if totals[0] else 0,
            'total_input_tokens': totals[1] if totals[1] else 0,
            'total_output_tokens': totals[2] if totals[2] else 0,
            'total_cached_tokens': totals[3] if totals[3] else 0,
            'total_requests': totals[4] if totals[4] else 0,
            'model_breakdown': model_costs
        }

    def update_project_auto_ai(self, project_id: int, auto_ai_enabled: bool) -> bool:
        """Update auto AI assistant setting for a project"""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE projects
            SET auto_ai_enabled = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (1 if auto_ai_enabled else 0, project_id))

        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success

    def get_project_cost_estimation(self, project_id: int) -> Dict:
        """Calculate cost estimation for completing the entire project"""
        conn = self.get_connection()
        cursor = conn.cursor()

        try:
            # Get project info
            project = self.get_project(project_id)
            if not project:
                return {}

            # Get current AI usage statistics
            cursor.execute('''
                SELECT
                    AVG(input_tokens) as avg_input_tokens,
                    AVG(output_tokens) as avg_output_tokens,
                    AVG(cached_tokens) as avg_cached_tokens,
                    AVG(total_cost) as avg_cost_per_request,
                    COUNT(*) as total_requests,
                    model
                FROM ai_usage
                WHERE project_id = ?
                GROUP BY model
                ORDER BY total_requests DESC
                LIMIT 1
            ''', (project_id,))

            usage_stats = cursor.fetchone()

            # Get sentence statistics
            cursor.execute('''
                SELECT
                    COUNT(*) as total_sentences,
                    SUM(CASE WHEN is_annotated THEN 1 ELSE 0 END) as annotated_sentences
                FROM sentences
                WHERE project_id = ?
            ''', (project_id,))

            sentence_stats = cursor.fetchone()

            # Get current total cost
            current_costs = self.get_project_ai_costs(project_id)

            if not usage_stats or not sentence_stats:
                return {
                    'current_cost': current_costs.get('total_cost', 0),
                    'estimated_remaining_cost': 0,
                    'estimated_total_cost': current_costs.get('total_cost', 0),
                    'remaining_sentences': sentence_stats[0] - sentence_stats[1] if sentence_stats else 0,
                    'has_usage_data': False,
                    'estimation_note': 'No AI usage data available for estimation'
                }

            total_sentences = sentence_stats[0]
            annotated_sentences = sentence_stats[1]
            remaining_sentences = total_sentences - annotated_sentences

            # Calculate averages
            avg_cost_per_request = usage_stats[3] or 0
            most_used_model = usage_stats[5]

            # Estimate remaining cost
            estimated_remaining_cost = remaining_sentences * avg_cost_per_request
            estimated_total_cost = current_costs.get('total_cost', 0) + estimated_remaining_cost

            return {
                'current_cost': current_costs.get('total_cost', 0),
                'estimated_remaining_cost': estimated_remaining_cost,
                'estimated_total_cost': estimated_total_cost,
                'remaining_sentences': remaining_sentences,
                'total_sentences': total_sentences,
                'annotated_sentences': annotated_sentences,
                'avg_cost_per_request': avg_cost_per_request,
                'most_used_model': most_used_model,
                'has_usage_data': True,
                'estimation_note': f'Based on {usage_stats[4]} AI requests using {most_used_model}'
            }

        finally:
            conn.close()
