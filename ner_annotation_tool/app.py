from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import os
import pandas as pd
from werkzeug.utils import secure_filename
from models.database import Database
from utils.file_processor import FileProcessor
from utils.export_utils import ExportUtils
import json
import re
from openai import OpenAI

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db = Database()

@app.route('/')
def index():
    """Home page showing all projects"""
    projects = db.get_all_projects()
    return render_template('index.html', projects=projects)

@app.route('/setup')
def setup():
    """Setup page for creating new projects"""
    return render_template('setup.html')

@app.route('/create_project', methods=['POST'])
def create_project():
    """Create a new project with entity tags and uploaded file"""
    try:
        # Get form data
        project_name = request.form.get('project_name', '').strip()
        project_description = request.form.get('project_description', '').strip()
        entity_config = request.form.get('entity_config', '').strip()

        # Get LLM configuration
        llm_enabled = request.form.get('llm_enabled') == 'on'
        llm_api_key = request.form.get('llm_api_key', '').strip() if llm_enabled else None
        llm_model = request.form.get('llm_model', 'gpt-4.1-mini') if llm_enabled else 'gpt-4.1-mini'
        llm_custom_instructions = request.form.get('llm_custom_instructions', '').strip() if llm_enabled else None

        if not project_name or not entity_config:
            flash('Project name and entity configuration are required', 'error')
            return redirect(url_for('setup'))

        # Validate LLM configuration if enabled
        if llm_enabled and not llm_api_key:
            flash('API key is required when AI assistant is enabled', 'error')
            return redirect(url_for('setup'))

        # Parse entity configuration
        try:
            entity_tags = json.loads(entity_config)
            if not entity_tags or not isinstance(entity_tags, list):
                raise ValueError("Invalid entity configuration")
        except (json.JSONDecodeError, ValueError):
            flash('Invalid entity configuration', 'error')
            return redirect(url_for('setup'))
        
        # Handle file upload
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('setup'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('setup'))
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process file and extract sentences
            processor = FileProcessor()
            sentences = processor.process_file(filepath)
            
            if not sentences:
                flash('No sentences found in the uploaded file', 'error')
                os.remove(filepath)  # Clean up
                return redirect(url_for('setup'))
            
            # Prepare LLM configuration
            llm_config = None
            if llm_enabled:
                llm_config = {
                    'enabled': True,
                    'api_key': llm_api_key,
                    'model': llm_model,
                    'custom_instructions': llm_custom_instructions
                }

            # Create project
            project_id = db.create_project(project_name, project_description, entity_tags, llm_config)
            
            # Add sentences to project
            sentence_count = db.add_sentences(project_id, sentences)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            flash(f'Project created successfully with {sentence_count} sentences', 'success')
            return redirect(url_for('annotate', project_id=project_id, sentence_index=0))
        
        else:
            flash('Invalid file type. Please upload .txt or .csv files only', 'error')
            return redirect(url_for('setup'))
    
    except Exception as e:
        flash(f'Error creating project: {str(e)}', 'error')
        return redirect(url_for('setup'))

@app.route('/project/<int:project_id>')
def project_detail(project_id):
    """Project detail page"""
    project = db.get_project(project_id)
    if not project:
        flash('Project not found', 'error')
        return redirect(url_for('index'))

    progress = db.get_project_progress(project_id)
    return render_template('project_detail.html', project=project, progress=progress)

@app.route('/project/<int:project_id>/analytics')
def project_analytics(project_id):
    """Project analytics dashboard"""
    project = db.get_project(project_id)
    if not project:
        flash('Project not found', 'error')
        return redirect(url_for('index'))

    analytics = db.get_project_analytics(project_id)
    ai_costs = db.get_project_ai_costs(project_id)
    analytics['ai_costs'] = ai_costs
    return render_template('analytics.html', analytics=analytics)

@app.route('/api/project/<int:project_id>/analytics')
def api_project_analytics(project_id):
    """API endpoint for project analytics data"""
    analytics = db.get_project_analytics(project_id)
    return jsonify(analytics)

@app.route('/api/project/<int:project_id>/entity-statistics')
def api_entity_statistics(project_id):
    """API endpoint for entity statistics (for charts)"""
    statistics = db.get_entity_statistics(project_id)
    return jsonify(statistics)

@app.route('/export/project/<int:project_id>/analytics')
def export_analytics(project_id):
    """Export analytics data as JSON file"""
    project = db.get_project(project_id)
    if not project:
        flash('Project not found', 'error')
        return redirect(url_for('index'))

    analytics = db.get_project_analytics(project_id)

    # Create response with JSON data
    response = jsonify(analytics)
    response.headers['Content-Disposition'] = f'attachment; filename=analytics_{project["name"]}.json'
    response.headers['Content-Type'] = 'application/json'

    return response

@app.route('/annotate/<int:project_id>/<int:sentence_index>')
def annotate(project_id, sentence_index):
    """Annotation page for a specific sentence"""
    project = db.get_project(project_id)
    if not project:
        flash('Project not found', 'error')
        return redirect(url_for('index'))
    
    sentence = db.get_sentence(project_id, sentence_index)
    if not sentence:
        flash('Sentence not found', 'error')
        return redirect(url_for('project_detail', project_id=project_id))
    
    # Get existing annotations for this sentence
    annotations = db.get_annotations(sentence['id'])
    
    # Get progress info
    total_sentences = db.get_sentence_count(project_id)
    progress = db.get_project_progress(project_id)
    
    return render_template('annotate.html', 
                         project=project, 
                         sentence=sentence, 
                         annotations=annotations,
                         sentence_index=sentence_index,
                         total_sentences=total_sentences,
                         progress=progress)

@app.route('/api/save_annotation', methods=['POST'])
def save_annotation():
    """API endpoint to save an annotation"""
    try:
        data = request.get_json()

        sentence_id = data.get('sentence_id')
        start_pos = data.get('start_pos')
        end_pos = data.get('end_pos')
        entity_label = data.get('entity_label')
        annotated_text = data.get('annotated_text')

        if not all([sentence_id, start_pos is not None, end_pos is not None, entity_label, annotated_text]):
            return jsonify({'error': 'Missing required fields'}), 400

        # Check for duplicate annotation
        existing_annotations = db.get_annotations_for_sentence(sentence_id)
        for annotation in existing_annotations:
            # Check for exact duplicate
            if (annotation['start_pos'] == start_pos and
                annotation['end_pos'] == end_pos and
                annotation['entity_label'] == entity_label and
                annotation['annotated_text'] == annotated_text):
                return jsonify({'error': 'This exact annotation already exists'}), 400

            # Check for overlapping annotation with same entity type
            if (annotation['entity_label'] == entity_label and
                start_pos < annotation['end_pos'] and end_pos > annotation['start_pos']):
                return jsonify({'error': f'This annotation overlaps with existing {entity_label} annotation: "{annotation["annotated_text"]}"'}), 400

        annotation_id = db.save_annotation(sentence_id, start_pos, end_pos, entity_label, annotated_text)

        return jsonify({
            'success': True,
            'annotation_id': annotation_id,
            'message': 'Annotation saved successfully'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete_annotation/<int:annotation_id>', methods=['DELETE'])
def delete_annotation(annotation_id):
    """API endpoint to delete an annotation"""
    try:
        success = db.delete_annotation(annotation_id)
        if success:
            return jsonify({'success': True, 'message': 'Annotation deleted successfully'})
        else:
            return jsonify({'error': 'Annotation not found'}), 404
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export/<int:project_id>')
def export_project(project_id):
    """Export project annotations as downloadable file"""
    try:
        project = db.get_project(project_id)
        if not project:
            flash('Project not found', 'error')
            return redirect(url_for('index'))

        exporter = ExportUtils(db)
        export_data = exporter.export_to_huggingface_format(project_id)

        # Create filename
        safe_project_name = "".join(c for c in project['name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_project_name = safe_project_name.replace(' ', '_')
        filename = f"{safe_project_name}_annotations.json"

        # Create response with file download
        response = app.response_class(
            response=json.dumps(export_data, indent=2),
            status=200,
            mimetype='application/json'
        )
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export/<int:project_id>/format/<format_type>')
def export_project_format(project_id, format_type):
    """Export project annotations in specific format"""
    try:
        project = db.get_project(project_id)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        exporter = ExportUtils(db)
        safe_project_name = "".join(c for c in project['name'] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_project_name = safe_project_name.replace(' ', '_')

        if format_type == 'jsonl':
            # Export as JSONL
            jsonl_data = exporter.export_to_jsonl(project_id, 'span')
            filename = f"{safe_project_name}_annotations.jsonl"

            response = app.response_class(
                response=jsonl_data,
                status=200,
                mimetype='application/jsonl'
            )
            response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response

        elif format_type == 'conll':
            # Export as CoNLL
            conll_data = exporter.export_to_conll(project_id)
            filename = f"{safe_project_name}_annotations.conll"

            response = app.response_class(
                response=conll_data,
                status=200,
                mimetype='text/plain'
            )
            response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
            return response

        else:
            return jsonify({'error': 'Unsupported format'}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/upload_to_hf/<int:project_id>', methods=['POST'])
def upload_to_huggingface(project_id):
    """Upload project annotations to Hugging Face Hub"""
    try:
        project = db.get_project(project_id)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        # Get form data
        hf_token = request.form.get('hf_token', '').strip()
        dataset_name = request.form.get('dataset_name', '').strip()
        description = request.form.get('description', '').strip()
        private = request.form.get('private') == 'on'

        if not hf_token or not dataset_name:
            return jsonify({'error': 'Hugging Face token and dataset name are required'}), 400

        # Upload to Hugging Face
        exporter = ExportUtils(db)
        result = exporter.upload_to_huggingface(project_id, hf_token, dataset_name, description, private)

        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/edit_entity_tags/<int:project_id>', methods=['POST'])
def edit_entity_tags(project_id):
    """Edit entity tags for a project"""
    try:
        project = db.get_project(project_id)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        # Get form data
        entity_config = request.form.get('entity_config', '').strip()
        tag_mappings_json = request.form.get('tag_mappings', '{}').strip()

        if not entity_config:
            return jsonify({'error': 'Entity configuration is required'}), 400

        try:
            entity_tags = json.loads(entity_config)
            tag_mappings = json.loads(tag_mappings_json)

            if not entity_tags or not isinstance(entity_tags, list):
                raise ValueError("Invalid entity configuration")
        except (json.JSONDecodeError, ValueError):
            return jsonify({'error': 'Invalid entity configuration'}), 400

        # Update entity tags
        success = db.update_project_entity_tags(project_id, entity_tags, tag_mappings)

        if success:
            return jsonify({'success': True, 'message': 'Entity tags updated successfully'})
        else:
            return jsonify({'error': 'Failed to update entity tags'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_project/<int:project_id>', methods=['POST'])
def delete_project(project_id):
    """Delete a project and all associated data"""
    try:
        project = db.get_project(project_id)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        # Delete project
        success = db.delete_project(project_id)

        if success:
            return jsonify({'success': True, 'message': 'Project deleted successfully'})
        else:
            return jsonify({'error': 'Failed to delete project'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/llm-suggest', methods=['POST'])
def llm_suggest():
    """Get LLM suggestions for entity annotations"""
    try:
        data = request.get_json()
        project_id = data.get('project_id')
        sentence_text = data.get('sentence_text', '').strip()

        if not project_id or not sentence_text:
            return jsonify({'error': 'Project ID and sentence text are required'}), 400

        # Get project configuration
        project = db.get_project(project_id)
        if not project:
            return jsonify({'error': 'Project not found'}), 404

        if not project.get('llm_enabled'):
            return jsonify({'error': 'LLM is not enabled for this project'}), 400

        api_key = project.get('llm_api_key')
        if not api_key:
            return jsonify({'error': 'No API key configured for this project'}), 400

        model = project.get('llm_model', 'gpt-3.5-turbo')
        custom_instructions = project.get('llm_custom_instructions', '')
        entity_tags = project.get('entity_tags', [])

        # Create entity tags list for the prompt
        entity_names = [tag['name'] for tag in entity_tags]
        entity_list = ', '.join(entity_names)

        # Build system message
        system_message = f"""You are an expert named entity recognition assistant. Your task is to identify and extract entities from the given text that match the specified entity types.

Entity types to identify: {entity_list}

Instructions:
1. Analyze the provided sentence carefully
2. Identify all entities that match the specified entity types
3. Return the results as a JSON array where each entity has:
   - "text": the exact text span of the entity as it appears in the sentence
   - "label": the entity type from the specified list

4. Only return entities that exactly match one of the specified entity types
5. Do NOT include character positions - only text and label
6. If no entities are found, return an empty array

{f"Additional instructions: {custom_instructions}" if custom_instructions else ""}

Example format:
[
  {{"text": "John Doe", "label": "PERSON"}},
  {{"text": "New York", "label": "LOCATION"}}
]"""

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Sentence: {sentence_text}"}
            ],
            temperature=0.1,
            max_tokens=1000
        )

        # Track AI usage and costs
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens

        # Safely extract cached tokens
        cached_tokens = 0
        if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
            if hasattr(usage.prompt_tokens_details, 'cached_tokens'):
                cached_tokens = usage.prompt_tokens_details.cached_tokens or 0

        # Track usage in database
        db.track_ai_usage(project_id, model, input_tokens, output_tokens, cached_tokens)

        # Parse response
        llm_response = response.choices[0].message.content.strip()

        # Try to extract JSON from the response
        try:
            # Look for JSON array in the response
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if json_match:
                suggestions = json.loads(json_match.group())
            else:
                # If no JSON array found, try parsing the entire response
                suggestions = json.loads(llm_response)

            # Validate suggestions format
            if not isinstance(suggestions, list):
                suggestions = []

            # Validate each suggestion
            valid_suggestions = []
            for suggestion in suggestions:
                if (isinstance(suggestion, dict) and
                    'text' in suggestion and
                    'label' in suggestion and
                    suggestion['label'] in entity_names):
                    valid_suggestions.append(suggestion)

            return jsonify({
                'suggestions': valid_suggestions,
                'model_used': model
            })

        except json.JSONDecodeError:
            return jsonify({
                'error': 'Failed to parse LLM response as JSON',
                'raw_response': llm_response
            }), 500

    except Exception as e:
        if 'api_key' in str(e).lower() or 'authentication' in str(e).lower():
            return jsonify({'error': 'Invalid API key or authentication failed'}), 401
        elif 'rate limit' in str(e).lower():
            return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
        elif 'quota' in str(e).lower():
            return jsonify({'error': 'API quota exceeded. Please check your OpenAI account.'}), 429
        else:
            return jsonify({'error': f'LLM request failed: {str(e)}'}), 500

@app.route('/api/project/<int:project_id>/toggle-auto-ai', methods=['POST'])
def toggle_auto_ai(project_id):
    """Toggle auto AI assistant for a project"""
    try:
        data = request.get_json()
        auto_ai_enabled = data.get('auto_ai_enabled', False)

        success = db.update_project_auto_ai(project_id, auto_ai_enabled)

        if success:
            return jsonify({
                'success': True,
                'auto_ai_enabled': auto_ai_enabled,
                'message': f'Auto AI assistant {"enabled" if auto_ai_enabled else "disabled"}'
            })
        else:
            return jsonify({'error': 'Failed to update auto AI setting'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/<int:project_id>/ai-costs')
def get_project_ai_costs(project_id):
    """Get AI usage costs for a project"""
    try:
        costs = db.get_project_ai_costs(project_id)
        return jsonify(costs)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/project/<int:project_id>/cost-estimation')
def get_project_cost_estimation(project_id):
    """Get cost estimation for completing the entire project"""
    try:
        estimation = db.get_project_cost_estimation(project_id)
        return jsonify(estimation)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'txt', 'csv'}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
