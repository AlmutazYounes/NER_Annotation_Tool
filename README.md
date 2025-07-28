# NER Annotation Tool

A comprehensive web-based Named Entity Recognition (NER) annotation tool built with Flask. This tool allows you to create custom entity annotation projects, upload text data, and export annotations in Hugging Face datasets format.

## 🚀 Quick Start

```bash
git clone https://github.com/AlmutazYounes/NER_Annotation_Tool.git
cd NER_Annotation_Tool
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ner_annotation_tool
python app.py
```

Then open `http://localhost:5001` in your browser and start annotating!

## Features

### 🚀 Core Functionality
- **Custom Entity Types**: Define your own entity labels (PERSON, ORGANIZATION, LOCATION, etc.)
- **File Upload Support**: Upload `.txt` (plain text) or `.csv` files
- **Interactive Annotation**: Select text spans and assign entity labels with an intuitive interface
- **Real-time Progress Tracking**: Monitor annotation progress with visual indicators
- **Auto-save**: Annotations are automatically saved as you work

### 📊 Data Management
- **SQLite Database**: Lightweight, file-based database for storing projects and annotations
- **Project Organization**: Manage multiple annotation projects
- **Sentence Navigation**: Navigate through sentences with next/previous/skip controls
- **Annotation History**: View and manage existing annotations

### 📤 Export Capabilities
- **Hugging Face Format**: Export annotations compatible with Hugging Face datasets
- **IOB/BIO Tagging**: Standard sequence labeling format
- **Span-based Format**: Entity span format with start/end positions
- **JSON/JSONL Export**: Multiple export formats supported

## Installation

### Prerequisites
- Python 3.7 or higher
- Virtual environment (recommended)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/AlmutazYounes/NER_Annotation_Tool.git
   cd NER_Annotation_Tool
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   cd ner_annotation_tool
   python app.py
   ```

5. **Access the application**
   Open your browser and go to: `http://localhost:5001`

## Usage Guide

### 1. Creating a New Project

1. Click "Create New Project" on the home page
2. Fill in project details:
   - **Project Name**: Descriptive name for your annotation task
   - **Description**: Optional context about the project
   - **Entity Tags**: Comma-separated list of entity types (e.g., PERSON, ORG, LOC)
3. Upload your data file:
   - **Text files (.txt)**: Plain text with sentences
   - **CSV files (.csv)**: Text data in columns named 'text', 'sentence', or first column
4. Click "Create Project & Start Annotating"

### 2. Annotating Text

1. **Select Text**: Click and drag to select text spans that represent entities
2. **Choose Entity Type**: A popup will appear with your defined entity types
3. **Assign Label**: Click on the appropriate entity type
4. **Navigate**: Use the navigation buttons or keyboard shortcuts:
   - `Ctrl + →`: Next sentence
   - `Ctrl + ←`: Previous sentence
   - `Esc`: Cancel current selection

### 3. Managing Annotations

- **View Current Annotations**: See all annotations for the current sentence in the sidebar
- **Delete Annotations**: Click the trash icon next to any annotation to remove it
- **Track Progress**: Monitor your progress with the progress bar and statistics

### 4. Exporting Data

1. Go to Project Details page
2. Click "Export Data"
3. Choose from available formats:
   - **IOB Format**: Token-level sequence labeling
   - **Span Format**: Entity spans with positions
   - **Hugging Face Compatible**: Ready for use with HF datasets library

## File Structure

```
ner_annotation_tool/
├── app.py                 # Main Flask application
├── models/
│   └── database.py        # Database models and operations
├── templates/             # HTML templates
│   ├── base.html         # Base template
│   ├── index.html        # Home page
│   ├── setup.html        # Project creation
│   ├── annotate.html     # Annotation interface
│   └── project_detail.html # Project overview
├── static/
│   ├── css/
│   │   └── style.css     # Custom styles
│   ├── js/
│   │   └── main.js       # JavaScript functionality
│   └── uploads/          # Temporary file uploads
└── utils/
    ├── file_processor.py # File parsing utilities
    └── export_utils.py   # Export functionality
```

## Data Formats

### Input Formats

**Text Files (.txt)**
```
Apple Inc. is a technology company. Microsoft Corporation is based in Seattle.
Google was founded by Larry Page and Sergey Brin.
```

**CSV Files (.csv)**
```csv
text
"Apple Inc. is a technology company."
"Microsoft Corporation is based in Seattle."
"Google was founded by Larry Page and Sergey Brin."
```

### Export Formats

**IOB Format**
```json
{
  "tokens": ["Apple", "Inc.", "is", "a", "technology", "company"],
  "ner_tags": ["B-ORG", "I-ORG", "O", "O", "O", "O"]
}
```

**Span Format**
```json
{
  "text": "Apple Inc. is a technology company.",
  "entities": [
    {
      "start": 0,
      "end": 10,
      "label": "ORG",
      "text": "Apple Inc."
    }
  ]
}
```

## Technical Details

### Database Schema

- **Projects**: Store project metadata and entity configurations
- **Sentences**: Individual sentences from uploaded files
- **Annotations**: Entity annotations with positions and labels

### Dependencies

- **Flask**: Web framework
- **SQLite**: Database
- **Pandas**: Data processing
- **Bootstrap**: UI framework
- **Font Awesome**: Icons

## Troubleshooting

### Common Issues

1. **Port 5000 in use**: The app runs on port 5001 by default to avoid conflicts with macOS AirPlay
2. **File upload errors**: Ensure files are .txt or .csv format and under 16MB
3. **Database issues**: Delete `ner_annotations.db` to reset the database

### Browser Compatibility

- Chrome/Chromium (recommended)
- Firefox
- Safari
- Edge

## Contributing

This is a standalone annotation tool. To extend functionality:

1. Add new entity types in the project setup
2. Modify export formats in `utils/export_utils.py`
3. Customize UI in templates and CSS files
4. Add new file formats in `utils/file_processor.py`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Sample Data

A sample text file (`sample_data.txt`) is included with technology company information for testing the annotation tool.
