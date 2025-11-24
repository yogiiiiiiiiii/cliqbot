import os
import json
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Configuration
TRELLO_API_KEY = os.getenv("TRELLO_API_KEY")
TRELLO_TOKEN = os.getenv("TRELLO_API_TOKEN")
TRELLO_BOARD_ID = os.getenv("TRELLO_BOARD_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# ============================================
# TRELLO API FUNCTIONS
# ============================================

def get_trello_cards():
    """Fetch all cards from Trello board"""
    try:
        url = f"https://api.trello.com/1/boards/{TRELLO_BOARD_ID}/cards"
        params = {
            'key': TRELLO_API_KEY,
            'token': TRELLO_TOKEN
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching Trello cards: {e}")
        return []

def get_trello_lists():
    """Fetch all lists from Trello board"""
    try:
        url = f"https://api.trello.com/1/boards/{TRELLO_BOARD_ID}/lists"
        params = {
            'key': TRELLO_API_KEY,
            'token': TRELLO_TOKEN
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching Trello lists: {e}")
        return []

def get_trello_labels():
    """Fetch all labels from Trello board"""
    try:
        url = f"https://api.trello.com/1/boards/{TRELLO_BOARD_ID}/labels"
        params = {
            'key': TRELLO_API_KEY,
            'token': TRELLO_TOKEN
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching labels: {e}")
        return []

def normalize_tasks(trello_cards, trello_lists):
    """Convert Trello cards to normalized task format"""
    list_map = {l['id']: l['name'] for l in trello_lists}
    
    normalized = []
    for card in trello_cards:
        task = {
            'id': card['id'],
            'title': card['name'],
            'description': card.get('desc', ''),
            'source': 'trello',
            'status': list_map.get(card['idList'], 'unknown'),
            'assignee': card.get('idMembers', []),
            'deadline': card.get('due', None),
            'url': card.get('url', ''),
            'labels': [l['name'] for l in card.get('labels', [])],
            'idList': card.get('idList')
        }
        normalized.append(task)
    
    return normalized

# ============================================
# PRIORITY CALCULATION ENGINE
# ============================================

def calculate_priority_score(task, all_tasks):
    """
    Calculate priority score (0-100) using algorithm
    
    Formula:
    Priority = (Urgency × 30%) + (Strategic × 20%) + 
               (Dependency × 25%) + (Capacity × 15%) + (Risk × 10%)
    """
    
    # 1. DEADLINE URGENCY (0-100)
    deadline = task.get('deadline')
    if deadline:
        try:
            deadline_date = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
            now = datetime.now(deadline_date.tzinfo)
            days_until = (deadline_date - now).days
            hours_until = (deadline_date - now).total_seconds() / 3600
            
            if hours_until < 0:
                urgency = 100  # Overdue
            elif hours_until < 24:
                urgency = 90   # Due today
            elif days_until == 1:
                urgency = 80   # Due tomorrow
            elif days_until <= 3:
                urgency = 60
            elif days_until <= 7:
                urgency = 40
            else:
                urgency = 20
        except:
            urgency = 30
    else:
        urgency = 10
    
    # 2. STRATEGIC ALIGNMENT (0-100)
    labels = task.get('labels', [])
    title = task.get('title', '').lower()
    
    if 'critical' in labels or 'blocker' in labels or 'emergency' in labels:
        strategic = 90
    elif 'high' in labels or 'urgent' in labels:
        strategic = 70
    elif 'bug' in title or 'fix' in title or 'error' in title:
        strategic = 75
    elif 'low' in labels or 'nice to have' in labels:
        strategic = 25
    else:
        strategic = 50
    
    # 3. DEPENDENCY IMPACT (0-100)
    # Count how many other tasks might depend on this one
    description = task.get('description', '').lower()
    if 'blocker' in description or 'blocks' in description:
        dependency_impact = 85
    elif task.get('status') in ['In Progress', 'In Review']:
        dependency_impact = 60
    else:
        dependency_impact = 30
    
    # 4. TEAM CAPACITY (0-100)
    assignees = len(task.get('assignee', []))
    if assignees == 0:
        team_capacity = 60  # Not assigned yet = medium priority
    elif assignees == 1:
        team_capacity = 70
    elif assignees == 2:
        team_capacity = 50
    else:
        team_capacity = 30
    
    # 5. RISK FACTOR (0-100)
    if task.get('status') == 'In Progress':
        risk = 40
    elif task.get('status') == 'In Review':
        risk = 50
    elif 'bug' in title:
        risk = 70
    elif 'deploy' in title or 'production' in title.lower():
        risk = 75
    else:
        risk = 30
    
    # FINAL CALCULATION
    priority_score = (
        (urgency * 0.30) +
        (strategic * 0.20) +
        (dependency_impact * 0.25) +
        (team_capacity * 0.15) +
        (risk * 0.10)
    )
    
    return round(min(100, max(0, priority_score)), 1)

# ============================================
# AI ANALYSIS WITH GEMINI
# ============================================

def analyze_task_with_ai(task, all_tasks):
    """Use Gemini to analyze task and provide insights"""
    try:
        prompt = f"""
Analyze this project task and provide a brief, actionable insight (2-3 sentences max):

Task: {task['title']}
Description: {task.get('description', 'No description')}
Status: {task['status']}
Due: {task.get('deadline', 'No deadline')}
Labels: {', '.join(task.get('labels', ['none']))}

Provide:
1. Risk assessment (low/medium/high)
2. Why this matters
3. One actionable suggestion

Keep it brief and professional.
"""
        response = model.generate_content(prompt, stream=False)
        return response.text
    except Exception as e:
        return f"Could not analyze: {str(e)[:100]}"

def predict_project_risk(tasks):
    """Use Gemini to predict if project is at risk"""
    try:
        done_count = sum(1 for t in tasks if t['status'] == 'Done')
        in_progress = sum(1 for t in tasks if t['status'] == 'In Progress')
        total = len(tasks)
        
        overdue = sum(1 for t in tasks if t.get('deadline') and 
                     datetime.fromisoformat(t['deadline'].replace('Z', '+00:00')) < datetime.now(datetime.now().astimezone().tzinfo))
        
        prompt = f"""
Analyze project health and predict risks:

Total Tasks: {total}
Completed: {done_count} ({round(done_count/total*100) if total else 0}%)
In Progress: {in_progress}
Overdue: {overdue}

Recent task titles: {[t['title'][:30] for t in tasks[:5]]}

Provide (2-3 sentences):
1. Risk level: LOW/MEDIUM/HIGH
2. Key concern
3. Recommended action

Be direct and concise.
"""
        response = model.generate_content(prompt, stream=False)
        return response.text
    except Exception as e:
        return f"Could not predict: {str(e)[:100]}"

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'OK',
        'message': 'ProActive Intelligence Hub is running',
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/api/tasks', methods=['GET'])
def get_all_tasks():
    """Get all tasks with priority scoring"""
    try:
        cards = get_trello_cards()
        lists = get_trello_lists()
        
        if not cards:
            return jsonify({
                'success': False,
                'error': 'No cards found. Check Trello credentials.',
                'cards_count': 0
            }), 200
        
        tasks = normalize_tasks(cards, lists)
        
        # Calculate priority
        for task in tasks:
            task['priority_score'] = calculate_priority_score(task, tasks)
        
        # Sort by priority (highest first)
        tasks.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return jsonify({
            'success': True,
            'total_tasks': len(tasks),
            'tasks': tasks
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/next-task', methods=['GET'])
def get_next_task():
    """Get the highest priority task"""
    try:
        cards = get_trello_cards()
        lists = get_trello_lists()
        
        if not cards:
            return jsonify({
                'success': True,
                'message': '✅ No tasks yet!'
            }), 200
        
        tasks = normalize_tasks(cards, lists)
        
        for task in tasks:
            task['priority_score'] = calculate_priority_score(task, tasks)
        
        tasks.sort(key=lambda x: x['priority_score'], reverse=True)
        
        top_task = tasks[0]
        
        return jsonify({
            'success': True,
            'task': top_task,
            'message': f"🎯 Top Priority: {top_task['title']} (Score: {top_task['priority_score']}/100)"
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get project summary"""
    try:
        cards = get_trello_cards()
        lists = get_trello_lists()
        
        tasks = normalize_tasks(cards, lists)
        
        # Count by status
        status_counts = {}
        for task in tasks:
            status = task['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        completion_rate = 0
        if tasks:
            done_count = status_counts.get('Done', 0)
            completion_rate = round((done_count / len(tasks)) * 100)
        
        return jsonify({
            'success': True,
            'summary': {
                'total_tasks': len(tasks),
                'completion_rate': completion_rate,
                'by_status': status_counts
            }
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_task_endpoint():
    """Analyze a specific task with AI"""
    try:
        data = request.json
        task_id = data.get('task_id')
        
        cards = get_trello_cards()
        lists = get_trello_lists()
        tasks = normalize_tasks(cards, lists)
        
        task = next((t for t in tasks if t['id'] == task_id), None)
        
        if not task:
            return jsonify({
                'success': False,
                'error': 'Task not found'
            }), 404
        
        analysis = analyze_task_with_ai(task, tasks)
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'task_title': task['title'],
            'analysis': analysis
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/risk', methods=['GET'])
def get_risk_prediction():
    """Get AI risk prediction for project"""
    try:
        cards = get_trello_cards()
        lists = get_trello_lists()
        tasks = normalize_tasks(cards, lists)
        
        if not tasks:
            return jsonify({
                'success': True,
                'message': 'No tasks to analyze'
            }), 200
        
        risk_analysis = predict_project_risk(tasks)
        
        return jsonify({
            'success': True,
            'risk_analysis': risk_analysis
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/blockers', methods=['GET'])
def get_blockers():
    """Get tasks that are blocking others"""
    try:
        cards = get_trello_cards()
        lists = get_trello_lists()
        tasks = normalize_tasks(cards, lists)
        
        blockers = [t for t in tasks if 'blocker' in t.get('labels', []) or 
                   'blocker' in t.get('description', '').lower()]
        
        blockers.sort(key=lambda x: calculate_priority_score(x, tasks), reverse=True)
        
        return jsonify({
            'success': True,
            'blockers': blockers,
            'count': len(blockers)
        }), 200
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============================================
# ERROR HANDLERS
# ============================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Server error', 'message': str(error)}), 500

# ============================================
# MAIN
# ============================================

if __name__ == '__main__':
    print("🚀 Starting ProActive Intelligence Hub...")
    print(f"📍 Running on http://localhost:5000")
    print(f"🧪 Test endpoint: http://localhost:5000/api/health")
    app.run(debug=True, port=5000, host='localhost')