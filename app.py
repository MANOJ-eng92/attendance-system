from flask import Flask, request, jsonify, render_template_string, Response
import os
import sys
import base64
import csv
import io
from datetime import datetime, date

sys.path.insert(0, os.path.dirname(__file__))
from database.db import (
    init_db, get_all_members, add_member, mark_attendance,
    get_attendance_today, get_attendance_stats, get_next_label,
    delete_member, get_attendance_history, get_member_by_label,
    get_attendance_by_date_range, bulk_add_members,
    get_attendance_by_date, get_member_attendance_on_date,
    get_member_by_id
)
from models.face_engine import (
    recognize_face, train_model, get_annotated_image,
    model_exists, extract_faces_from_bytes
)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# ─── Serve Frontend ────────────────────────────────────────────────────────────

@app.route('/')
def index():
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/uploads/photo/<int:member_id>')
def serve_member_photo(member_id):
    """Serve a member's photo, stored as base64 in the database."""
    member = get_member_by_id(member_id)
    if not member or not member.get('photo_data'):
        return '', 404
    photo_bytes = base64.b64decode(member['photo_data'])
    return Response(photo_bytes, mimetype='image/jpeg')

# ─── Member Management ─────────────────────────────────────────────────────────

@app.route('/api/members', methods=['GET'])
def api_get_members():
    members = get_all_members()
    return jsonify({'success': True, 'members': members})

@app.route('/api/members', methods=['POST'])
def api_add_member():
    try:
        name = request.form.get('name', '').strip()
        employee_id = request.form.get('employee_id', '').strip()
        department = request.form.get('department', 'General').strip()

        if not name or not employee_id:
            return jsonify({'success': False, 'error': 'Name and Employee ID are required.'}), 400

        if 'photo' not in request.files:
            return jsonify({'success': False, 'error': 'Photo is required.'}), 400

        photo = request.files['photo']
        if photo.filename == '':
            return jsonify({'success': False, 'error': 'No photo selected.'}), 400

        photo_bytes = photo.read()

        # Validate face in photo
        img, faces = extract_faces_from_bytes(photo_bytes)
        if not faces:
            return jsonify({'success': False, 'error': 'No face detected in the uploaded photo. Please use a clear frontal face photo.'}), 400

        # Store photo as base64 text in the database (persists across restarts)
        photo_data = base64.b64encode(photo_bytes).decode('ascii')

        label = get_next_label()
        member = add_member(name, employee_id, department, photo_data, label)

        # Retrain model
        success, msg = train_model()

        return jsonify({
            'success': True,
            'member': member,
            'message': f"Member '{name}' added successfully. {msg}"
        })

    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/members/<int:member_id>', methods=['DELETE'])
def api_delete_member(member_id):
    try:
        delete_member(member_id)
        train_model()
        return jsonify({'success': True, 'message': 'Member removed successfully.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/members/import', methods=['POST'])
def api_import_members():
    """
    Bulk import members from a CSV file.
    Expected columns: name, employee_id, department (optional, defaults to 'General')
    Members imported this way have no photo yet — add one later via
    the Members page to enable face recognition for them.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No CSV file provided.'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected.'}), 400

        if not file.filename.lower().endswith('.csv'):
            return jsonify({'success': False, 'error': 'Please upload a .csv file.'}), 400

        raw = file.read()
        try:
            text = raw.decode('utf-8-sig')
        except UnicodeDecodeError:
            text = raw.decode('latin-1')

        reader = csv.DictReader(io.StringIO(text))
        # Normalize header names (case-insensitive, strip spaces)
        if reader.fieldnames:
            reader.fieldnames = [ (f or '').strip().lower().replace(' ', '_') for f in reader.fieldnames ]

        rows = []
        for row in reader:
            normalized = { (k or '').strip().lower().replace(' ', '_'): (v or '').strip() for k, v in row.items() }
            rows.append(normalized)

        if not rows:
            return jsonify({'success': False, 'error': 'CSV file is empty.'}), 400

        added, skipped = bulk_add_members(rows)

        message = f"Imported {len(added)} member(s)."
        if skipped:
            message += f" Skipped {len(skipped)} row(s)."
        if added:
            message += " Note: imported members have no photo yet — add a photo on the Members page to enable face recognition."

        return jsonify({
            'success': True,
            'added': added,
            'skipped': skipped,
            'message': message
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/members/export', methods=['GET'])
def api_export_members():
    members = get_all_members()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Name', 'Employee ID', 'Department', 'Has Photo', 'Created At'])
    for m in members:
        writer.writerow([
            m.get('name', ''),
            m.get('employee_id', ''),
            m.get('department', ''),
            'Yes' if m.get('photo_data') else 'No',
            m.get('created_at', '')
        ])

    filename = f"members_{date.today().isoformat()}.csv"
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'}
    )

@app.route('/api/members/import/template', methods=['GET'])
def api_members_import_template():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['name', 'employee_id', 'department'])
    writer.writerow(['Alice Johnson', 'EMP001', 'Engineering'])
    writer.writerow(['Bob Smith', 'EMP002', 'Sales'])

    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename="members_import_template.csv"'}
    )

# ─── Attendance ─────────────────────────────────────────────────────────────────

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    try:
@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    try:
        if not model_exists():          # ← ADD THIS
            train_model()               # ← ADD THIS
        if not model_exists():
            return jsonify({'success': False, 'error': 'No trained model found. Please add members first.'}), 400

        if 'image' not in request.files and 'image_data' not in request.json if request.is_json else True:
            if 'image' not in request.files:
                return jsonify({'success': False, 'error': 'No image provided.'}), 400

        if 'image' in request.files:
            img_bytes = request.files['image'].read()
        else:
            data = request.json
            img_data = data.get('image_data', '')
            if ',' in img_data:
                img_data = img_data.split(',')[1]
            img_bytes = base64.b64decode(img_data)

        results, error = recognize_face(img_bytes)
        if error and not results:
            return jsonify({'success': False, 'error': error}), 400

        if not results:
            return jsonify({'success': False, 'error': 'No face detected.'}), 400

        # Take best result
        best = max(results, key=lambda r: r['confidence'])

        if not best['recognized']:
            return jsonify({
                'success': True,
                'recognized': False,
                'confidence': best['confidence'],
                'message': 'Face not recognized. Please register first.'
            })

        member = get_member_by_label(best['label'])
        if not member:
            return jsonify({'success': False, 'error': 'Member not found in database.'}), 404

        # Mark attendance
        action, record = mark_attendance(member['id'], best['confidence'])

        # Annotated image
        annotated = get_annotated_image(img_bytes, results, member['name'])
        annotated_b64 = base64.b64encode(annotated).decode() if annotated else None

        return jsonify({
            'success': True,
            'recognized': True,
            'action': action,
            'member': member,
            'confidence': best['confidence'],
            'record': record,
            'annotated_image': f'data:image/jpeg;base64,{annotated_b64}' if annotated_b64 else None,
            'message': f"{'✅ Check-in' if action=='check_in' else '🔒 Check-out' if action=='check_out' else '⚠️ Already complete'} — {member['name']} ({best['confidence']:.0f}% match)"
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/attendance/today', methods=['GET'])
def api_attendance_today():
    records = get_attendance_today()
    return jsonify({'success': True, 'records': records})

@app.route('/api/attendance/by-date', methods=['GET'])
def api_attendance_by_date():
    """
    Get attendance for ALL members on a specific date.
    Query params: date (YYYY-MM-DD), defaults to today.
    """
    try:
        date_str = request.args.get('date', '').strip()
        target_date = date.fromisoformat(date_str) if date_str else date.today()

        present, absent = get_attendance_by_date(target_date.isoformat())

        return jsonify({
            'success': True,
            'date': target_date.isoformat(),
            'present': present,
            'absent': absent,
            'present_count': len(present),
            'absent_count': len(absent)
        })
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/attendance/member/<int:member_id>', methods=['GET'])
def api_member_attendance_on_date(member_id):
    """
    Check a specific member's attendance status on a specific date.
    Query params: date (YYYY-MM-DD), defaults to today.
    """
    try:
        date_str = request.args.get('date', '').strip()
        target_date = date.fromisoformat(date_str) if date_str else date.today()

        member, record = get_member_attendance_on_date(member_id, target_date.isoformat())

        if member is None:
            return jsonify({'success': False, 'error': 'Member not found.'}), 404

        return jsonify({
            'success': True,
            'date': target_date.isoformat(),
            'member': member,
            'present': record is not None,
            'record': record
        })
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/attendance/history', methods=['GET'])
def api_attendance_history():
    days = int(request.args.get('days', 30))
    records = get_attendance_history(days)
    return jsonify({'success': True, 'records': records})

@app.route('/api/attendance/export', methods=['GET'])
def api_export_attendance():
    """
    Export attendance records as a CSV file for a custom date range.
    Query params:
      start (YYYY-MM-DD) - defaults to 30 days ago
      end   (YYYY-MM-DD) - defaults to today
    """
    try:
        end_str = request.args.get('end', '').strip()
        start_str = request.args.get('start', '').strip()

        end_date = date.fromisoformat(end_str) if end_str else date.today()
        if start_str:
            start_date = date.fromisoformat(start_str)
        else:
            start_date = date.fromisoformat((datetime.combine(end_date, datetime.min.time())).date().isoformat())

        if start_date > end_date:
            start_date, end_date = end_date, start_date

        records = get_attendance_by_date_range(start_date.isoformat(), end_date.isoformat())

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Date', 'Name', 'Employee ID', 'Department', 'Check In', 'Check Out', 'Status', 'Confidence (%)'])
        for r in records:
            writer.writerow([
                r.get('date', ''),
                r.get('name', ''),
                r.get('employee_id', ''),
                r.get('department', ''),
                r.get('check_in', '') or '',
                r.get('check_out', '') or '',
                r.get('status', '') or '',
                r.get('confidence', '') if r.get('confidence') is not None else ''
            ])

        filename = f"attendance_{start_date.isoformat()}_to_{end_date.isoformat()}.csv"
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{filename}"'}
        )

    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid date format. Use YYYY-MM-DD.'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def api_stats():
    stats = get_attendance_stats()
    return jsonify({'success': True, 'stats': stats})

@app.route('/api/train', methods=['POST'])
def api_train():
    success, msg = train_model()
    return jsonify({'success': success, 'message': msg})

# Initialize database and train face recognition model at startup
init_db()
try:
    success, msg = train_model()
    print(f"Startup training: {msg}")
except Exception as e:
    print(f"Startup training failed: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(debug=False, host='0.0.0.0', port=port)
