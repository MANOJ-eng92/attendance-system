import sqlite3
import os
from datetime import datetime, date

DB_PATH = os.path.join(os.path.dirname(__file__), 'attendance.db')

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS members (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            employee_id TEXT UNIQUE NOT NULL,
            department TEXT DEFAULT 'General',
            photo_path TEXT,
            label INTEGER UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            active INTEGER DEFAULT 1
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            member_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            check_in TEXT,
            check_out TEXT,
            status TEXT DEFAULT 'present',
            confidence REAL,
            FOREIGN KEY (member_id) REFERENCES members(id),
            UNIQUE(member_id, date)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_date TEXT NOT NULL,
            total_members INTEGER DEFAULT 0,
            present_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()

def get_all_members():
    conn = get_db()
    members = conn.execute(
        'SELECT * FROM members WHERE active=1 ORDER BY name'
    ).fetchall()
    conn.close()
    return [dict(m) for m in members]

def get_member_by_label(label):
    conn = get_db()
    member = conn.execute(
        'SELECT * FROM members WHERE label=?', (label,)
    ).fetchone()
    conn.close()
    return dict(member) if member else None

def add_member(name, employee_id, department, photo_path, label):
    conn = get_db()
    try:
        conn.execute(
            'INSERT INTO members (name, employee_id, department, photo_path, label) VALUES (?,?,?,?,?)',
            (name, employee_id, department, photo_path, label)
        )
        conn.commit()
        member = conn.execute('SELECT * FROM members WHERE employee_id=?', (employee_id,)).fetchone()
        conn.close()
        return dict(member)
    except sqlite3.IntegrityError as e:
        conn.close()
        raise ValueError(f"Member with employee_id '{employee_id}' already exists.")

def mark_attendance(member_id, confidence):
    conn = get_db()
    today = date.today().isoformat()
    now = datetime.now().strftime('%H:%M:%S')
    existing = conn.execute(
        'SELECT * FROM attendance WHERE member_id=? AND date=?',
        (member_id, today)
    ).fetchone()

    if existing:
        if not existing['check_out']:
            conn.execute(
                'UPDATE attendance SET check_out=? WHERE member_id=? AND date=?',
                (now, member_id, today)
            )
            conn.commit()
            action = 'check_out'
        else:
            conn.close()
            return 'already_complete', dict(existing)
    else:
        conn.execute(
            'INSERT INTO attendance (member_id, date, check_in, confidence) VALUES (?,?,?,?)',
            (member_id, today, now, confidence)
        )
        conn.commit()
        action = 'check_in'

    record = conn.execute(
        'SELECT * FROM attendance WHERE member_id=? AND date=?',
        (member_id, today)
    ).fetchone()
    conn.close()
    return action, dict(record)

def get_attendance_today():
    conn = get_db()
    today = date.today().isoformat()
    records = conn.execute('''
        SELECT a.*, m.name, m.employee_id, m.department, m.photo_path
        FROM attendance a
        JOIN members m ON a.member_id = m.id
        WHERE a.date = ?
        ORDER BY a.check_in DESC
    ''', (today,)).fetchall()
    conn.close()
    return [dict(r) for r in records]

def get_attendance_stats():
    conn = get_db()
    today = date.today().isoformat()
    total_members = conn.execute('SELECT COUNT(*) FROM members WHERE active=1').fetchone()[0]
    present_today = conn.execute(
        'SELECT COUNT(*) FROM attendance WHERE date=?', (today,)
    ).fetchone()[0]

    # Overall attendance percentage per member
    member_stats = conn.execute('''
        SELECT m.name, m.employee_id, m.department,
               COUNT(a.id) as days_present,
               (SELECT COUNT(DISTINCT date) FROM attendance) as total_days,
               ROUND(COUNT(a.id) * 100.0 / MAX((SELECT COUNT(DISTINCT date) FROM attendance), 1), 1) as percentage
        FROM members m
        LEFT JOIN attendance a ON m.id = a.member_id
        WHERE m.active=1
        GROUP BY m.id
        ORDER BY percentage DESC
    ''').fetchall()

    # Last 7 days trend
    trend = conn.execute('''
        SELECT date, COUNT(*) as count
        FROM attendance
        WHERE date >= date('now', '-6 days')
        GROUP BY date
        ORDER BY date
    ''').fetchall()

    conn.close()
    return {
        'total_members': total_members,
        'present_today': present_today,
        'absent_today': total_members - present_today,
        'attendance_rate': round(present_today * 100.0 / max(total_members, 1), 1),
        'member_stats': [dict(r) for r in member_stats],
        'trend': [dict(r) for r in trend]
    }

def get_next_label():
    conn = get_db()
    max_label = conn.execute('SELECT MAX(label) FROM members').fetchone()[0]
    conn.close()
    return (max_label or 0) + 1

def delete_member(member_id):
    conn = get_db()
    conn.execute('UPDATE members SET active=0 WHERE id=?', (member_id,))
    conn.commit()
    conn.close()

def get_attendance_history(days=30):
    conn = get_db()
    records = conn.execute('''
        SELECT a.date, a.check_in, a.check_out, a.confidence,
               m.name, m.employee_id, m.department
        FROM attendance a
        JOIN members m ON a.member_id = m.id
        WHERE a.date >= date('now', ?)
        ORDER BY a.date DESC, a.check_in DESC
    ''', (f'-{days} days',)).fetchall()
    conn.close()
    return [dict(r) for r in records]

def get_attendance_by_date_range(start_date, end_date):
    conn = get_db()
    records = conn.execute('''
        SELECT a.date, a.check_in, a.check_out, a.confidence, a.status,
               m.name, m.employee_id, m.department
        FROM attendance a
        JOIN members m ON a.member_id = m.id
        WHERE a.date >= ? AND a.date <= ?
        ORDER BY a.date DESC, m.name ASC
    ''', (start_date, end_date)).fetchall()
    conn.close()
    return [dict(r) for r in records]

def bulk_add_members(rows):
    """
    Bulk add members from CSV rows.
    Each row is a dict with keys: name, employee_id, department (optional).
    Returns (added_list, skipped_list) where skipped contains rows that
    failed (e.g. duplicate employee_id or missing required fields).
    """
    conn = get_db()
    added = []
    skipped = []

    max_label = conn.execute('SELECT MAX(label) FROM members').fetchone()[0] or 0

    for row in rows:
        name = (row.get('name') or '').strip()
        employee_id = (row.get('employee_id') or '').strip()
        department = (row.get('department') or 'General').strip() or 'General'

        if not name or not employee_id:
            skipped.append({'row': row, 'reason': 'Missing name or employee_id'})
            continue

        existing = conn.execute(
            'SELECT id FROM members WHERE employee_id=?', (employee_id,)
        ).fetchone()
        if existing:
            skipped.append({'row': row, 'reason': f"Employee ID '{employee_id}' already exists"})
            continue

        max_label += 1
        try:
            conn.execute(
                'INSERT INTO members (name, employee_id, department, photo_path, label) VALUES (?,?,?,?,?)',
                (name, employee_id, department, None, max_label)
            )
            added.append({'name': name, 'employee_id': employee_id, 'department': department})
        except sqlite3.IntegrityError as e:
            max_label -= 1
            skipped.append({'row': row, 'reason': str(e)})

    conn.commit()
    conn.close()
    return added, skipped
