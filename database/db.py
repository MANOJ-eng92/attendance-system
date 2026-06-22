import os
import psycopg2
import psycopg2.extras
from datetime import datetime, date, timedelta

DATABASE_URL = os.environ.get('DATABASE_URL', '')


def get_db():
    """
    Returns a PostgreSQL connection with dict-like row access
    (similar to sqlite3.Row via RealDictCursor).
    """
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS members (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            employee_id TEXT UNIQUE NOT NULL,
            department TEXT DEFAULT 'General',
            photo_data TEXT,
            label INTEGER UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            active INTEGER DEFAULT 1
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id SERIAL PRIMARY KEY,
            member_id INTEGER NOT NULL REFERENCES members(id),
            date TEXT NOT NULL,
            check_in TEXT,
            check_out TEXT,
            status TEXT DEFAULT 'present',
            confidence REAL,
            UNIQUE(member_id, date)
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id SERIAL PRIMARY KEY,
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
    c = conn.cursor()
    c.execute('SELECT * FROM members WHERE active=1 ORDER BY name')
    members = c.fetchall()
    conn.close()
    return [dict(m) for m in members]


def get_member_by_label(label):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM members WHERE label=%s', (label,))
    member = c.fetchone()
    conn.close()
    return dict(member) if member else None


def get_member_by_id(member_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('SELECT * FROM members WHERE id=%s', (member_id,))
    member = c.fetchone()
    conn.close()
    return dict(member) if member else None


def add_member(name, employee_id, department, photo_data, label):
    """
    photo_data: base64-encoded string of the member's photo (or None).
    """
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute(
            'INSERT INTO members (name, employee_id, department, photo_data, label) VALUES (%s,%s,%s,%s,%s)',
            (name, employee_id, department, photo_data, label)
        )
        conn.commit()
        c.execute('SELECT * FROM members WHERE employee_id=%s', (employee_id,))
        member = c.fetchone()
        conn.close()
        return dict(member)
    except psycopg2.IntegrityError:
        conn.rollback()
        conn.close()
        raise ValueError(f"Member with employee_id '{employee_id}' already exists.")


def mark_attendance(member_id, confidence):
    conn = get_db()
    c = conn.cursor()
    today = date.today().isoformat()
    now = datetime.now().strftime('%H:%M:%S')

    c.execute('SELECT * FROM attendance WHERE member_id=%s AND date=%s', (member_id, today))
    existing = c.fetchone()

    if existing:
        if not existing['check_out']:
            c.execute(
                'UPDATE attendance SET check_out=%s WHERE member_id=%s AND date=%s',
                (now, member_id, today)
            )
            conn.commit()
            action = 'check_out'
        else:
            conn.close()
            return 'already_complete', dict(existing)
    else:
        c.execute(
            'INSERT INTO attendance (member_id, date, check_in, confidence) VALUES (%s,%s,%s,%s)',
            (member_id, today, now, confidence)
        )
        conn.commit()
        action = 'check_in'

    c.execute('SELECT * FROM attendance WHERE member_id=%s AND date=%s', (member_id, today))
    record = c.fetchone()
    conn.close()
    return action, dict(record)


def get_attendance_today():
    conn = get_db()
    c = conn.cursor()
    today = date.today().isoformat()
    c.execute('''
        SELECT a.*, m.name, m.employee_id, m.department, m.photo_data
        FROM attendance a
        JOIN members m ON a.member_id = m.id
        WHERE a.date = %s
        ORDER BY a.check_in DESC
    ''', (today,))
    records = c.fetchall()
    conn.close()
    return [dict(r) for r in records]


def get_attendance_stats():
    conn = get_db()
    c = conn.cursor()
    today = date.today().isoformat()

    c.execute('SELECT COUNT(*) AS count FROM members WHERE active=1')
    total_members = c.fetchone()['count']

    c.execute('SELECT COUNT(*) AS count FROM attendance WHERE date=%s', (today,))
    present_today = c.fetchone()['count']

    c.execute('''
        SELECT m.name, m.employee_id, m.department,
               COUNT(a.id) as days_present,
               (SELECT COUNT(DISTINCT date) FROM attendance) as total_days,
               ROUND(COUNT(a.id) * 100.0 / GREATEST((SELECT COUNT(DISTINCT date) FROM attendance), 1), 1) as percentage
        FROM members m
        LEFT JOIN attendance a ON m.id = a.member_id
        WHERE m.active=1
        GROUP BY m.id, m.name, m.employee_id, m.department
        ORDER BY percentage DESC
    ''')
    member_stats = c.fetchall()

    cutoff = (date.today() - timedelta(days=6)).isoformat()
    c.execute('''
        SELECT date, COUNT(*) as count
        FROM attendance
        WHERE date >= %s
        GROUP BY date
        ORDER BY date
    ''', (cutoff,))
    trend = c.fetchall()

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
    c = conn.cursor()
    c.execute('SELECT MAX(label) AS max_label FROM members')
    max_label = c.fetchone()['max_label']
    conn.close()
    return (max_label or 0) + 1


def delete_member(member_id):
    conn = get_db()
    c = conn.cursor()
    c.execute('UPDATE members SET active=0 WHERE id=%s', (member_id,))
    conn.commit()
    conn.close()


def get_attendance_history(days=30):
    conn = get_db()
    c = conn.cursor()
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    c.execute('''
        SELECT a.date, a.check_in, a.check_out, a.confidence,
               m.name, m.employee_id, m.department
        FROM attendance a
        JOIN members m ON a.member_id = m.id
        WHERE a.date >= %s
        ORDER BY a.date DESC, a.check_in DESC
    ''', (cutoff,))
    records = c.fetchall()
    conn.close()
    return [dict(r) for r in records]


def get_attendance_by_date_range(start_date, end_date):
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        SELECT a.date, a.check_in, a.check_out, a.confidence, a.status,
               m.name, m.employee_id, m.department
        FROM attendance a
        JOIN members m ON a.member_id = m.id
        WHERE a.date >= %s AND a.date <= %s
        ORDER BY a.date DESC, m.name ASC
    ''', (start_date, end_date))
    records = c.fetchall()
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
    c = conn.cursor()
    added = []
    skipped = []

    c.execute('SELECT MAX(label) AS max_label FROM members')
    max_label = c.fetchone()['max_label'] or 0

    for row in rows:
        name = (row.get('name') or '').strip()
        employee_id = (row.get('employee_id') or '').strip()
        department = (row.get('department') or 'General').strip() or 'General'

        if not name or not employee_id:
            skipped.append({'row': row, 'reason': 'Missing name or employee_id'})
            continue

        c.execute('SELECT id FROM members WHERE employee_id=%s', (employee_id,))
        existing = c.fetchone()
        if existing:
            skipped.append({'row': row, 'reason': f"Employee ID '{employee_id}' already exists"})
            continue

        max_label += 1
        try:
            c.execute(
                'INSERT INTO members (name, employee_id, department, photo_data, label) VALUES (%s,%s,%s,%s,%s)',
                (name, employee_id, department, None, max_label)
            )
            added.append({'name': name, 'employee_id': employee_id, 'department': department})
        except psycopg2.IntegrityError as e:
            conn.rollback()
            max_label -= 1
            skipped.append({'row': row, 'reason': str(e)})

    conn.commit()
    conn.close()
    return added, skipped


def get_attendance_by_date(target_date):
    """
    Return attendance records (present members) for a specific date,
    plus a list of members who were absent on that date.
    """
    conn = get_db()
    c = conn.cursor()

    c.execute('''
        SELECT a.*, m.name, m.employee_id, m.department, m.photo_data
        FROM attendance a
        JOIN members m ON a.member_id = m.id
        WHERE a.date = %s
        ORDER BY m.name ASC
    ''', (target_date,))
    present = c.fetchall()

    present_ids = [r['member_id'] for r in present]

    if present_ids:
        c.execute('''
            SELECT id, name, employee_id, department, photo_data
            FROM members
            WHERE active=1 AND id != ALL(%s)
            ORDER BY name ASC
        ''', (present_ids,))
    else:
        c.execute('''
            SELECT id, name, employee_id, department, photo_data
            FROM members
            WHERE active=1
            ORDER BY name ASC
        ''')
    absent = c.fetchall()

    conn.close()
    return [dict(r) for r in present], [dict(r) for r in absent]


def get_member_attendance_on_date(member_id, target_date):
    """
    Return the attendance record for a specific member on a specific date,
    or None if the member was absent that day.
    Also returns basic member info regardless.
    """
    conn = get_db()
    c = conn.cursor()

    c.execute('SELECT * FROM members WHERE id=%s', (member_id,))
    member = c.fetchone()

    if not member:
        conn.close()
        return None, None

    c.execute('SELECT * FROM attendance WHERE member_id=%s AND date=%s', (member_id, target_date))
    record = c.fetchone()

    conn.close()
    return dict(member), (dict(record) if record else None)


def save_model_to_db(model_data, labels_data):
    """Save trained model binary data to database."""
    conn = get_db()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS model_store (
            id INTEGER PRIMARY KEY DEFAULT 1,
            model_data BYTEA,
            labels_data BYTEA,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('DELETE FROM model_store')
    c.execute('INSERT INTO model_store (id, model_data, labels_data) VALUES (1, %s, %s)',
              (psycopg2.Binary(model_data), psycopg2.Binary(labels_data)))
    conn.commit()
    conn.close()


def load_model_from_db():
    """Load trained model binary data from database."""
    try:
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT model_data, labels_data FROM model_store WHERE id = 1')
        row = c.fetchone()
        conn.close()
        if row:
            return bytes(row['model_data']), bytes(row['labels_data'])
        return None, None
    except Exception:
        return None, None
