"""
PMS (Planned Maintenance System) MCP Server
=============================================
Loads 6 Excel reports into SQLite, exposes 8 tools for Qwen3 tool calling.

Tables:
  - equipment        (2,278 rows)  — Equipment registry: codes, makers, models, types
  - job_plan         (3,901 rows)  — Scheduled maintenance: frequencies, due dates, descriptions
  - pending_jobs     (3,320 rows)  — Jobs due/overdue: status, priority, due dates
  - completed_jobs  (33,033 rows)  — Maintenance history: dates, reports, condition
  - running_hours   (12,154 rows)  — Equipment running hour readings over time
  - spare_parts     (28,071 rows)  — Spare parts inventory: ROB, min, reorder, mapped equipment

Usage:
  python mcp_pms.py

Runs on port 5011.
"""

from flask import Flask, request, jsonify
import sqlite3
import pandas as pd
import os
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask('mcp_pms')

DB_PATH = r"C:\Users\User\Desktop\Main Engine Diagnostics\pms_data.db"
PMS_FOLDER = r"C:\Users\User\Desktop\Main Engine Diagnostics\Clemens Schulte (IMO 9665671)\PMS Reports"

# ============================================================
# EXCEL → SQLITE LOADER
# ============================================================

def load_excel_to_sqlite():
    """Parse all 6 PMS Excel files and load into SQLite"""
    
    if os.path.exists(DB_PATH):
        logger.info(f"PMS database already exists at {DB_PATH}")
        # Verify tables exist
        conn = sqlite3.connect(DB_PATH)
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        conn.close()
        if len(tables) >= 6:
            logger.info(f"All tables present: {tables}")
            return
        logger.info("Rebuilding database — missing tables")
    
    logger.info("Loading PMS Excel files into SQLite...")
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # 1. EQUIPMENT QUERY
        logger.info("Loading Equipment Query...")
        df = pd.read_excel(
            os.path.join(PMS_FOLDER, "Equipment Query.xls"),
            engine='xlrd', header=None
        )
        col_map = {1: 'sno', 2: 'equipment_code', 3: 'equipment_name', 7: 'maker',
                   8: 'model', 9: 'builder', 10: 'serial_number', 11: 'type',
                   12: 'class_reference', 14: 'safety_level', 16: 'print_equipment',
                   17: 'equipment_type', 19: 'location', 23: 'department'}
        equip_data = df.iloc[13:].rename(columns=col_map)[[v for v in col_map.values()]]
        equip_data = equip_data.dropna(subset=['equipment_code'])
        equip_data['equipment_name'] = equip_data['equipment_name'].str.strip()
        equip_data.to_sql('equipment', conn, if_exists='replace', index=False)
        logger.info(f"  Equipment: {len(equip_data)} rows")
        
        # 2. JOB PLAN
        logger.info("Loading Job Plan...")
        df = pd.read_excel(
            os.path.join(PMS_FOLDER, "Job Plan.xls"),
            engine='xlrd', header=None
        )
        col_map = {0: 'sno', 3: 'equipment_code', 4: 'equipment_name', 5: 'job_title',
                   6: 'job_description', 9: 'job_type', 10: 'frequency', 11: 'job_code',
                   12: 'last_done_date', 13: 'next_due_date', 14: 'last_done_hours',
                   15: 'next_due_hrs', 16: 'discipline', 17: 'present_reading',
                   19: 'rhrs_days_since_last', 20: 'remaining_rhrs_days',
                   21: 'critical_to_safety', 23: 'risk_assessment_required',
                   24: 'form_attached', 25: 'procedures', 27: 'remarks'}
        plan_data = df.iloc[14:].rename(columns=col_map)[[v for v in col_map.values()]]
        plan_data = plan_data.dropna(subset=['equipment_code'])
        plan_data['equipment_name'] = plan_data['equipment_name'].str.strip()
        plan_data['job_title'] = plan_data['job_title'].str.strip()
        plan_data.to_sql('job_plan', conn, if_exists='replace', index=False)
        logger.info(f"  Job Plan: {len(plan_data)} rows")
        
        # 3. PENDING JOBS
        logger.info("Loading Pending Jobs...")
        df = pd.read_excel(
            os.path.join(PMS_FOLDER, "Job_history_Pending_Job.xls"),
            engine='xlrd', header=None
        )
        col_map = {1: 'job_order_no', 3: 'job_title', 6: 'equipment_code',
                   7: 'equipment_name', 9: 'job_description', 12: 'next_due_date',
                   14: 'next_due_hrs', 15: 'interval', 16: 'frequency',
                   17: 'last_done_date', 19: 'present_rhrs', 21: 'job_status',
                   23: 'job_priority', 26: 'previous_job_report', 27: 'group_job',
                   28: 'class_reference', 30: 'discipline', 31: 'incident_number'}
        pending_data = df.iloc[11:].rename(columns=col_map)[[v for v in col_map.values()]]
        pending_data = pending_data.dropna(subset=['job_order_no'])
        pending_data['job_title'] = pending_data['job_title'].str.strip()
        pending_data['equipment_name'] = pending_data['equipment_name'].str.strip()
        pending_data.to_sql('pending_jobs', conn, if_exists='replace', index=False)
        logger.info(f"  Pending Jobs: {len(pending_data)} rows")
        
        # 4. COMPLETED JOBS
        logger.info("Loading Completed Jobs...")
        df = pd.read_excel(
            os.path.join(PMS_FOLDER, "Job History_Completed_Job.xls"),
            engine='xlrd', header=None
        )
        col_map = {1: 'sno', 4: 'job_order_no', 7: 'job_title', 8: 'equipment_code',
                   10: 'equipment_name', 11: 'interval', 13: 'frequency',
                   14: 'job_start_date', 15: 'job_end_date', 16: 'job_done_hrs',
                   17: 'job_status', 18: 'job_report', 19: 'special_note',
                   20: 'condition_after_job', 21: 'overdue_reason', 22: 'done_by',
                   23: 'cancellation_date', 25: 'reviewed_by', 26: 'job_assigned_to',
                   27: 'emp_name'}
        completed_data = df.iloc[15:].rename(columns=col_map)[[v for v in col_map.values()]]
        completed_data = completed_data.dropna(subset=['job_order_no'])
        completed_data['job_title'] = completed_data['job_title'].str.strip()
        completed_data['equipment_name'] = completed_data['equipment_name'].str.strip()
        completed_data.to_sql('completed_jobs', conn, if_exists='replace', index=False)
        logger.info(f"  Completed Jobs: {len(completed_data)} rows")
        
        # 5. RUNNING HOURS
        logger.info("Loading Running Hours...")
        rh_data = pd.read_excel(
            os.path.join(PMS_FOLDER, "RUNNING HOUR HISTORY.xlsx"),
            engine='openpyxl'
        )
        rh_data.columns = ['vessel_name', 'equipment_code', 'equipment_name',
                           'counter_reading', 'reading_date']
        rh_data['reading_date'] = pd.to_datetime(rh_data['reading_date']).dt.strftime('%Y-%m-%d')
        rh_data.to_sql('running_hours', conn, if_exists='replace', index=False)
        logger.info(f"  Running Hours: {len(rh_data)} rows")
        
        # 6. SPARE PARTS
        logger.info("Loading Spare Parts...")
        df = pd.read_excel(
            os.path.join(PMS_FOLDER, "Sapres_spare_query.xlsx"),
            engine='openpyxl', header=None
        )
        col_map = {1: 'part_number', 4: 'item_description', 5: 'storage_location',
                   7: 'item_specification', 9: 'plate_drawing_number', 10: 'uom',
                   11: 'normal_stock', 12: 'reconditioned_stock', 13: 'rob',
                   15: 'min_stock', 16: 'reorder', 17: 'max_stock',
                   18: 'class_essential', 20: 'item_code', 22: 'mapped_equipment'}
        spares_data = df.iloc[12:].rename(columns=col_map)[[v for v in col_map.values()]]
        spares_data = spares_data.dropna(subset=['item_description'])
        spares_data['item_description'] = spares_data['item_description'].str.strip()
        # Clean numeric columns
        for col in ['normal_stock', 'reconditioned_stock', 'rob', 'min_stock', 'reorder', 'max_stock']:
            spares_data[col] = pd.to_numeric(spares_data[col], errors='coerce').fillna(0).astype(int)
        spares_data.to_sql('spare_parts', conn, if_exists='replace', index=False)
        logger.info(f"  Spare Parts: {len(spares_data)} rows")
        
        # CREATE INDEXES for fast searching
        logger.info("Creating indexes...")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_equip_code ON equipment(equipment_code)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_equip_name ON equipment(equipment_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jp_equip ON job_plan(equipment_code)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jp_type ON job_plan(job_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pj_equip ON pending_jobs(equipment_code)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pj_status ON pending_jobs(job_status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pj_priority ON pending_jobs(job_priority)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cj_equip ON completed_jobs(equipment_code)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cj_status ON completed_jobs(job_status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rh_equip ON running_hours(equipment_code)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sp_desc ON spare_parts(item_description)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sp_mapped ON spare_parts(mapped_equipment)")
        conn.commit()
        
        logger.info("PMS database built successfully!")
        
    except Exception as e:
        logger.error(f"Error loading PMS data: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    finally:
        conn.close()


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================
# TOOL DEFINITIONS — what Qwen3 sees
# ============================================================

PMS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_equipment",
            "description": "Search the vessel equipment registry. Find equipment by name, code, maker, type, department, or location. Use when asked about what equipment is on board, equipment details, maker info, serial numbers, or equipment in a specific area.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Search keyword (e.g., 'turbocharger', 'boiler', 'pump', 'MAN', 'engine room')"
                    },
                    "field": {
                        "type": "string",
                        "enum": ["all", "equipment_name", "equipment_code", "maker", "equipment_type", "department", "location"],
                        "description": "Which field to search. Use 'all' for general search."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results to return (default: 20)"
                    }
                },
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_maintenance_schedule",
            "description": "Get planned/scheduled maintenance jobs for equipment. Shows job title, description, frequency, job type, next due date, and remaining days/hours. Use when asked about maintenance schedule, planned jobs, job frequency, what maintenance is due, or inspection schedule.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Equipment name or code to search (e.g., 'turbocharger', '651.053', 'boiler')"
                    },
                    "job_type": {
                        "type": "string",
                        "enum": ["all", "INSPECTION", "TESTING", "OVERHAUL", "SERVICE", "MAINTENANCE", "CALIBRATION", "REPLACEMENT"],
                        "description": "Filter by job type (default: all)"
                    },
                    "critical_only": {
                        "type": "boolean",
                        "description": "If true, only show jobs marked critical to safety"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 20)"
                    }
                },
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_pending_jobs",
            "description": "Get pending/due maintenance jobs. Shows overdue and upcoming jobs with due dates, priority, and status. Use when asked about overdue maintenance, pending work, upcoming jobs, due jobs, or what needs to be done.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Equipment name or code, or 'all' for all pending jobs"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["all", "URGENT", "HIGH", "NORMAL", "LOW"],
                        "description": "Filter by priority (default: all)"
                    },
                    "overdue_only": {
                        "type": "boolean",
                        "description": "If true, only show overdue jobs (past due date)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 30)"
                    }
                },
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_job_history",
            "description": "Get completed maintenance job history for equipment. Shows what was done, when, job reports, condition after job, who did it. Use when asked about maintenance history, last maintenance, what was done to equipment, job reports, or completed work.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Equipment name or code to search (e.g., 'main engine', '551.001')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 20, max: 50)"
                    }
                },
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_running_hours",
            "description": "Get equipment running hour readings and trends. Shows counter readings over time. Use when asked about running hours, equipment hours, how long something has been running, or running hour trends.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Equipment name or code (e.g., 'main engine', 'generator', '551.001')"
                    },
                    "latest_only": {
                        "type": "boolean",
                        "description": "If true, only return the latest reading per equipment"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 20)"
                    }
                },
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_spare_parts",
            "description": "Search spare parts inventory. Shows part numbers, descriptions, stock levels (ROB), minimum stock, reorder levels, and which equipment they belong to. Use when asked about spare parts, stock levels, parts availability, reorder needs, or parts for specific equipment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search_term": {
                        "type": "string",
                        "description": "Part name, description, or equipment name (e.g., 'gasket', 'bearing', 'turbocharger')"
                    },
                    "low_stock_only": {
                        "type": "boolean",
                        "description": "If true, only show parts where ROB <= minimum stock"
                    },
                    "equipment_code": {
                        "type": "string",
                        "description": "Filter by mapped equipment code (e.g., '651.053')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 25)"
                    }
                },
                "required": ["search_term"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_maintenance_summary",
            "description": "Get a high-level maintenance summary/dashboard. Shows counts of pending, overdue, completed jobs, critical equipment status, and low-stock spares. Use when asked for maintenance overview, PMS status, dashboard, summary, or general maintenance health.",
            "parameters": {
                "type": "object",
                "properties": {
                    "include_details": {
                        "type": "boolean",
                        "description": "If true, include lists of overdue jobs and low-stock parts (default: false)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_equipment_full_status",
            "description": "Get COMPLETE status of a specific equipment: its details, scheduled maintenance, pending jobs, recent job history, running hours, and available spare parts — all in one call. Use when asked about the full status or complete picture of a specific equipment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "equipment_code": {
                        "type": "string",
                        "description": "Equipment code (e.g., '651.053', '644.001')"
                    },
                    "equipment_name": {
                        "type": "string",
                        "description": "Equipment name to search if code unknown (e.g., 'turbocharger')"
                    }
                },
                "required": []
            }
        }
    }
]


# ============================================================
# TOOL HANDLERS
# ============================================================

def handle_search_equipment(args):
    search = args.get('search_term', '')
    field = args.get('field', 'all')
    limit = min(args.get('limit', 20), 50)
    
    conn = get_db()
    try:
        if field == 'all':
            query = """
                SELECT equipment_code, equipment_name, maker, model, serial_number,
                       equipment_type, class_reference, safety_level, location, department
                FROM equipment
                WHERE equipment_code LIKE ? OR equipment_name LIKE ? OR maker LIKE ?
                      OR equipment_type LIKE ? OR department LIKE ? OR location LIKE ?
                LIMIT ?
            """
            term = f"%{search}%"
            rows = conn.execute(query, (term, term, term, term, term, term, limit)).fetchall()
        else:
            query = f"""
                SELECT equipment_code, equipment_name, maker, model, serial_number,
                       equipment_type, class_reference, safety_level, location, department
                FROM equipment WHERE {field} LIKE ? LIMIT ?
            """
            rows = conn.execute(query, (f"%{search}%", limit)).fetchall()
        
        results = [dict(r) for r in rows]
        return {"search_term": search, "field": field, "count": len(results), "equipment": results}
    finally:
        conn.close()


def handle_maintenance_schedule(args):
    search = args.get('search_term', '')
    job_type = args.get('job_type', 'all')
    critical_only = args.get('critical_only', False)
    limit = min(args.get('limit', 20), 50)
    
    conn = get_db()
    try:
        conditions = ["(equipment_code LIKE ? OR equipment_name LIKE ? OR job_title LIKE ?)"]
        params = [f"%{search}%", f"%{search}%", f"%{search}%"]
        
        if job_type != 'all':
            conditions.append("job_type LIKE ?")
            params.append(f"%{job_type}%")
        
        if critical_only:
            conditions.append("critical_to_safety = 'YES'")
        
        where = " AND ".join(conditions)
        query = f"""
            SELECT equipment_code, equipment_name, job_title, job_description,
                   job_type, frequency, job_code, last_done_date, next_due_date,
                   remaining_rhrs_days, critical_to_safety, discipline, remarks
            FROM job_plan WHERE {where}
            ORDER BY next_due_date ASC
            LIMIT ?
        """
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        results = [dict(r) for r in rows]
        
        return {"search_term": search, "job_type": job_type, "count": len(results), "jobs": results}
    finally:
        conn.close()


def handle_pending_jobs(args):
    search = args.get('search_term', '')
    priority = args.get('priority', 'all')
    overdue_only = args.get('overdue_only', False)
    limit = min(args.get('limit', 30), 100)
    
    conn = get_db()
    try:
        conditions = []
        params = []
        
        if search.lower() != 'all':
            conditions.append("(equipment_code LIKE ? OR equipment_name LIKE ? OR job_title LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
        
        if priority != 'all':
            conditions.append("job_priority LIKE ?")
            params.append(f"%{priority}%")
        
        if overdue_only:
            conditions.append("next_due_date < date('now')")
        
        where = " AND ".join(conditions) if conditions else "1=1"
        query = f"""
            SELECT job_order_no, job_title, equipment_code, equipment_name,
                   job_description, next_due_date, frequency, interval,
                   last_done_date, job_status, job_priority, class_reference, discipline
            FROM pending_jobs WHERE {where}
            ORDER BY next_due_date ASC
            LIMIT ?
        """
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        results = [dict(r) for r in rows]
        
        # Count overdue
        overdue_count = sum(1 for r in results if r.get('next_due_date') and r['next_due_date'] < datetime.now().strftime('%d-%b-%Y'))
        
        return {
            "search_term": search, "count": len(results),
            "priority_filter": priority, "overdue_only": overdue_only,
            "jobs": results
        }
    finally:
        conn.close()


def handle_job_history(args):
    search = args.get('search_term', '')
    limit = min(args.get('limit', 20), 50)
    
    conn = get_db()
    try:
        query = """
            SELECT job_order_no, job_title, equipment_code, equipment_name,
                   frequency, job_start_date, job_end_date, job_status,
                   job_report, special_note, condition_after_job,
                   done_by, reviewed_by, emp_name
            FROM completed_jobs
            WHERE equipment_code LIKE ? OR equipment_name LIKE ? OR job_title LIKE ?
            ORDER BY job_end_date DESC
            LIMIT ?
        """
        term = f"%{search}%"
        rows = conn.execute(query, (term, term, term, limit)).fetchall()
        results = [dict(r) for r in rows]
        
        return {"search_term": search, "count": len(results), "history": results}
    finally:
        conn.close()


def handle_running_hours(args):
    search = args.get('search_term', '')
    latest_only = args.get('latest_only', False)
    limit = min(args.get('limit', 20), 100)
    
    conn = get_db()
    try:
        if latest_only:
            query = """
                SELECT equipment_code, equipment_name, counter_reading, reading_date
                FROM running_hours
                WHERE (equipment_code LIKE ? OR equipment_name LIKE ?)
                AND reading_date = (
                    SELECT MAX(rh2.reading_date) FROM running_hours rh2
                    WHERE rh2.equipment_code = running_hours.equipment_code
                )
                GROUP BY equipment_code
                ORDER BY counter_reading DESC
                LIMIT ?
            """
        else:
            query = """
                SELECT equipment_code, equipment_name, counter_reading, reading_date
                FROM running_hours
                WHERE equipment_code LIKE ? OR equipment_name LIKE ?
                ORDER BY equipment_code, reading_date DESC
                LIMIT ?
            """
        
        term = f"%{search}%"
        rows = conn.execute(query, (term, term, limit)).fetchall()
        results = [dict(r) for r in rows]
        
        return {"search_term": search, "latest_only": latest_only, "count": len(results), "readings": results}
    finally:
        conn.close()


def handle_spare_parts(args):
    search = args.get('search_term', '')
    low_stock_only = args.get('low_stock_only', False)
    equipment_code = args.get('equipment_code', '')
    limit = min(args.get('limit', 25), 100)
    
    conn = get_db()
    try:
        conditions = ["(part_number LIKE ? OR item_description LIKE ? OR mapped_equipment LIKE ?)"]
        params = [f"%{search}%", f"%{search}%", f"%{search}%"]
        
        if equipment_code:
            conditions.append("mapped_equipment LIKE ?")
            params.append(f"%{equipment_code}%")
        
        if low_stock_only:
            conditions.append("rob <= min_stock")
        
        where = " AND ".join(conditions)
        query = f"""
            SELECT part_number, item_description, storage_location, uom,
                   normal_stock, rob, min_stock, reorder, max_stock,
                   class_essential, item_code, mapped_equipment
            FROM spare_parts WHERE {where}
            LIMIT ?
        """
        params.append(limit)
        rows = conn.execute(query, params).fetchall()
        results = [dict(r) for r in rows]
        
        low_stock_count = sum(1 for r in results if r['rob'] <= r['min_stock'])
        
        return {
            "search_term": search, "count": len(results),
            "low_stock_count": low_stock_count,
            "parts": results
        }
    finally:
        conn.close()


def handle_maintenance_summary(args):
    include_details = args.get('include_details', False)
    
    conn = get_db()
    try:
        summary = {}
        
        # Total equipment
        summary['total_equipment'] = conn.execute("SELECT COUNT(*) FROM equipment").fetchone()[0]
        
        # Job plan stats
        summary['total_planned_jobs'] = conn.execute("SELECT COUNT(*) FROM job_plan").fetchone()[0]
        safety_critical = conn.execute("SELECT COUNT(*) FROM job_plan WHERE critical_to_safety = 'YES'").fetchone()[0]
        summary['safety_critical_jobs'] = safety_critical
        
        # Pending jobs
        summary['total_pending_jobs'] = conn.execute("SELECT COUNT(*) FROM pending_jobs").fetchone()[0]
        
        # Priority breakdown
        priorities = conn.execute("""
            SELECT job_priority, COUNT(*) as cnt FROM pending_jobs
            GROUP BY job_priority ORDER BY cnt DESC
        """).fetchall()
        summary['pending_by_priority'] = {r['job_priority']: r['cnt'] for r in priorities}
        
        # Completed jobs
        summary['total_completed_jobs'] = conn.execute("SELECT COUNT(*) FROM completed_jobs").fetchone()[0]
        
        # Spare parts
        summary['total_spare_parts'] = conn.execute("SELECT COUNT(*) FROM spare_parts").fetchone()[0]
        low_stock = conn.execute("SELECT COUNT(*) FROM spare_parts WHERE rob <= min_stock AND min_stock > 0").fetchone()[0]
        summary['low_stock_parts'] = low_stock
        zero_stock = conn.execute("SELECT COUNT(*) FROM spare_parts WHERE rob = 0 AND min_stock > 0").fetchone()[0]
        summary['zero_stock_parts'] = zero_stock
        
        # Running hours - equipment count
        rh_equip = conn.execute("SELECT COUNT(DISTINCT equipment_code) FROM running_hours").fetchone()[0]
        summary['equipment_with_running_hours'] = rh_equip
        
        if include_details:
            # Top overdue jobs (by priority)
            overdue_rows = conn.execute("""
                SELECT job_order_no, job_title, equipment_name, next_due_date, job_priority
                FROM pending_jobs
                WHERE job_priority IN ('URGENT', 'HIGH')
                ORDER BY next_due_date ASC
                LIMIT 15
            """).fetchall()
            summary['high_priority_pending'] = [dict(r) for r in overdue_rows]
            
            # Low stock parts
            low_rows = conn.execute("""
                SELECT part_number, item_description, rob, min_stock, mapped_equipment
                FROM spare_parts
                WHERE rob <= min_stock AND min_stock > 0
                ORDER BY (min_stock - rob) DESC
                LIMIT 15
            """).fetchall()
            summary['low_stock_details'] = [dict(r) for r in low_rows]
            
            # Recent completed jobs
            recent = conn.execute("""
                SELECT job_title, equipment_name, job_end_date, job_status
                FROM completed_jobs
                ORDER BY job_end_date DESC
                LIMIT 10
            """).fetchall()
            summary['recent_completed'] = [dict(r) for r in recent]
        
        return summary
    finally:
        conn.close()


def handle_equipment_full_status(args):
    equip_code = args.get('equipment_code', '')
    equip_name = args.get('equipment_name', '')
    
    conn = get_db()
    try:
        result = {}
        
        # Find equipment
        if equip_code:
            search_code = equip_code
            equip_rows = conn.execute(
                "SELECT * FROM equipment WHERE equipment_code = ? OR equipment_code LIKE ?",
                (equip_code, f"{equip_code}%")
            ).fetchall()
        elif equip_name:
            equip_rows = conn.execute(
                "SELECT * FROM equipment WHERE equipment_name LIKE ?",
                (f"%{equip_name}%",)
            ).fetchall()
            search_code = equip_rows[0]['equipment_code'] if equip_rows else equip_name
        else:
            return {"error": "Provide equipment_code or equipment_name"}
        
        if equip_rows:
            result['equipment'] = [dict(r) for r in equip_rows[:5]]
            search_code = equip_rows[0]['equipment_code']
        else:
            result['equipment'] = []
            search_code = equip_code or equip_name
        
        # Scheduled maintenance
        plan_rows = conn.execute("""
            SELECT job_title, job_type, frequency, next_due_date,
                   remaining_rhrs_days, critical_to_safety
            FROM job_plan
            WHERE equipment_code LIKE ?
            ORDER BY next_due_date ASC LIMIT 10
        """, (f"%{search_code}%",)).fetchall()
        result['maintenance_schedule'] = [dict(r) for r in plan_rows]
        
        # Pending jobs
        pending_rows = conn.execute("""
            SELECT job_order_no, job_title, next_due_date, job_priority, job_status
            FROM pending_jobs
            WHERE equipment_code LIKE ?
            ORDER BY next_due_date ASC LIMIT 10
        """, (f"%{search_code}%",)).fetchall()
        result['pending_jobs'] = [dict(r) for r in pending_rows]
        
        # Recent completed jobs
        completed_rows = conn.execute("""
            SELECT job_title, job_start_date, job_end_date, job_report,
                   condition_after_job, done_by
            FROM completed_jobs
            WHERE equipment_code LIKE ?
            ORDER BY job_end_date DESC LIMIT 10
        """, (f"%{search_code}%",)).fetchall()
        result['recent_history'] = [dict(r) for r in completed_rows]
        
        # Running hours
        rh_rows = conn.execute("""
            SELECT counter_reading, reading_date
            FROM running_hours
            WHERE equipment_code LIKE ?
            ORDER BY reading_date DESC LIMIT 5
        """, (f"%{search_code}%",)).fetchall()
        result['running_hours'] = [dict(r) for r in rh_rows]
        
        # Spare parts
        spare_rows = conn.execute("""
            SELECT part_number, item_description, rob, min_stock, reorder
            FROM spare_parts
            WHERE mapped_equipment LIKE ?
            LIMIT 15
        """, (f"%{search_code}%",)).fetchall()
        result['spare_parts'] = [dict(r) for r in spare_rows]
        result['spare_parts_count'] = len(spare_rows)
        
        return result
    finally:
        conn.close()


# ============================================================
# FLASK ENDPOINTS
# ============================================================

@app.route('/mcp/tools', methods=['GET'])
def list_tools():
    return jsonify({"tools": PMS_TOOLS})


@app.route('/mcp/call', methods=['POST'])
def call_tool():
    try:
        data = request.get_json()
        tool_name = data.get('name')
        arguments = data.get('arguments', {})
        
        logger.info(f"PMS Tool call: {tool_name} with args: {arguments}")
        
        handlers = {
            "search_equipment": handle_search_equipment,
            "get_maintenance_schedule": handle_maintenance_schedule,
            "get_pending_jobs": handle_pending_jobs,
            "get_job_history": handle_job_history,
            "get_running_hours": handle_running_hours,
            "search_spare_parts": handle_spare_parts,
            "get_maintenance_summary": handle_maintenance_summary,
            "get_equipment_full_status": handle_equipment_full_status,
        }
        
        handler = handlers.get(tool_name)
        if not handler:
            return jsonify({"error": f"Unknown tool: {tool_name}"}), 400
        
        result = handler(arguments)
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"PMS Tool call error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/mcp/health', methods=['GET'])
def health():
    conn = get_db()
    try:
        tables = {}
        for table in ['equipment', 'job_plan', 'pending_jobs', 'completed_jobs', 'running_hours', 'spare_parts']:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            tables[table] = count
        
        return jsonify({
            "status": "healthy",
            "database": DB_PATH,
            "tables": tables,
            "total_records": sum(tables.values())
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500
    finally:
        conn.close()


# ============================================================
# STARTUP
# ============================================================

if __name__ == '__main__':
    logger.info("Starting PMS MCP Server...")
    logger.info(f"PMS folder: {PMS_FOLDER}")
    logger.info(f"Database: {DB_PATH}")
    
    # Load Excel → SQLite on first run
    load_excel_to_sqlite()
    
    app.run(host='0.0.0.0', port=5011, debug=False)