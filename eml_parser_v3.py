"""
EML Mail Parser v3 - Dual Engine (Regex + LLM)
================================================
Architecture:
    Raw EML → Thread Split → Per-email processing:
        Each email → Regex engine (fast, deterministic)
                   → LLM engine (smart, contextual)
                   → Decision engine (compare, choose, update)
    
    Decision logic:
        Both agree + high confidence  → auto-update
        Both agree + low confidence   → auto-update (lower threshold)
        Disagree + one high           → trust the high one
        Disagree + both high          → flag for human review
        Both low / neither finds      → skip (no update)

Usage:
    # With LLM (needs gpu_service running on port 5005)
    python eml_parser_v3.py "C:/path/to/email.eml"
    
    # Regex-only mode (no LLM needed)
    python eml_parser_v3.py "C:/path/to/email.eml" --regex-only
    
    # Folder
    python eml_parser_v3.py "C:/path/to/folder/"

Live files (updated after each email - refresh to see progress):
    live_tracker.csv  - Current tracker state
    live_audit.txt    - Decision trail
"""

import email
import email.policy
import os
import sys
import re
import json
import csv
import requests
import logging
from datetime import datetime
from email import policy
from email.parser import BytesParser
from pathlib import Path
from collections import defaultdict, OrderedDict
from typing import List, Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

KNOWN_VESSELS = [
    "molly schulte",
    "clemens schulte",
    "maren schulte",
    "anna schulte",
    "MAGDALENA SCHULTE",
    "MARY SCHULTE"
    # Add your vessels here
]

PRODUCTS = ["MDC", "Transbox", "Electrical Panel", "Flowmeter", "Energy Meter", "SPM"]

STAGE_ORDER = ["procurement", "install", "commission", "fat"]
STATUS_PRIORITY = {"pending": 0, "in_progress": 1, "done": 2}

PRODUCT_PATTERNS = {
    "MDC": [r'\bMDC\b', r'\bMDC\s+PC\b', r'\bdata\s+collector\b'],
    "Transbox": [r'\btransbox\b', r'\btrans[\s-]?box\b'],
    "Electrical Panel": [
        r'(?:MariApps|Memphis|mariapps)\s+(?:electrical\s+)?panel',
        r'electrical\s+panel',
    ],
    "Flowmeter": [r'\bflow\s*meter\b', r'\bF050\b', r'\bF025\b', r'\bPromass\b'],
    "Energy Meter": [r'\benergy\s+meter\b', r'\bRogowski\s+coil', r'\benergy\s+meters\b'],
    "SPM": [r'\bSPM\b', r'\bshaft\s+power\b'],
}

NOISE_PATTERNS = [
    r"Thanks\s*[&and]*\s*Regards.*",
    r"Sincerely,.*",
    r"Best\s+[Rr]egards.*",
    r"Kind\s+[Rr]egards.*",
    r"MariApps\s+(?:Marine|House).*",
    r"www\.(?:mariapps|memphis-marine)\.com.*",
    r"(?:Telex|FBB\s+Phone|IP\s+Phone):?.*",
    r"Email:\s*master@.*",
    r"\|.*(?:Engineer|Manager|Director).*@.*",
    r"(?:Plot\s+No|SmartCity|Kerala).*",
    r"Mob:\s*\+?\d+",
    r"\[.*?Logo.*?\]",
    r"\[.*?Description\s+automatically.*?\]",
    r"CLICK\s+HERE.*",
    r"_{3,}",
    r"<mailto:[^>]+>",
    r"<https?://[^>]+>",
    r"mailto:\S+",
]

REGEX_RULES = [
    # PROCUREMENT
    (r'(?:parts|equipment|hardware|items|cargo|goods)\s+(?:arrived|received|on\s+board|delivered)',
     'procurement', 'done', 0.9, 'Items received on board'),
    (r'(?:procurement|delivery)\s+completed',
     'procurement', 'done', 0.95, 'Procurement completed'),
    (r'vessel\s+has\s+not\s+yet\s+received',
     'procurement', 'pending', 0.8, 'Items not yet received'),
    (r'not\s+(?:yet\s+)?(?:received|supplied|delivered)',
     'procurement', 'pending', 0.7, 'Items not yet received'),
    (r'not\s+supplied',
     'procurement', 'pending', 0.75, 'Not yet supplied'),
    (r'(?:ordered|purchase\s+order|PO\s+raised)',
     'procurement', 'in_progress', 0.8, 'Order placed'),
    (r'we\s+will\s+supply',
     'procurement', 'in_progress', 0.6, 'Supply planned'),

    # INSTALL
    (r'installation\s+(?:of\s+.{3,50}\s+)?(?:completed|done|finished)',
     'install', 'done', 0.95, 'Installation completed'),
    (r'(?:following|below\s+mentioned)\s+jobs?\s+has\s+been\s+completed',
     'install', 'done', 0.9, 'Jobs completed'),
    (r'(?:cable\s+laying|LAN\s+cable|termination).*(?:completed|done)',
     'install', 'done', 0.85, 'Cable/wiring work completed'),
    (r'panel\s+installation\s+completed',
     'install', 'done', 0.95, 'Panel installed'),
    (r'(?:connected|installed|mounted)\s+(?:to|on|at|in)\s+(?:port\s+\d|rack|location|panel|switch)',
     'install', 'done', 0.85, 'Hardware connected'),
    (r'connections?\s+were\s+made',
     'install', 'done', 0.85, 'Connections completed'),
    (r'LAN\s+cable[s]?\s+.*terminated',
     'install', 'done', 0.85, 'Cables terminated'),
    (r'we\s+will\s+(?:proceed|carry\s+out|start)\s+(?:with\s+)?(?:the\s+)?(?:install|cable|LAN)',
     'install', 'in_progress', 0.7, 'Installation starting'),
    (r'(?:started|commenced)\s+.*(?:install|connection|cable|wiring)',
     'install', 'in_progress', 0.75, 'Installation in progress'),
    (r'we\s+have\s+(?:started|identified|proposed)',
     'install', 'in_progress', 0.6, 'Installation planning'),
    (r'(?:please\s+)?(?:proceed|install|connect|mount)\s+(?:the|with)',
     'install', 'in_progress', 0.4, 'Installation instructions given'),
    (r'(?:could\s+you\s+please|kindly)\s+(?:install|connect|mount|lay)',
     'install', 'in_progress', 0.35, 'Installation requested'),

    # COMMISSION
    (r'(?:commissioned|commissioning\s+completed|system\s+(?:online|operational))',
     'commission', 'done', 0.95, 'Commissioning completed'),
    (r'(?:remote\s+)?access\s+(?:working|established|successful)',
     'commission', 'done', 0.9, 'Remote access working'),
    (r'data\s+(?:flowing|receiving|coming)',
     'commission', 'done', 0.9, 'Data flow established'),
    (r'telemetry\s+(?:active|working|online)',
     'commission', 'done', 0.95, 'Telemetry active'),
    (r'successfully\s+(?:configured|connected|accessed)',
     'commission', 'done', 0.85, 'Configuration successful'),
    (r'unable\s+to\s+access',
     'commission', 'in_progress', 0.75, 'Access issues'),
    (r'(?:cannot|can\'t|could\s+not)\s+(?:access|connect|reach)',
     'commission', 'in_progress', 0.75, 'Connection issues'),
    (r'(?:network|connectivity|access)\s+issue',
     'commission', 'in_progress', 0.7, 'Network issue'),
    (r'not\s+(?:been\s+)?powered\s+(?:ON|on)',
     'commission', 'in_progress', 0.65, 'Power issue'),
    (r'(?:log\s*in|credentials|username|password)',
     'commission', 'in_progress', 0.6, 'Login/access being configured'),
    (r'(?:try|trying)\s+(?:to\s+)?access',
     'commission', 'in_progress', 0.5, 'Attempting access'),
    (r'share\s+(?:a\s+)?screenshot',
     'commission', 'in_progress', 0.4, 'Diagnostic info requested'),

    # FAT
    (r'FAT\s+(?:completed|done|passed|signed)',
     'fat', 'done', 0.95, 'FAT completed'),
    (r'(?:acceptance\s+test|FAT)\s+(?:in\s+progress|scheduled|ongoing)',
     'fat', 'in_progress', 0.8, 'FAT in progress'),
     # Catch "X installed for all DG" / "X installed" patterns
    (r'(?:meters?|coils?|cables?|panel|equipment)\s+installed',
    'install', 'done', 0.9, 'Equipment installed'),
    (r'(?:installed|connected|laid)\s+for\s+all',
    'install', 'done', 0.9, 'Installed for all units'),
    (r'(?:LAN\s+cable|cable)\s+(?:already\s+)?laid',
    'install', 'done', 0.85, 'Cable laid'),
    (r'already\s+installed\s+inside',
    'install', 'done', 0.85, 'Already installed'),
]

BLOCKER_PATTERNS = [
    (r'no\s+space\s+(?:on|in)\s+(?:the\s+)?rail', 'No space on rail for installation'),
    (r'(?:not\s+(?:yet\s+)?received|vessel\s+has\s+not.*received)\s+.*(?:switch|equipment|parts)',
     'Equipment/parts not yet received'),
    (r'network\s+issue', 'Network connectivity issue'),
    (r'unable\s+to\s+access', 'Remote access issue'),
    (r'not\s+(?:been\s+)?powered\s+(?:on|ON)', 'Equipment not powered on'),
    (r'(?:require|need)\s+(?:clarification|additional)', 'Clarification needed'),
]

EMAIL_TYPE_PATTERNS = [
    (r'(?:could\s+you|kindly|please)\s+(?:let\s+us\s+know|update|share|confirm)', 'query'),
    (r'(?:noted\s+with\s+thanks|well\s+noted|acknowledged)', 'acknowledgment'),
    (r'(?:following|below)\s+jobs?\s+(?:has|have)\s+been\s+completed', 'status_update'),
    (r'(?:installation|cable\s+laying|termination).*completed', 'status_update'),
    (r'(?:we\s+will\s+(?:proceed|carry|start))', 'commitment'),
    (r'(?:unable|cannot|issue|problem|not\s+working)', 'issue_report'),
    (r'(?:please\s+(?:proceed|install|connect|do))', 'instruction'),
]


# =============================================================================
# THREAD SPLITTER & CLEANER
# =============================================================================

class EmailCleaner:
    def __init__(self):
        self.noise_compiled = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in NOISE_PATTERNS]

    def split_thread(self, body):
        parts = re.split(r'(?=^From:\s*.+\r?\n\s*Sent:\s*)', body, flags=re.MULTILINE)
        parts = [p.strip() for p in parts if p.strip() and len(p.strip()) > 30]
        if not parts:
            parts = [body]
        parts.reverse()
        return parts

    def extract_sender(self, text):
        result = {'name': None, 'email': None, 'role': None}
        m = re.search(r'From:\s*(?:"?([^"<\r\n]+)"?\s*)?<?([^>\r\n\s]+@[^>\r\n\s]+)>?', text[:1500])
        if m:
            result['name'] = (m.group(1) or '').strip().strip('"') or None
            result['email'] = (m.group(2) or '').strip() or None
        if result['email']:
            if 'bsmfleet.com' in result['email'] or 'bs-shipmanagement' in result['email']:
                result['role'] = 'ship'
            elif 'mariapps.com' in result['email'] or 'memphis-marine.com' in result['email']:
                result['role'] = 'office'
        if re.search(r'(?:C/E|Chief\s+Engineer)', text[:1000]):
            if result['role'] != 'office':
                result['role'] = 'ship'
        if not result['name']:
            ce = re.search(r'(?:CE|C/E)[\s.]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', text[:1000])
            if ce:
                result['name'] = f"CE {ce.group(1)}"
                result['role'] = 'ship'
        return result

    def extract_date(self, text):
        for pat in [r'Sent:\s*(?:\w+,\s*)?(\w+\s+\d{1,2},?\s+\d{4}(?:\s+\d{1,2}:\d{2}(?:\s*[AP]M)?)?)',
                    r'Sent:\s*(\d{1,2}\s+\w+\s+\d{4}(?:\s+\d{1,2}:\d{2})?)',
                    r'Sent:\s*(?:\w+,\s*)?(\d{1,2}\s+\w+\s+\d{4}\s+\d{2}:\d{2})']:
            m = re.search(pat, text[:600])
            if m:
                return self._parse_date(m.group(1).strip().rstrip(','))
        # ← ADD HERE:
        m = re.search(r'On\s+\w+,?\s+(\d{1,2}\s+\w+\s+\d{4})', text[:1000])
        if m:
            return self._parse_date(m.group(1).strip())    
        return None

    def _parse_date(self, s):
        for fmt in ['%B %d, %Y %I:%M %p', '%B %d, %Y', '%B %d %Y %I:%M %p', '%B %d %Y',
                     '%d %B %Y %H:%M', '%d %B %Y', '%b %d, %Y %I:%M %p', '%b %d, %Y',
                     '%d %b %Y %H:%M', '%d %b %Y']:
            try:
                return datetime.strptime(re.sub(r'\s+', ' ', s).strip(), fmt)
            except ValueError:
                continue
        return None

    def clean_body(self, text):
        text = re.sub(r'^(?:From|To|Cc|Subject|Sent):.*$', '', text, flags=re.MULTILINE)
        for p in self.noise_compiled:
            text = p.sub('', text)
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

    def extract_products(self, text):
        found = []
        for product, patterns in PRODUCT_PATTERNS.items():
            for p in patterns:
                if re.search(p, text, re.IGNORECASE):
                    found.append(product)
                    break
        return found

    def extract_remarks(self, clean_text):
        remarks = []
        for line in clean_text.split('\n'):
            line = line.strip()
            if not line or len(line) < 15 or len(line) > 250:
                continue
            if re.match(r'^(Dear|Good\s+day|Hi|Hello|Noted\s+with\s+thanks)', line, re.IGNORECASE):
                continue
            if re.match(r'^[\d\.\)]+\s*$', line):
                continue
            remarks.append(line)
            if len(remarks) >= 3:
                break
        return remarks

    def identify_vessel(self, subject, from_addr, body_start):
        search = f"{subject} {from_addr} {body_start}".lower()
        for vessel in KNOWN_VESSELS:
            if vessel.lower() in search:
                return vessel.title()
        m = re.search(r'PROJECT[- ](.+?)(?:\s*$|\s*-)', subject, re.IGNORECASE)
        if m:
            return m.group(1).strip().title()
        m = re.search(r'mv\s+"?([^"]+)"?', body_start, re.IGNORECASE)
        if m:
            return m.group(1).strip().title()
        return None


# =============================================================================
# REGEX ENGINE
# =============================================================================

class RegexEngine:
    def analyze(self, clean_text, products):
        result = {'stage_updates': defaultdict(dict), 'blockers': [], 'email_type': 'other'}
        for pat, etype in EMAIL_TYPE_PATTERNS:
            if re.search(pat, clean_text, re.IGNORECASE):
                result['email_type'] = etype
                break
        for pat, stage, status, confidence, desc in REGEX_RULES:
            if re.search(pat, clean_text, re.IGNORECASE):
                targets = products if products else ['General']
                for product in targets:
                    if stage not in result['stage_updates'][product]:
                        result['stage_updates'][product][stage] = {
                            'status': status, 'confidence': confidence, 'description': desc}
                    else:
                        ex = result['stage_updates'][product][stage]
                        if (STATUS_PRIORITY.get(status, 0) > STATUS_PRIORITY.get(ex['status'], 0) or
                                (status == ex['status'] and confidence > ex['confidence'])):
                            result['stage_updates'][product][stage] = {
                                'status': status, 'confidence': confidence, 'description': desc}
        for pat, desc in BLOCKER_PATTERNS:
            if re.search(pat, clean_text, re.IGNORECASE):
                result['blockers'].append({'products': products or ['General'], 'blocker': desc})
        return result


# =============================================================================
# LLM ENGINE
# =============================================================================

class LLMEngine:
    def __init__(self, gpu_url="http://localhost:5005"):
        self.gpu_url = gpu_url
        self.available = self._check_availability()

    def _check_availability(self):
        try:
            return requests.get(f"{self.gpu_url}/gpu/health", timeout=5).status_code == 200
        except Exception:
            return False

    def analyze(self, clean_text, products, sender_role, current_state):
        if not self.available:
            return None

        products_str = ', '.join(products) if products else 'None detected'
        state_str = json.dumps(current_state, indent=2) if current_state else 'No previous state'

        prompt = f"""You are analyzing a single email from a marine vessel telemetry project.

PRODUCTS being tracked: {', '.join(PRODUCTS)}
STAGES in order: Procurement → Install → Commission → FAT
STATUS options: pending, in_progress, done

Products mentioned in this email: {products_str}
Sender type: {sender_role or 'unknown'}

Current tracker state:
{state_str}

EMAIL CONTENT:
{clean_text}

TASK: Analyze this email and output ONLY a JSON object. No explanation, no markdown, no extra text.

Rules:
1. Only update products explicitly mentioned or clearly referenced
2. Stages only move forward: pending → in_progress → done
3. If just a question/follow-up with no status change, set "updates" to empty
4. "confidence" must be "high", "medium", or "low"
5. "email_type": status_update, query, acknowledgment, commitment, issue_report, instruction, other
6. "stage" must be: procurement, install, commission, or fat
7. "status" must be: pending, in_progress, or done
8. "blockers" must be a list of strings

Output ONLY this JSON:
{{"email_type": "...", "updates": [{{"product": "...", "stage": "...", "status": "...", "confidence": "...", "reason": "..."}}], "blockers": ["..."]}}"""

        try:
            resp = requests.post(
                f"{self.gpu_url}/gpu/llm/generate",
                json={"messages": [{"role": "user", "content": prompt}], "response_type": "email_analysis"},
                timeout=150
            )
            raw = resp.json().get('response', '')
            json_str = self._extract_json(raw)
            if not json_str:
                logger.warning(f"LLM returned no valid JSON: {raw[:200]}")
                return None

            data = json.loads(json_str)
            result = {'stage_updates': defaultdict(dict), 'blockers': [], 'email_type': data.get('email_type', 'other')}
            conf_map = {'high': 0.9, 'medium': 0.6, 'low': 0.3}

            for u in data.get('updates', []):
                prod = u.get('product', '')
                stage = u.get('stage', '').lower()
                status = u.get('status', '').lower()
                conf = u.get('confidence', 'low').lower()
                reason = u.get('reason', '')
                if prod and prod in PRODUCTS and stage in STAGE_ORDER and status in STATUS_PRIORITY:
                    result['stage_updates'][prod][stage] = {
                        'status': status, 'confidence': conf_map.get(conf, 0.3), 'description': reason}

            for b in data.get('blockers', []):
                if b:
                    result['blockers'].append({
                        'products': products,
                        'blocker': b if isinstance(b, str) else b.get('blocker', str(b))
                    })
            return result

        except requests.exceptions.ConnectionError:
            logger.warning("LLM service not available")
            self.available = False
            return None
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}")
            return None

    def _extract_json(self, text):
        text = text.strip()
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass

        fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if fence:
            try:
                json.loads(fence.group(1))
                return fence.group(1)
            except json.JSONDecodeError:
                pass

        brace_count = 0
        start = None
        for i, c in enumerate(text):
            if c == '{':
                if start is None:
                    start = i
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0 and start is not None:
                    candidate = text[start:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        start = None
        return None


# =============================================================================
# DECISION ENGINE
# =============================================================================

class DecisionEngine:
    AUTO_UPDATE_THRESHOLD = 0.7
    FLAG_THRESHOLD = 0.5

    def decide(self, regex_result, llm_result, products):
        decisions = {'stage_updates': {}, 'blockers': [], 'email_type': 'other', 'flags': [], 'audit': []}

        regex_type = regex_result.get('email_type', 'other')
        llm_type = llm_result.get('email_type', 'other') if llm_result else None
        decisions['email_type'] = llm_type if llm_type and llm_type != 'other' else regex_type

        all_products = set(regex_result.get('stage_updates', {}).keys())
        if llm_result:
            all_products |= set(llm_result.get('stage_updates', {}).keys())

        for product in all_products:
            r_stages = regex_result.get('stage_updates', {}).get(product, {})
            l_stages = llm_result.get('stage_updates', {}).get(product, {}) if llm_result else {}

            for stage in set(list(r_stages.keys()) + list(l_stages.keys())):
                if stage not in STAGE_ORDER:
                    continue
                d = self._decide_single(product, stage, r_stages.get(stage), l_stages.get(stage))
                decisions['audit'].append(d['audit'])
                if d['action'] == 'update':
                    if product not in decisions['stage_updates']:
                        decisions['stage_updates'][product] = {}
                    decisions['stage_updates'][product][stage] = {
                        'status': d['status'], 'confidence': d['confidence'],
                        'source': d['source'], 'description': d['description']}
                elif d['action'] == 'flag':
                    decisions['flags'].append(d['flag_detail'])

        seen = set()
        for b in regex_result.get('blockers', []):
            key = b['blocker'] if isinstance(b['blocker'], str) else str(b['blocker'])
            if key not in seen:
                seen.add(key)
                decisions['blockers'].append(b)
        if llm_result:
            for b in llm_result.get('blockers', []):
                key = b['blocker'] if isinstance(b['blocker'], str) else str(b['blocker'])
                if key not in seen:
                    seen.add(key)
                    decisions['blockers'].append(b)
        return decisions

    def _decide_single(self, product, stage, r, l):
        rs = r['status'] if r else None
        rc = r['confidence'] if r else 0.0
        rd = r.get('description', '') if r else ''
        ls = l['status'] if l else None
        lc = l['confidence'] if l else 0.0
        ld = l.get('description', '') if l else ''
        audit = f"{product}/{stage}: regex={rs}({rc:.2f}) llm={ls}({lc:.2f})"

        if rs and ls and rs == ls:
            cc = min(max(rc, lc) + 0.1, 1.0)
            return {'action': 'update', 'status': rs, 'confidence': cc,
                    'source': 'both_agree', 'description': ld or rd,
                    'audit': f"{audit} → AGREE → update({cc:.2f})"}
        if rs and not ls:
            if rc >= self.AUTO_UPDATE_THRESHOLD:
                return {'action': 'update', 'status': rs, 'confidence': rc,
                        'source': 'regex_only', 'description': rd,
                        'audit': f"{audit} → REGEX_ONLY → update({rc:.2f})"}
            elif rc >= self.FLAG_THRESHOLD:
                return {'action': 'update', 'status': rs, 'confidence': rc,
                        'source': 'regex_low', 'description': rd,
                        'audit': f"{audit} → REGEX_LOW → update({rc:.2f})"}
            return {'action': 'skip', 'audit': f"{audit} → REGEX_TOO_LOW → skip"}
        if ls and not rs:
            if lc >= self.AUTO_UPDATE_THRESHOLD:
                return {'action': 'update', 'status': ls, 'confidence': lc,
                        'source': 'llm_only', 'description': ld,
                        'audit': f"{audit} → LLM_ONLY → update({lc:.2f})"}
            elif lc >= self.FLAG_THRESHOLD:
                return {'action': 'update', 'status': ls, 'confidence': lc,
                        'source': 'llm_low', 'description': ld,
                        'audit': f"{audit} → LLM_LOW → update({lc:.2f})"}
            return {'action': 'skip', 'audit': f"{audit} → LLM_TOO_LOW → skip"}
        if rs and ls and rs != ls:
            if rc >= 0.7 and lc >= 0.7:
                return {'action': 'flag', 'flag_detail': {
                    'product': product, 'stage': stage,
                    'regex_says': rs, 'regex_conf': rc, 'llm_says': ls, 'llm_conf': lc},
                    'audit': f"{audit} → DISAGREE_BOTH_HIGH → flag"}
            if rc >= lc:
                return {'action': 'update', 'status': rs, 'confidence': rc,
                        'source': 'regex_wins', 'description': rd,
                        'audit': f"{audit} → DISAGREE → regex_wins({rc:.2f})"}
            return {'action': 'update', 'status': ls, 'confidence': lc,
                    'source': 'llm_wins', 'description': ld,
                    'audit': f"{audit} → DISAGREE → llm_wins({lc:.2f})"}
        return {'action': 'skip', 'audit': f"{audit} → NOTHING → skip"}


# =============================================================================
# VESSEL TRACKER v3
# =============================================================================

class VesselTrackerV3:
    def __init__(self):
        self.state = defaultdict(lambda: defaultdict(lambda: OrderedDict([
            ('procurement', {'status': 'pending', 'date': None, 'by': None, 'description': '', 'confidence': 0, 'source': ''}),
            ('install', {'status': 'pending', 'date': None, 'by': None, 'description': '', 'confidence': 0, 'source': ''}),
            ('commission', {'status': 'pending', 'date': None, 'by': None, 'description': '', 'confidence': 0, 'source': ''}),
            ('fat', {'status': 'pending', 'date': None, 'by': None, 'description': '', 'confidence': 0, 'source': ''}),
        ])))
        self.history = []
        self.blockers = []
        self.flags = []
        self.audit_log = []

    def update(self, vessel, decisions, date, sender, sender_role):
        for product, stages in decisions.get('stage_updates', {}).items():
            for stage, info in stages.items():
                if stage not in STAGE_ORDER:
                    continue
                current = self.state[vessel][product][stage]
                new_status = info['status']
                if STATUS_PRIORITY.get(new_status, 0) > STATUS_PRIORITY.get(current['status'], 0):
                    old = current['status']
                    current['status'] = new_status
                    current['date'] = date
                    current['by'] = sender
                    current['description'] = info.get('description', '')
                    current['confidence'] = info.get('confidence', 0)
                    current['source'] = info.get('source', '')
                    self.history.append({
                        'date': date, 'vessel': vessel, 'product': product, 'stage': stage,
                        'old_status': old, 'new_status': new_status,
                        'confidence': info.get('confidence', 0),
                        'decision_source': info.get('source', ''),
                        'description': info.get('description', ''),
                        'sender': sender, 'sender_role': sender_role})
        for b in decisions.get('blockers', []):
            self.blockers.append({'date': date, 'vessel': vessel,
                                  'products': b.get('products', []),
                                  'blocker': b['blocker'], 'sender': sender})
        for f in decisions.get('flags', []):
            f['date'] = date
            f['vessel'] = vessel
            f['sender'] = sender
            self.flags.append(f)
        for a in decisions.get('audit', []):
            self.audit_log.append(f"[{date}] {sender}: {a}")

    def get_current_state_for_vessel(self, vessel):
        if vessel not in self.state:
            return {}
        return {prod: {stg: info['status'] for stg, info in stages.items()}
                for prod, stages in self.state[vessel].items()}

    def export_live(self):
        """Write live_tracker.csv and live_audit.txt (overwritten each call)."""
        rows = []
        for v, products in self.state.items():
            for prod, stages in products.items():
                row = {'vessel': v, 'product': prod}
                for stg, info in stages.items():
                    row[f'{stg}_status'] = info['status']
                    row[f'{stg}_date'] = info['date'] or ''
                    row[f'{stg}_confidence'] = f"{info['confidence']:.2f}" if info['confidence'] else ''
                    row[f'{stg}_source'] = info['source'] or ''
                rows.append(row)
        if rows:
            with open("live_tracker.csv", 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
        with open("live_audit.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.audit_log))

    def print_summary(self):
        print("\n" + "=" * 95)
        print("  VESSEL PROJECT TRACKER v3 - DUAL ENGINE")
        print("=" * 95)
        for vessel, products in self.state.items():
            print(f"\n{'─' * 95}")
            print(f"  VESSEL: {vessel}")
            print(f"{'─' * 95}")
            for product in sorted(products.keys(), key=lambda x: (x == 'General', x)):
                stages = products[product]
                print(f"\n  📦 {product}")
                for stage, info in stages.items():
                    s = info['status']
                    d = info['date'] or 'N/A'
                    c = info['confidence']
                    src = info['source'] or ''
                    desc = info['description'] or ''
                    icon = {'pending': '⬚', 'in_progress': '◐', 'done': '✅'}.get(s, '?')
                    bar = '█' * int(c * 5) + '░' * (5 - int(c * 5))
                    line = f"    {icon} {stage.upper():15s} → {s.upper().replace('_',' '):15s} ({d})"
                    if c > 0:
                        line += f"  [{bar} {c:.0%}]"
                    if src:
                        line += f"  [{src}]"
                    print(line)
                    if desc:
                        print(f"      {desc}")
        if self.flags:
            print(f"\n{'─' * 95}")
            print("  🚩 FLAGS FOR HUMAN REVIEW:")
            print(f"{'─' * 95}")
            for f in self.flags:
                print(f"    [{f.get('date','?')}] {f['product']}/{f['stage']}: "
                      f"regex:{f['regex_says']}({f['regex_conf']:.0%}) vs llm:{f['llm_says']}({f['llm_conf']:.0%})")
        if self.blockers:
            print(f"\n{'─' * 95}")
            print("  ⚠️  BLOCKERS:")
            print(f"{'─' * 95}")
            seen = set()
            for b in self.blockers:
                key = f"{b['vessel']}|{b['blocker']}"
                if key not in seen:
                    seen.add(key)
                    prods = ', '.join(b['products']) if b['products'] else 'General'
                    print(f"    [{b['date']}] {prods}: {b['blocker']} ({b['sender']})")
        if self.history:
            print(f"\n{'─' * 95}")
            print("  📋 CHANGE HISTORY:")
            print(f"{'─' * 95}")
            for h in self.history:
                t = '🚢' if h['sender_role'] == 'ship' else '🏢'
                print(f"    [{h['date']}] {t} {h['product']} | {h['stage'].upper()} | "
                      f"{h['old_status']}→{h['new_status']} [{h['confidence']:.0%} {h['decision_source']}] "
                      f"{h['description']}")
        print("\n" + "=" * 95)

    def export_csv(self, prefix=""):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = {}
        fname = f"{prefix}tracker_v3_{ts}.csv"
        rows = []
        for v, products in self.state.items():
            for prod, stages in products.items():
                row = {'vessel': v, 'product': prod}
                for stg, info in stages.items():
                    row[f'{stg}_status'] = info['status']
                    row[f'{stg}_date'] = info['date'] or ''
                    row[f'{stg}_by'] = info['by'] or ''
                    row[f'{stg}_confidence'] = f"{info['confidence']:.2f}" if info['confidence'] else ''
                    row[f'{stg}_source'] = info['source'] or ''
                rows.append(row)
        if rows:
            with open(fname, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
            files['tracker'] = fname
        if self.history:
            fname = f"{prefix}history_v3_{ts}.csv"
            with open(fname, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=self.history[0].keys())
                w.writeheader()
                w.writerows(self.history)
            files['history'] = fname
        if self.flags:
            fname = f"{prefix}flags_v3_{ts}.csv"
            with open(fname, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=self.flags[0].keys())
                w.writeheader()
                w.writerows(self.flags)
            files['flags'] = fname
        if self.audit_log:
            fname = f"{prefix}audit_v3_{ts}.txt"
            with open(fname, 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.audit_log))
            files['audit'] = fname
        return files
    def export_html(self, prefix=""):
        """Export a visual HTML report per vessel with status cards, blockers, and timeline."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        files = {}
        for vessel, products in self.state.items():
            safe_name = re.sub(r'[^\w\s-]', '', vessel).strip().replace(' ', '_')
            fname = f"{prefix}report_{safe_name}_{ts}.html"
            sorted_history = sorted(self.history,
                key=lambda h: h['date'] if h['date'] != 'Unknown' else '9999-99-99')
            vessel_history = [h for h in sorted_history if h['vessel'] == vessel]
            vessel_blockers = []
            seen_b = set()
            for b in self.blockers:
                if b['vessel'] == vessel:
                    key = b['blocker']
                    if key not in seen_b:
                        seen_b.add(key)
                        vessel_blockers.append(b)
            vessel_flags = [f for f in self.flags if f.get('vessel') == vessel]
            cards_html = ""
            sorted_products = sorted(products.keys(), key=lambda x: (x == 'General', x))
            for product in sorted_products:
                stages = products[product]
                rows = ""
                for stage, info in stages.items():
                    s = info['status']
                    badge_class = {'done': 'done', 'in_progress': 'progress', 'pending': 'pending'}.get(s, 'pending')
                    label = s.upper().replace('_', ' ')
                    date_str = info['date'] or ''
                    by_str = info['by'] or ''
                    conf = info['confidence']
                    src = info['source'] or ''
                    detail = ''
                    if date_str:
                        detail += f'{date_str}'
                    if by_str:
                        detail += f' ({by_str})'
                    if conf > 0:
                        detail += f' [{conf:.0%} {src}]'
                    rows += f"""<tr>
                        <td>{stage.upper()}</td>
                        <td><span class="badge {badge_class}">{label}</span></td>
                        <td class="detail">{detail}</td>
                    </tr>\n"""
                cards_html += f"""<div class="card">
                    <h3>📦 {product}</h3>
                    <table>
                        <tr><th>Stage</th><th>Status</th><th>Details</th></tr>
                        {rows}
                    </table>
                </div>\n"""
            blockers_html = ""
            if vessel_blockers:
                items = ""
                for b in vessel_blockers:
                    prods = ', '.join(b['products']) if b['products'] else 'General'
                    items += f'<div class="blocker-item"><strong>{prods}:</strong> {b["blocker"]} <span class="timestamp">({b["date"]}, {b["sender"]})</span></div>\n'
                blockers_html = f"""<div class="blocker-section">
                    <h2>⚠️ Active Blockers</h2>
                    {items}
                </div>"""
            flags_html = ""
            if vessel_flags:
                items = ""
                for fl in vessel_flags:
                    items += f'<div class="flag-item">🚩 <strong>{fl["product"]}/{fl["stage"]}</strong>: Regex says {fl["regex_says"]} ({fl["regex_conf"]:.0%}) vs LLM says {fl["llm_says"]} ({fl["llm_conf"]:.0%}) <span class="timestamp">({fl.get("date","?")})</span></div>\n'
                flags_html = f"""<div class="flag-section">
                    <h2>🚩 Flags for Human Review</h2>
                    {items}
                </div>"""
            timeline_rows = ""
            for h in vessel_history:
                role_icon = '🚢' if h['sender_role'] == 'ship' else '🏢' if h['sender_role'] == 'office' else '❓'
                old_badge = {'done': 'done', 'in_progress': 'progress', 'pending': 'pending'}.get(h['old_status'], 'pending')
                new_badge = {'done': 'done', 'in_progress': 'progress', 'pending': 'pending'}.get(h['new_status'], 'pending')
                old_label = h['old_status'].upper().replace('_', ' ')
                new_label = h['new_status'].upper().replace('_', ' ')
                timeline_rows += f"""<tr>
                    <td class="timestamp">{h['date']}</td>
                    <td>{role_icon} {h['sender']}</td>
                    <td><strong>{h['product']}</strong> / {h['stage'].upper()}</td>
                    <td><span class="badge {old_badge}">{old_label}</span> → <span class="badge {new_badge}">{new_label}</span></td>
                    <td>{h['description'] or ''} <span class="confidence">[{h['confidence']:.0%} {h['decision_source']}]</span></td>
                </tr>\n"""
            total_stages = done_count = progress_count = pending_count = 0
            for prod, stages in products.items():
                for stg, info in stages.items():
                    total_stages += 1
                    if info['status'] == 'done': done_count += 1
                    elif info['status'] == 'in_progress': progress_count += 1
                    else: pending_count += 1
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Vessel Report: {vessel}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; background-color: #f4f7f9; margin: 0; padding: 20px; }}
        .container {{ max-width: 1100px; margin: auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
        header {{ border-bottom: 2px solid #0056b3; margin-bottom: 20px; padding-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }}
        h1 {{ color: #0056b3; margin: 0; font-size: 24px; }}
        h2 {{ color: #444; margin-top: 30px; border-bottom: 1px solid #eee; padding-bottom: 8px; }}
        .vessel-tag {{ background: #0056b3; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; }}
        .stats {{ display: flex; gap: 15px; margin-bottom: 20px; }}
        .stat-box {{ padding: 10px 20px; border-radius: 6px; text-align: center; font-weight: bold; }}
        .stat-done {{ background: #d4edda; color: #155724; }}
        .stat-progress {{ background: #fff3cd; color: #856404; }}
        .stat-pending {{ background: #f8d7da; color: #721c24; }}
        .status-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{ border: 1px solid #e1e4e8; border-radius: 6px; padding: 15px; background: #fff; }}
        .card h3 {{ margin-top: 0; border-bottom: 1px solid #eee; padding-bottom: 8px; color: #444; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; text-transform: uppercase; }}
        .done {{ background-color: #d4edda; color: #155724; }}
        .progress {{ background-color: #fff3cd; color: #856404; }}
        .pending {{ background-color: #f8d7da; color: #721c24; }}
        .blocker-section {{ background-color: #fff5f5; border: 1px solid #feb2b2; border-radius: 6px; padding: 20px; margin-bottom: 20px; }}
        .blocker-section h2 {{ color: #c53030; margin-top: 0; font-size: 18px; border: none; }}
        .blocker-item {{ margin-bottom: 8px; padding-left: 15px; border-left: 3px solid #c53030; font-size: 14px; }}
        .flag-section {{ background-color: #fffbeb; border: 1px solid #fbbf24; border-radius: 6px; padding: 20px; margin-bottom: 20px; }}
        .flag-section h2 {{ color: #b45309; margin-top: 0; font-size: 18px; border: none; }}
        .flag-item {{ margin-bottom: 8px; padding-left: 15px; border-left: 3px solid #fbbf24; font-size: 14px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
        th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #eee; }}
        th {{ background-color: #f8f9fa; font-weight: 600; }}
        .timestamp {{ color: #666; font-size: 12px; }}
        .detail {{ color: #666; font-size: 12px; }}
        .confidence {{ color: #999; font-size: 11px; }}
        footer {{ margin-top: 30px; font-size: 12px; color: #999; text-align: center; border-top: 1px solid #eee; padding-top: 15px; }}
    </style>
</head>
<body>
<div class="container">
    <header>
        <h1>Vessel Project Tracker v3</h1>
        <span class="vessel-tag">🚢 {vessel}</span>
    </header>
    <div class="stats">
        <div class="stat-box stat-done">✅ Done: {done_count}</div>
        <div class="stat-box stat-progress">◐ In Progress: {progress_count}</div>
        <div class="stat-box stat-pending">⬚ Pending: {pending_count}</div>
    </div>
    {blockers_html}
    {flags_html}
    <h2>System Components Status</h2>
    <div class="status-grid">
        {cards_html}
    </div>
    <h2>📋 Project Timeline (Ascending)</h2>
    <table>
        <thead>
            <tr><th>Date</th><th>Source</th><th>Product / Stage</th><th>Transition</th><th>Details</th></tr>
        </thead>
        <tbody>
            {timeline_rows}
        </tbody>
    </table>
    <footer>
        Report Generated: {datetime.now().strftime('%B %d, %Y %H:%M')} | Dual Engine (Regex + LLM) | {len(vessel_history)} state changes tracked
    </footer>
</div>
</body>
</html>"""
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(html)
            files[f'html_{safe_name}'] = fname
        return files


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class EmailPipeline:
    def __init__(self, gpu_url="http://localhost:5005", regex_only=False):
        self.cleaner = EmailCleaner()
        self.regex_engine = RegexEngine()
        self.llm_engine = None if regex_only else LLMEngine(gpu_url)
        self.decision_engine = DecisionEngine()
        self.tracker = VesselTrackerV3()
        self.regex_only = regex_only
        mode = "REGEX ONLY" if regex_only else (
            "REGEX + LLM" if self.llm_engine and self.llm_engine.available else "REGEX (LLM unavailable)")
        logger.info(f"Pipeline initialized: {mode}")

    def process_eml(self, file_path):
        file_path = Path(file_path)
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)

        subject = str(msg['subject'] or '')
        from_addr = str(msg['from'] or '')
        top_date = str(msg['date'] or '')
        body = self._extract_body(msg)
        vessel = self.cleaner.identify_vessel(subject, from_addr, body[:500])
        thread_parts = self.cleaner.split_thread(body)
        logger.info(f"Split into {len(thread_parts)} emails for vessel: {vessel}")

        email_results = []
        for i, raw_text in enumerate(thread_parts):
            sender = self.cleaner.extract_sender(raw_text)
            date_obj = self.cleaner.extract_date(raw_text)
            if i == len(thread_parts) - 1 and not date_obj:
                date_obj = self.cleaner.extract_date(f"Sent: {top_date}")
            date_str = date_obj.strftime('%Y-%m-%d') if date_obj else 'Unknown'
            clean_text = self.cleaner.clean_body(raw_text)
            products = self.cleaner.extract_products(raw_text)
            remarks = self.cleaner.extract_remarks(clean_text)

            if not clean_text or len(clean_text) < 20:
                continue

            regex_result = self.regex_engine.analyze(clean_text, products)

            llm_result = None
            if self.llm_engine and self.llm_engine.available:
                current_state = self.tracker.get_current_state_for_vessel(vessel) if vessel else {}
                llm_result = self.llm_engine.analyze(clean_text, products,
                                                     sender.get('role', 'unknown'), current_state)

            decisions = self.decision_engine.decide(regex_result, llm_result, products)

            if vessel:
                self.tracker.update(vessel, decisions, date_str,
                                    sender.get('name', 'Unknown'), sender.get('role', 'unknown'))

            email_results.append({
                'index': i, 'date': date_str, 'sender': sender, 'products': products,
                'email_type': decisions['email_type'],
                'decisions': {'updates': decisions['stage_updates'], 'flags': decisions['flags']},
                'blockers': decisions['blockers'], 'remarks': remarks})

            # === LIVE CONSOLE OUTPUT ===
            role_icon = '🚢' if sender.get('role') == 'ship' else '🏢' if sender.get('role') == 'office' else '❓'
            src = 'BOTH' if llm_result and regex_result else ('LLM' if llm_result else 'REGEX')
            print(f"  {role_icon} [{date_str}] {sender.get('name', '?')} | products:{products} | {src}", flush=True)
            for prod, stages in decisions.get('stage_updates', {}).items():
                for stg, info in stages.items():
                    if stg in STAGE_ORDER:
                        print(f"      → {prod}/{stg.upper()}: {info['status']} "
                              f"[{info.get('confidence', 0):.0%} {info.get('source', '')}]", flush=True)

            # === LIVE FILE EXPORT ===
            self.tracker.export_live()

        return {'file': file_path.name, 'vessel': vessel,
                'thread_count': len(thread_parts), 'emails_processed': len(email_results),
                'email_details': email_results}

    def process_folder(self, folder_path):
        folder = Path(folder_path)
        results = []
        eml_files = sorted(list(folder.glob('*.eml')) + list(folder.glob('*.EML')),
                           key=lambda f: f.stat().st_mtime)
        if not eml_files:
            print(f"No .eml files found in {folder_path}")
            return results
        print(f"Found {len(eml_files)} EML files")
        for eml_file in eml_files:
            try:
                result = self.process_eml(str(eml_file))
                results.append(result)
                print(f"  ✓ {eml_file.name} → {result.get('vessel', '?')} ({result['emails_processed']} emails)")
            except Exception as e:
                print(f"  ✗ {eml_file.name} → {e}")
                results.append({'file': eml_file.name, 'error': str(e)})
        return results

    def _extract_body(self, msg):
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                ct = part.get_content_type()
                disp = str(part.get('Content-Disposition', ''))
                if ct == 'text/plain' and 'attachment' not in disp:
                    try:
                        p = part.get_payload(decode=True)
                        if p:
                            body += p.decode(part.get_content_charset() or 'utf-8', errors='replace')
                    except Exception:
                        pass
                elif ct == 'text/html' and not body and 'attachment' not in disp:
                    try:
                        p = part.get_payload(decode=True)
                        if p:
                            body = self._html_to_text(p.decode(part.get_content_charset() or 'utf-8', errors='replace'))
                    except Exception:
                        pass
        else:
            try:
                p = msg.get_payload(decode=True)
                if p:
                    cs = msg.get_content_charset() or 'utf-8'
                    body = self._html_to_text(p.decode(cs, errors='replace')) if msg.get_content_type() == 'text/html' else p.decode(cs, errors='replace')
            except Exception:
                body = str(msg.get_payload())
        return body

    def _html_to_text(self, html):
        t = re.sub(r'<br\s*/?>', '\n', html, flags=re.IGNORECASE)
        t = re.sub(r'<p[^>]*>', '\n', t, flags=re.IGNORECASE)
        t = re.sub(r'<[^>]+>', '', t)
        t = re.sub(r'&nbsp;', ' ', t)
        t = re.sub(r'&amp;', '&', t)
        t = re.sub(r'&lt;|&gt;', '', t)
        t = re.sub(r'&#\d+;', '', t)
        return t.strip()

    def print_results(self, results):
        for result in results:
            if 'error' in result:
                continue
            print(f"\n{'═' * 95}")
            print(f"  FILE: {result['file']}")
            print(f"  VESSEL: {result.get('vessel', 'Unknown')}")
            print(f"  THREAD: {result['thread_count']} emails → {result['emails_processed']} processed")
            print(f"  MODE: {'REGEX ONLY' if self.regex_only else 'REGEX + LLM'}")
            print(f"{'═' * 95}")
            for em in result.get('email_details', []):
                sender = em['sender']
                role_icon = '🚢' if sender.get('role') == 'ship' else '🏢' if sender.get('role') == 'office' else '❓'
                print(f"\n  {role_icon} [{em['date']}] {sender.get('name', '?')} ({em['email_type']})")
                if em['products']:
                    print(f"     Products: {', '.join(em['products'])}")
                for prod, stages in em['decisions'].get('updates', {}).items():
                    for stg, info in stages.items():
                        icon = {'done': '✅', 'in_progress': '◐', 'pending': '⬚'}.get(info['status'], '?')
                        print(f"     {icon} {prod}/{stg.upper()}: {info['status']} "
                              f"[{info.get('confidence', 0):.0%} {info.get('source', '')}] "
                              f"{info.get('description', '')}")
                for f in em['decisions'].get('flags', []):
                    print(f"     🚩 FLAG: {f['product']}/{f['stage']} - "
                          f"regex:{f['regex_says']} vs llm:{f['llm_says']}")
                for b in em.get('blockers', []):
                    bt = b['blocker'] if isinstance(b['blocker'], str) else str(b['blocker'])
                    print(f"     ⚠️  {bt}")
                for r in em.get('remarks', [])[:2]:
                    print(f"     💬 {r[:100]}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = sys.argv[1]
    regex_only = '--regex-only' in sys.argv
    gpu_url = "http://localhost:5005"
    for arg in sys.argv:
        if arg.startswith('--gpu='):
            gpu_url = arg.split('=', 1)[1]

    pipeline = EmailPipeline(gpu_url=gpu_url, regex_only=regex_only)

    if os.path.isfile(target):
        results = [pipeline.process_eml(target)]
    elif os.path.isdir(target):
        results = pipeline.process_folder(target)
    else:
        print(f"Error: '{target}' not found")
        sys.exit(1)

    pipeline.print_results(results)
    pipeline.tracker.print_summary()

    files = pipeline.tracker.export_csv()
    for ftype, fname in files.items():
        print(f"✓ {ftype}: {fname}")
    html_files = pipeline.tracker.export_html()
    for ftype, fname in html_files.items():
        print(f"✓ {ftype}: {fname}")    

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jf = f"report_v3_{ts}.json"
    with open(jf, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✓ JSON report: {jf}")
    total = sum(r.get('emails_processed', 0) for r in results if 'error' not in r)
    print(f"\nDone! {total} emails processed across {len(results)} file(s)")


if __name__ == "__main__":
    main()