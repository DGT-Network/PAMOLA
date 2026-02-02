#!/usr/bin/env python3
"""
Synthetic Banking Dataset Generator - Multi-Table Version
PAMOLA.CORE - Epic 2 Testing Suite

Generates realistic linked banking data across 6 tables:
- CUSTOMERS: Customer PII and demographics
- ACCOUNTS: Bank accounts
- TRANSACTIONS: Account transactions
- LOANS: Loan applications and status
- CREDIT_CARDS: Credit card accounts
- FEEDBACK: Customer feedback and complaints

All data is 100% synthetic - no real PII included.
"""

import csv
import json
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import uuid

# Seed for reproducibility
SEED = 42
random.seed(SEED)

# === CONFIGURATION ===

NUM_CUSTOMERS = 1500
NUM_ACCOUNTS = 2100  # ~1.4 per customer
NUM_TRANSACTIONS = 10000
NUM_LOANS = 800
NUM_CREDIT_CARDS = 1200
NUM_FEEDBACK = 500

# Data quality issue rates
OUTLIER_RATE = 0.02
FORMAT_VIOLATION_RATE = 0.03
MISSING_VALUE_RATE = 0.02

# Fingerprint markers
FINGERPRINT_PREFIX = "PMLA"

# === REFERENCE DATA ===

CANADIAN_CITIES = [
    ("Toronto", "ON", "M"), ("Vancouver", "BC", "V"), ("Montreal", "QC", "H"),
    ("Calgary", "AB", "T"), ("Edmonton", "AB", "T"), ("Ottawa", "ON", "K"),
    ("Winnipeg", "MB", "R"), ("Quebec City", "QC", "G"), ("Hamilton", "ON", "L"),
    ("Kitchener", "ON", "N"), ("Victoria", "BC", "V"), ("Halifax", "NS", "B"),
    ("London", "ON", "N"), ("Windsor", "ON", "N"), ("Saskatoon", "SK", "S"),
    ("Regina", "SK", "S"), ("St. John's", "NL", "A"), ("Barrie", "ON", "L"),
    ("Kelowna", "BC", "V"), ("Sherbrooke", "QC", "J"), ("Guelph", "ON", "N"),
    ("Moncton", "NB", "E"), ("Charlottetown", "PE", "C"), ("Whitehorse", "YT", "Y")
]

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Elizabeth", "David", "Barbara", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
    "Jean", "Pierre", "Marie", "François", "Sophie", "Luc", "Isabelle", "Marc",
    "Wei", "Yan", "Ming", "Ling", "Raj", "Priya", "Ahmed", "Fatima",
    "Emma", "Olivia", "Liam", "Noah", "Ava", "Sophia", "Lucas", "Charlotte"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Wilson", "Anderson", "Taylor", "Thomas", "Moore", "Jackson",
    "Tremblay", "Roy", "Gagnon", "Bouchard", "Côté", "Gauthier", "Morin", "Lavoie",
    "Chen", "Wang", "Li", "Singh", "Patel", "Kim", "Lee", "Wong",
    "Thompson", "White", "Harris", "Martin", "Clark", "Lewis", "Walker", "Hall"
]

STREET_NAMES = [
    "Main", "Oak", "Maple", "Cedar", "Pine", "Elm", "Park", "Lake", "Hill", "River",
    "King", "Queen", "Church", "Victoria", "Albert", "George", "James", "John",
    "Wellington", "Dundas", "Yonge", "Bloor", "College", "Spadina", "Bay", "Front"
]

STREET_TYPES = ["Street", "Avenue", "Road", "Drive", "Boulevard", "Lane", "Way", "Crescent"]

EMPLOYERS = [
    "TD Bank", "RBC", "Scotiabank", "BMO", "CIBC", "Desjardins",
    "Shopify", "Telus", "Rogers", "Bell Canada", "Air Canada", "Loblaws",
    "Government of Canada", "City of Toronto", "University of Toronto",
    "Amazon Canada", "Google Canada", "Microsoft Canada", "IBM Canada",
    "Suncor", "Enbridge", "CN Rail", "CP Rail", "Bombardier", "CAE"
]

EMAIL_DOMAINS = [
    "gmail.com", "yahoo.ca", "hotmail.com", "outlook.com", "icloud.com",
    "mail.com", "protonmail.com", "fastmail.com"
]

# === HELPER FUNCTIONS ===

def generate_id(prefix: str, index: int, suffix: str = "SYN") -> str:
    """Generate synthetic ID with fingerprint."""
    return f"{FINGERPRINT_PREFIX}-{prefix}{index:06d}-{suffix}"


def generate_uuid() -> str:
    """Generate UUID."""
    return str(uuid.uuid4())


def generate_phone() -> str:
    """Generate Canadian phone number."""
    area_codes = ["416", "647", "437", "905", "289", "365",  # GTA
                  "604", "778", "236",  # Vancouver
                  "514", "438", "450",  # Montreal
                  "403", "587", "825",  # Calgary
                  "613", "343"]  # Ottawa
    return f"+1-{random.choice(area_codes)}-{random.randint(200,999)}-{random.randint(1000,9999)}"


def generate_email(first_name: str, last_name: str, index: int) -> str:
    """Generate email address."""
    patterns = [
        f"{first_name.lower()}.{last_name.lower()}",
        f"{first_name.lower()}{last_name.lower()}",
        f"{first_name[0].lower()}{last_name.lower()}",
        f"{first_name.lower()}.{last_name.lower()}{random.randint(1,99)}",
        f"{first_name.lower()}_{last_name.lower()}"
    ]
    return f"{random.choice(patterns)}@{random.choice(EMAIL_DOMAINS)}"


def generate_address() -> str:
    """Generate street address."""
    return f"{random.randint(1, 9999)} {random.choice(STREET_NAMES)} {random.choice(STREET_TYPES)}"


def generate_postal_code(fsa_prefix: str) -> str:
    """Generate Canadian postal code."""
    letters = "ABCEGHJKLMNPRSTVWXYZ"
    digits = "0123456789"
    fsa = f"{fsa_prefix}{random.choice(digits)}{random.choice(letters)}"
    ldu = f"{random.choice(digits)}{random.choice(letters)}{random.choice(digits)}"
    return f"{fsa} {ldu}"


def generate_date_in_range(start: datetime, end: datetime) -> datetime:
    """Generate random date in range."""
    delta = end - start
    random_days = random.randint(0, delta.days)
    return start + timedelta(days=random_days)


def weighted_choice(choices: List[Tuple[Any, float]]) -> Any:
    """Select item based on weights."""
    items, weights = zip(*choices)
    return random.choices(items, weights=weights, k=1)[0]


def introduce_format_violation(value: str, violation_type: str) -> str:
    """Introduce intentional format violations."""
    if violation_type == "case":
        return value.lower() if random.random() > 0.5 else value.upper()
    elif violation_type == "whitespace":
        return f"  {value}  " if random.random() > 0.5 else f"{value}\t"
    elif violation_type == "encoding":
        replacements = {"a": "а", "e": "е", "o": "о"}  # Cyrillic lookalikes
        for lat, cyr in replacements.items():
            if lat in value.lower() and random.random() > 0.7:
                value = value.replace(lat, cyr, 1)
                break
        return value
    elif violation_type == "truncation":
        return value[:len(value)//2] if len(value) > 4 else value
    return value


# === DATA GENERATORS ===

def generate_customers(num: int) -> List[Dict]:
    """Generate customer records with PII."""
    customers = []
    
    for i in range(num):
        city, province, fsa_prefix = random.choice(CANADIAN_CITIES)
        first_name = random.choice(FIRST_NAMES)
        last_name = random.choice(LAST_NAMES)
        birth_date = generate_date_in_range(
            datetime(1945, 1, 1),
            datetime(2005, 12, 31)
        )
        
        # Format violations
        has_format_issue = random.random() < FORMAT_VIOLATION_RATE
        violation_type = random.choice(["case", "whitespace", "encoding"]) if has_format_issue else None
        
        customer = {
            "customer_id": generate_id("C", i),
            "first_name": first_name if not has_format_issue else introduce_format_violation(first_name, violation_type),
            "last_name": last_name,
            "birth_date": birth_date.strftime("%Y-%m-%d"),
            "age": (datetime.now() - birth_date).days // 365,
            "gender": weighted_choice([("Male", 0.48), ("Female", 0.48), ("Other", 0.03), ("Prefer not to say", 0.01)]),
            "address": generate_address(),
            "city": city,
            "province": province,
            "postal_code": generate_postal_code(fsa_prefix),
            "country": "Canada",
            "phone": generate_phone(),
            "email": generate_email(first_name, last_name, i),
            "employer": random.choice(EMPLOYERS) if random.random() > 0.1 else None,
            "employment_status": weighted_choice([
                ("Employed", 0.65), ("Self-employed", 0.12), ("Retired", 0.10),
                ("Student", 0.08), ("Unemployed", 0.05)
            ]),
            "annual_income": round(random.lognormvariate(mu=10.8, sigma=0.5), 2) if random.random() > 0.05 else None,
            "segment": weighted_choice([
                ("RETAIL", 0.60), ("SMALL_BUSINESS", 0.20), ("PREMIUM", 0.15), ("PRIVATE", 0.05)
            ]),
            "kyc_level": weighted_choice([("BASIC", 0.30), ("STANDARD", 0.50), ("ENHANCED", 0.20)]),
            "risk_rating": weighted_choice([("LOW", 0.60), ("MEDIUM", 0.30), ("HIGH", 0.10)]),
            "customer_since": generate_date_in_range(datetime(2000, 1, 1), datetime(2023, 12, 31)).strftime("%Y-%m-%d"),
            "is_active": random.random() > 0.05,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "_has_format_issue": has_format_issue,
            "_violation_type": violation_type
        }
        customers.append(customer)
    
    return customers


def generate_accounts(customers: List[Dict], num: int) -> List[Dict]:
    """Generate bank account records."""
    accounts = []
    account_types = [
        ("CHECKING", 0.40), ("SAVINGS", 0.35), ("BUSINESS", 0.15), ("JOINT", 0.10)
    ]
    
    # Distribute accounts across customers
    customer_ids = [c["customer_id"] for c in customers]
    
    for i in range(num):
        customer_id = random.choice(customer_ids)
        open_date = generate_date_in_range(datetime(2005, 1, 1), datetime(2024, 6, 1))
        
        is_outlier = random.random() < OUTLIER_RATE
        balance = round(random.uniform(100000, 2000000), 2) if is_outlier else round(random.lognormvariate(mu=8.5, sigma=1.2), 2)
        
        account = {
            "account_id": generate_id("A", i),
            "customer_id": customer_id,
            "account_type": weighted_choice(account_types),
            "account_number": f"****{random.randint(1000, 9999)}",  # Masked
            "currency": weighted_choice([("CAD", 0.95), ("USD", 0.05)]),
            "balance": balance,
            "available_balance": round(balance * random.uniform(0.8, 1.0), 2),
            "hold_amount": round(random.uniform(0, 500), 2) if random.random() > 0.8 else 0.0,
            "interest_rate": round(random.uniform(0.01, 4.5), 2),
            "overdraft_limit": random.choice([0, 500, 1000, 2000, 5000]),
            "opened_date": open_date.strftime("%Y-%m-%d"),
            "last_activity_date": generate_date_in_range(open_date, datetime(2024, 11, 30)).strftime("%Y-%m-%d"),
            "status": weighted_choice([("ACTIVE", 0.90), ("DORMANT", 0.05), ("CLOSED", 0.03), ("FROZEN", 0.02)]),
            "branch_id": f"BR-{random.randint(1, 50):04d}",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "_is_outlier": is_outlier
        }
        accounts.append(account)
    
    return accounts


def generate_transactions(accounts: List[Dict], num: int) -> List[Dict]:
    """Generate transaction records."""
    transactions = []
    
    txn_types = [
        ("DEPOSIT", 0.18), ("WITHDRAWAL", 0.20), ("TRANSFER_OUT", 0.15),
        ("TRANSFER_IN", 0.15), ("BILL_PAYMENT", 0.12), ("POS_PURCHASE", 0.10),
        ("ATM_WITHDRAWAL", 0.05), ("INTERAC_ETRANSFER", 0.03), ("PAYROLL", 0.02)
    ]
    
    channels = [("ONLINE", 0.35), ("MOBILE", 0.30), ("BRANCH", 0.15), ("ATM", 0.15), ("PHONE", 0.05)]
    statuses = [("COMPLETED", 0.94), ("PENDING", 0.03), ("FAILED", 0.02), ("REVERSED", 0.01)]
    
    merchant_categories = [
        "GROCERY", "RESTAURANT", "GAS_STATION", "RETAIL", "UTILITIES",
        "HEALTHCARE", "ENTERTAINMENT", "TRAVEL", "INSURANCE", "TELECOM"
    ]
    
    account_ids = [a["account_id"] for a in accounts]
    base_date = datetime(2024, 6, 1)
    
    for i in range(num):
        account_id = random.choice(account_ids)
        txn_type = weighted_choice(txn_types)
        channel = weighted_choice(channels)
        
        is_outlier = random.random() < OUTLIER_RATE
        has_format_issue = random.random() < FORMAT_VIOLATION_RATE
        has_missing = random.random() < MISSING_VALUE_RATE
        
        # Amount based on type
        if is_outlier:
            amount = round(random.uniform(50000, 500000), 2)
        elif txn_type == "PAYROLL":
            amount = round(random.uniform(2000, 15000), 2)
        elif txn_type in ["POS_PURCHASE", "ATM_WITHDRAWAL"]:
            amount = round(random.lognormvariate(mu=3.5, sigma=1.0), 2)
        else:
            amount = round(random.lognormvariate(mu=5.0, sigma=1.5), 2)
        
        timestamp = base_date + timedelta(
            days=random.randint(0, 180),
            hours=random.choices(range(24), weights=[1,1,1,1,1,2,3,5,8,10,10,9,8,9,10,10,8,6,4,3,2,2,1,1], k=1)[0],
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        description = f"{txn_type} - {channel}"
        if has_format_issue:
            violation = random.choice(["case", "whitespace", "truncation"])
            description = introduce_format_violation(description, violation)
        
        txn = {
            "transaction_id": generate_uuid(),
            "account_id": account_id,
            "transaction_type": txn_type,
            "amount": amount,
            "currency": "CAD" if random.random() > 0.04 else "USD",
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "channel": channel,
            "status": weighted_choice(statuses),
            "description": description if not has_missing else None,
            "merchant_name": f"Merchant_{random.randint(1000, 9999)}" if txn_type == "POS_PURCHASE" else None,
            "merchant_category": random.choice(merchant_categories) if txn_type == "POS_PURCHASE" else None,
            "reference_number": f"REF{random.randint(100000000, 999999999)}",
            "balance_after": round(random.uniform(100, 100000), 2),
            "ip_address": f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}" if channel in ["ONLINE", "MOBILE"] else None,
            "device_id": hashlib.md5(f"{account_id}-{random.randint(1,3)}".encode()).hexdigest()[:16] if channel in ["ONLINE", "MOBILE"] else None,
            "geolocation": f"{round(random.uniform(43.0, 53.0), 4)},{round(random.uniform(-130.0, -60.0), 4)}" if channel in ["ONLINE", "MOBILE"] and random.random() > 0.3 else None,
            "is_international": random.random() < 0.04,
            "fx_rate": round(random.uniform(1.30, 1.40), 4) if random.random() < 0.04 else 1.0,
            "anomaly_flag": 1 if is_outlier or random.random() < 0.03 else 0,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "_is_outlier": is_outlier,
            "_has_format_issue": has_format_issue
        }
        transactions.append(txn)
    
    # Sort by timestamp
    transactions.sort(key=lambda x: x["timestamp"])
    return transactions


def generate_loans(customers: List[Dict], num: int) -> List[Dict]:
    """Generate loan records."""
    loans = []
    
    loan_types = [
        ("MORTGAGE", 0.30), ("AUTO", 0.25), ("PERSONAL", 0.25),
        ("LINE_OF_CREDIT", 0.10), ("STUDENT", 0.05), ("BUSINESS", 0.05)
    ]
    
    statuses = [
        ("APPROVED", 0.45), ("ACTIVE", 0.25), ("CLOSED", 0.15),
        ("REJECTED", 0.10), ("DEFAULTED", 0.03), ("WRITTEN_OFF", 0.02)
    ]
    
    customer_ids = [c["customer_id"] for c in customers]
    
    for i in range(num):
        customer_id = random.choice(customer_ids)
        loan_type = weighted_choice(loan_types)
        status = weighted_choice(statuses)
        
        # Amount based on type
        if loan_type == "MORTGAGE":
            amount = round(random.uniform(200000, 1500000), 2)
            term_months = random.choice([60, 120, 180, 240, 300, 360])
        elif loan_type == "AUTO":
            amount = round(random.uniform(15000, 80000), 2)
            term_months = random.choice([36, 48, 60, 72, 84])
        elif loan_type == "PERSONAL":
            amount = round(random.uniform(5000, 50000), 2)
            term_months = random.choice([12, 24, 36, 48, 60])
        elif loan_type == "LINE_OF_CREDIT":
            amount = round(random.uniform(10000, 100000), 2)
            term_months = 0  # Revolving
        elif loan_type == "STUDENT":
            amount = round(random.uniform(10000, 100000), 2)
            term_months = random.choice([60, 120, 180])
        else:  # BUSINESS
            amount = round(random.uniform(25000, 500000), 2)
            term_months = random.choice([36, 60, 84, 120])
        
        application_date = generate_date_in_range(datetime(2018, 1, 1), datetime(2024, 6, 1))
        approval_date = application_date + timedelta(days=random.randint(1, 30)) if status != "REJECTED" else None
        
        loan = {
            "loan_id": generate_id("L", i),
            "customer_id": customer_id,
            "loan_type": loan_type,
            "principal_amount": amount,
            "interest_rate": round(random.uniform(2.5, 12.0), 2),
            "term_months": term_months,
            "monthly_payment": round(amount / max(term_months, 1) * 1.05, 2) if term_months > 0 else 0,
            "outstanding_balance": round(amount * random.uniform(0.1, 0.9), 2) if status == "ACTIVE" else (0 if status == "CLOSED" else amount),
            "application_date": application_date.strftime("%Y-%m-%d"),
            "approval_date": approval_date.strftime("%Y-%m-%d") if approval_date else None,
            "disbursement_date": (approval_date + timedelta(days=random.randint(1, 14))).strftime("%Y-%m-%d") if approval_date and status not in ["REJECTED"] else None,
            "maturity_date": (approval_date + timedelta(days=term_months * 30)).strftime("%Y-%m-%d") if approval_date and term_months > 0 else None,
            "status": status,
            "collateral_type": random.choice(["PROPERTY", "VEHICLE", "SAVINGS", "NONE"]) if loan_type in ["MORTGAGE", "AUTO", "BUSINESS"] else "NONE",
            "credit_score_at_application": random.randint(580, 850),
            "debt_to_income_ratio": round(random.uniform(0.15, 0.55), 2),
            "late_payments_count": random.randint(0, 5) if status in ["ACTIVE", "DEFAULTED"] else 0,
            "days_past_due": random.randint(0, 90) if status == "DEFAULTED" else 0,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        loans.append(loan)
    
    return loans


def generate_credit_cards(customers: List[Dict], num: int) -> List[Dict]:
    """Generate credit card records."""
    cards = []
    
    card_types = [
        ("VISA", 0.35), ("MASTERCARD", 0.35), ("AMEX", 0.20), ("DISCOVER", 0.10)
    ]
    
    card_tiers = [
        ("CLASSIC", 0.40), ("GOLD", 0.30), ("PLATINUM", 0.20), ("INFINITE", 0.10)
    ]
    
    customer_ids = [c["customer_id"] for c in customers]
    
    for i in range(num):
        customer_id = random.choice(customer_ids)
        card_type = weighted_choice(card_types)
        tier = weighted_choice(card_tiers)
        
        # Credit limit based on tier
        if tier == "CLASSIC":
            credit_limit = random.randint(500, 5000)
        elif tier == "GOLD":
            credit_limit = random.randint(5000, 15000)
        elif tier == "PLATINUM":
            credit_limit = random.randint(15000, 50000)
        else:  # INFINITE
            credit_limit = random.randint(50000, 200000)
        
        balance = round(credit_limit * random.uniform(0, 1.1), 2)  # Can exceed limit slightly
        issued_date = generate_date_in_range(datetime(2015, 1, 1), datetime(2024, 1, 1))
        
        card = {
            "card_id": generate_id("CC", i),
            "customer_id": customer_id,
            "card_type": card_type,
            "card_tier": tier,
            "card_number_masked": f"****-****-****-{random.randint(1000, 9999)}",
            "credit_limit": credit_limit,
            "current_balance": balance,
            "available_credit": max(0, credit_limit - balance),
            "minimum_payment_due": round(max(25, balance * 0.02), 2) if balance > 0 else 0,
            "payment_due_date": (datetime.now() + timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
            "last_payment_date": generate_date_in_range(datetime(2024, 1, 1), datetime(2024, 11, 30)).strftime("%Y-%m-%d") if random.random() > 0.1 else None,
            "last_payment_amount": round(random.uniform(50, 5000), 2) if random.random() > 0.1 else None,
            "interest_rate_purchase": round(random.uniform(19.99, 29.99), 2),
            "interest_rate_cash_advance": round(random.uniform(22.99, 34.99), 2),
            "annual_fee": random.choice([0, 0, 0, 79, 99, 120, 150, 399, 599]),
            "rewards_points": random.randint(0, 500000),
            "rewards_program": random.choice(["CASHBACK", "TRAVEL", "POINTS", "NONE"]),
            "issued_date": issued_date.strftime("%Y-%m-%d"),
            "expiry_date": (issued_date + timedelta(days=random.randint(365*3, 365*5))).strftime("%Y-%m"),
            "status": weighted_choice([("ACTIVE", 0.85), ("SUSPENDED", 0.05), ("CLOSED", 0.07), ("LOST_STOLEN", 0.03)]),
            "utilization_rate": round(balance / credit_limit * 100, 1) if credit_limit > 0 else 0,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        cards.append(card)
    
    return cards


def generate_feedback(customers: List[Dict], num: int) -> List[Dict]:
    """Generate customer feedback records."""
    feedback_records = []
    
    feedback_types = [
        ("COMPLAINT", 0.35), ("SUGGESTION", 0.25), ("PRAISE", 0.20),
        ("INQUIRY", 0.15), ("DISPUTE", 0.05)
    ]
    
    categories = [
        "ACCOUNT_SERVICE", "CARD_SERVICE", "LOAN_SERVICE", "ONLINE_BANKING",
        "MOBILE_APP", "BRANCH_SERVICE", "FEES_CHARGES", "FRAUD", "OTHER"
    ]
    
    channels = ["PHONE", "EMAIL", "BRANCH", "CHAT", "SOCIAL_MEDIA", "MAIL"]
    
    resolutions = [
        ("RESOLVED", 0.60), ("PENDING", 0.20), ("ESCALATED", 0.10),
        ("CLOSED_NO_ACTION", 0.05), ("DUPLICATE", 0.05)
    ]
    
    customer_ids = [c["customer_id"] for c in customers]
    
    for i in range(num):
        customer_id = random.choice(customer_ids)
        feedback_type = weighted_choice(feedback_types)
        submitted_date = generate_date_in_range(datetime(2023, 1, 1), datetime(2024, 11, 30))
        resolution_status = weighted_choice(resolutions)
        
        feedback = {
            "feedback_id": generate_id("FB", i),
            "customer_id": customer_id,
            "feedback_type": feedback_type,
            "category": random.choice(categories),
            "channel": random.choice(channels),
            "subject": f"RE: {feedback_type.title()} - Case #{random.randint(10000, 99999)}",
            "description": f"Synthetic feedback description for testing. Type: {feedback_type}. Category: {random.choice(categories)}.",
            "submitted_date": submitted_date.strftime("%Y-%m-%d"),
            "resolution_status": resolution_status,
            "resolution_date": (submitted_date + timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d") if resolution_status in ["RESOLVED", "CLOSED_NO_ACTION"] else None,
            "assigned_to": f"Agent_{random.randint(100, 999)}",
            "priority": weighted_choice([("LOW", 0.30), ("MEDIUM", 0.50), ("HIGH", 0.15), ("CRITICAL", 0.05)]),
            "satisfaction_score": random.randint(1, 5) if resolution_status == "RESOLVED" else None,
            "response_time_hours": random.randint(1, 72),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        feedback_records.append(feedback)
    
    return feedback_records


def calculate_statistics(data: Dict) -> Dict:
    """Calculate dataset statistics."""
    stats = {
        "customers": {
            "total": len(data["customers"]),
            "by_segment": {},
            "by_province": {},
            "avg_age": 0,
            "format_issues": sum(1 for c in data["customers"] if c.get("_has_format_issue"))
        },
        "accounts": {
            "total": len(data["accounts"]),
            "by_type": {},
            "by_status": {},
            "outliers": sum(1 for a in data["accounts"] if a.get("_is_outlier"))
        },
        "transactions": {
            "total": len(data["transactions"]),
            "by_type": {},
            "by_channel": {},
            "outliers": sum(1 for t in data["transactions"] if t.get("_is_outlier")),
            "format_issues": sum(1 for t in data["transactions"] if t.get("_has_format_issue"))
        },
        "loans": {
            "total": len(data["loans"]),
            "by_type": {},
            "by_status": {}
        },
        "credit_cards": {
            "total": len(data["credit_cards"]),
            "by_type": {},
            "by_tier": {}
        },
        "feedback": {
            "total": len(data["feedback"]),
            "by_type": {},
            "by_status": {}
        }
    }
    
    # Calculate distributions
    for c in data["customers"]:
        stats["customers"]["by_segment"][c["segment"]] = stats["customers"]["by_segment"].get(c["segment"], 0) + 1
        stats["customers"]["by_province"][c["province"]] = stats["customers"]["by_province"].get(c["province"], 0) + 1
    stats["customers"]["avg_age"] = round(sum(c["age"] for c in data["customers"]) / len(data["customers"]), 1)
    
    for a in data["accounts"]:
        stats["accounts"]["by_type"][a["account_type"]] = stats["accounts"]["by_type"].get(a["account_type"], 0) + 1
        stats["accounts"]["by_status"][a["status"]] = stats["accounts"]["by_status"].get(a["status"], 0) + 1
    
    for t in data["transactions"]:
        stats["transactions"]["by_type"][t["transaction_type"]] = stats["transactions"]["by_type"].get(t["transaction_type"], 0) + 1
        stats["transactions"]["by_channel"][t["channel"]] = stats["transactions"]["by_channel"].get(t["channel"], 0) + 1
    
    for l in data["loans"]:
        stats["loans"]["by_type"][l["loan_type"]] = stats["loans"]["by_type"].get(l["loan_type"], 0) + 1
        stats["loans"]["by_status"][l["status"]] = stats["loans"]["by_status"].get(l["status"], 0) + 1
    
    for cc in data["credit_cards"]:
        stats["credit_cards"]["by_type"][cc["card_type"]] = stats["credit_cards"]["by_type"].get(cc["card_type"], 0) + 1
        stats["credit_cards"]["by_tier"][cc["card_tier"]] = stats["credit_cards"]["by_tier"].get(cc["card_tier"], 0) + 1
    
    for f in data["feedback"]:
        stats["feedback"]["by_type"][f["feedback_type"]] = stats["feedback"]["by_type"].get(f["feedback_type"], 0) + 1
        stats["feedback"]["by_status"][f["resolution_status"]] = stats["feedback"]["by_status"].get(f["resolution_status"], 0) + 1
    
    return stats


def create_passport(data: Dict, stats: Dict) -> Dict:
    """Create dataset passport with metadata."""
    return {
        "dataset_name": "BANK_SYNTHETIC_MULTI_TABLE",
        "version": "2.0.0",
        "generated_date": datetime.now().strftime("%Y-%m-%d"),
        "generator": "PAMOLA Epic 2 - generate_bank_txs_v2.py",
        "seed": SEED,
        "synthetic_notice": {
            "is_synthetic": True,
            "contains_real_pii": False,
            "derived_from_real_data": False,
            "collected_from_external": False,
            "statement": "All data is artificially generated. No real personal information included."
        },
        "tables": {
            "CUSTOMERS": {"records": stats["customers"]["total"], "file": "CUSTOMERS.csv"},
            "ACCOUNTS": {"records": stats["accounts"]["total"], "file": "ACCOUNTS.csv"},
            "TRANSACTIONS": {"records": stats["transactions"]["total"], "file": "TRANSACTIONS.csv"},
            "LOANS": {"records": stats["loans"]["total"], "file": "LOANS.csv"},
            "CREDIT_CARDS": {"records": stats["credit_cards"]["total"], "file": "CREDIT_CARDS.csv"},
            "FEEDBACK": {"records": stats["feedback"]["total"], "file": "FEEDBACK.csv"}
        },
        "relationships": [
            {"from": "ACCOUNTS.customer_id", "to": "CUSTOMERS.customer_id", "type": "many-to-one"},
            {"from": "TRANSACTIONS.account_id", "to": "ACCOUNTS.account_id", "type": "many-to-one"},
            {"from": "LOANS.customer_id", "to": "CUSTOMERS.customer_id", "type": "many-to-one"},
            {"from": "CREDIT_CARDS.customer_id", "to": "CUSTOMERS.customer_id", "type": "many-to-one"},
            {"from": "FEEDBACK.customer_id", "to": "CUSTOMERS.customer_id", "type": "many-to-one"}
        ],
        "data_quality_issues": {
            "outliers": {
                "accounts": stats["accounts"]["outliers"],
                "transactions": stats["transactions"]["outliers"],
                "description": "High-value records for anomaly detection testing"
            },
            "format_violations": {
                "customers": stats["customers"]["format_issues"],
                "transactions": stats["transactions"]["format_issues"],
                "types": ["case", "whitespace", "encoding", "truncation"],
                "description": "Intentional format issues for data quality testing"
            }
        },
        "fingerprints": {
            "prefix": FINGERPRINT_PREFIX,
            "pattern": "PMLA-{TABLE_CODE}{INDEX}-SYN",
            "description": "Synthetic markers for data lineage tracking"
        },
        "statistics": stats
    }


def write_csv(data: List[Dict], filepath: str, exclude_fields: List[str] = None):
    """Write data to CSV, excluding internal fields."""
    if not data:
        return
    
    exclude = exclude_fields or []
    exclude.extend([k for k in data[0].keys() if k.startswith("_")])
    
    fieldnames = [k for k in data[0].keys() if k not in exclude]
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            filtered_row = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(filtered_row)


def main():
    print("=" * 60)
    print("Synthetic Banking Dataset Generator - Multi-Table Version")
    print("=" * 60)
    print(f"Seed: {SEED}")
    print()
    
    # Generate all tables
    print("Generating CUSTOMERS...")
    customers = generate_customers(NUM_CUSTOMERS)
    print(f"  -> {len(customers)} records")
    
    print("Generating ACCOUNTS...")
    accounts = generate_accounts(customers, NUM_ACCOUNTS)
    print(f"  -> {len(accounts)} records")
    
    print("Generating TRANSACTIONS...")
    transactions = generate_transactions(accounts, NUM_TRANSACTIONS)
    print(f"  -> {len(transactions)} records")
    
    print("Generating LOANS...")
    loans = generate_loans(customers, NUM_LOANS)
    print(f"  -> {len(loans)} records")
    
    print("Generating CREDIT_CARDS...")
    credit_cards = generate_credit_cards(customers, NUM_CREDIT_CARDS)
    print(f"  -> {len(credit_cards)} records")
    
    print("Generating FEEDBACK...")
    feedback = generate_feedback(customers, NUM_FEEDBACK)
    print(f"  -> {len(feedback)} records")
    
    # Collect all data
    data = {
        "customers": customers,
        "accounts": accounts,
        "transactions": transactions,
        "loans": loans,
        "credit_cards": credit_cards,
        "feedback": feedback
    }
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats = calculate_statistics(data)
    
    # Write CSVs
    print("\nWriting CSV files...")
    output_dir = "/home/claude"
    
    write_csv(customers, f"{output_dir}/CUSTOMERS.csv")
    write_csv(accounts, f"{output_dir}/ACCOUNTS.csv")
    write_csv(transactions, f"{output_dir}/TRANSACTIONS.csv")
    write_csv(loans, f"{output_dir}/LOANS.csv")
    write_csv(credit_cards, f"{output_dir}/CREDIT_CARDS.csv")
    write_csv(feedback, f"{output_dir}/FEEDBACK.csv")
    
    # Write passport
    print("Writing passport...")
    passport = create_passport(data, stats)
    with open(f"{output_dir}/BANK_TXS_passport.json", "w", encoding="utf-8") as f:
        json.dump(passport, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)
    print(f"Customers:    {stats['customers']['total']:,}")
    print(f"Accounts:     {stats['accounts']['total']:,}")
    print(f"Transactions: {stats['transactions']['total']:,}")
    print(f"Loans:        {stats['loans']['total']:,}")
    print(f"Credit Cards: {stats['credit_cards']['total']:,}")
    print(f"Feedback:     {stats['feedback']['total']:,}")
    print("-" * 60)
    print(f"Data Quality Issues:")
    print(f"  Outliers:         {stats['accounts']['outliers'] + stats['transactions']['outliers']}")
    print(f"  Format Issues:    {stats['customers']['format_issues'] + stats['transactions']['format_issues']}")
    print("=" * 60)
    
    return data, passport, stats


if __name__ == "__main__":
    main()
