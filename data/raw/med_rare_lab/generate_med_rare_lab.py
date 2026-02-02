#!/usr/bin/env python3
"""
Synthetic Medical Rare Disease & Laboratory Dataset Generator
PAMOLA.CORE - Epic 2 Testing Suite

Generates two linked datasets for federated learning testing:
1. MED_RARE_TXP_REGISTRY - Rare disease and transplant registry
2. MED_LAB - Laboratory results system

Key feature: ~2,000-2,500 patients appear in both datasets via link_shared_id

All data is 100% synthetic - no real PHI included.
"""

import csv
import json
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import uuid

# Seed for reproducibility
SEED = 42
random.seed(SEED)

# === CONFIGURATION ===

NUM_REGISTRY_RECORDS = 10000
NUM_LAB_RECORDS = 10000
NUM_UNIQUE_REGISTRY_PATIENTS = 5100
NUM_UNIQUE_LAB_PATIENTS = 3600
NUM_SHARED_PATIENTS = 2200  # Appear in both datasets

# Data quality rates
PHI_RATE_REGISTRY = 0.48
PHI_RATE_LAB = 0.16
MISSING_COORDS_RATE = 0.21

# Fingerprint
FINGERPRINT_PREFIX = "PMLA"

# === REFERENCE DATA ===

CANADIAN_PROVINCES = ["ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE", "NT", "YT", "NU"]
PROVINCE_WEIGHTS = [0.38, 0.23, 0.13, 0.12, 0.04, 0.03, 0.03, 0.02, 0.01, 0.005, 0.002, 0.002, 0.001]

CITIES_BY_PROVINCE = {
    "ON": ["Toronto", "Ottawa", "Mississauga", "Hamilton", "London", "Kitchener"],
    "QC": ["Montreal", "Quebec City", "Laval", "Gatineau", "Sherbrooke"],
    "BC": ["Vancouver", "Victoria", "Surrey", "Burnaby", "Richmond"],
    "AB": ["Calgary", "Edmonton", "Red Deer", "Lethbridge"],
    "MB": ["Winnipeg", "Brandon"],
    "SK": ["Saskatoon", "Regina"],
    "NS": ["Halifax", "Dartmouth"],
    "NB": ["Moncton", "Saint John", "Fredericton"],
    "NL": ["St. John's"],
    "PE": ["Charlottetown"],
    "NT": ["Yellowknife"],
    "YT": ["Whitehorse"],
    "NU": ["Iqaluit"]
}

RARE_DISEASES = [
    ("Hemophilia A", "F8", "c.6545C>T"),
    ("Idiopathic Pulmonary Fibrosis", "MUC5B", "rs35705950"),
    ("Sickle Cell Disease", "HBB", "c.20A>T"),
    ("Cystic Fibrosis", "CFTR", "F508del"),
    ("ALS", "SOD1", "p.Ala4Val"),
    ("Phenylketonuria", "PAH", "p.Arg408Trp"),
    ("Huntington Disease", "HTT", "CAG expansion"),
    ("Marfan Syndrome", "FBN1", "c.4621G>A"),
    ("Wilson Disease", "ATP7B", "p.His1069Gln"),
    ("Gaucher Disease", "GBA", "N370S"),
    ("Fabry Disease", "GLA", "p.Arg227Ter"),
    ("Pompe Disease", "GAA", "c.-32-13T>G")
]

TRANSPLANT_TYPES = ["Kidney", "Liver", "Heart", "Lung", "Pancreas", "HSCT"]

WAITLIST_STATUSES = [
    ("Not Listed", 0.55),
    ("Active", 0.25),
    ("Inactive", 0.12),
    ("Removed", 0.08)
]

IMMUNOSUPPRESSION_DRUGS = [
    "Tacrolimus", "Cyclosporine", "Mycophenolate", "Prednisone",
    "Sirolimus", "Everolimus", "Azathioprine", "Basiliximab"
]

# Laboratory tests with LOINC codes
LAB_TESTS = [
    ("19123-9", "Hemoglobin", "g/dL", "12.0-17.5", 14.5, 2.0),
    ("777-3", "Platelets", "10*3/uL", "150-400", 250, 60),
    ("2951-2", "Sodium", "mmol/L", "136-145", 140, 3),
    ("1920-8", "AST", "U/L", "10-40", 25, 12),
    ("1742-6", "ALT", "U/L", "7-56", 30, 15),
    ("2160-0", "Creatinine", "mg/dL", "0.7-1.3", 1.0, 0.3),
    ("3094-0", "BUN", "mg/dL", "7-20", 14, 4),
    ("2345-7", "Glucose", "mg/dL", "70-100", 95, 20),
    ("2085-9", "HDL Cholesterol", "mg/dL", "40-60", 50, 12),
    ("13457-7", "LDL Cholesterol", "mg/dL", "0-100", 100, 30),
    ("6690-2", "WBC", "10*3/uL", "4.5-11.0", 7.5, 2.0),
    ("4544-3", "Hematocrit", "%", "36-48", 42, 4),
    ("14749-6", "Glucose (fasting)", "mg/dL", "70-99", 90, 15),
    ("2339-0", "Glucose (random)", "mg/dL", "70-140", 105, 25),
    ("17861-6", "Calcium", "mg/dL", "8.5-10.5", 9.5, 0.5),
    ("2823-3", "Potassium", "mmol/L", "3.5-5.0", 4.2, 0.4),
    ("1751-7", "Albumin", "g/dL", "3.5-5.0", 4.2, 0.4),
    ("1975-2", "Bilirubin Total", "mg/dL", "0.1-1.2", 0.7, 0.3),
    ("6768-6", "Alkaline Phosphatase", "U/L", "44-147", 80, 25),
    ("2532-0", "LDH", "U/L", "140-280", 200, 40)
]

SPECIMEN_TYPES = ["Blood", "Serum", "Plasma", "Urine", "Sputum"]

TRANSPLANT_CENTERS = [
    ("Toronto General Hospital", "Toronto", "ON"),
    ("University Health Network", "Toronto", "ON"),
    ("Hospital for Sick Children", "Toronto", "ON"),
    ("CHUM", "Montreal", "QC"),
    ("McGill University Health Centre", "Montreal", "QC"),
    ("Vancouver General Hospital", "Vancouver", "BC"),
    ("Foothills Medical Centre", "Calgary", "AB"),
    ("University of Alberta Hospital", "Edmonton", "AB"),
    ("QEII Health Sciences Centre", "Halifax", "NS"),
    ("London Health Sciences Centre", "London", "ON")
]

# === HELPER FUNCTIONS ===

def generate_id(prefix: str, index: int) -> str:
    """Generate synthetic ID with fingerprint."""
    return f"{FINGERPRINT_PREFIX}-{prefix}{index:06d}-SYN"


def generate_hash_id(seed_str: str) -> str:
    """Generate consistent hash-based ID."""
    return hashlib.sha256(f"PAMOLA-{seed_str}".encode()).hexdigest()[:16].upper()


def generate_postal_fsa(province: str) -> str:
    """Generate Canadian postal FSA."""
    fsa_map = {
        "ON": ["M", "K", "L", "N", "P"],
        "QC": ["H", "G", "J"],
        "BC": ["V"],
        "AB": ["T"],
        "MB": ["R"],
        "SK": ["S"],
        "NS": ["B"],
        "NB": ["E"],
        "NL": ["A"],
        "PE": ["C"],
        "NT": ["X"],
        "YT": ["Y"],
        "NU": ["X"]
    }
    letters = "ABCEGHJKLMNPRSTVWXYZ"
    prefix = random.choice(fsa_map.get(province, ["X"]))
    return f"{prefix}{random.randint(0,9)}{random.choice(letters)}"


def generate_hla_typing() -> str:
    """Generate HLA typing string."""
    a_alleles = ["A*01:01", "A*02:01", "A*03:01", "A*11:01", "A*24:02", "A*26:01"]
    b_alleles = ["B*07:02", "B*08:01", "B*15:01", "B*27:05", "B*35:01", "B*44:02"]
    dr_alleles = ["DRB1*01:01", "DRB1*03:01", "DRB1*04:01", "DRB1*07:01", "DRB1*11:01", "DRB1*15:01"]
    
    return f"{random.choice(a_alleles)},{random.choice(a_alleles)};{random.choice(b_alleles)},{random.choice(b_alleles)};{random.choice(dr_alleles)},{random.choice(dr_alleles)}"


def generate_date_in_range(start: datetime, end: datetime) -> datetime:
    """Generate random date in range."""
    delta = end - start
    random_days = random.randint(0, max(1, delta.days))
    return start + timedelta(days=random_days)


def weighted_choice(choices: List[Tuple[Any, float]]) -> Any:
    """Select item based on weights."""
    items, weights = zip(*choices)
    return random.choices(items, weights=weights, k=1)[0]


def generate_clinical_note(has_phi: bool, disease: str = None) -> Tuple[str, int, int, int]:
    """Generate clinical note with optional PHI."""
    templates = [
        "Patient presents with {symptoms}. {disease_note} Current medications reviewed. Plan: {plan}",
        "Follow-up visit for {disease_note}. Vitals stable. Labs reviewed - {lab_note}. Continue current regimen.",
        "Routine monitoring visit. {disease_note} No acute concerns. {plan}",
        "Patient reports {symptoms}. Physical exam unremarkable. {disease_note} Will continue monitoring.",
        "Transplant follow-up: {disease_note} Graft function stable. Immunosuppression levels therapeutic."
    ]
    
    symptoms = random.choice([
        "fatigue and mild dyspnea",
        "stable condition, no new complaints",
        "mild discomfort, well-controlled",
        "improved symptoms since last visit",
        "occasional pain, manageable with current therapy"
    ])
    
    disease_note = f"Managing {disease}." if disease else "Chronic condition stable."
    
    lab_note = random.choice([
        "within normal limits",
        "minor abnormalities noted",
        "stable compared to previous",
        "improved from baseline"
    ])
    
    plan = random.choice([
        "Continue current therapy",
        "Adjust medications as needed",
        "Return in 3 months",
        "Order follow-up labs",
        "Refer to specialist if symptoms worsen"
    ])
    
    note = random.choice(templates).format(
        symptoms=symptoms,
        disease_note=disease_note,
        lab_note=lab_note,
        plan=plan
    )
    
    email_count = 0
    phone_count = 0
    date_count = 0
    
    if has_phi:
        # Inject PHI
        phi_additions = []
        
        if random.random() < 0.3:
            email = f"patient{random.randint(100,999)}@email.com"
            phi_additions.append(f"Contact: {email}")
            email_count = 1
        
        if random.random() < 0.4:
            phone = f"{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
            phi_additions.append(f"Callback: {phone}")
            phone_count = 1
        
        if random.random() < 0.5:
            month = random.choice(["January", "February", "March", "April", "May", "June"])
            day = random.randint(1, 28)
            phi_additions.append(f"Next appointment: {month} {day}, 2025")
            date_count = 1
        
        if phi_additions:
            note += " " + " ".join(phi_additions)
    
    return note, email_count, phone_count, date_count


def generate_lab_note(has_phi: bool) -> Tuple[Optional[str], int, int, int]:
    """Generate lab result note with optional PHI."""
    if random.random() > 0.16:  # 84% have no note
        return None, 0, 0, 0
    
    templates = [
        "Sample quality acceptable.",
        "Repeat testing recommended.",
        "Critical value - physician notified.",
        "Hemolyzed specimen - results may be affected.",
        "Fasting sample confirmed."
    ]
    
    note = random.choice(templates)
    email_count = 0
    phone_count = 0
    date_count = 0
    
    if has_phi:
        if random.random() < 0.2:
            phone = f"{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
            note += f" Called {phone}."
            phone_count = 1
    
    return note, email_count, phone_count, date_count


# === DATA GENERATORS ===

def create_patient_pool():
    """Create pool of patients for both datasets."""
    patients = []
    
    total_patients = NUM_UNIQUE_REGISTRY_PATIENTS + NUM_UNIQUE_LAB_PATIENTS - NUM_SHARED_PATIENTS
    
    for i in range(total_patients):
        province = random.choices(CANADIAN_PROVINCES, weights=PROVINCE_WEIGHTS, k=1)[0]
        city = random.choice(CITIES_BY_PROVINCE.get(province, ["Unknown"]))
        
        patient = {
            "gt_person_id": generate_id("P", i),
            "birth_year": random.randint(1930, 2015),
            "sex": weighted_choice([("F", 0.52), ("M", 0.47), ("X", 0.01)]),
            "home_province": province,
            "home_city": city,
            "home_postal_fsa": generate_postal_fsa(province),
            "language_pref": weighted_choice([("EN", 0.60), ("FR", 0.30), ("EN/FR", 0.10)]),
            "in_registry": False,
            "in_lab": False,
            "link_shared_id": None
        }
        patients.append(patient)
    
    # Assign patients to datasets
    random.shuffle(patients)
    
    # First NUM_SHARED_PATIENTS go to both
    for i in range(NUM_SHARED_PATIENTS):
        patients[i]["in_registry"] = True
        patients[i]["in_lab"] = True
        patients[i]["link_shared_id"] = generate_hash_id(f"LINK-{i}")
    
    # Next batch: registry only
    registry_only_count = NUM_UNIQUE_REGISTRY_PATIENTS - NUM_SHARED_PATIENTS
    for i in range(NUM_SHARED_PATIENTS, NUM_SHARED_PATIENTS + registry_only_count):
        patients[i]["in_registry"] = True
        # Some have link_shared_id even if not in lab (realistic scenario)
        if random.random() < 0.1:
            patients[i]["link_shared_id"] = generate_hash_id(f"LINK-{i}")
    
    # Rest: lab only
    lab_only_count = NUM_UNIQUE_LAB_PATIENTS - NUM_SHARED_PATIENTS
    for i in range(NUM_SHARED_PATIENTS + registry_only_count, 
                   NUM_SHARED_PATIENTS + registry_only_count + lab_only_count):
        patients[i]["in_lab"] = True
        if random.random() < 0.1:
            patients[i]["link_shared_id"] = generate_hash_id(f"LINK-{i}")
    
    return patients


def generate_registry_records(patients: List[Dict]) -> List[Dict]:
    """Generate rare disease registry records."""
    records = []
    
    registry_patients = [p for p in patients if p["in_registry"]]
    
    # Calculate records per patient to reach NUM_REGISTRY_RECORDS
    base_records = NUM_REGISTRY_RECORDS // len(registry_patients)
    extra_records = NUM_REGISTRY_RECORDS % len(registry_patients)
    
    record_idx = 0
    for i, patient in enumerate(registry_patients):
        num_records = base_records + (1 if i < extra_records else 0)
        
        # Assign disease and transplant type for this patient
        disease, gene, variant = random.choice(RARE_DISEASES)
        transplant_type = random.choice(TRANSPLANT_TYPES)
        center = random.choice(TRANSPLANT_CENTERS)
        
        # Determine if patient is transplanted
        waitlist_status = weighted_choice(WAITLIST_STATUSES)
        is_transplanted = waitlist_status == "Removed" or (waitlist_status == "Active" and random.random() < 0.3)
        
        diagnosis_date = generate_date_in_range(datetime(2015, 1, 1), datetime(2023, 12, 31))
        referral_date = diagnosis_date + timedelta(days=random.randint(30, 365))
        transplant_date = referral_date + timedelta(days=random.randint(180, 730)) if is_transplanted else None
        
        for j in range(num_records):
            has_phi = random.random() < PHI_RATE_REGISTRY
            note, email_count, phone_count, date_count = generate_clinical_note(has_phi, disease)
            
            # Generate immunosuppression regimen
            num_drugs = random.randint(2, 4)
            immunosuppression = ";".join(random.sample(IMMUNOSUPPRESSION_DRUGS, num_drugs))
            
            record = {
                "registry_row_id": generate_id("RR", record_idx),
                "registry_internal_id": generate_hash_id(f"REG-{patient['gt_person_id']}"),
                "gt_person_id": patient["gt_person_id"],
                "link_shared_id": patient["link_shared_id"],
                "birth_year": patient["birth_year"],
                "sex": patient["sex"],
                "home_postal_fsa": patient["home_postal_fsa"],
                "home_province": patient["home_province"],
                "language_pref": patient["language_pref"],
                "disease_group": disease,
                "transplant_type": transplant_type,
                "diagnosis_date": diagnosis_date.strftime("%Y-%m-%d"),
                "referral_date": referral_date.strftime("%Y-%m-%d"),
                "waitlist_status": waitlist_status,
                "transplant_date": transplant_date.strftime("%Y-%m-%d") if transplant_date else None,
                "donor_type": random.choice(["Deceased", "Living"]) if is_transplanted else None,
                "hla_typing": generate_hla_typing(),
                "pra_percent": round(random.uniform(0, 100), 1),
                "genotype_gene": gene,
                "genotype_variant": variant,
                "immunosuppression": immunosuppression if is_transplanted else None,
                "center_id": generate_id("CTR", TRANSPLANT_CENTERS.index(center)),
                "center_name": center[0],
                "center_city": center[1],
                "center_province": center[2],
                "center_lat": round(random.uniform(43.0, 53.0), 4) if random.random() > MISSING_COORDS_RATE else None,
                "center_lng": round(random.uniform(-130.0, -60.0), 4) if random.random() > MISSING_COORDS_RATE else None,
                "clinician_note": note,
                "has_phi_in_text": has_phi,
                "phi_email_count": email_count,
                "phi_phone_count": phone_count,
                "phi_date_count": date_count,
                "hospitalized_30d": random.random() < 0.15,
                "graft_failure_1y": random.random() < 0.02 if is_transplanted else False,
                "mortality_1y": random.random() < 0.02,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_system": "PAMOLA_REGISTRY_GEN_V1"
            }
            records.append(record)
            record_idx += 1
    
    return records


def generate_lab_records(patients: List[Dict]) -> List[Dict]:
    """Generate laboratory result records."""
    records = []
    
    lab_patients = [p for p in patients if p["in_lab"]]
    
    # Calculate records per patient
    base_records = NUM_LAB_RECORDS // len(lab_patients)
    extra_records = NUM_LAB_RECORDS % len(lab_patients)
    
    record_idx = 0
    site_idx = 0
    
    for i, patient in enumerate(lab_patients):
        num_records = base_records + (1 if i < extra_records else 0)
        
        # Assign lab site for this patient
        province = patient["home_province"]
        city = patient["home_city"]
        site_name = f"{city} Clinical Laboratory"
        site_id = generate_id("LAB", site_idx % 100)
        site_idx += 1
        
        for j in range(num_records):
            # Pick a random test
            test = random.choice(LAB_TESTS)
            loinc, name, units, ref_range, mean_val, std_val = test
            
            # Generate result value
            result_value = round(random.gauss(mean_val, std_val), 2)
            
            # Determine if abnormal
            try:
                ref_parts = ref_range.split("-")
                ref_low = float(ref_parts[0])
                ref_high = float(ref_parts[1])
                is_abnormal = result_value < ref_low or result_value > ref_high
            except:
                is_abnormal = random.random() < 0.07
            
            has_phi = random.random() < PHI_RATE_LAB
            note, email_count, phone_count, date_count = generate_lab_note(has_phi)
            
            collection_time = generate_date_in_range(datetime(2024, 1, 1), datetime(2025, 8, 1))
            received_time = collection_time + timedelta(hours=random.randint(1, 24))
            
            record = {
                "lab_row_id": generate_id("LR", record_idx),
                "lab_internal_id": generate_hash_id(f"LAB-{patient['gt_person_id']}"),
                "gt_person_id": patient["gt_person_id"],
                "link_shared_id": patient["link_shared_id"],
                "order_id": generate_id("ORD", record_idx),
                "birth_year": patient["birth_year"],
                "sex": patient["sex"],
                "home_postal_fsa": patient["home_postal_fsa"],
                "home_province": patient["home_province"],
                "language_pref": patient["language_pref"],
                "site_id": site_id,
                "site_name": site_name,
                "site_city": city,
                "site_province": province,
                "site_lat": round(random.uniform(43.0, 53.0), 4) if random.random() > MISSING_COORDS_RATE else None,
                "site_lng": round(random.uniform(-130.0, -60.0), 4) if random.random() > MISSING_COORDS_RATE else None,
                "test_loinc": loinc,
                "test_name": name,
                "specimen_type": random.choice(SPECIMEN_TYPES),
                "collection_ts": collection_time.strftime("%Y-%m-%d %H:%M:%S"),
                "received_ts": received_time.strftime("%Y-%m-%d %H:%M:%S"),
                "result_value": result_value,
                "result_units": units,
                "ref_range": ref_range,
                "abnormal_flag": "A" if is_abnormal else "N",
                "note": note,
                "has_phi_in_text": has_phi and note is not None,
                "phi_email_count": email_count,
                "phi_phone_count": phone_count,
                "phi_date_count": date_count,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_system": "PAMOLA_LAB_GEN_V1"
            }
            records.append(record)
            record_idx += 1
    
    return records


def calculate_statistics(registry: List[Dict], lab: List[Dict], patients: List[Dict]) -> Dict:
    """Calculate dataset statistics."""
    
    # Count shared patients
    registry_gt_ids = set(r["gt_person_id"] for r in registry)
    lab_gt_ids = set(r["gt_person_id"] for r in lab)
    shared_ids = registry_gt_ids & lab_gt_ids
    
    # Disease distribution
    disease_counts = {}
    for r in registry:
        d = r["disease_group"]
        disease_counts[d] = disease_counts.get(d, 0) + 1
    
    # Transplant stats
    transplanted = sum(1 for r in registry if r["transplant_date"])
    waitlist_active = sum(1 for r in registry if r["waitlist_status"] == "Active")
    graft_failures = sum(1 for r in registry if r["graft_failure_1y"])
    mortality = sum(1 for r in registry if r["mortality_1y"])
    
    # Lab stats
    abnormal_labs = sum(1 for r in lab if r["abnormal_flag"] == "A")
    phi_in_registry = sum(1 for r in registry if r["has_phi_in_text"])
    phi_in_lab = sum(1 for r in lab if r["has_phi_in_text"])
    
    # Linkage stats
    registry_with_link = sum(1 for r in registry if r["link_shared_id"])
    lab_with_link = sum(1 for r in lab if r["link_shared_id"])
    
    return {
        "registry": {
            "total_records": len(registry),
            "unique_patients": len(registry_gt_ids),
            "disease_distribution": disease_counts,
            "transplanted_count": transplanted,
            "transplanted_pct": round(transplanted / len(registry) * 100, 2),
            "waitlist_active_pct": round(waitlist_active / len(registry) * 100, 2),
            "graft_failure_1y_pct": round(graft_failures / max(transplanted, 1) * 100, 2),
            "mortality_1y_pct": round(mortality / len(registry) * 100, 2),
            "phi_in_text_pct": round(phi_in_registry / len(registry) * 100, 2),
            "with_link_shared_id_pct": round(registry_with_link / len(registry) * 100, 2)
        },
        "lab": {
            "total_records": len(lab),
            "unique_patients": len(lab_gt_ids),
            "abnormal_rate_pct": round(abnormal_labs / len(lab) * 100, 2),
            "phi_in_text_pct": round(phi_in_lab / len(lab) * 100, 2),
            "with_link_shared_id_pct": round(lab_with_link / len(lab) * 100, 2)
        },
        "linkage": {
            "total_unique_patients": len(registry_gt_ids | lab_gt_ids),
            "shared_patients": len(shared_ids),
            "registry_only": len(registry_gt_ids - lab_gt_ids),
            "lab_only": len(lab_gt_ids - registry_gt_ids),
            "overlap_pct": round(len(shared_ids) / len(registry_gt_ids | lab_gt_ids) * 100, 2)
        }
    }


def create_passports(stats: Dict) -> Tuple[Dict, Dict]:
    """Create passport files for both datasets."""
    
    base_passport = {
        "generated_date": datetime.now().strftime("%Y-%m-%d"),
        "generator": "PAMOLA Epic 2 - generate_med_rare_lab.py",
        "seed": SEED,
        "synthetic_notice": {
            "is_synthetic": True,
            "contains_real_phi": False,
            "derived_from_real_data": False,
            "collected_from_external": False,
            "statement": "All data is artificially generated. No real patient information included."
        }
    }
    
    registry_passport = {
        **base_passport,
        "dataset_name": "MED_RARE_TXP_REGISTRY",
        "version": "1.0.0",
        "file": "MED_RARE_TXP_REGISTRY_10k.csv",
        "statistics": stats["registry"],
        "linkage": stats["linkage"],
        "fingerprints": {
            "prefix": FINGERPRINT_PREFIX,
            "row_pattern": "PMLA-RR{index}-SYN",
            "center_pattern": "PMLA-CTR{index}-SYN"
        }
    }
    
    lab_passport = {
        **base_passport,
        "dataset_name": "MED_LAB",
        "version": "1.0.0",
        "file": "MED_LAB_10k.csv",
        "statistics": stats["lab"],
        "linkage": stats["linkage"],
        "fingerprints": {
            "prefix": FINGERPRINT_PREFIX,
            "row_pattern": "PMLA-LR{index}-SYN",
            "site_pattern": "PMLA-LAB{index}-SYN"
        }
    }
    
    return registry_passport, lab_passport


def write_csv(data: List[Dict], filepath: str):
    """Write data to CSV."""
    if not data:
        return
    
    fieldnames = list(data[0].keys())
    
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def main():
    print("=" * 70)
    print("Synthetic Medical Rare Disease & Laboratory Dataset Generator")
    print("=" * 70)
    print(f"Seed: {SEED}")
    print()
    
    # Create shared patient pool
    print("Creating patient pool...")
    patients = create_patient_pool()
    print(f"  Total patients: {len(patients)}")
    print(f"  Registry patients: {sum(1 for p in patients if p['in_registry'])}")
    print(f"  Lab patients: {sum(1 for p in patients if p['in_lab'])}")
    print(f"  Shared patients: {sum(1 for p in patients if p['in_registry'] and p['in_lab'])}")
    
    # Generate registry records
    print("\nGenerating MED_RARE_TXP_REGISTRY...")
    registry_records = generate_registry_records(patients)
    print(f"  -> {len(registry_records)} records")
    
    # Generate lab records
    print("Generating MED_LAB...")
    lab_records = generate_lab_records(patients)
    print(f"  -> {len(lab_records)} records")
    
    # Calculate statistics
    print("\nCalculating statistics...")
    stats = calculate_statistics(registry_records, lab_records, patients)
    
    # Write files
    output_dir = "/home/claude"
    
    print("\nWriting CSV files...")
    write_csv(registry_records, f"{output_dir}/MED_RARE_TXP_REGISTRY_10k.csv")
    write_csv(lab_records, f"{output_dir}/MED_LAB_10k.csv")
    
    print("Writing passport files...")
    registry_passport, lab_passport = create_passports(stats)
    
    with open(f"{output_dir}/MED_RARE_TXP_REGISTRY_passport.json", "w") as f:
        json.dump(registry_passport, f, indent=2)
    
    with open(f"{output_dir}/MED_LAB_passport.json", "w") as f:
        json.dump(lab_passport, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("Generation Complete!")
    print("=" * 70)
    print(f"\nMED_RARE_TXP_REGISTRY:")
    print(f"  Records:           {stats['registry']['total_records']:,}")
    print(f"  Unique patients:   {stats['registry']['unique_patients']:,}")
    print(f"  Transplanted:      {stats['registry']['transplanted_pct']}%")
    print(f"  PHI in text:       {stats['registry']['phi_in_text_pct']}%")
    print(f"  With link_id:      {stats['registry']['with_link_shared_id_pct']}%")
    
    print(f"\nMED_LAB:")
    print(f"  Records:           {stats['lab']['total_records']:,}")
    print(f"  Unique patients:   {stats['lab']['unique_patients']:,}")
    print(f"  Abnormal results:  {stats['lab']['abnormal_rate_pct']}%")
    print(f"  PHI in text:       {stats['lab']['phi_in_text_pct']}%")
    print(f"  With link_id:      {stats['lab']['with_link_shared_id_pct']}%")
    
    print(f"\nCross-Dataset Linkage:")
    print(f"  Total unique patients: {stats['linkage']['total_unique_patients']:,}")
    print(f"  Shared patients:       {stats['linkage']['shared_patients']:,}")
    print(f"  Overlap:               {stats['linkage']['overlap_pct']}%")
    print("=" * 70)
    
    return registry_records, lab_records, stats


if __name__ == "__main__":
    main()
