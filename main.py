from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import re
import os
from scipy.sparse import hstack, csr_matrix

app = FastAPI()

# ==========================================
# 1. LOAD MODELS & DICTIONARIES
# ==========================================
print("Loading Models & Dictionaries...")

# --- RULE DARI NOTEBOOK (CELL 14) ---
KEYWORD_KLINIK_UMUM = [
    'demam', 'meriang', 'pusing', 'sakit kepala',
    'batuk pilek', 'pilek', 'flu',
    'panas', 'tidak enak badan', 'ga enak badan', 'nggak enak badan'
]

def load_dictionary(file_name, key_col, value_col):
    if not os.path.exists(file_name):
        return {}
    df = pd.read_csv(file_name, dtype=str).fillna('')
    return dict(zip(df[key_col].str.lower(), df[value_col].str.lower()))

try:
    final_model = joblib.load('model_final_optimized.pkl')
    tfidf = joblib.load('tfidf_final.pkl')
    selector = joblib.load('selector_final.pkl')
    le = joblib.load('encoder_final.pkl')
    lgb_model = joblib.load('lgb_final.pkl')
    scaler = joblib.load('scaler_final.pkl')
    
    # Load Dictionaries (Harus SAMA dengan train.py)
    KAMUS_SINGKATAN = load_dictionary('kamus_singkatan.csv', 'Singkatan', 'Bentuk Lengkap')
    kamus_medis_df = pd.read_csv('kamus_medis.csv', dtype=str).fillna('')
    KAMUS_MEDIS_FULL = dict(zip(kamus_medis_df['istilah_lama'].str.lower(), kamus_medis_df['istilah_standar'].str.lower()))
    KAMUS_MEDIS_PHRASE = {k: v for k, v in KAMUS_MEDIS_FULL.items() if " " in k}
    KAMUS_MEDIS_TOKEN  = {k: v for k, v in KAMUS_MEDIS_FULL.items() if " " not in k}
    KAMUS_KEYWORD_TRIAGE = load_dictionary('kamus_keyword_poli.csv', 'Keyword Keluhan', 'Poli Tujuan')
    print("Sistem siap!")
except Exception as e:
    print(f"CRITICAL ERROR: {e}")

# ==========================================
# 2. HELPER FUNCTIONS (Harus SAMA dengan train.py)
# ==========================================
# Copy paste helper function (medical_expansions, clean_text, enhance_medical_text, dll)
# yang SAMA PERSIS dengan train.py agar transformasi data konsisten.
medical_expansions = {
    'jantung': 'jantung cardiac cardio heart arrhythmia coronary angina pectoris cvd cardiovascular',
    'hipertensi': 'hipertensi hypertension htn tekanan tinggi darah tinggi blood pressure',
    'nyeri dada': 'nyeri dada chest pain angina pectoris cardiac ischemia',
    'sesak': 'sesak dyspnea dispnea shortness breath napas pendek respiratory breathing difficulty',
    'batuk': 'batuk cough tussis respiratory paru lung bronchitis pneumonia',
    'asma': 'asma asthma wheezing mengi bronchial bronchospasm',
    'paru': 'paru lung pulmonary respiratory chest thorax',
    'mual': 'mual nausea muntah vomit emesis gastric gastrointestinal',
    'diare': 'diare diarrhea gastroenteritis gastric loose stool',
    'perut': 'perut abdomen stomach gastric abdominal belly',
    'maag': 'maag gastritis peptic ulcer gastric dyspepsia',
    'lambung': 'lambung stomach gastric gastritis ulcer',
    'pusing': 'pusing dizziness vertigo neurologic headache cephalgia',
    'stroke': 'stroke cva cerebrovascular neurologic paralysis hemiparesis',
    'kejang': 'kejang seizure convulsion epilepsy neurologic',
    'saraf': 'saraf nerve neurologic neurology neuropathy',
    'diabetes': 'diabetes mellitus dm endocrine gula darah hyperglycemia',
    'gula': 'gula sugar diabetes glucose hyperglycemia',
    'anak': 'anak child pediatric pediatri bayi infant neonatus',
    'bayi': 'bayi infant baby neonatus newborn pediatric',
    'demam': 'demam fever febris panas pyrexia temperature elevated',
    'nyeri': 'nyeri pain sakit ache painful',
    'lemas': 'lemas weak weakness fatigue lethargy tired',
    'mata': 'mata eye ocular ophthalmology vision penglihatan',
    'gigi': 'gigi teeth dental tooth oral mulut',
    'kulit': 'kulit skin dermatology dermato rash ruam',
    'tulang': 'tulang bone orthopedi fracture patah sendi joint',
    'bedah': 'bedsah surgery surgical operasi operation',
    'gawat': 'gawat emergency urgent critical igd er acute severe',
    'darurat': 'darurat emergency critical urgent acute life-threatening'
}

def remove_corrupt_patterns(text):
    text = str(text)
    text = re.sub(r'_x000D_', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'x000d', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\\x00', ' ', text)
    text = re.sub(r'\\u000d', ' ', text)
    text = re.sub(r'[;,]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def clean_text_with_dictionaries(text):
    if not text: return ""
    text = remove_corrupt_patterns(text).lower()
    text = re.sub(r'\brspb\b', '', text)
    text = re.sub(r'\bkeluhan\b', '', text)
    text = re.sub(r'[,:;]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    for old, new in KAMUS_MEDIS_PHRASE.items():
        if old in text: text = text.replace(old, new)
    tokens = text.split()
    tokens = [KAMUS_SINGKATAN.get(t, t) for t in tokens]
    tokens = [KAMUS_MEDIS_TOKEN.get(t, t) for t in tokens]
    return " ".join(tokens)

def enhance_medical_text(text):
    if not text: return ""
    text = str(text).lower()
    text = clean_text_with_dictionaries(text)
    for term, expansion in medical_expansions.items():
        if term in text: text += " " + expansion
    stop_small = {'dan','atau','yang','ini','itu','ke','di','dgn','dgnnya'}
    tokens = [w for w in text.split() if len(w) >= 2 and w not in stop_small]
    return ' '.join(tokens)

def find_poli_from_keyword(keluhan):
    if not keluhan: return ''
    keluhan = str(keluhan).lower()
    for keyword, poli in KAMUS_KEYWORD_TRIAGE.items():
        if keyword in keluhan: return poli.strip()
    return ''

def create_weighted_text(row):
    # Logic ini HARUS SAMA dengan train.py
    diagnosa_text = str(row.get('diagnosa_enhanced', ''))
    keluhan_text = str(row.get('keluhan_enhanced', ''))
    riwayat_text = str(row.get('riwayat_enhanced', ''))
    poli_triage = str(row.get('poli_triage', ''))

    text = (diagnosa_text + " ") * 2
    text += (riwayat_text + " ") * 3
    text += (keluhan_text + " ")
    
    if poli_triage:
        text += " " + poli_triage.replace(' ', '_').upper() * 5

    demo_text = f"{row.get('age_category','')} {row.get('jenis_kelamin','')} {row.get('fever_level','')} {row.get('resp_level','')}"
    text += " " + (demo_text + " ") * 3
    
    if row.get('vital_severity', 0) >= 4: text += " CRITICAL_CASE HIGH_ACUITY EMERGENCY URGENT_CARE SEVERE_CONDITION " * 2
    if row.get('pediatric_case', 0) == 1: text += " PEDIATRIC_SPECIALTY CHILD_MEDICINE ANAK_CARE PEDIATRI " * 2
    if row.get('geriatric_case', 0) == 1: text += " GERIATRIC_SPECIALTY ELDERLY_CARE LANSIA_CARE " * 2
    return text.replace('nan','').strip()

# LOGIC BARU: RULE_KLINIK_UMUM (Dari Notebook Cell 14)
def check_rule_klinik_umum(keluhan_raw, poli_triage, max_confidence, threshold):
    text = str(keluhan_raw).lower().strip()
    poli_triage = str(poli_triage).upper().strip()
    # Rule 1: Kamus Triage bilang umum
    if poli_triage == 'KLINIK UMUM': return True
    # Rule 2: Keluhan tidak spesifik & confidence rendah
    if any(kw in text for kw in KEYWORD_KLINIK_UMUM) and max_confidence < threshold:
        return True
    return False

# ==========================================
# 3. API ENDPOINT
# ==========================================
class PatientRequest(BaseModel):
    keluhan: str
    usia: int = 30
    jenis_kelamin: str = "L"
    riwayat_penyakit: str = ""

@app.post("/predict")
def predict_poli(data: PatientRequest):
    try:
        # 1. Feature Engineering Numerik
        input_df = pd.DataFrame([{
            'usia': data.usia, 'jenis_kelamin': data.jenis_kelamin, 'riwayat_penyakit': data.riwayat_penyakit,
            'keluhan': data.keluhan, 'diagnosa': '', 'pernafasan_x_per_menit': 20.0, 'suhu_tubuh': 36.5
        }])

        input_df['age_category'] = pd.cut(input_df['usia'], bins=[-1,2,12,18,45,65,200], labels=['infant','child','adolescent','adult','senior','geriatric']).astype(str)
        input_df['fever_level'] = pd.cut(input_df['suhu_tubuh'], bins=[-1,36.5,37.5,38.5,60], labels=['hypothermia','normal','lowfever','highfever']).astype(str)
        input_df['resp_level'] = pd.cut(input_df['pernafasan_x_per_menit'], bins=[-1,12,20,30,200], labels=['bradypnea','normal','tachypnea','severe']).astype(str)
        
        input_df['critical_fever'] = (input_df['suhu_tubuh'] > 38.5).astype(int)
        input_df['critical_resp'] = (input_df['pernafasan_x_per_menit'] > 30).astype(int)
        input_df['pediatric_case'] = (input_df['usia'] < 18).astype(int)
        input_df['geriatric_case'] = (input_df['usia'] > 65).astype(int)
        input_df['adult_case'] = ((input_df['usia'] >= 18) & (input_df['usia'] <= 65)).astype(int)
        
        input_df['vital_severity'] = (input_df['critical_fever']*2 + input_df['critical_resp']*2 + (input_df['suhu_tubuh']>37.5).astype(int) + (input_df['pernafasan_x_per_menit']>20).astype(int)).astype(int)
        input_df['age_risk'] = (input_df['pediatric_case']*2 + input_df['geriatric_case']*3).astype(int)

        # 2. Text Engineering
        input_df['keluhan_enhanced'] = input_df['keluhan'].apply(enhance_medical_text)
        input_df['diagnosa_enhanced'] = input_df['diagnosa'].apply(enhance_medical_text)
        input_df['riwayat_enhanced'] = input_df['riwayat_penyakit'].apply(enhance_medical_text)
        input_df['poli_triage'] = input_df.get('keluhan', '').apply(find_poli_from_keyword)
        input_df['weighted_text'] = input_df.apply(create_weighted_text, axis=1)

        # 3. Rule Based Check (Priority 1)
        poli_triage_input = str(input_df.loc[0, 'poli_triage']).strip()
        if poli_triage_input:
            poli_display = poli_triage_input.upper()
            if 'KLINIK GIGI' in poli_display: poli_display = 'KLINIK GIGI & MULUT'
            return {"status": "success", "rekomendasi_poli": poli_display, "confidence": 1.0, "source": "triage_keyword"}

        # 4. Model Prediction
        text_vec = tfidf.transform(input_df['weighted_text'])
        text_sel = selector.transform(text_vec)

        numerical_features = ['usia','pernafasan_x_per_menit','suhu_tubuh','critical_fever','critical_resp','pediatric_case','geriatric_case','vital_severity','age_risk','adult_case']
        for col in numerical_features:
            if col not in input_df.columns: input_df[col] = 0
            
        num_pred = lgb_model.predict(input_df[numerical_features].astype(float))
        num_scaled = scaler.transform(num_pred)
        final_vec = hstack([csr_matrix(num_scaled * 2.0), text_sel * 1.2])

        scores = final_model.decision_function(final_vec)[0]
        proba_vector = scores if np.ndim(scores) > 0 else np.array([scores])
        max_confidence = float(np.max(proba_vector))
        CONFIDENCE_THRESHOLD = 1.0 

        # 5. Rule Based Check (Priority 2: Klinik Umum & Ambiguity)
        if check_rule_klinik_umum(data.keluhan, poli_triage_input, max_confidence, CONFIDENCE_THRESHOLD):
            return {"status": "success", "rekomendasi_poli": "KLINIK UMUM", "confidence": max_confidence, "source": "rule_ambiguity"}
        
        if max_confidence < CONFIDENCE_THRESHOLD:
            return {"status": "success", "rekomendasi_poli": "KLINIK UMUM", "confidence": max_confidence, "source": "rule_low_confidence"}

        # 6. Final Result & Adult Check
        pred_class = np.argmax(proba_vector)
        poli_pred = le.inverse_transform([pred_class])[0]
        poli_display = re.sub(r'\s*\([A-Z]\)\s*$', '', poli_pred).strip()

        if 'KLINIK GIGI' in poli_display.upper(): poli_display = 'KLINIK GIGI & MULUT'

        # Rule Adult Check
        if data.usia >= 18 and 'ANAK' in poli_pred.upper():
            sorted_indices = np.argsort(proba_vector)[::-1]
            found_alt = False
            for idx in sorted_indices[1:]:
                alt_poli = le.inverse_transform([idx])[0]
                if 'ANAK' not in alt_poli.upper():
                    poli_display = re.sub(r'\s*\([A-Z]\)\s*$', '', alt_poli).strip()
                    found_alt = True
                    break
            if not found_alt:
                poli_display = 'KLINIK PENYAKIT DALAM' # Fallback aman

        return {"status": "success", "rekomendasi_poli": poli_display.upper(), "confidence": max_confidence, "source": "model"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Menjalankan di port 5000 sesuai permintaan
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)