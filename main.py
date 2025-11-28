# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import re
from scipy.sparse import hstack, csr_matrix

# ==========================================
# 1. INITIALIZATION & LOAD MODELS
# ==========================================
app = FastAPI()

print("Loading Models...")
try:
    svm_model = joblib.load('model_final_optimized.pkl')
    tfidf = joblib.load('tfidf_final.pkl')
    selector = joblib.load('selector_final.pkl')
    le = joblib.load('encoder_final.pkl')
    lgb_model = joblib.load('lgb_final.pkl')
    scaler = joblib.load('scaler_final.pkl')
    print("Semua model berhasil dimuat!")
except Exception as e:
    print(f"CRITICAL ERROR: Gagal memuat model. Pastikan file .pkl ada. {e}")

# ==========================================
# 2. HELPER FUNCTIONS (Copy dari Training Script)
# ==========================================
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

def enhance_medical_text(text):
    if pd.isna(text) or text is None:
        return ''
    text = str(text).lower()
    for term, expansion in medical_expansions.items():
        if term in text:
            text += " " + expansion
    stop_small = {'dan','atau','yang','ini','itu','ke','di','dgn','dgnnya'}
    tokens = [w for w in text.split() if len(w) >= 2 and w not in stop_small]
    return ' '.join(tokens)

def create_weighted_text(row):
    diagnosa_text = str(row.get('diagnosa_enhanced', ''))
    keluhan_text = str(row.get('keluhan_enhanced', ''))
    riwayat_text = str(row.get('riwayat_enhanced', ''))
    
    text = (diagnosa_text + " ") * 15
    text += (riwayat_text + " ") * 8
    text += (keluhan_text + " ") * 5
    
    demo_text = f"{row.get('age_category','')} {row.get('jenis_kelamin','')} {row.get('fever_level','')} {row.get('resp_level','')}"
    text += (demo_text + " ") * 3
    
    if row.get('vital_severity', 0) >= 4:
        text += "CRITICAL_CASE HIGH_ACUITY EMERGENCY URGENT_CARE SEVERE_CONDITION " * 2
    if row.get('pediatric_case', 0) == 1:
        text += "PEDIATRIC_SPECIALTY CHILD_MEDICINE ANAK_CARE PEDIATRI " * 2
    if row.get('geriatric_case', 0) == 1:
        text += "GERIATRIC_SPECIALTY ELDERLY_CARE LANSIA_CARE " * 2
        
    return text.replace('nan','').strip()

# ==========================================
# 3. REQUEST SCHEMA
# ==========================================
class PatientRequest(BaseModel):
    keluhan: str
    usia: int = 30  # Default value jika tidak dikirim
    jenis_kelamin: str = "L" # Default L
    riwayat_penyakit: str = ""

# ==========================================
# 4. PREDICTION ENDPOINT
# ==========================================
@app.post("/predict")
def predict_poli(data: PatientRequest):
    try:
        # 1. Siapkan DataFrame dari Input
        # Default nilai vital sign (karena user belum diperiksa)
        nafas = 20.0
        suhu = 36.5
        
        input_df = pd.DataFrame([{
            'usia': data.usia,
            'jenis_kelamin': data.jenis_kelamin,
            'riwayat_penyakit': data.riwayat_penyakit,
            'keluhan': data.keluhan,
            'diagnosa': data.keluhan, # Proxy: anggap keluhan sebagai diagnosa awal
            'pernafasan_x_per_menit': nafas,
            'suhu_tubuh': suhu
        }])

        # 2. Feature Engineering (Harus PERSIS sama dengan training)
        input_df['age_category'] = pd.cut(input_df['usia'], bins=[-1,2,12,18,45,65,200],
                                        labels=['infant','child','adolescent','adult','senior','geriatric']).astype(str)
        input_df['fever_level'] = pd.cut(input_df['suhu_tubuh'], bins=[-1,36.5,37.5,38.5,60],
                                        labels=['hypothermia','normal','lowfever','highfever']).astype(str)
        input_df['resp_level'] = pd.cut(input_df['pernafasan_x_per_menit'], bins=[-1,12,20,30,200],
                                       labels=['bradypnea','normal','tachypnea','severe']).astype(str)
        
        input_df['critical_fever'] = (input_df['suhu_tubuh'] > 38.5).astype(int)
        input_df['critical_resp'] = (input_df['pernafasan_x_per_menit'] > 30).astype(int)
        input_df['pediatric_case'] = (input_df['usia'] < 18).astype(int)
        input_df['geriatric_case'] = (input_df['usia'] > 65).astype(int)
        input_df['adult_case'] = ((input_df['usia'] >= 18) & (input_df['usia'] <= 65)).astype(int)
        
        input_df['vital_severity'] = (input_df['critical_fever']*2 + input_df['critical_resp']*2 +
                                     (input_df['suhu_tubuh']>37.5).astype(int) + 
                                     (input_df['pernafasan_x_per_menit']>20).astype(int)).astype(int)
        input_df['age_risk'] = (input_df['pediatric_case']*2 + input_df['geriatric_case']*3).astype(int)

        # 3. Text Processing
        input_df['keluhan_enhanced'] = input_df['keluhan'].apply(enhance_medical_text)
        input_df['diagnosa_enhanced'] = input_df['diagnosa'].apply(enhance_medical_text)
        input_df['riwayat_enhanced'] = input_df['riwayat_penyakit'].apply(enhance_medical_text)
        input_df['weighted_text'] = input_df.apply(create_weighted_text, axis=1)

        # 4. Vectorization (TF-IDF)
        text_vec = tfidf.transform(input_df['weighted_text'])
        text_sel = selector.transform(text_vec)

        # 5. Numeric Prediction (LightGBM)
        numerical_features = ['usia','pernafasan_x_per_menit','suhu_tubuh','critical_fever','critical_resp',
                              'pediatric_case','geriatric_case','vital_severity','age_risk','adult_case']
        
        # Pastikan kolom numerik lengkap
        for col in numerical_features:
            if col not in input_df.columns: input_df[col] = 0
            
        num_pred = lgb_model.predict(input_df[numerical_features].astype(float))
        num_scaled = scaler.transform(num_pred)

        # 6. Fusion (Gabung output LGBM + TF-IDF)
        final_vec = hstack([csr_matrix(num_scaled * 2.0), text_sel * 1.2])

        # 7. Final Prediction (SVM)
        pred_class = svm_model.predict(final_vec)[0]
        poli_pred = le.inverse_transform([pred_class])[0]

        # 8. Post-Processing (Validasi Umur & Nama Poli)
        # Bersihkan nama poli dari suffix seperti (C), (A)
        poli_display = re.sub(r'\s*\([A-Z]\)\s*$', '', poli_pred).strip()

        # Logic: Jika dewasa tapi masuk poli anak -> paksa ganti
        if data.usia >= 18 and 'ANAK' in poli_pred.upper():
            # Cek probabilitas tertinggi kedua
            proba = svm_model.decision_function(final_vec)[0]
            sorted_indices = np.argsort(proba)[::-1]
            
            found_alt = False
            for idx in sorted_indices[1:]:
                alt_poli = le.inverse_transform([idx])[0]
                if 'ANAK' not in alt_poli.upper():
                    poli_display = re.sub(r'\s*\([A-Z]\)\s*$', '', alt_poli).strip()
                    found_alt = True
                    break
            
            if not found_alt:
                poli_display = "POLI PENYAKIT DALAM" # Fallback

        return {
            "status": "success",
            "rekomendasi_poli": poli_display.upper(),
            "original_class": poli_pred
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# Run dengan: uvicorn main:app --reload --port 5000