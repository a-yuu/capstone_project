"""
FINAL BORE-UP VERSION + INTERACTIVE INPUT
Target: ~83% accuracy + langsung bisa dipakai dokter/admin
Fitur baru: Setelah training → input interaktif → prediksi Poli + contoh ICD-10
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
import joblib
import os

print(" ULTRA BORE-UP FINAL + INTERACTIVE MODE")
print(" Target akurasi 83%+ | Setelah training langsung bisa prediksi real-time")
print("="*80)

# ================== LOAD DATA ==================
print("Loading data...")
df = pd.read_excel('data_diagnosa_filtered.xlsx', dtype=str)
kamus = pd.read_csv('icd10_poli_clean.csv', dtype=str)

df.columns = [c.strip() for c in df.columns]

# ================== EXPANDED MEDICAL DICTIONARY ==================
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

# ================== PREPROCESSING ==================
# ... (seluruh kode preprocessing, feature engineering, weighting, TF-IDF, LGBM, fusion, SVM)

# Numeric handling
for col in ['usia', 'pernafasan_x_per_menit', 'suhu_tubuh']:
    if col not in df.columns:
        df[col] = np.nan
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
    median_val = df[col].median(skipna=True)
    df[col].fillna(median_val if not np.isnan(median_val) else 0, inplace=True)

# Feature engineering
df['age_category'] = pd.cut(df['usia'].astype(float), bins=[-1,2,12,18,45,65,200],
                           labels=['infant','child','adolescent','adult','senior','geriatric']).astype(str)
df['fever_level'] = pd.cut(df['suhu_tubuh'].astype(float), bins=[-1,36.5,37.5,38.5,60],
                           labels=['hypothermia','normal','lowfever','highfever']).astype(str)
df['resp_level'] = pd.cut(df['pernafasan_x_per_menit'].astype(float), bins=[-1,12,20,30,200],
                          labels=['bradypnea','normal','tachypnea','severe']).astype(str)

df['critical_fever'] = (df['suhu_tubuh'].astype(float) > 38.5).astype(int)
df['critical_resp'] = (df['pernafasan_x_per_menit'].astype(float) > 30).astype(int)
df['pediatric_case'] = (df['usia'].astype(float) < 18).astype(int)
df['geriatric_case'] = (df['usia'].astype(float) > 65).astype(int)
df['adult_case'] = ((df['usia'].astype(float) >= 18) & (df['usia'].astype(float) <= 65)).astype(int)
df['vital_severity'] = (df['critical_fever']*2 + df['critical_resp']*2 + 
                       (df['suhu_tubuh'].astype(float)>37.5).astype(int) + 
                       (df['pernafasan_x_per_menit'].astype(float)>20).astype(int)).astype(int)
df['age_risk'] = (df['pediatric_case']*2 + df['geriatric_case']*3).astype(int)

if 'riwayat_penyakit' not in df.columns:
    df['riwayat_penyakit'] = ''
else:
    df['riwayat_penyakit'].fillna('', inplace=True)

# Mapping poli
map_poli = dict(zip(kamus['list_icd'], kamus['poli_name']))
df['poli'] = df.get('list_icd_10', df.get('list_icd', pd.Series(['']*len(df)))).map(map_poli)
df = df.dropna(subset=['poli']).copy()

# TOP 8 poli only
major_specialties = df['poli'].value_counts().nlargest(8).index.tolist()
df_focused = df[df['poli'].isin(major_specialties)].copy()

print(f"Fokus pada TOP {len(major_specialties)} poli terbanyak")

# Text enhancement & weighted text
df_focused['keluhan_enhanced'] = df_focused.get('keluhan', '').apply(enhance_medical_text)
df_focused['diagnosa_enhanced'] = df_focused.get('diagnosa', '').apply(enhance_medical_text)
df_focused['riwayat_enhanced'] = df_focused.get('riwayat_penyakit', '').apply(enhance_medical_text)

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

df_focused['weighted_text'] = df_focused.apply(create_weighted_text, axis=1)

# Encoding & split
le = LabelEncoder()
y = le.fit_transform(df_focused['poli'])
X_train, X_test, y_train, y_test = train_test_split(df_focused, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,5), min_df=2, max_df=0.92,
                        sublinear_tf=True, token_pattern=r'[a-zA-Z_]{2,}')
X_train_tfidf = tfidf.fit_transform(X_train['weighted_text'])
X_test_tfidf = tfidf.transform(X_test['weighted_text'])

selector = SelectKBest(chi2, k=min(12000, X_train_tfidf.shape[1]))
X_train_tfidf_selected = selector.fit_transform(X_train_tfidf, y_train)
X_test_tfidf_selected = selector.transform(X_test_tfidf)

# LightGBM numeric only
numerical_features = ['usia','pernafasan_x_per_menit','suhu_tubuh','critical_fever','critical_resp',
                      'pediatric_case','geriatric_case','vital_severity','age_risk','adult_case']
for col in numerical_features:
    if col not in X_train.columns: X_train[col] = 0
    if col not in X_test.columns: X_test[col] = 0

lgb_params = {
    'objective': 'multiclass', 'num_class': len(le.classes_), 'learning_rate': 0.08,
    'num_leaves': 127, 'max_depth': 10, 'min_child_samples': 10, 'feature_fraction': 0.95,
    'bagging_fraction': 0.95, 'bagging_freq': 3, 'reg_alpha': 0.03, 'reg_lambda': 0.03,
    'verbose': -1, 'seed': 42, 'force_col_wise': True
}
lgb_train = lgb.Dataset(X_train[numerical_features].astype(float), label=y_train)
lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=400)

emb_train = lgb_model.predict(X_train[numerical_features].astype(float))
emb_test = lgb_model.predict(X_test[numerical_features].astype(float))
scaler = StandardScaler()
emb_train_scaled = scaler.fit_transform(emb_train)
emb_test_scaled = scaler.transform(emb_test)

# Final fusion
X_train_final = hstack([csr_matrix(emb_train_scaled * 2.0), X_train_tfidf_selected * 1.2])
X_test_final = hstack([csr_matrix(emb_test_scaled * 2.0), X_test_tfidf_selected * 1.2])

# SVM
svm = SGDClassifier(loss='hinge', alpha=3e-5, max_iter=2500, tol=5e-5,
                    class_weight='balanced', random_state=42, n_jobs=-1)
svm.fit(X_train_final, y_train)

# Evaluation
y_pred = svm.predict(X_test_final)
acc = accuracy_score(y_test, y_pred)
print("\n" + "="*80)
print(f"FINAL AKURASI: {acc*100:.2f}%")
print("="*80)
print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

# Save all
#joblib.dump(svm, 'model_final_optimized.pkl')
#joblib.dump(tfidf, 'tfidf_final.pkl')
#joblib.dump(selector, 'selector_final.pkl')
#joblib.dump(le, 'encoder_final.pkl')
#joblib.dump(lgb_model, 'lgb_final.pkl')
#joblib.dump(scaler, 'scaler_final.pkl')

# ================== MAPPING POLI KE ICD-10 (REAL DATA) ==================
# Buat dictionary: poli -> list ICD yang paling sering muncul
print("Membangun mapping Poli -> ICD-10 dari data training...")
poli_to_icd = {}
for poli_name in major_specialties:
    # Ambil semua ICD-10 yang terkait dengan poli ini dari data asli
    poli_data = df[df['poli'] == poli_name]
    icd_col = poli_data.get('list_icd_10', poli_data.get('list_icd', pd.Series([])))
    icd_counts = icd_col.value_counts().head(5)  
    if len(icd_counts) > 0:
        poli_to_icd[poli_name] = icd_counts.index.tolist()
    else:
        poli_to_icd[poli_name] = ["Tidak ada data ICD-10"]

print(f"Mapping selesai untuk {len(poli_to_icd)} poli\n")

# ================== INTERACTIVE PREDICTION ==================
print("\n" + "MODE INTERAKTIF SIAP!".center(80))
print("Ketik data pasien baru (atau ketik 'exit' untuk keluar)\n")

def predict_poli():
    while True:
        print("-" * 60)
        try:
            usia = input("Usia (tahun)                  : ").strip()
            if usia.lower() == 'exit': break
            usia = float(usia)

            jk = input("Jenis Kelamin (L/P)            : ").strip().upper()
            if jk not in ['L','P']: raise ValueError("Harus L atau P")
            
            riwayat = input("Riwayat Penyakit (kosongkan jika tidak ada) : ").strip()
            
            keluhan = input("Keluhan Utama                  : ").strip()
            if not keluhan: raise ValueError("Keluhan wajib diisi")
            
            # Set default values untuk pernafasan dan suhu
            nafas = 20.0  # nilai normal default
            suhu = 36.5   # nilai normal default

            # Reload model setiap kali (biar aman jika file di-update)
            svm = joblib.load('model_final_optimized.pkl')
            tfidf = joblib.load('tfidf_final.pkl')
            selector = joblib.load('selector_final.pkl')
            le = joblib.load('encoder_final.pkl')
            lgb_model = joblib.load('lgb_final.pkl')
            scaler = joblib.load('scaler_final.pkl')

            # Buat dataframe input
            input_df = pd.DataFrame([{
                'usia': usia, 'jenis_kelamin': jk, 'riwayat_penyakit': riwayat,
                'keluhan': keluhan, 'diagnosa': keluhan,  # proxy diagnosa = keluhan
                'pernafasan_x_per_menit': nafas, 'suhu_tubuh': suhu
            }])

            # Feature engineering sama persis
            input116 = input_df.copy()
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

            input_df['keluhan_enhanced'] = input_df['keluhan'].apply(enhance_medical_text)
            input_df['diagnosa_enhanced'] = input_df['diagnosa'].apply(enhance_medical_text)
            input_df['riwayat_enhanced'] = input_df['riwayat_penyakit'].apply(enhance_medical_text)
            input_df['weighted_text'] = input_df.apply(create_weighted_text, axis=1)

            text_vec = tfidf.transform(input_df['weighted_text'])
            text_sel = selector.transform(text_vec)

            num_pred = lgb_model.predict(input_df[numerical_features].astype(float))
            num_scaled = scaler.transform(num_pred)

            final_vec = hstack([csr_matrix(num_scaled * 2.0), text_sel * 1.2])
            pred_class = svm.predict(final_vec)[0]
            poli_pred = le.inverse_transform([pred_class])[0]

            # Hilangkan suffix seperti (C), (A), (B) dari nama poli
            import re
            poli_display = re.sub(r'\s*\([A-Z]\)\s*$', '', poli_pred).strip()

            # VALIDASI: Jika umur >= 18 tahun dan prediksi poli anak, ganti ke poli dewasa
            if usia >= 18 and 'ANAK' in poli_pred.upper():
                # Ambil prediksi kedua atau fallback ke INTERNIS
                proba = svm.decision_function(final_vec)[0]
                sorted_indices = np.argsort(proba)[::-1]
                for idx in sorted_indices[1:]:  # Skip prediksi pertama (poli anak)
                    alternative_poli = le.inverse_transform([idx])[0]
                    if 'ANAK' not in alternative_poli.upper():
                        poli_pred = alternative_poli
                        poli_display = re.sub(r'\s*\([A-Z]\)\s*$', '', poli_pred).strip()
                        break
                else:
                    # Jika semua alternatif poli anak, paksa ke INTERNIS
                    for i, poli_name in enumerate(le.classes_):
                        if 'INTERNIS' in poli_name.upper() or 'PENYAKIT DALAM' in poli_name.upper():
                            poli_pred = poli_name
                            poli_display = re.sub(r'\s*\([A-Z]\)\s*$', '', poli_pred).strip()
                            break

            # Ambil ICD-10 yang relevan untuk poli ini
            icd_list = poli_to_icd.get(poli_pred, ["Data ICD-10 tidak tersedia"])
            
            # Coba prediksi ICD-10 yang paling cocok berdasarkan keluhan
            keluhan_lower = keluhan.lower()
            best_icd = icd_list[0]  # Default: ICD paling sering
            
            # Smart ICD selection berdasarkan keyword matching
            for icd in icd_list:
                icd_str = str(icd).lower()
                # Cari ICD yang paling match dengan keluhan
                # Ini bisa diperbaiki dengan model ML tersendiri jika perlu
                if any(kw in keluhan_lower for kw in ['demam', 'fever', 'panas']) and 'r50' in icd_str:
                    best_icd = icd
                    break
                elif any(kw in keluhan_lower for kw in ['batuk', 'cough']) and ('j' in icd_str[:1] or 'r05' in icd_str):
                    best_icd = icd
                    break
                elif any(kw in keluhan_lower for kw in ['nyeri', 'pain', 'sakit']) and 'r' in icd_str[:1]:
                    best_icd = icd
                    break
                elif any(kw in keluhan_lower for kw in ['diabetes', 'gula']) and 'e' in icd_str[:1]:
                    best_icd = icd
                    break
                elif any(kw in keluhan_lower for kw in ['jantung', 'cardiac', 'chest']) and 'i' in icd_str[:1]:
                    best_icd = icd
                    break
            
            print("\n" + "="*60)
            print(f"REKOMENDASI POLI       : {poli_display.upper()}")
            print(f"KODE ICD-10 PREDIKSI   : {best_icd}")
            print(f"ALTERNATIF ICD-10      : {', '.join(map(str, icd_list[:3]))}")
            print(f"AKURASI MODEL (test)   : {acc*100:.2f}%")
            print("="*60)

        except Exception as e:
            print(f"Error: {e}. Coba lagi atau ketik 'exit'")

predict_poli()
print("\nTerima kasih! Model tetap aktif untuk prediksi kapan saja.")