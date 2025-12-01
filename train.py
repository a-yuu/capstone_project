import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import accuracy_score, classification_report
import lightgbm as lgb
import joblib
import re
import os
import gc  # Garbage Collector untuk hemat RAM
import warnings

warnings.filterwarnings('ignore')

print("="*50)
print("MEMULAI PROSES TRAINING MODEL AI (OPTIMIZED)")
print("="*50)

# ==========================================
# 1. HELPER FUNCTIONS & DICTIONARIES
# ==========================================
try:
    kamus_singkatan = pd.read_csv('kamus_singkatan.csv', dtype=str).fillna('')
    KAMUS_SINGKATAN = dict(zip(kamus_singkatan['Singkatan'].str.lower(), kamus_singkatan['Bentuk Lengkap'].str.lower()))
    
    kamus_medis = pd.read_csv('kamus_medis.csv', dtype=str).fillna('')
    KAMUS_MEDIS_FULL = dict(zip(kamus_medis['istilah_lama'].str.lower(), kamus_medis['istilah_standar'].str.lower()))
    
    # Pisahkan phrase dan token (Logic baru Cell 5)
    KAMUS_MEDIS_PHRASE = {k: v for k, v in KAMUS_MEDIS_FULL.items() if " " in k}
    KAMUS_MEDIS_TOKEN  = {k: v for k, v in KAMUS_MEDIS_FULL.items() if " " not in k}
    
    kamus_triage = pd.read_csv('kamus_keyword_poli.csv', dtype=str).fillna('')
    KAMUS_KEYWORD_TRIAGE = dict(zip(kamus_triage['Keyword Keluhan'].str.lower(), kamus_triage['Poli Tujuan'].str.lower()))
except:
    print("Warning: File kamus tidak lengkap.")
    KAMUS_SINGKATAN = {}
    KAMUS_MEDIS_PHRASE = {}
    KAMUS_MEDIS_TOKEN = {}
    KAMUS_KEYWORD_TRIAGE = {}

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

def extract_core_keluhan(text):
    """
    Fungsi baru dari Cell 5: Mengambil inti keluhan, membuang 'KU: Sedang' dll.
    """
    if pd.isna(text): return ""
    s = str(text).lower().replace('\n', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    
    # Cari pola 'kel' atau 'keluhan'
    m = re.search(r'\bkel(?:uhan)?\b[^a-z0-9]*[:;\-]*\s*(.*)', s)
    if m:
        core = m.group(1).strip()
    else:
        core = s
        
    core = core.strip()
    core = re.sub(r'^[\-\.,:]+$', '', core).strip()
    return core

def remove_corrupt_patterns(text):
    text = str(text)
    text = re.sub(r'_x000D_', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'x000d', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\\x00', ' ', text)
    text = re.sub(r'\\u000d', ' ', text)
    text = re.sub(r'[;,]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def clean_text_with_dictionaries(text):
    """Update logika cleaning sesuai Cell 5"""
    if pd.isna(text): return ""
    text = remove_corrupt_patterns(text).lower()
    text = re.sub(r'\brspb\b', '', text)
    text = re.sub(r'\bkeluhan\b', '', text)
    text = re.sub(r'[,:;]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 1. Ganti Phrase (Multi-kata) dulu
    for old, new in KAMUS_MEDIS_PHRASE.items():
        if old in text: text = text.replace(old, new)
    
    # 2. Ganti Token (Per-kata)
    tokens = text.split()
    tokens = [KAMUS_SINGKATAN.get(t, t) for t in tokens]
    tokens = [KAMUS_MEDIS_TOKEN.get(t, t) for t in tokens]
    return " ".join(tokens)

def enhance_medical_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = clean_text_with_dictionaries(text)
    for term, expansion in medical_expansions.items():
        if term in text: text += " " + expansion
    stop_small = {'dan','atau','yang','ini','itu','ke','di','dgn','dgnnya'}
    tokens = [w for w in text.split() if len(w) >= 2 and w not in stop_small]
    return ' '.join(tokens)

def find_poli_from_keyword(keluhan):
    """Versi Dictionary untuk Training (Cell 5)"""
    if not keluhan: return ''
    keluhan = str(keluhan).lower()
    for keyword, poli in KAMUS_KEYWORD_TRIAGE.items():
        if keyword in keluhan: return poli.strip()
    return ''

def clean_poli_label(poli):
    if pd.isna(poli): return np.nan
    text = str(poli)
    text = re.sub(r'\bRSPB\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*\([A-Z]\)\s*$', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if 'KLINIK GIGI' in text.upper(): return 'KLINIK GIGI & MULUT'
    if 'INTERNIS' in text.upper(): return 'KLINIK PENYAKIT DALAM'
    if 'NEUROLOGI' in text.upper(): return 'KLINIK NEUROLOGI'
    if 'THT' in text.upper(): return 'KLINIK THT'
    return text.upper()

# ==========================================
# 2. LOAD & PREPROCESS DATA
# ==========================================
print("[1/8] Loading Dataset...")
try:
    df = pd.read_excel('data_diagnosa_filtered.xlsx', dtype=str, engine='openpyxl')
    kamus = pd.read_csv('icd10_poli_clean.csv', dtype=str)
    df.columns = [c.strip() for c in df.columns]
except FileNotFoundError as e:
    print(f"ERROR: File data tidak ditemukan! {e}")
    exit()

print("[2/8] Cleaning & Filtering...")
if 'keluhan' in df.columns:
    mask_kontrol = df['keluhan'].astype(str).str.lower().str.contains(r'\bkontrol\b|\bcontrol\b|\bfollow\s*up\b|\bmcu\b', regex=True)
    df = df[~mask_kontrol].copy()

map_poli = dict(zip(kamus['list_icd'], kamus['poli_name']))
df['poli'] = df.get('list_icd_10', df.get('list_icd', pd.Series(['']*len(df)))).map(map_poli)
df['poli'] = df['poli'].apply(clean_poli_label)
df = df.dropna(subset=['poli']).copy()

# Filter Top Poli
top_poli = df['poli'].value_counts().head(8).index.tolist() # Mengikuti notebook: 8 poli
df = df[df['poli'].isin(top_poli)].copy()

# Feature Engineering
for col in ['usia', 'pernafasan_x_per_menit', 'suhu_tubuh']:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
    median_val = df[col].median(skipna=True)
    df[col].fillna(median_val if not np.isnan(median_val) else 0, inplace=True)

df['age_category'] = pd.cut(df['usia'], bins=[-1,2,12,18,45,65,200], labels=['infant','child','adolescent','adult','senior','geriatric']).astype(str)
df['fever_level'] = pd.cut(df['suhu_tubuh'], bins=[-1,36.5,37.5,38.5,60], labels=['hypothermia','normal','lowfever','highfever']).astype(str)
df['resp_level'] = pd.cut(df['pernafasan_x_per_menit'], bins=[-1,12,20,30,200], labels=['bradypnea','normal','tachypnea','severe']).astype(str)

df['critical_fever'] = (df['suhu_tubuh'] > 38.5).astype(int)
df['critical_resp'] = (df['pernafasan_x_per_menit'] > 30).astype(int)
df['pediatric_case'] = (df['usia'] < 18).astype(int)
df['geriatric_case'] = (df['usia'] > 65).astype(int)
df['adult_case'] = ((df['usia'] >= 18) & (df['usia'] <= 65)).astype(int)
df['vital_severity'] = (df['critical_fever']*2 + df['critical_resp']*2 + (df['suhu_tubuh']>37.5).astype(int) + (df['pernafasan_x_per_menit']>20).astype(int)).astype(int)
df['age_risk'] = (df['pediatric_case']*2 + df['geriatric_case']*3).astype(int)

# Text Enrichment (Sesuai Cell 8)
df['keluhan_core'] = df.get('keluhan', '').apply(extract_core_keluhan) # NEW STEP
df = df[df['keluhan_core'].astype(str).str.strip() != ''].copy() # Drop empty core

df['keluhan_enhanced'] = df['keluhan_core'].apply(enhance_medical_text) # Use core
df['diagnosa_enhanced'] = df.get('diagnosa', '').apply(enhance_medical_text)
df['riwayat_enhanced'] = df.get('riwayat_penyakit', '').astype(str).apply(enhance_medical_text)
df['poli_triage'] = df['keluhan_core'].apply(find_poli_from_keyword) # Use core

def create_weighted_text(row):
    text = (str(row['diagnosa_enhanced']) + " ") * 2
    text += (str(row['riwayat_enhanced']) + " ") * 3
    text += (str(row['keluhan_enhanced']) + " ")
    if str(row['poli_triage']):
        text += " " + str(row['poli_triage']).replace(' ', '_').upper() * 5
    demo_text = f"{row['age_category']} {row['jenis_kelamin']} {row['fever_level']} {row['resp_level']}"
    text += " " + (demo_text + " ") * 3
    if row['vital_severity'] >= 4: text += " CRITICAL_CASE HIGH_ACUITY " * 2
    if row['pediatric_case'] == 1: text += " PEDIATRIC_SPECIALTY CHILD_MEDICINE " * 2
    if row['geriatric_case'] == 1: text += " GERIATRIC_SPECIALTY ELDERLY_CARE " * 2
    return text.replace('nan','').strip()

df['weighted_text'] = df.apply(create_weighted_text, axis=1)

# ==========================================
# 3. TRAINING & SAVING (MEMORY OPTIMIZED)
# ==========================================
print("[3/8] Encoding & Splitting...")
le = LabelEncoder()
y = le.fit_transform(df['poli'])
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)

# [MEMORY FIX] Hapus df besar
del df
gc.collect()

print("[4/8] Vectorizing Text (TF-IDF)...")
# [MEMORY FIX] Kurangi max_features jadi 10000 agar muat di RAM
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,5), min_df=2, max_df=0.92, sublinear_tf=True, token_pattern=r'[a-zA-Z_]{2,}')
X_train_tfidf = tfidf.fit_transform(X_train['weighted_text'])
X_test_tfidf = tfidf.transform(X_test['weighted_text'])

# [MEMORY FIX] Kurangi k SelectKBest
selector = SelectKBest(chi2, k=min(8000, X_train_tfidf.shape[1]))
X_train_tfidf_selected = selector.fit_transform(X_train_tfidf, y_train)
X_test_tfidf_selected = selector.transform(X_test_tfidf)

# [MEMORY FIX] Hapus variable tfidf mentah
del X_train_tfidf, X_test_tfidf
gc.collect()

print("[5/8] Training LightGBM (Numeric)...")
numerical_features = ['usia','pernafasan_x_per_menit','suhu_tubuh','critical_fever','critical_resp','pediatric_case','geriatric_case','vital_severity','age_risk','adult_case']
lgb_params = {'objective': 'multiclass', 'num_class': len(le.classes_), 'learning_rate': 0.08, 'num_leaves': 127, 'max_depth': 10, 'verbose': -1, 'seed': 42}
lgb_train_data = lgb.Dataset(X_train[numerical_features].astype(float), label=y_train)
lgb_model = lgb.train(lgb_params, lgb_train_data, num_boost_round=400)

emb_train = lgb_model.predict(X_train[numerical_features].astype(float))
emb_test = lgb_model.predict(X_test[numerical_features].astype(float))

scaler = StandardScaler()
emb_train_scaled = scaler.fit_transform(emb_train)
emb_test_scaled = scaler.transform(emb_test)

print("[6/8] Fusion & Training SVM...")
# [MEMORY FIX] Konversi ke float32 untuk hemat 50% memori sebelum hstack
emb_train_scaled = emb_train_scaled.astype(np.float32)
emb_test_scaled = emb_test_scaled.astype(np.float32)
X_train_tfidf_selected = X_train_tfidf_selected.astype(np.float32)
X_test_tfidf_selected = X_test_tfidf_selected.astype(np.float32)

X_train_final = hstack([csr_matrix(emb_train_scaled * 2.0), X_train_tfidf_selected * 1.2])
X_test_final = hstack([csr_matrix(emb_test_scaled * 2.0), X_test_tfidf_selected * 1.2])

# [MEMORY FIX] Clean up
del emb_train_scaled, X_train_tfidf_selected, emb_test_scaled, X_test_tfidf_selected
gc.collect()

svm = SGDClassifier(loss='hinge', alpha=3e-5, max_iter=2500, tol=5e-5, class_weight='balanced', random_state=42, n_jobs=-1)
svm.fit(X_train_final, y_train)

print("[7/8] Evaluasi Model...")
y_pred = svm.predict(X_test_final)
acc = accuracy_score(y_test, y_pred)
print(f"   Akurasi Final: {acc*100:.2f}%")

print("[8/8] Menyimpan Model (.pkl)...")
joblib.dump(svm, 'model_final_optimized.pkl')
joblib.dump(tfidf, 'tfidf_final.pkl')
joblib.dump(selector, 'selector_final.pkl')
joblib.dump(le, 'encoder_final.pkl')
joblib.dump(lgb_model, 'lgb_final.pkl')
joblib.dump(scaler, 'scaler_final.pkl')

print("SELESAI! Model disimpan.")