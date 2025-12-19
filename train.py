import pandas as pd
import numpy as np
import joblib
import re
import os
import pathlib
import gc
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import accuracy_score
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ==========================================
# 1. KONFIGURASI PATH ABSOLUT
# ==========================================
# Ini memastikan script membaca file di folder script berada
BASE_DIR = pathlib.Path(__file__).parent.absolute()
print(f"Working Directory: {BASE_DIR}")

# ==========================================
# 2. LOAD DICTIONARIES & HELPER FUNCTIONS
# ==========================================
print("[1/7] Loading Dictionaries...")

def load_dictionary(file_name, key_col, value_col):
    path = os.path.join(BASE_DIR, file_name)
    if not os.path.exists(path): return {}
    df = pd.read_csv(path, dtype=str).fillna('')
    return dict(zip(df[key_col].str.lower(), df[value_col].str.lower()))

# Load Dictionary Files
try:
    KAMUS_SINGKATAN = load_dictionary('kamus_singkatan.csv', 'Singkatan', 'Bentuk Lengkap')
    kamus_medis_path = os.path.join(BASE_DIR, 'kamus_medis.csv')
    if os.path.exists(kamus_medis_path):
        kamus_medis_df = pd.read_csv(kamus_medis_path, dtype=str).fillna('')
        KAMUS_MEDIS_FULL = dict(zip(kamus_medis_df['istilah_lama'].str.lower(), kamus_medis_df['istilah_standar'].str.lower()))
        KAMUS_MEDIS_PHRASE = {k: v for k, v in KAMUS_MEDIS_FULL.items() if " " in k}
        KAMUS_MEDIS_TOKEN  = {k: v for k, v in KAMUS_MEDIS_FULL.items() if " " not in k}
    else:
        print("WARNING: kamus_medis.csv tidak ditemukan! Text processing mungkin kurang optimal.")
        KAMUS_MEDIS_FULL, KAMUS_MEDIS_PHRASE, KAMUS_MEDIS_TOKEN = {}, {}, {}
        
    KAMUS_KEYWORD_TRIAGE = load_dictionary('kamus_keyword_poli.csv', 'Keyword Keluhan', 'Poli Tujuan')
except Exception as e:
    print(f"Warning: Gagal load kamus. Error: {e}")

# Medical Expansions
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

# --- Functions (Sama Persis dengan Main.py) ---

def remove_corrupt_patterns(text):
    text = str(text)
    text = re.sub(r'_x000D_', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'[;,]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def clean_text_with_dictionaries(text):
    if not text or pd.isna(text): return ""
    text = remove_corrupt_patterns(text).lower()
    text = re.sub(r'\brspb\b', '', text)
    text = re.sub(r'\bkeluhan\b', '', text)
    text = re.sub(r'[,:;]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 1. Phrase Replacement
    for old, new in KAMUS_MEDIS_PHRASE.items():
        if old in text: text = text.replace(old, new)
    
    # 2. Token Replacement
    tokens = text.split()
    tokens = [KAMUS_SINGKATAN.get(t, t) for t in tokens]
    tokens = [KAMUS_MEDIS_TOKEN.get(t, t) for t in tokens]
    return " ".join(tokens)

def enhance_medical_text(text):
    if not text or pd.isna(text): return ""
    text = str(text).lower()
    text = clean_text_with_dictionaries(text)
    for term, expansion in medical_expansions.items():
        if term in text: text += " " + expansion
    stop_small = {'dan','atau','yang','ini','itu','ke','di','dgn','dgnnya'}
    tokens = [w for w in text.split() if len(w) >= 2 and w not in stop_small]
    return ' '.join(tokens)

def extract_core_keluhan(text):
    if not text or pd.isna(text): return ""
    s = str(text).lower().replace('\n', ' ')
    s = re.sub(r'\s+', ' ', s).strip()
    m = re.search(r'\bkel(?:uhan)?\b[^a-z0-9]*[:;\-]*\s*(.*)', s)
    if m: core = m.group(1).strip()
    else: core = s
    return re.sub(r'^[\-\.,:]+$', '', core).strip()

def find_poli_from_keyword(keluhan):
    if not keluhan or pd.isna(keluhan): return ''
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

def create_weighted_text(row):
    # Urutan ini HARUS sama dengan main.py
    text = (str(row.get('diagnosa_enhanced', '')) + " ") * 2
    text += (str(row.get('riwayat_enhanced', '')) + " ") * 3
    text += (str(row.get('keluhan_enhanced', '')) + " ")
    
    poli_triage = str(row.get('poli_triage', ''))
    if poli_triage:
        text += " " + poli_triage.replace(' ', '_').upper() * 5
        
    demo_text = f"{row.get('age_category','')} {row.get('jenis_kelamin','')} {row.get('fever_level','')} {row.get('resp_level','')}"
    text += " " + (demo_text + " ") * 3
    
    if row.get('vital_severity', 0) >= 4: text += " CRITICAL_CASE HIGH_ACUITY EMERGENCY URGENT_CARE SEVERE_CONDITION " * 2
    if row.get('pediatric_case', 0) == 1: text += " PEDIATRIC_SPECIALTY CHILD_MEDICINE ANAK_CARE PEDIATRI " * 2
    if row.get('geriatric_case', 0) == 1: text += " GERIATRIC_SPECIALTY ELDERLY_CARE LANSIA_CARE " * 2
    return text.replace('nan','').strip()

# ==========================================
# 3. DATA LOADING & PROCESSING
# ==========================================
print("[2/7] Loading & Preprocessing Data...")

try:
    df_path = os.path.join(BASE_DIR, 'data_diagnosa_filtered.xlsx')
    if not os.path.exists(df_path):
        raise FileNotFoundError(f"File excel tidak ditemukan di: {df_path}")
    
    df = pd.read_excel(df_path, dtype=str, engine='openpyxl')
    kamus_poli_path = os.path.join(BASE_DIR, 'icd10_poli_clean.csv')
    kamus = pd.read_csv(kamus_poli_path, dtype=str)
    
    df.columns = [c.strip() for c in df.columns]
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    exit()

# Filter Kontrol
if 'keluhan' in df.columns:
    mask_kontrol = df['keluhan'].astype(str).str.lower().str.contains(r'\bkontrol\b|\bcontrol\b|\bfollow\s*up\b|\bmcu\b', regex=True)
    df = df[~mask_kontrol].copy()

# Feature Engineering (Numeric)
numeric_cols = ['usia', 'pernafasan_x_per_menit', 'suhu_tubuh']
for col in numeric_cols:
    if col not in df.columns: df[col] = np.nan
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True), errors='coerce')
    df[col].fillna(df[col].median(skipna=True), inplace=True)

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

# Label Processing
map_poli = dict(zip(kamus['list_icd'], kamus['poli_name']))
df['poli'] = df['list_icd_10'].map(map_poli).apply(clean_poli_label)
df = df.dropna(subset=['poli']).copy()

top_poli = df['poli'].value_counts().head(8).index.tolist()
df = df[df['poli'].isin(top_poli)].copy()

# Text Processing
print("    > Enriching Text Features...")
df['keluhan_core'] = df.get('keluhan', '').apply(extract_core_keluhan)
df = df[df['keluhan_core'].astype(str).str.strip() != ''].copy()

df['keluhan_enhanced'] = df['keluhan_core'].apply(enhance_medical_text)
df['diagnosa_enhanced'] = df.get('diagnosa', '').apply(enhance_medical_text)
df['riwayat_enhanced'] = df.get('riwayat_penyakit', '').astype(str).apply(enhance_medical_text)
df['poli_triage'] = df['keluhan_core'].apply(find_poli_from_keyword)
df['weighted_text'] = df.apply(create_weighted_text, axis=1)

# ==========================================
# 4. TRAINING PIPELINE
# ==========================================
print("[3/7] Splitting Data...")
le = LabelEncoder()
y = le.fit_transform(df['poli'])
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42, stratify=y)

del df
gc.collect()

print("[4/7] Training TF-IDF & Selector...")
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1,5), min_df=2, max_df=0.92, sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train['weighted_text'])
X_test_tfidf = tfidf.transform(X_test['weighted_text'])

selector = SelectKBest(chi2, k=min(6000, X_train_tfidf.shape[1]))
X_train_tfidf_selected = selector.fit_transform(X_train_tfidf, y_train)
X_test_tfidf_selected = selector.transform(X_test_tfidf)

del X_train_tfidf, X_test_tfidf
gc.collect()

print("[5/7] Training LightGBM (Numeric Embedding)...")
numerical_features = ['usia','pernafasan_x_per_menit','suhu_tubuh','critical_fever','critical_resp','pediatric_case','geriatric_case','vital_severity','age_risk','adult_case']

# Pastikan kolom numeric ada
for col in numerical_features:
    if col not in X_train.columns: X_train[col] = 0
    if col not in X_test.columns: X_test[col] = 0

lgb_params = {
    'objective': 'multiclass', 'num_class': len(le.classes_), 
    'learning_rate': 0.08, 'num_leaves': 127, 'max_depth': 10, 
    'verbose': -1, 'seed': 42
}
lgb_train_data = lgb.Dataset(X_train[numerical_features].astype(float), label=y_train)
lgb_model = lgb.train(lgb_params, lgb_train_data, num_boost_round=300)

emb_train = lgb_model.predict(X_train[numerical_features].astype(float))
emb_test = lgb_model.predict(X_test[numerical_features].astype(float))

scaler = StandardScaler()
emb_train_scaled = scaler.fit_transform(emb_train)
emb_test_scaled = scaler.transform(emb_test)

print("[6/7] Fusion & Training Final Model (SGD)...")
# Casting float32 untuk hemat memori
X_train_final = hstack([csr_matrix(emb_train_scaled.astype(np.float32) * 2.0), X_train_tfidf_selected.astype(np.float32) * 1.2])
X_test_final = hstack([csr_matrix(emb_test_scaled.astype(np.float32) * 2.0), X_test_tfidf_selected.astype(np.float32) * 1.2])

svm = SGDClassifier(loss='hinge', alpha=3e-5, max_iter=2000, tol=1e-4, class_weight='balanced', random_state=42, n_jobs=-1)
svm.fit(X_train_final, y_train)

y_pred = svm.predict(X_test_final)
print(f"    > Akurasi Test: {accuracy_score(y_test, y_pred)*100:.2f}%")

# ==========================================
# 5. SAVING MODELS (FILENAME MATCHING MAIN.PY)
# ==========================================
print("[7/7] Saving Models to Disk...")

def save_model(obj, name):
    path = os.path.join(BASE_DIR, name)
    joblib.dump(obj, path)
    print(f"    Saved: {name}")

save_model(svm, 'best_poli_model.joblib')           # PENTING: Nama file harus ini
save_model(le, 'label_encoder.joblib')
save_model(tfidf, 'tfidf_vectorizer.joblib')
save_model(selector, 'chi2_selector.joblib')
save_model(lgb_model, 'lgb_embedding_model.joblib')
save_model(scaler, 'lgb_scaler.joblib')
save_model(numerical_features, 'numerical_features.joblib')

print("="*60)
print("SUKSES! Semua file model siap digunakan oleh main.py")
print("Silakan jalankan: python main.py")
print("="*60)