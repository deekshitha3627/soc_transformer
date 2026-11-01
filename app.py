import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="SOC Console | Threat Detection", page_icon="🛡️", layout="wide")
ROOT = Path(__file__).resolve().parent
DATA, RESULTS = ROOT / "data", ROOT / "results"

st.markdown("""
<style>
.stApp{background:#081321;color:#e8eef8}
[data-testid="stSidebar"]{background:#0b1728;border-right:1px solid #263a53}
.hero{background:linear-gradient(110deg,#18324d,#12243a);border:1px solid #294866;border-radius:16px;padding:24px 28px;margin-bottom:25px}
.eyebrow,.section{color:#72b5ff;font-size:11px;font-weight:800;letter-spacing:2px}
.hero h1{color:#f4f7fc;font-size:34px;margin:7px 0}
.hero p,.muted{color:#aebfd4}
.agent{background:linear-gradient(145deg,#122238,#101d30);border:1px solid #2b405a;border-radius:14px;padding:20px;min-height:155px}
.agent h3{color:#f2f6fc}.agent p{color:#b5c5d9;line-height:1.5}
.tag{display:inline-block;color:#8cc7ff;border:1px solid #315a7e;border-radius:20px;padding:4px 10px;font-size:10px;font-weight:800}
.metric{background:#111f31;border:1px solid #2b405a;border-radius:13px;padding:15px;min-height:100px}
.metric-label{color:#9aabc0;font-size:12px}.metric-value{color:#f4f7fc;font-size:30px;font-weight:650}
.status{background:#0c2b29;color:#53d6a1;border-radius:9px;padding:15px;border:1px solid #145044}
</style>
""", unsafe_allow_html=True)

def jload(path):
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return None

def nload(path):
    try: return np.load(path, allow_pickle=True)
    except Exception: return None

def arr(data, keys):
    if data is None: return None
    for k in keys:
        if k in data.files: return np.asarray(data[k])
    return None

def nested_value(obj, keys):
    if not isinstance(obj, dict): return None
    norm = {str(k).lower().replace("-","_").replace(" ","_"):v for k,v in obj.items()}
    for k in keys:
        if k in norm: return norm[k]
    for v in obj.values():
        if isinstance(v,dict):
            got=nested_value(v,keys)
            if got is not None:return got
    return None

def report_metric(report, key):
    aliases={"accuracy":["accuracy","test_accuracy"],"precision":["precision","test_precision"],
             "recall":["recall","test_recall"],"f1":["f1","f1_score","macro_f1","weighted_f1","test_f1"]}
    v=nested_value(report,aliases[key])
    if v is None:return "—"
    try:
        v=float(v)
        if v<=1:v*=100
        return f"{v:.2f}%"
    except:return str(v)

def card(label,value):
    st.markdown(f'<div class="metric"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>',unsafe_allow_html=True)

def mapping():
    m=jload(DATA/"agent2_processed"/"class_mapping.json") or jload(DATA/"processed"/"class_mapping.json")
    if isinstance(m,dict):
        try:
            if all(str(k).isdigit() for k in m):return {int(k):str(v) for k,v in m.items()}
            return {int(v):str(k) for k,v in m.items()}
        except:pass
    return {0:"Botnet ARES",1:"Brute Force",2:"Dos/DDos",3:"Infiltration",4:"Normal",5:"PortScan",6:"Web Attack"}

def label(v, m):
    try:return m.get(int(v),str(v))
    except:return str(v)

with st.sidebar:
    st.markdown("## 🛡️ SOC CONSOLE")
    st.markdown('<div class="eyebrow">PROJECT WORKSPACE</div>',unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**CURRENT OBJECTIVE**")
    st.info("01 · Threat Detection")
    st.markdown("**DATA SOURCE**")
    st.caption("IDS2025")
    st.markdown("**DETECTION AGENTS**")
    st.markdown("🟦 FT-Transformer")
    st.markdown("🟩 Autoencoder")
    st.markdown("---")
    st.caption("NETWORK SECURITY MONITOR")

st.markdown('<div class="hero"><div class="eyebrow">SECURITY OPERATIONS CENTRE · NETWORK DEFENCE</div><h1>Explainable Multi-Agent AI for SOC Threat Detection</h1><p>IDS2025 · Network traffic monitoring · Independent detection agents</p></div>',unsafe_allow_html=True)
st.markdown("## Detection workspace")
st.caption("Displays saved project outputs. This dashboard does not retrain either agent.")

a1data=nload(DATA/"processed"/"dataset.npz")
a2data=nload(DATA/"agent2_processed"/"autoencoder_data.npz")
r1=jload(RESULTS/"agent1_performance.json") or jload(RESULTS/"transformer_performance.json")
r2=jload(RESULTS/"agent2_performance.json") or jload(RESULTS/"autoencoder_performance.json")
p1=nload(RESULTS/"transformer_predictions.npz")
p2=nload(RESULTS/"autoencoder_predictions.npz")
m=mapping()

# 01 Dataset overview: counts are calculated from the saved test split.
st.markdown('<div class="section">01 / DATASET OVERVIEW</div>',unsafe_allow_html=True)
# Prefer the complete Agent 1 split (train + validation + test) for dataset totals.
# Agent 1 labels are multiclass; class_mapping identifies the Normal class.
split_labels = []
split_features = []
if a1data is not None:
    for key in ["y_train", "y_val", "y_test"]:
        if key in a1data.files:
            split_labels.append(np.asarray(a1data[key]).reshape(-1))
    for key in ["X_train", "X_val", "X_test"]:
        if key in a1data.files:
            split_features.append(np.asarray(a1data[key]))
if split_labels:
    y = np.concatenate(split_labels)
    flows = len(y)
    features = split_features[0].shape[1] if split_features and split_features[0].ndim > 1 else None
    benign_label = next((k for k, v in m.items() if str(v).lower() == "normal"), None)
    benign = int(np.sum(y == benign_label)) if benign_label is not None else 0
    attacks = int(np.sum(y != benign_label)) if benign_label is not None else None
else:
    y = arr(a2data, ["y_test"]) if a2data is not None else None
    x = arr(a2data, ["X_test"]) if a2data is not None else None
    flows = len(y) if y is not None else None
    features = x.shape[1] if x is not None and x.ndim > 1 else None
    benign = int(np.sum(y == 0)) if y is not None else None
    attacks = int(np.sum(y != 0)) if y is not None else None
cols=st.columns(4)
with cols[0]:card("Usable network flows",f"{flows:,}" if flows is not None else "—")
with cols[1]:card("Numeric features",f"{features:,}" if features is not None else "—")
with cols[2]:card("Benign / normal traffic",f"{benign:,}" if benign is not None else "—")
with cols[3]:card("Attack traffic",f"{attacks:,}" if attacks is not None else "—")
left,right=st.columns(2)
with left:
    st.markdown("**Traffic distribution**")
    if benign is not None:st.dataframe(pd.DataFrame({"Traffic type":["Benign / Normal","Attack"],"Flows":[benign,attacks]}),hide_index=True,use_container_width=True)
    else:st.info("Processed dataset files are needed to show traffic counts.")
with right:
    st.markdown("**Dataset composition**")
    if benign is not None and flows:
        st.write(f"Attack share: {attacks/flows*100:.1f}%");st.progress(attacks/flows)
        st.write(f"Benign share: {benign/flows*100:.1f}%");st.progress(benign/flows)
    else:st.info("Composition appears when processed labels are available.")
st.markdown("**Threat intelligence · Attack types in dataset**")
# Use multiclass labels from Agent 1 test set for class distribution.
y_multi=np.concatenate([np.asarray(a1data[k]).reshape(-1) for k in ["y_train","y_val","y_test"] if a1data is not None and k in a1data.files]) if a1data is not None and any(k in a1data.files for k in ["y_train","y_val","y_test"]) else None
if y_multi is not None:
    counts=pd.Series(y_multi.reshape(-1)).value_counts().sort_index()
    rows=[{"Threat label in dataset":label(k,m),"Flows":int(v)} for k,v in counts.items() if label(k,m).lower()!="normal"]
    st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
else:st.info("Threat distribution appears when Agent 1 processed labels are available.")
st.caption("FT-Transformer predicts IDS2025 traffic categories. Autoencoder classifies traffic as Normal / Anomaly. Counts above use the complete processed dataset split (train + validation + test) when those arrays are available.")
st.markdown("---")

# 02 Detection agent overview
st.markdown('<div class="section">02 / DETECTION AGENT OVERVIEW</div>',unsafe_allow_html=True)
c1,c2=st.columns(2)
with c1:
    st.markdown('<div class="agent"><h3>01 · FT-Transformer</h3><p>Supervised machine learning · Multiclass threat classification</p><span class="tag">CLASSIFICATION AGENT</span><p>Learns from labelled network traffic to classify flows into IDS2025 traffic categories.</p></div>',unsafe_allow_html=True)
with c2:
    st.markdown('<div class="agent"><h3>02 · Autoencoder</h3><p>Unsupervised learning · Anomaly detection</p><span class="tag">ANOMALY AGENT</span><p>Learns the normal traffic profile and flags flows with higher reconstruction errors as anomalous.</p></div>',unsafe_allow_html=True)
st.markdown("---")

# 03 Individual outputs
st.markdown('<div class="section">03 / INDIVIDUAL DETECTION RESULTS</div>',unsafe_allow_html=True)
st.markdown("## Detection agent outputs")
yp1=arr(p1,["y_pred","predictions","y_predicted","predicted_labels"])
yp2=arr(p2,["y_pred","predictions","anomaly_predictions"])
c1,c2=st.columns(2)
with c1:
    st.markdown("**01 · FT-Transformer**")
    nattack=sum(label(v,m).lower()!="normal" for v in yp1) if yp1 is not None else None
    card("Attack predictions",f"{nattack:,}" if nattack is not None else "—")
with c2:
    st.markdown("**02 · Autoencoder**")
    nanom=int(np.sum(yp2.reshape(-1)==1)) if yp2 is not None else None
    card("Anomalous flows",f"{nanom:,}" if nanom is not None else "—")
st.markdown("**Multiclass classification · IDS2025 categories**")
st.markdown("**Unsupervised anomaly detection · Normal / Anomaly**")
st.markdown("**Sample network-flow predictions**")
# Construct a table only from arrays actually saved by the model scripts.
table=None
true1=arr(p1,["y_true","y_test","actual"]);conf=arr(p1,["confidence","confidences","max_probability"])
prob=arr(p1,["probabilities","y_prob"]);err=arr(p2,["reconstruction_errors","errors","anomaly_scores"])
n=min([len(z) for z in [yp1,yp2,true1,err] if z is not None and np.ndim(z)>0],default=0)
if n:
    table=pd.DataFrame({"Sample":np.arange(1,n+1)})
    if true1 is not None:table["Actual label"]=[label(v,m) for v in true1[:n]]
    if yp1 is not None:table["FT-Transformer prediction"]=[label(v,m) for v in yp1[:n]]
    if conf is not None:table["FT-Transformer confidence"]=conf[:n].reshape(-1)
    elif prob is not None and prob.ndim>1:table["FT-Transformer confidence"]=prob[:n].max(axis=1)
    if yp2 is not None:table["Autoencoder result"]=["Anomaly" if int(v)==1 else "Normal" for v in yp2[:n]]
    if err is not None:table["Reconstruction error"]=err[:n].reshape(-1)
if table is not None:
    st.dataframe(table.head(100),use_container_width=True,height=360)
    st.download_button("⬇ DOWNLOAD AGENT PREDICTIONS · CSV",table.to_csv(index=False).encode(),file_name="soc_agent_predictions.csv",mime="text/csv")
else:st.info("Prediction table appears after both agent prediction files are saved in the results folder.")
st.markdown("---")

# 04 Basic performance overview
st.markdown('<div class="section">04 / BASIC PERFORMANCE OVERVIEW</div>',unsafe_allow_html=True)
st.markdown("## Independent model evaluation")
rows=[]
for name,key in [("Accuracy","accuracy"),("Precision","precision"),("Recall","recall"),("F1-score","f1")]:
    rows.append({"Metric":name,"FT-Transformer":report_metric(r1,key),"Autoencoder":report_metric(r2,key)})
st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
st.caption("FT-Transformer performs multiclass classification. Autoencoder is trained on normal traffic; anomaly outputs are compared with labelled test data for evaluation.")
st.markdown("---")

# 05 Detection status
st.markdown('<div class="section">05 / DETECTION STATUS</div>',unsafe_allow_html=True)
if p1 is not None or p2 is not None:
    st.markdown('<div class="status">Detection outputs loaded from the project results folder.</div>',unsafe_allow_html=True)
else:
    st.warning("Prediction output files are not available yet. Add the generated files to the results folder.")
with st.expander("Expected project file locations"):
    st.code("data/processed/dataset.npz\ndata/agent2_processed/autoencoder_data.npz\nresults/transformer_predictions.npz\nresults/autoencoder_predictions.npz\nresults/agent1_performance.json\nresults/agent2_performance.json")
st.caption("SOC Console · IDS2025 · Independent detection agents")
