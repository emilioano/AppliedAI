"""
Telco Churn Prediction Demo
============================
Interaktiv app för att visa logistisk regression-modellens prediktioner.
Körs med: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------- Page config ----------
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📞",
    layout="wide"
)

# ---------- Ladda modell ----------
@st.cache_resource
def load_artifacts():
    with open('churn_artifacts.pkl', 'rb') as f:
        return pickle.load(f)

artifacts = load_artifacts()
model = artifacts['model']
scaler = artifacts['scaler']
feature_columns = artifacts['feature_columns']
numeric_cols = artifacts['numeric_cols']
optimal_threshold = artifacts['optimal_threshold']
top10 = artifacts['top10_customers']

# ---------- Prediktionsfunktion ----------
def predict_churn(customer_dict):
    """Tar en dict med kund-attribut, returnerar churn-sannolikhet."""

    # Bygg en tom rad med alla feature_columns satta till 0
    row = {col: 0 for col in feature_columns}

    # Sätt numeriska värden direkt
    row['tenure'] = customer_dict['tenure']
    row['MonthlyCharges'] = customer_dict['MonthlyCharges']
    row['TotalCharges'] = customer_dict['TotalCharges']
    row['SeniorCitizen'] = customer_dict['SeniorCitizen']

    # Sätt dummies manuellt baserat på vad användaren valde
    # OBS: vi måste matcha exakt namnet på dummy-kolumnerna
    if customer_dict['gender'] == 'Male':
        row['gender_Male'] = 1
    if customer_dict['Partner'] == 'Yes':
        row['Partner_Yes'] = 1
    if customer_dict['Dependents'] == 'Yes':
        row['Dependents_Yes'] = 1
    if customer_dict['PhoneService'] == 'Yes':
        row['PhoneService_Yes'] = 1
    if customer_dict['MultipleLines'] == 'Yes':
        row['MultipleLines_Yes'] = 1

    # InternetService – referens är DSL
    if customer_dict['InternetService'] == 'Fiber optic':
        row['InternetService_Fiber optic'] = 1
    elif customer_dict['InternetService'] == 'No':
        row['InternetService_No'] = 1

    if customer_dict['OnlineSecurity'] == 'Yes':
        row['OnlineSecurity_Yes'] = 1
    if customer_dict['OnlineBackup'] == 'Yes':
        row['OnlineBackup_Yes'] = 1
    if customer_dict['DeviceProtection'] == 'Yes':
        row['DeviceProtection_Yes'] = 1
    if customer_dict['TechSupport'] == 'Yes':
        row['TechSupport_Yes'] = 1
    if customer_dict['StreamingTV'] == 'Yes':
        row['StreamingTV_Yes'] = 1
    if customer_dict['StreamingMovies'] == 'Yes':
        row['StreamingMovies_Yes'] = 1

    # Contract – referens är Month-to-month
    if customer_dict['Contract'] == 'One year':
        row['Contract_One year'] = 1
    elif customer_dict['Contract'] == 'Two year':
        row['Contract_Two year'] = 1

    if customer_dict['PaperlessBilling'] == 'Yes':
        row['PaperlessBilling_Yes'] = 1

    # PaymentMethod – referens är Bank transfer (automatic)
    if customer_dict['PaymentMethod'] == 'Credit card (automatic)':
        row['PaymentMethod_Credit card (automatic)'] = 1
    elif customer_dict['PaymentMethod'] == 'Electronic check':
        row['PaymentMethod_Electronic check'] = 1
    elif customer_dict['PaymentMethod'] == 'Mailed check':
        row['PaymentMethod_Mailed check'] = 1

    # Konvertera till DataFrame
    df_kund = pd.DataFrame([row])[feature_columns]

    # Skala numeriska
    df_kund_scaled = df_kund.copy()
    df_kund_scaled[numeric_cols] = scaler.transform(df_kund[numeric_cols])

    # Lägg till const
    df_kund_scaled.insert(0, 'const', 1.0)

    # Predicera
    return float(model.predict(df_kund_scaled)[0])


def klassificera_risk(prob):
    """Klassificera baserat på Youden-tröskel"""
    if prob >= optimal_threshold + 0.30:
        return "🔴 KRITISK RISK", "#c0392b"
    elif prob >= optimal_threshold + 0.15:
        return "🟠 MYCKET HÖG RISK", "#e67e22"
    elif prob >= optimal_threshold:
        return "🟡 HÖG RISK – ÅTGÄRDA", "#f1c40f"
    elif prob >= optimal_threshold - 0.10:
        return "🟢 MEDEL RISK", "#27ae60"
    else:
        return "✅ LÅG RISK", "#16a085"


# ---------- Förinställda profiler ----------
PROFILES = {
    "🆕 Custom (fyll i själv)": None,
    "👤 Anna – nyfiken Month-to-month-kund": {
        'tenure': 5, 'MonthlyCharges': 89.0, 'TotalCharges': 445.0,
        'SeniorCitizen': 0, 'gender': 'Female', 'Partner': 'No',
        'Dependents': 'No', 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'Fiber optic', 'OnlineSecurity': 'No',
        'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
        'StreamingTV': 'Yes', 'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check'
    },
    "👨 Bert – lojal långtidskund": {
        'tenure': 60, 'MonthlyCharges': 65.0, 'TotalCharges': 3900.0,
        'SeniorCitizen': 0, 'gender': 'Male', 'Partner': 'Yes',
        'Dependents': 'Yes', 'PhoneService': 'Yes', 'MultipleLines': 'Yes',
        'InternetService': 'DSL', 'OnlineSecurity': 'Yes',
        'OnlineBackup': 'Yes', 'DeviceProtection': 'Yes', 'TechSupport': 'Yes',
        'StreamingTV': 'No', 'StreamingMovies': 'No',
        'Contract': 'Two year', 'PaperlessBilling': 'No',
        'PaymentMethod': 'Bank transfer (automatic)'
    },
    "👵 Carla – pensionär med basabonnemang": {
        'tenure': 24, 'MonthlyCharges': 25.0, 'TotalCharges': 600.0,
        'SeniorCitizen': 1, 'gender': 'Female', 'Partner': 'No',
        'Dependents': 'No', 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'No', 'OnlineSecurity': 'No',
        'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
        'StreamingTV': 'No', 'StreamingMovies': 'No',
        'Contract': 'One year', 'PaperlessBilling': 'No',
        'PaymentMethod': 'Mailed check'
    },
}

# Lägg till top-10 från testdata
for i, kund in enumerate(top10):
    cid = kund['customerID']
    proba_pct = kund['churn_proba'] * 100
    PROFILES[f"⚠️ #{i+1}: {cid} ({proba_pct:.0f}% risk)"] = "TOP10_PLACEHOLDER"


# ---------- HEADER ----------
st.title("📞 Telco Churn Predictor")
st.markdown(
    "**Logistisk regression** med Youden-optimerad tröskel ("
    f"**{optimal_threshold:.3f}**). "
    "Recall: 82% • ROC-AUC: 0.85 • Datasetstorlek: 7032 kunder."
)

# ---------- LAYOUT: Sidebar för profilval, huvudområde för input ----------
st.sidebar.header("👥 Snabbval: kundprofil")
profile_choice = st.sidebar.selectbox(
    "Välj en förinställd profil eller börja från scratch:",
    options=list(PROFILES.keys()),
    index=1  # default = Anna
)

# Information om Top10
with st.sidebar.expander("ℹ️ Om Top-10-listan"):
    st.markdown(
        "De 10 kunder från testdatan som modellen flaggat som **högst risk**. "
        "Av dessa **churnade faktiskt åtta** – modellens prioritering "
        "är alltså direkt operationaliserbar."
    )

# Sätt upp default-värden
if PROFILES[profile_choice] and PROFILES[profile_choice] != "TOP10_PLACEHOLDER":
    defaults = PROFILES[profile_choice]
elif PROFILES[profile_choice] == "TOP10_PLACEHOLDER":
    defaults = PROFILES["👤 Anna – nyfiken Month-to-month-kund"]  # fallback
    st.sidebar.warning("Top-10-detaljer kräver utökad datakoppling – använder Anna som mall.")
else:
    # Custom: rimliga defaults
    defaults = {
        'tenure': 12, 'MonthlyCharges': 65.0, 'TotalCharges': 780.0,
        'SeniorCitizen': 0, 'gender': 'Female', 'Partner': 'No',
        'Dependents': 'No', 'PhoneService': 'Yes', 'MultipleLines': 'No',
        'InternetService': 'DSL', 'OnlineSecurity': 'No',
        'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No',
        'StreamingTV': 'No', 'StreamingMovies': 'No',
        'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check'
    }

# ---------- Huvudområde: två kolumner för input ----------
col_input, col_result = st.columns([3, 2])

with col_input:
    st.subheader("🛠️ Justera kundens attribut")

    # Demografi
    with st.expander("👤 Demografi", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            gender = st.radio("Kön", ["Female", "Male"],
                             index=["Female", "Male"].index(defaults['gender']))
            senior = st.radio("Pensionär", ["Nej", "Ja"],
                             index=defaults['SeniorCitizen'])
        with c2:
            partner = st.radio("Partner", ["No", "Yes"],
                              index=["No", "Yes"].index(defaults['Partner']))
            dependents = st.radio("Anhöriga", ["No", "Yes"],
                                  index=["No", "Yes"].index(defaults['Dependents']))

    # Kontrakt och pris
    with st.expander("📋 Kontrakt & Pris", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            contract = st.selectbox(
                "Kontraktstyp",
                ["Month-to-month", "One year", "Two year"],
                index=["Month-to-month", "One year", "Two year"].index(defaults['Contract'])
            )
            tenure = st.slider("Tenure (månader)", 0, 72, defaults['tenure'])
        with c2:
            monthly = st.slider("MonthlyCharges ($)", 18.0, 120.0,
                                float(defaults['MonthlyCharges']), step=1.0)
            # TotalCharges som funktion av tenure och monthly + lite slumpmässigt
            total = st.slider("TotalCharges ($)", 0.0, 9000.0,
                              float(defaults['TotalCharges']), step=10.0)

    # Telefoni & Internet
    with st.expander("📞 Telefoni & Internet"):
        c1, c2 = st.columns(2)
        with c1:
            phone = st.radio("PhoneService", ["No", "Yes"],
                             index=["No", "Yes"].index(defaults['PhoneService']))
            multilines = st.radio("MultipleLines", ["No", "Yes"],
                                  index=["No", "Yes"].index(defaults['MultipleLines']))
        with c2:
            internet = st.selectbox(
                "InternetService",
                ["DSL", "Fiber optic", "No"],
                index=["DSL", "Fiber optic", "No"].index(defaults['InternetService'])
            )

    # Tilläggstjänster
    with st.expander("🛡️ Tilläggstjänster"):
        c1, c2 = st.columns(2)
        with c1:
            online_sec = st.checkbox("OnlineSecurity",
                                     value=defaults['OnlineSecurity'] == 'Yes')
            online_backup = st.checkbox("OnlineBackup",
                                        value=defaults['OnlineBackup'] == 'Yes')
            device_prot = st.checkbox("DeviceProtection",
                                      value=defaults['DeviceProtection'] == 'Yes')
        with c2:
            tech_sup = st.checkbox("TechSupport",
                                   value=defaults['TechSupport'] == 'Yes')
            stream_tv = st.checkbox("StreamingTV",
                                    value=defaults['StreamingTV'] == 'Yes')
            stream_movies = st.checkbox("StreamingMovies",
                                        value=defaults['StreamingMovies'] == 'Yes')

    # Fakturering
    with st.expander("💰 Fakturering"):
        c1, c2 = st.columns(2)
        with c1:
            paperless = st.radio("PaperlessBilling", ["No", "Yes"],
                                 index=["No", "Yes"].index(defaults['PaperlessBilling']))
        with c2:
            payment = st.selectbox(
                "PaymentMethod",
                ["Bank transfer (automatic)", "Credit card (automatic)",
                 "Electronic check", "Mailed check"],
                index=["Bank transfer (automatic)", "Credit card (automatic)",
                       "Electronic check", "Mailed check"].index(defaults['PaymentMethod'])
            )

# ---------- Bygg kund-dict och predicera ----------
customer = {
    'tenure': tenure,
    'MonthlyCharges': monthly,
    'TotalCharges': total,
    'SeniorCitizen': 1 if senior == "Ja" else 0,
    'gender': gender,
    'Partner': partner,
    'Dependents': dependents,
    'PhoneService': phone,
    'MultipleLines': multilines,
    'InternetService': internet,
    'OnlineSecurity': 'Yes' if online_sec else 'No',
    'OnlineBackup': 'Yes' if online_backup else 'No',
    'DeviceProtection': 'Yes' if device_prot else 'No',
    'TechSupport': 'Yes' if tech_sup else 'No',
    'StreamingTV': 'Yes' if stream_tv else 'No',
    'StreamingMovies': 'Yes' if stream_movies else 'No',
    'Contract': contract,
    'PaperlessBilling': paperless,
    'PaymentMethod': payment
}

# DEBUG: visa vad som faktiskt skickas till modellen
st.write("🔧 Debug:", {
    'InternetService': customer['InternetService'],
    'MonthlyCharges': customer['MonthlyCharges']
})

prob = predict_churn(customer)
risk_text, risk_color = klassificera_risk(prob)

# ---------- Resultat-kolumn ----------
with col_result:
    st.subheader("🎯 Modellens prediktion")

    # Stor sannolikhets-display
    st.markdown(
        f"""
        <div style='text-align:center; padding:20px; background-color:{risk_color};
                    color:white; border-radius:10px; margin-bottom:10px;'>
            <div style='font-size:14px;'>CHURN-SANNOLIKHET</div>
            <div style='font-size:54px; font-weight:bold;'>{prob:.1%}</div>
            <div style='font-size:18px;'>{risk_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Tröskel-information
    st.markdown(
        f"**Tröskel (Youden's J):** {optimal_threshold:.3f}\n\n"
        f"**Modellens beslut:** {'CHURN' if prob >= optimal_threshold else 'KVAR'}"
    )

    # What-if knappar för snabb scenario-test
    st.markdown("---")
    st.subheader("🔮 Snabba what-if-scenarier")

    scenarios = []

    # Scenario 1: Konvertera till tvåårskontrakt
    if contract != "Two year":
        cust2 = customer.copy()
        cust2['Contract'] = 'Two year'
        prob2 = predict_churn(cust2)
        diff = (prob - prob2) * 100
        scenarios.append((
            "📋 Två-årskontrakt",
            prob2,
            diff,
            "Erbjud 2-årskontrakt"
        ))

    # Scenario 2: Byt till automatisk betalning
    if payment in ["Electronic check", "Mailed check"]:
        cust3 = customer.copy()
        cust3['PaymentMethod'] = 'Bank transfer (automatic)'
        prob3 = predict_churn(cust3)
        diff = (prob - prob3) * 100
        scenarios.append((
            "🏦 Automatisk betalning",
            prob3,
            diff,
            "Konvertera till autogiro"
        ))

    # Scenario 3: Lägg till security-paket
    if not (online_sec and tech_sup):
        cust4 = customer.copy()
        cust4['OnlineSecurity'] = 'Yes'
        cust4['TechSupport'] = 'Yes'
        prob4 = predict_churn(cust4)
        diff = (prob - prob4) * 100
        scenarios.append((
            "🛡️ Security + TechSupport",
            prob4,
            diff,
            "Push säkerhetspaket"
        ))

    if scenarios:
        for namn, ny_prob, diff_pp, atgard in scenarios:
            color = "#27ae60" if diff_pp > 0 else "#c0392b"
            tecken = "↓" if diff_pp > 0 else "↑"
            st.markdown(
                f"**{namn}** → {ny_prob:.1%} "
                f"<span style='color:{color}'>({tecken} {abs(diff_pp):.1f} pp)</span>",
                unsafe_allow_html=True
            )
            st.caption(f"Åtgärd: *{atgard}*")
            st.markdown("")
    else:
        st.info("Den här kunden är redan optimerad för låg risk!")

# ---------- Footer ----------
st.markdown("---")
with st.expander("ℹ️ Om modellen"):
    st.markdown(
        f"""
        - **Modell:** Logistisk regression (statsmodels)
        - **Tröskel:** {optimal_threshold:.3f} (optimerad med Youden's J)
        - **ROC-AUC:** 0.85 på testdata (n = 2113)
        - **Recall (Churn) vid tröskeln:** 82%
        - **Precision (Churn):** 50%

        Modellens **referenskategorier** (osynliga, värde 0 i alla dummies):
        - Contract: Month-to-month
        - InternetService: DSL
        - PaymentMethod: Bank transfer (automatic)
        - gender: Female

        Effekter rapporteras alltid **relativt referenskategorin**.
        """
    )