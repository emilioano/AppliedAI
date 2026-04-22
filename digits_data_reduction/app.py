import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import plotly.express as px
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from umap import UMAP



# ==========================================
# Sidkonfiguration
# ==========================================
st.set_page_config(
    page_title="Digits PCA Explorer",
    page_icon="📈",
    layout="centered"
)

st.title("Digits PCA Explorer")
st.markdown("Interaktiv utforskning av PCA på sklearns Digits-dataset")

# ==========================================
# Ladda data (cachad för prestanda)
# ==========================================
@st.cache_data
def load_data():
    digits = load_digits()
    return digits.data, digits.target, digits.images

x, y, images = load_data()

# ==========================================
# Sidebar — kontroller som påverkar allt
# ==========================================
st.sidebar.header("Inställningar")

scaler_choice = st.sidebar.radio(
    "Skalningsmetod",
    ["StandardScaler", "MinMaxScaler", "Ingen skalning"]
)

# Applicera vald skalning
if scaler_choice == "StandardScaler":
    x_processed = StandardScaler().fit_transform(x)
elif scaler_choice == "MinMaxScaler":
    x_processed = MinMaxScaler().fit_transform(x)
else:
    x_processed = x.copy()

# ==========================================
# Kör PCA
# ==========================================
@st.cache_data
def run_pca(data):
    pca = PCA()
    x_pca = pca.fit_transform(data)
    return x_pca, pca.explained_variance_ratio_

x_pca, variance_ratio = run_pca(x_processed)

# ==========================================
# Huvudinnehåll — flikar
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Översikt",
    "Scree plot",
    "2D-scatter",
    "3D-scatter",
    "UMAP"
])

with tab1:
    st.header("Översikt av datasetet")
    col1, col2, col3 = st.columns(3)
    col1.metric("Antal bilder", len(x))
    col2.metric("Features (pixlar)", x.shape[1])
    col3.metric("Klasser", len(np.unique(y)))
    
    st.info(f"Vald skalning: **{scaler_choice}**")

with tab2:
    st.header("Scree plot — hur viktig är varje komponent?")
    
    st.markdown("""
    Scree plot visar hur mycket varje huvudkomponent bidrar till den totala variansen. 
    De första komponenterna fångar mest, sen planar kurvan ut.
    """)
    
    # Slider för att välja antal komponenter
    n_components = st.slider(
        "Välj antal komponenter att fokusera på",
        min_value=1,
        max_value=64,
        value=10,
        help="Linjen på plotten markerar ditt val"
    )
    
    # Beräkna kumulativ varians
    cumulative = np.cumsum(variance_ratio)
    captured = cumulative[n_components - 1]
    

    # Scree plot
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(1, len(variance_ratio) + 1), variance_ratio, 
            'o-', color='steelblue', markersize=4)
    ax.axvline(n_components, color='red', linestyle='--', 
               label=f'N = {n_components}')
    ax.set_xlabel('Komponentnummer')
    ax.set_ylabel('Andel varians')
    ax.set_title('Scree plot')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, width="content")

    # Kumulativ varians
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(range(1, len(cumulative) + 1), cumulative,
            'o-', color='darkorange', markersize=4)
    ax.axvline(n_components, color='red', linestyle='--')
    ax.axhline(captured, color='red', linestyle=':', alpha=0.5)
    ax.set_xlabel('Antal komponenter')
    ax.set_ylabel('Kumulativ varians')
    ax.set_title('Kumulativ förklarad varians')
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, width="content")
    
    # Stor metrik-rad nedanför
    col_a, col_b, col_c = st.columns(3)
    col_a.metric(
        "Vald N",
        n_components
    )
    col_b.metric(
        "Fångad varians",
        f"{captured:.1%}"
    )
    col_c.metric(
        "Dimensionsreduktion",
        f"{64 - n_components} dimensioner borttagna"
    )
    
    # Praktiska trösklar
    st.markdown("### Hur många komponenter för olika tröskelvärden?")
    thresholds = [0.80, 0.90, 0.95, 0.99]
    threshold_cols = st.columns(len(thresholds))
    
    for col, threshold in zip(threshold_cols, thresholds):
        n_needed = int(np.argmax(cumulative >= threshold)) + 1
        col.metric(f"För {threshold:.0%}", f"{n_needed} komp.")

with tab3:
    st.header("2D-scatter av PCA-komponenter")
    
    st.markdown("""
    Plotta vilka två principalkomponenter som helst mot varandra. 
    Varje punkt är en handskriven siffra — färgad efter vilken siffra den föreställer.
    """)
    
    # Kontrollpanel — tre kolumner för val
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_component = st.selectbox(
            "X-axel",
            options=list(range(1, 21)),  # PC1 till PC20
            index=0,  # default PC1
            format_func=lambda i: f"PC{i}"
        )
    
    with col2:
        y_component = st.selectbox(
            "Y-axel",
            options=list(range(1, 21)),
            index=1,  # default PC2
            format_func=lambda i: f"PC{i}"
        )
    
    with col3:
        selected_digits = st.multiselect(
            "Visa siffror",
            options=list(range(10)),
            default=list(range(10)),  # alla som default
            format_func=lambda d: f"Siffra {d}"
        )
    
    # Varning om användaren avmarkerat alla siffror
    if not selected_digits:
        st.warning("Välj minst en siffra att visa!")
    else:
        # Filtrera datan baserat på val
        mask = np.isin(y, selected_digits)
        
        # Bygg DataFrame för plotly
        df_plot = pd.DataFrame({
            f'PC{x_component}': x_pca[mask, x_component - 1],
            f'PC{y_component}': x_pca[mask, y_component - 1],
            'Siffra': y[mask].astype(str)
        })
        
        # Scatter plot
        fig = px.scatter(
            df_plot,
            x=f'PC{x_component}',
            y=f'PC{y_component}',
            color='Siffra',
            color_discrete_sequence=px.colors.qualitative.T10,
            opacity=0.7,
            title=f'PC{x_component} vs PC{y_component}',
            labels={
                f'PC{x_component}': f'PC{x_component} ({variance_ratio[x_component-1]:.1%} varians)',
                f'PC{y_component}': f'PC{y_component} ({variance_ratio[y_component-1]:.1%} varians)'
            }
        )
        fig.update_traces(marker=dict(size=6, line=dict(width=0.3, color='white')))
        fig.update_layout(height=600)
        
        st.plotly_chart(fig, width="content")
        
        # Info nedanför
        total_variance = variance_ratio[x_component-1] + variance_ratio[y_component-1]
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Antal punkter", mask.sum())
        col_b.metric("Visade klasser", len(selected_digits))
        col_c.metric("Total varians i plotten", f"{total_variance:.1%}")

with tab4:
    st.header("3D-scatter — rotera och utforska")
    
    st.markdown("""
    Tre komponenter ger mer information än två. Dra i plotten för att rotera, 
    scrolla för att zooma, och klicka på siffrorna i legenden för att dölja/visa klasser.
    """)
    
    # Kontrollpanel — val av komponenter
    col1, col2, col3 = st.columns(3)
    
    with col1:
        x_comp_3d = st.selectbox(
            "X-axel",
            options=list(range(1, 21)),
            index=0,
            format_func=lambda i: f"PC{i}",
            key="x_3d"  # unik key för att undvika krock med 2D-fliken
        )
    
    with col2:
        y_comp_3d = st.selectbox(
            "Y-axel",
            options=list(range(1, 21)),
            index=1,
            format_func=lambda i: f"PC{i}",
            key="y_3d"
        )
    
    with col3:
        z_comp_3d = st.selectbox(
            "Z-axel",
            options=list(range(1, 21)),
            index=2,
            format_func=lambda i: f"PC{i}",
            key="z_3d"
        )
    
    # Filtrering av klasser
    selected_digits_3d = st.multiselect(
        "Visa siffror",
        options=list(range(10)),
        default=list(range(10)),
        format_func=lambda d: f"Siffra {d}",
        key="digits_3d"
    )
    
    # Markörstorlek
    marker_size = st.slider(
        "Markörstorlek",
        min_value=1, max_value=10, value=3,
        key="size_3d"
    )
    
    if not selected_digits_3d:
        st.warning("Välj minst en siffra att visa.")
    else:
        # Filtrera
        mask = np.isin(y, selected_digits_3d)
        
        # DataFrame för plotly
        df_3d = pd.DataFrame({
            f'PC{x_comp_3d}': x_pca[mask, x_comp_3d - 1],
            f'PC{y_comp_3d}': x_pca[mask, y_comp_3d - 1],
            f'PC{z_comp_3d}': x_pca[mask, z_comp_3d - 1],
            'Siffra': y[mask].astype(str)
        })
        
        # 3D-scatter
        fig = px.scatter_3d(
            df_3d,
            x=f'PC{x_comp_3d}',
            y=f'PC{y_comp_3d}',
            z=f'PC{z_comp_3d}',
            color='Siffra',
            color_discrete_sequence=px.colors.qualitative.T10,
            opacity=0.7,
            title=f'PC{x_comp_3d} vs PC{y_comp_3d} vs PC{z_comp_3d}'
        )
        fig.update_traces(marker=dict(size=marker_size))
        fig.update_layout(
            height=700,
            scene=dict(
                xaxis_title=f'PC{x_comp_3d} ({variance_ratio[x_comp_3d-1]:.1%})',
                yaxis_title=f'PC{y_comp_3d} ({variance_ratio[y_comp_3d-1]:.1%})',
                zaxis_title=f'PC{z_comp_3d} ({variance_ratio[z_comp_3d-1]:.1%})'
            )
        )
        
        st.plotly_chart(fig, width="content")
        
        # Info
        total_variance_3d = (variance_ratio[x_comp_3d-1] + 
                             variance_ratio[y_comp_3d-1] + 
                             variance_ratio[z_comp_3d-1])
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Antal punkter", mask.sum())
        col_b.metric("Visade klasser", len(selected_digits_3d))
        col_c.metric("Total varians", f"{total_variance_3d:.1%}")

with tab5:
    st.header("UMAP — uniform manifold approximation and projection")
    
    st.markdown("""
    UMAP är en icke-linjär teknik som ofta ger tydligare klasseparation än ren PCA. 
    Vi matar in de N första PCA-komponenterna som input, sedan klämmer 
    UMAP ner dem till 2D eller 3D.
    """)
    
    # === Rad 1: Antal komponenter in, och ut ===
    col1, col2 = st.columns(2)
    
    with col1:
        n_pca_for_umap = st.slider(
            "Antal PCA-komponenter som input",
            min_value=2, max_value=50, value=20,
            help="Fler komponenter = mer info men mer brus"
        )
    
    with col2:
        output_dims = st.radio(
            "Output-dimensioner",
            options=[2, 3],
            horizontal=True,
            format_func=lambda n: f"{n}D"
        )
    
    # === Rad 2: De två viktigaste hyperparametrarna ===
    col3, col4 = st.columns(2)
    
    with col3:
        n_neighbors = st.slider(
            "n_neighbors",
            min_value=2, max_value=100, value=15,
            help="Lågt = fokus på lokal struktur. Högt = global struktur."
        )
    
    with col4:
        min_dist = st.slider(
            "min_dist",
            min_value=0.0, max_value=1.0, value=0.1, step=0.05,
            help="Lågt = tighta kluster. Högt = mer spridning."
        )
    
    # === Rad 3: Metric och random state ===
    col5, col6 = st.columns(2)
    
    with col5:
        metric = st.selectbox(
            "Avståndsmått (metric)",
            options=["euclidean", "manhattan", "cosine", "chebyshev", "correlation"],
            index=0,
            help="Hur avstånd mellan punkter beräknas"
        )
    
    with col6:
        random_state = st.number_input(
            "Random state",
            min_value=0, max_value=9999, value=42, step=1,
            help="Samma värde ger samma resultat. Byt för att se andra lösningar."
        )
    
    # Visa input-info
    captured_for_umap = variance_ratio[:n_pca_for_umap].sum()
    st.info(f"Input till UMAP: {n_pca_for_umap} PCA-komponenter som fångar {captured_for_umap:.1%} av variansen")
    
    # Filtrering av klasser
    selected_digits_umap = st.multiselect(
        "Visa siffror",
        options=list(range(10)),
        default=list(range(10)),
        format_func=lambda d: f"Siffra {d}",
        key="digits_umap"
    )
    
    # Cachad UMAP-funktion
    @st.cache_data
    def compute_umap(_x_pca, n_components_input, n_out, n_neigh, min_d, metric_name, rand_state):
        x_input = _x_pca[:, :n_components_input]
        reducer = UMAP(
            n_components=n_out,
            n_neighbors=n_neigh,
            min_dist=min_d,
            metric=metric_name,
            random_state=rand_state
        )
        return reducer.fit_transform(x_input)
    
    if not selected_digits_umap:
        st.warning("Välj minst en siffra att visa.")
    else:
        with st.spinner("Kör UMAP... (kan ta några sekunder)"):
            x_umap = compute_umap(
                x_pca, n_pca_for_umap, output_dims,
                n_neighbors, min_dist, metric, random_state
            )
        
        # Filtrera punkter
        mask = np.isin(y, selected_digits_umap)
        
        title_str = (f'UMAP ({output_dims}D, metric={metric}, '
                     f'n_neighbors={n_neighbors}, min_dist={min_dist})')
        
        if output_dims == 2:
            df_umap = pd.DataFrame({
                'UMAP1': x_umap[mask, 0],
                'UMAP2': x_umap[mask, 1],
                'Siffra': y[mask].astype(str)
            })
            
            fig = px.scatter(
                df_umap, x='UMAP1', y='UMAP2',
                color='Siffra',
                color_discrete_sequence=px.colors.qualitative.T10,
                opacity=0.7,
                title=title_str
            )
            fig.update_traces(marker=dict(size=5, line=dict(width=0.3, color='white')))
            fig.update_layout(height=600)
        
        else:  # 3D
            df_umap = pd.DataFrame({
                'UMAP1': x_umap[mask, 0],
                'UMAP2': x_umap[mask, 1],
                'UMAP3': x_umap[mask, 2],
                'Siffra': y[mask].astype(str)
            })
            
            fig = px.scatter_3d(
                df_umap, x='UMAP1', y='UMAP2', z='UMAP3',
                color='Siffra',
                color_discrete_sequence=px.colors.qualitative.T10,
                opacity=0.7,
                title=title_str
            )
            fig.update_traces(marker=dict(size=3))
            fig.update_layout(height=700)
        
        st.plotly_chart(fig, width="stretch")
        
        # Info
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Antal punkter", mask.sum())
        col_b.metric("PCA-varians som input", f"{captured_for_umap:.1%}")
        col_c.metric("UMAP output", f"{output_dims}D")
        
        # Hjälpsam förklaring
        with st.expander("Vad gör alla parametrar?"):
            st.markdown("""
            **n_neighbors** — hur stor omgivning UMAP tittar på för varje punkt
            - Lågt (2-10): lokala detaljer, tighta små kluster
            - Högt (30-100): global struktur, bredare överblick
            
            **min_dist** — minsta avstånd mellan punkter i output
            - Lågt (0.0-0.1): tightare kluster, tydligare separation
            - Högt (0.5-1.0): mer spridning, mjukare övergångar
            
            **metric** — hur UMAP mäter avstånd mellan datapunkter
            - `euclidean` — raka avståndet (standard). Fungerar bra för de flesta kontinuerliga data.
            - `manhattan` — summa av absolut skillnad per dimension ("taxavstånd"). Mindre känslig för outliers.
            - `cosine` — vinkel mellan vektorer, ignorerar magnitud. Bra för textdata och bilder där riktning spelar större roll än storlek.
            - `chebyshev` — max-avstånd i någon dimension. Fokuserar på den värsta skillnaden.
            - `correlation` — baseras på korrelation mellan vektorer. Bra när datan har mönster snarare än absoluta värden.
            
            **random_state** — slumpfröet
            - UMAP har slumpmässiga inslag. Samma random_state → samma resultat varje gång.
            - Prova olika värden för att se hur stabil strukturen är. Om klustren är konsistenta över olika random states är de "riktiga".
            
            **n_components (output-dimensioner)** — antal dimensioner i resultatet
            - 2D för klassisk visualisering
            - 3D för rotation och mer separation, men svårare att fånga på en statisk bild
            """)