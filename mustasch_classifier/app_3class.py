import os
import io
import time
import urllib.parse
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import MTCNN

mtcnn = MTCNN(image_size=178, margin=40, post_process=False)

# --------------------------------------------------
# Page config
# --------------------------------------------------

st.set_page_config(
    page_title="Mustaschkraft™ (3-klass test)",
    page_icon="🥸",
    layout="centered"
)

# --------------------------------------------------
# Load models
# --------------------------------------------------

@st.cache_resource
def load_models():
    mustache_model = tf.keras.models.load_model(
        "models/mustache_detector_3.keras"
    )
    epic_model = tf.keras.models.load_model(
        "models/epic_detector_3class.keras"
    )
    return mustache_model, epic_model


mustache_model, epic_model = load_models()

# Ordningen image_dataset_from_directory sorterar mappar alfabetiskt:
# epic, medium, thin
CLASS_NAMES = ["epic", "medium", "thin"]

# --------------------------------------------------
# Constants
# --------------------------------------------------

IMG_SIZE = (178, 178)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.image("assets/logo.png", use_container_width=True)

st.markdown(
    "<p style='text-align: center; color: gray; font-size: 0.875rem;'>"
    "Officiell ansiktshårsbedömning "
    "av Internationella Mustaschmyndigheten"
    "</p>",
    unsafe_allow_html=True
)

st.warning("🧪 TESTVERSION — 3-klassmodell (episk/respektabel/tunn)", icon="🧪")

# --------------------------------------------------
# Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Skicka in provet för analys",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def weighted_epic_score(p_epic, p_medium, p_thin):
    """Viktad poäng 0-100 baserat på de tre klassernas sannolikheter.

    Om epic leder: medium/thin-närvaro KOSTAR poäng (subtraheras) istället för
    att läggas till — en epic-bedömning ska straffas av tvivel, inte gynnas.
    Om medium eller thin leder: vanlig viktad summa, med liten golv-vikt på
    thin (8) så ett rent thin-fall inte slås ner till noll.
    Straff (×0.85) när ingen klass har klar majoritet (p_max < 0.7) — fångar
    genuint osäkra/blandade fall som annars fick orättvist hög poäng bara
    genom att vara den största av tre svaga alternativ.
    """
    p_max = max(p_epic, p_medium, p_thin)

    if p_epic == p_max:
        # Epic leder. Medium läggs alltid till (är inte "tvivel"), bara thin straffar.
        score = p_epic * 100 + p_medium * 50 - p_thin * 100
    elif p_medium == p_max:
        score = p_medium * 60 + p_epic * 40 - p_thin * 30
    else:
        score = p_thin * 8 + p_epic * 100 + p_medium * 50

    if p_max < 0.7:
        score *= 0.92  # generöst — appen ska vara kul att dela, inte sträng

    return float(np.clip(score, 0, 100))


def prepare_image(image):
    image = image.convert("RGB")

    face = mtcnn(image)

    if face is not None:
        # post_process=False → tensor i [0, 255] direkt
        arr = face.permute(1, 2, 0).numpy().astype(np.uint8)
    else:
        # Fallback om inget ansikte hittas
        arr = np.array(image.resize(IMG_SIZE))

    arr = np.expand_dims(arr, axis=0)
    return arr


def classify_epicness(score):
    if score >= 95:
        return (
            "🏆 Legendarisk Mustasch",
            "Internationella Mustaschmyndigheten är mållös.",
            "assets/epic.mp4"
        )
    elif score >= 80:
        return (
            "🎩 Episk Mustasch",
            "En sann prestation för överläppen.",
            "assets/medium.mp4"
        )
    elif score >= 50:
        return (
            "🧔 Respektabel Mustasch",
            "Godkänd. Inte historisk, men godkänd.",
            "assets/medium.mp4"
        )
    elif score >= 25:
        return (
            "🌱 Lovande Mustasch",
            "Mustaschtillväxten befinner sig fortfarande i betatest.",
            "assets/medium.mp4"
        )
    else:
        return (
            "🪶 Fjunig Mustasch",
            "Mustaschen existerar mest som ett teoretiskt koncept.",
            "assets/fjunig.mp4"
        )


def _load_font(size):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def create_share_image(photo, score, title, logo_path="assets/mustaschkampen_logo.png"):
    """Bygger en delningsbar bild: foto + poäng/titel-banner + Mustaschkampen-logga."""
    photo = photo.convert("RGB")

    width = 1080
    ratio = width / photo.width
    height = int(photo.height * ratio)
    photo = photo.resize((width, height))

    banner_height = 220
    canvas = Image.new("RGB", (width, height + banner_height), "white")
    canvas.paste(photo, (0, 0))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, height, width, height + banner_height], fill="#1a1a1a")

    score_font = _load_font(90)
    title_font = _load_font(40)

    score_text = f"{score:.0f}/100"
    draw.text((40, height + 25), score_text, font=score_font, fill="white")
    draw.text((40, height + 130), title, font=title_font, fill="#f5c542")

    if logo_path and os.path.exists(logo_path):
        logo = Image.open(logo_path).convert("RGBA")
        logo_h = 160
        logo_w = int(logo.width * (logo_h / logo.height))
        logo = logo.resize((logo_w, logo_h))
        canvas.paste(
            logo,
            (width - logo_w - 30, height + (banner_height - logo_h) // 2),
            logo
        )

    return canvas


def animate_scanner():
    labels = [
        "🔬 Skannar överläppsregionen...",
        "🧬 Analyserar hårdensitetsmatris...",
        "⚗️  Korsreferens mot mustaschdatabasen...",
        "🏛️  Konsulterar Myndighetens arkiv...",
        "📊 Beräknar mustaschkraft...",
    ]

    progress_bar = st.progress(0)
    status = st.empty()
    meter1 = st.empty()
    meter2 = st.empty()

    for i, label in enumerate(labels):
        status.markdown(f"**{label}**")

        for v in list(range(0, 101, 5)) + list(range(100, -1, -5)):
            meter1.progress(v / 100)
            meter2.progress(abs(v - 100) / 100)
            time.sleep(0.015)

        progress_bar.progress((i + 1) / len(labels))

    status.empty()
    meter1.empty()
    meter2.empty()
    progress_bar.empty()

# --------------------------------------------------
# Analysis
# --------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Inskickat prov", use_container_width=True)

    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        analyze = st.button("🔬 Starta analys", use_container_width=True)

    if analyze:

        img_array = prepare_image(image)

        animate_scanner()

        mustache_prob = float(
            mustache_model.predict(img_array, verbose=0)[0][0]
        )

        st.markdown("---")

        st.write("mustasch_sannolikhet:", mustache_prob)

        if mustache_prob < 0.4:
            st.error("❌ INGEN CERTIFIERAD MUSTASCH UPPTÄCKT")
            st.write(
                "Utlåtande: Överläppen verkar för närvarande "
                "sakna tillräcklig auktoritet."
            )
            if os.path.exists("assets/no.mp4"):
                st.video("assets/no.mp4")

        else:
            preds = epic_model.predict(img_array, verbose=0)[0]
            p_epic, p_medium, p_thin = float(preds[0]), float(preds[1]), float(preds[2])

            epic_score = weighted_epic_score(p_epic, p_medium, p_thin)

            st.write("p_episk:", p_epic)
            st.write("p_respektabel:", p_medium)
            st.write("p_tunn:", p_thin)
            st.write("mustaschkraft_poäng:", epic_score)

            title, description, video_file = classify_epicness(epic_score)

            # Video överst om det finns en
            if video_file and os.path.exists(video_file):
                st.video(video_file)

            st.markdown("<br>", unsafe_allow_html=True)

            st.metric(
                "Mustaschkraft™",
                f"{epic_score:.1f} / 100"
            )
            st.progress(float(epic_score) / 100)

            st.markdown("<br>", unsafe_allow_html=True)

            st.subheader(title)
            st.write(description)

            if epic_score >= 95:
                st.balloons()

            st.markdown("<br>", unsafe_allow_html=True)

            share_image = create_share_image(image, epic_score, title)

            buf = io.BytesIO()
            share_image.save(buf, format="PNG")
            buf.seek(0)

            st.image(share_image, caption="Förhandsvisning av delningsbild", use_container_width=True)

            col_dl, col_fb = st.columns(2)

            with col_dl:
                st.download_button(
                    "⬇️ Ladda ner bild",
                    data=buf,
                    file_name="mustaschkraft.png",
                    mime="image/png",
                    use_container_width=True
                )

            with col_fb:
                share_text = urllib.parse.quote(
                    f"Min mustasch fick {epic_score:.0f}/100 — {title}! Stötta Mustaschkampen 🥸"
                )
                fb_url = f"https://www.facebook.com/sharer/sharer.php?u=https://mustaschkampen.se&quote={share_text}"
                st.link_button("📘 Dela på Facebook", fb_url, use_container_width=True)

            st.caption(
                "Tips: ladda ner bilden och dela den via din mobils delningsmeny "
                "för Instagram, Snapchat eller andra appar."
            )

            st.divider()
            st.markdown(
                "<p style='text-align: center; color: gray; font-size: 0.875rem;'>"
                "Resultat certifierat av "
                "Internationella Mustaschmyndigheten™"
                "</p>",
                unsafe_allow_html=True
            )
