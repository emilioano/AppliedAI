"""
Helt fristående, minimal Streamlit-app för TRANSFER LEARNING på loggdata: finjustera
en riktig FÖRTRÄNAD textmodell (DistilBERT/BERT, tränad av andra på stora mängder
allmän engelsk text) för att klassificera loggrader som normala eller avvikande --
istället för att träna en egen liten modell från grunden.

Fem steg:

  1. Ladda upp loggfiler + ange ord/fraser att filtrera bort -> ett enkelt, textbaserat
     facit (normal/anomal).

  2. Bygg dataset: en SUPERVISED finjustering -- klassificeraren måste se exempel på
     BÅDA klasserna för att lära sig en gränsyta. Både normal- och anomal-raderna
     delas därför var för sig, kronologiskt, i tränings-/validerings-/testdel. Ingen
     mall-mining inblandad -- vi använder basmodellens EGEN subword-tokenizer direkt
     på loggtexten (efter att ha plockat bort en ev. tidsstämpel/header).

  3. Träna (finjustera): laddar en FÖRTRÄNAD modell från HuggingFace och byter ut dess
     språkmodellerings-huvud mot ett NYTT, slumpmässigt initierat klassificerings-
     huvud -- det är "egna lager" i uppgiftens mening. De understa transformer-
     blocken FRYSES (vikterna ändras inte alls under träningen); bara de översta
     blocken + det nya klassificeringshuvudet finjusteras. Det är kärnan i transfer
     learning: återanvänd vad modellen redan lärt sig om språk i stort, anpassa bara
     toppen till vår specifika uppgift.

  4. Validera: precision/recall/F1/ROC-AUC, en confusion matrix och ett histogram
     över normal- vs. anomal-sannolikheter.

  5. Analysera en ny loggfil: ladda upp en ny fil, få en lista över vilka rader
     klassificeraren tror är avvikande, sorterad efter sannolikhet.

VIKTIGT: modellen laddas ner från HuggingFace Hub vid första körningen (kräver
internetåtkomst, några hundra MB). På den här maskinen krävdes paketet
`pip-system-certs` för att det skulle fungera alls -- utan det misslyckas nedladdningen
med ett SSL-certifikatfel, eftersom Python annars inte litar på samma lokala
TLS-inspekterande proxy/antivirus-certifikat som Windows (och t.ex. curl) redan gör.
Se requirements.txt.

Kör med:
    ./.venv/Scripts/streamlit run streamlit_transfer_app.py
"""
from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).parent
DATASET_DIR = ROOT / "data" / "streamlit_transfer" / "dataset"

# Kända radformat -- byt/lägg till ett eget om din logg ser annorlunda ut. "raw"
# betyder "inget känt format", varvid hela raden behandlas som meddelandet.
HEADER_FORMATS = {
    "syslog": re.compile(
        r"^(?P<ts>\S+)\s+(?P<host>\S+)\s+(?P<process>[^:\[]+)(?:\[(?P<pid>\d+)\])?:\s?(?P<message>.*)$"
    ),
    "hdfs": re.compile(
        r"^\d{6}\s+\d{6}\s+\d+\s+(?P<level>\w+)\s+(?P<component>[^:]+):\s*(?P<message>.*)$"
    ),
    "raw": None,
}
HEADER_LABELS = {
    "syslog": "syslog / journalctl (TIMESTAMP HOST PROCESS[PID]: meddelande)",
    "hdfs": "HDFS-loggformat (t.ex. 081109 203518 143 INFO comp: meddelande)",
    "raw": "Okänt format -- hela raden är meddelandet",
}

# De förtränade basmodellerna man kan välja mellan, och hur många transformer-block
# var och en har (behövs för att sätta gränser på "hur många block ska frysas"-reglaget
# INNAN modellen faktiskt laddas ner).
MODEL_CHOICES = {
    "distilbert-base-uncased": {
        "label": "DistilBERT (66M parametrar, 6 block -- bra balans mellan kvalitet och CPU-hastighet)",
        "num_layers": 6,
    },
    "bert-base-uncased": {
        "label": "BERT base (110M parametrar, 12 block -- \"den riktiga\" BERT, långsammare på CPU)",
        "num_layers": 12,
    },
}


def extract_message(raw_line: str, header_re: re.Pattern | None) -> str:
    raw_line = raw_line.rstrip("\n")
    if header_re is None:
        return raw_line
    m = header_re.match(raw_line)
    return m.group("message") if m else raw_line


def get_encoder_layers(model) -> tuple[nn.Module, list[nn.Module]]:
    """Hittar embedding-lagret och listan av transformer-block i den underliggande
    förtränade modellen -- oavsett om det är en DistilBERT- eller BERT-arkitektur
    (de har olika attributnamn internt: `.transformer.layer` respektive
    `.encoder.layer`). `model.base_model` ger den förtränade encodern utan
    klassificeringshuvudet, oavsett vilken av de två det är."""
    base = model.base_model
    if hasattr(base, "transformer"):  # DistilBERT
        return base.embeddings, list(base.transformer.layer)
    if hasattr(base, "encoder"):  # BERT och de flesta andra
        return base.embeddings, list(base.encoder.layer)
    raise ValueError(f"Okänd modellarkitektur ({type(base).__name__}) -- vet inte hur man fryser lager för den.")


def freeze_bottom_layers(model, num_frozen: int) -> None:
    """Fryser embedding-lagret (alltid) plus de `num_frozen` understa transformer-
    blocken -- deras vikter uppdateras aldrig under träningen. Klassificeringshuvudet
    (nytt, slumpmässigt initierat) och de översta blocken förblir tränbara."""
    embeddings, layers = get_encoder_layers(model)
    for p in embeddings.parameters():
        p.requires_grad = False
    for layer in layers[:num_frozen]:
        for p in layer.parameters():
            p.requires_grad = False


def finetune(model, tokenizer, texts: list[str], labels: list[int], *, epochs: int, batch_size: int,
             lr: float, max_length: int, log_fn=print) -> float:
    """Finjusterar modellen på (text, etikett)-par. Klassvikter kompenserar för att
    normal/anomal oftast är olika stora grupper (annars lär sig modellen bara gissa
    majoritetsklassen). Returnerar sista epokens genomsnittliga loss."""
    device = torch.device("cpu")
    model.to(device)

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    class_weight = torch.tensor(
        [len(labels) / (2 * max(n_neg, 1)), len(labels) / (2 * max(n_pos, 1))],
        dtype=torch.float32,
    )
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    # Bara de parametrar som INTE är frysta (requires_grad=True) skickas till optimizern.
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)

    indices = list(range(len(texts)))
    avg_loss = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(indices)
        total_loss, n_batches = 0.0, 0
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_texts = [texts[i] for i in batch_idx]
            batch_labels = torch.tensor([labels[i] for i in batch_idx], dtype=torch.long, device=device)
            enc = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            outputs = model(**enc)
            loss = loss_fn(outputs.logits, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        log_fn(f"  epoch {epoch}/{epochs}  loss={avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def predict_anomal_proba(model, tokenizer, texts: list[str], max_length: int, batch_size: int = 32) -> list[float]:
    """Kör texterna genom modellen och returnerar P(anomal) (softmax av logiterna,
    klass 1) för var och en -- det här är avvikelse-scoren."""
    model.eval()
    probs: list[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        logits = model(**enc).logits
        probs.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
    return probs


# ============================================================================
# Streamlit-UI
# ============================================================================
st.set_page_config(page_title="LogBERT: Transfer Learning", page_icon="🔁", layout="wide")
st.title("LogBERT via Transfer Learning")
st.caption(
    "Finjustera en riktig förtränad textmodell (DistilBERT/BERT) för att klassificera "
    "loggrader som normala eller avvikande -- transfer learning istället för att träna "
    "en egen modell från grunden."
)

for _key in ("dataset", "trained", "eval", "analysis"):
    st.session_state.setdefault(_key, None)


# ----------------------------------------------------------------------------
# Steg 1: Ladda upp + filtrera
# ----------------------------------------------------------------------------
st.header("1. Ladda upp loggar och filtrera")
st.write(
    "Rader som innehåller något av orden nedan räknas som **anomal**, resten som "
    "**normal**. BÅDA klasserna används vid träningen (steg 3) -- en klassificerare "
    "behöver se exempel på det den ska lära sig skilja ut."
)

uploaded_files = st.file_uploader("Loggfiler (en eller flera textfiler)", accept_multiple_files=True)
keywords_raw = st.text_input(
    "Ord/fraser att filtrera bort (kommaseparerat, skiftlägesokänsligt)",
    value="error, failed, exception",
)
header_format = st.selectbox(
    "Radformat", options=list(HEADER_FORMATS.keys()),
    format_func=lambda k: HEADER_LABELS.get(k, k),
)

col_a, col_b = st.columns(2)
with col_a:
    val_fraction = st.slider("Andel som valideringsdel (av både normal och anomal)", 0.05, 0.3, 0.15)
with col_b:
    test_fraction = st.slider("Andel som testdel (av både normal och anomal)", 0.05, 0.3, 0.15)

if st.button("Bygg dataset", type="primary", disabled=not uploaded_files):
    keywords = [k.strip().lower() for k in keywords_raw.split(",") if k.strip()]
    if not keywords:
        st.warning("Inga filtreringsord angivna -- alla rader blir normala (ingen anomal-data att träna/validera mot).")

    # ---- Filtrera: dela varje uppladdad fil i normala/anomala rader ----------
    normal_lines: list[str] = []
    anomal_lines: list[str] = []
    per_file_rows = []
    normal_dir = DATASET_DIR / "normal"
    anomal_dir = DATASET_DIR / "anomal"
    normal_dir.mkdir(parents=True, exist_ok=True)
    anomal_dir.mkdir(parents=True, exist_ok=True)

    for uf in uploaded_files:
        text = uf.getvalue().decode("utf-8", errors="replace")
        lines = [line for line in text.splitlines() if line.strip()]
        f_normal = [line for line in lines if not any(kw in line.lower() for kw in keywords)]
        f_anomal = [line for line in lines if any(kw in line.lower() for kw in keywords)]
        normal_lines.extend(f_normal)
        anomal_lines.extend(f_anomal)
        per_file_rows.append({"fil": uf.name, "rader": len(lines), "normal": len(f_normal), "anomal": len(f_anomal)})
        (normal_dir / uf.name).write_text("\n".join(f_normal), encoding="utf-8")
        (anomal_dir / uf.name).write_text("\n".join(f_anomal), encoding="utf-8")

    total = len(normal_lines) + len(anomal_lines)
    if total == 0:
        st.error("Inga rader hittades i de uppladdade filerna.")
        st.stop()
    if len(anomal_lines) < 10:
        st.error(f"För få anomal-rader ({len(anomal_lines)}) för att träna en klassificerare -- lägg till fler filtreringsord eller mer data.")
        st.stop()

    st.write(
        f"**{total} rader totalt** -- {len(normal_lines)} normala ({len(normal_lines) / total * 100:.0f}%), "
        f"{len(anomal_lines)} anomala ({len(anomal_lines) / total * 100:.0f}%). "
        f"Sparade under `{DATASET_DIR}/normal/` respektive `/anomal/`."
    )
    st.dataframe(per_file_rows, use_container_width=True, hide_index=True)

    # ---- Dela normal OCH anomal var för sig, KRONOLOGISKT, i tre delar -------
    # Kronologiskt (inte slumpmässigt): sista biten av loggen används för
    # validering/test, så vi mäter hur bra modellen klarar mönster den ännu inte
    # sett, inte bara mönster den "kikat" på under träningen.
    header_re = HEADER_FORMATS[header_format]

    def split_and_extract(lines: list[str]) -> tuple[list[str], list[str], list[str]]:
        n = len(lines)
        n_test = max(1, int(n * test_fraction))
        n_val = max(1, int(n * val_fraction))
        train = lines[: n - n_val - n_test]
        val = lines[n - n_val - n_test : n - n_test]
        test = lines[n - n_test :]
        extract = lambda ls: [extract_message(l, header_re) for l in ls]
        return extract(train), extract(val), extract(test)

    normal_train, normal_val, normal_test = split_and_extract(normal_lines)
    anomal_train, anomal_val, anomal_test = split_and_extract(anomal_lines)

    dataset = {
        "header_re": header_re,
        "header_format": header_format,
        "train_texts": normal_train + anomal_train,
        "train_labels": [0] * len(normal_train) + [1] * len(anomal_train),
        "val_texts": normal_val + anomal_val,
        "val_labels": [0] * len(normal_val) + [1] * len(anomal_val),
        "test_texts": normal_test + anomal_test,
        "test_labels": [0] * len(normal_test) + [1] * len(anomal_test),
    }
    st.session_state.dataset = dataset
    st.session_state.trained = None  # ny data -> en ev. gammal modell gäller inte längre
    st.session_state.eval = None
    st.session_state.analysis = None

    st.success(
        f"Dataset byggt. Träning: {len(dataset['train_texts'])} rader "
        f"({sum(dataset['train_labels'])} anomala). Validering: {len(dataset['val_texts'])} rader. "
        f"Test: {len(dataset['test_texts'])} rader."
    )


# ----------------------------------------------------------------------------
# Steg 2: Träna (finjustera)
# ----------------------------------------------------------------------------
st.header("2. Finjustera en förtränad modell")
dataset = st.session_state.dataset
if dataset is None:
    st.info("Bygg ett dataset i steg 1 först.")
else:
    st.write(
        "Vi laddar en modell som redan är tränad på stora mängder allmän text, byter "
        "ut dess huvud mot ett nytt, slumpmässigt initierat klassificeringshuvud, "
        "fryser de understa transformer-blocken (de ändras inte alls) och finjusterar "
        "bara resten på våra loggrader."
    )
    model_choice = st.selectbox(
        "Basmodell", options=list(MODEL_CHOICES.keys()),
        format_func=lambda k: MODEL_CHOICES[k]["label"],
    )
    num_layers = MODEL_CHOICES[model_choice]["num_layers"]

    tc1, tc2, tc3 = st.columns(3)
    with tc1:
        num_frozen = st.slider(
            "Antal frysta transformer-block (räknat från botten)", 0, num_layers,
            value=max(0, num_layers - 2),
            help="Fler frysta block = snabbare träning men mindre anpassning till loggdatan. "
                 "0 = finjustera hela modellen, alla utom klassificeringshuvudet är förtränat ändå.",
        )
        epochs = st.number_input("Epoker", min_value=1, max_value=50, value=3)
    with tc2:
        max_length = st.number_input("Max antal tokens per rad", min_value=8, max_value=256, value=64)
        batch_size = st.number_input("Batchstorlek", min_value=4, max_value=128, value=16)
    with tc3:
        lr = st.select_slider("Inlärningshastighet", options=[1e-5, 2e-5, 5e-5, 1e-4], value=2e-5)
        percentile = st.slider("Tröskel-percentil (av valideringsdelens P(anomal) för normala rader)", 50.0, 100.0, 95.0)

    max_train = st.number_input(
        "Max antal träningsrader (slumpmässigt urval om fler -- håller CPU-träningen snabb)",
        min_value=100, max_value=20000, value=2000,
    )

    if st.button("Träna", type="primary"):
        train_texts, train_labels = dataset["train_texts"], dataset["train_labels"]
        if len(train_texts) > max_train:
            idx = random.sample(range(len(train_texts)), max_train)
            train_texts = [train_texts[i] for i in idx]
            train_labels = [train_labels[i] for i in idx]
            st.write(f"Tränar på ett slumpmässigt urval av {max_train} rader (av {len(dataset['train_texts'])}).")

        with st.spinner(f"Laddar {model_choice} (laddas ner första gången, kan ta en stund)..."):
            tokenizer = AutoTokenizer.from_pretrained(model_choice)
            model = AutoModelForSequenceClassification.from_pretrained(model_choice, num_labels=2)
            freeze_bottom_layers(model, num_frozen)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        st.write(f"Tränbara parametrar: {trainable:,} av {total_params:,} totalt ({trainable / total_params * 100:.1f}%).")

        loss_history: list[float] = []
        chart_placeholder = st.empty()
        progress = st.progress(0.0)

        def log_fn(msg: str) -> None:
            if "loss=" in msg:
                try:
                    loss_history.append(float(msg.split("loss=")[1]))
                except ValueError:
                    return
                chart_placeholder.line_chart(pd.DataFrame({"loss": loss_history}))
                progress.progress(min(len(loss_history) / epochs, 1.0))

        with st.spinner("Finjusterar..."):
            final_loss = finetune(
                model, tokenizer, train_texts, train_labels,
                epochs=epochs, batch_size=batch_size, lr=lr, max_length=max_length, log_fn=log_fn,
            )

            normal_val_texts = [t for t, y in zip(dataset["val_texts"], dataset["val_labels"]) if y == 0]
            val_probs_normal = predict_anomal_proba(model, tokenizer, normal_val_texts, max_length) if normal_val_texts else []
            threshold = float(np.percentile(val_probs_normal, percentile)) if val_probs_normal else 0.5

        st.session_state.trained = {
            "model": model, "tokenizer": tokenizer, "model_choice": model_choice,
            "max_length": max_length, "threshold": threshold, "final_loss": final_loss,
        }
        st.session_state.eval = None
        st.session_state.analysis = None
        st.success(f"Klar. Slutlig loss={final_loss:.4f}, tröskel={threshold:.3f} (percentil {percentile:.0f}%).")


# ----------------------------------------------------------------------------
# Steg 3: Validera
# ----------------------------------------------------------------------------
st.header("3. Validera")
trained = st.session_state.trained
if dataset is None or trained is None:
    st.info("Bygg dataset och träna en modell först.")
else:
    if st.button("Validera", type="primary"):
        model, tokenizer = trained["model"], trained["tokenizer"]
        max_length, threshold = trained["max_length"], trained["threshold"]

        with st.spinner("Poängsätter testdelen..."):
            probs = predict_anomal_proba(model, tokenizer, dataset["test_texts"], max_length)

        y_true = dataset["test_labels"]
        y_pred = [1 if p > threshold else 0 for p in probs]

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        roc_auc = roc_auc_score(y_true, probs) if len(set(y_true)) > 1 else None
        # labels=[0, 1] tvingar fram en fullständig 2x2-matris även om testdelen råkar
        # sakna en av klasserna, istället för att sklearn tyst krymper matrisen.
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        st.session_state.eval = {
            "normal_probs": [p for p, y in zip(probs, y_true) if y == 0],
            "anomal_probs": [p for p, y in zip(probs, y_true) if y == 1],
            "precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc,
            "confusion_matrix": cm,
        }

    ev = st.session_state.eval
    if ev:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{ev['precision'] * 100:.1f}%")
        c2.metric("Recall", f"{ev['recall'] * 100:.1f}%")
        c3.metric("F1", f"{ev['f1'] * 100:.1f}%")
        c4.metric("ROC-AUC", f"{ev['roc_auc']:.3f}" if ev["roc_auc"] is not None else "-- (bara en klass i testdata)")

        st.write("**Confusion matrix:**")
        cm = ev["confusion_matrix"]
        (tn, fp), (fn, tp) = cm
        cm_df = pd.DataFrame(
            cm, index=["Faktisk: Normal", "Faktisk: Anomal"], columns=["Predikterad: Normal", "Predikterad: Anomal"],
        )

        # Manuell cellfärgning per rad (inget matplotlib-beroende, till skillnad från
        # pandas' inbyggda .background_gradient()): mörkare blå ju högre andel av
        # RADEN (dvs. av den faktiska klassen) cellen utgör.
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # undvik division med noll om en klass saknas i testdelen
        shade = (cm / row_sums * 255).astype(int)

        def style_row(row: pd.Series) -> list[str]:
            row_idx = list(cm_df.index).index(row.name)
            return [
                f"background-color: rgb({255 - shade[row_idx, j]}, {255 - shade[row_idx, j]}, 255); "
                "color: black; font-weight: 600"
                for j in range(len(row))
            ]

        st.dataframe(cm_df.style.apply(style_row, axis=1), use_container_width=True)
        st.caption(
            f"TP={tp} (avvikande korrekt flaggade), FP={fp} (normala felaktigt flaggade), "
            f"FN={fn} (avvikande som missades), TN={tn} (normala korrekt lämnade ifred). "
            f"Precision = TP/(TP+FP) = {tp}/{tp + fp}. Recall = TP/(TP+FN) = {tp}/{tp + fn}."
        )

        st.write("**Score-fördelning (P(anomal) enligt modellen): normal vs. anomal**")
        bins = np.linspace(0, 1, 21)
        normal_hist, _ = np.histogram(ev["normal_probs"], bins=bins)
        anomal_hist, _ = np.histogram(ev["anomal_probs"], bins=bins)
        hist_df = pd.DataFrame({"Normal": normal_hist, "Anomal": anomal_hist}, index=[f"{b:.2f}" for b in bins[:-1]])
        st.bar_chart(hist_df)
        st.caption(
            f"Tröskel={trained['threshold']:.3f} -- rader med högre P(anomal) än så flaggas som "
            "avvikande. En bra modell ska ha normal-fördelningen samlad nära 0 och "
            "anomal-fördelningen förskjuten mot 1."
        )


# ----------------------------------------------------------------------------
# Steg 4: Använd modellen -- analysera en ny loggfil
# ----------------------------------------------------------------------------
st.header("4. Analysera en ny loggfil")
if dataset is None or trained is None:
    st.info("Bygg dataset och träna en modell först.")
else:
    st.write("Ladda upp en (eller flera) loggfiler så klassificeras varje rad, och du får en lista över de som ser avvikande ut.")
    analyze_files = st.file_uploader("Loggfil(er) att analysera", accept_multiple_files=True, key="analyze_files")

    if st.button("Analysera", type="primary", disabled=not analyze_files):
        model, tokenizer = trained["model"], trained["tokenizer"]
        max_length, threshold = trained["max_length"], trained["threshold"]
        header_re = dataset["header_re"]

        all_rows = []
        per_file_summary = []
        for uf in analyze_files:
            text = uf.getvalue().decode("utf-8", errors="replace")
            lines = [line for line in text.splitlines() if line.strip()]
            if not lines:
                continue

            with st.spinner(f"Analyserar {uf.name} ({len(lines)} rader)..."):
                texts = [extract_message(l, header_re) for l in lines]
                probs = predict_anomal_proba(model, tokenizer, texts, max_length)

            for line, prob in zip(lines, probs):
                all_rows.append({"fil": uf.name, "P(anomal)": round(prob, 3), "avvikande": prob > threshold, "radtext": line[:200]})

            flagged_in_file = sum(1 for p in probs if p > threshold)
            per_file_summary.append({"fil": uf.name, "rader": len(lines), "flaggade": flagged_in_file})

        st.session_state.analysis = {"rows": all_rows, "per_file": per_file_summary}

    analysis = st.session_state.analysis
    if analysis and analysis["rows"]:
        rows = analysis["rows"]
        flagged_rows = [r for r in rows if r["avvikande"]]

        st.write("**Sammanfattning per fil:**")
        st.dataframe(analysis["per_file"], use_container_width=True, hide_index=True)

        c1, c2 = st.columns(2)
        c1.metric("Rader totalt", len(rows))
        c2.metric("Flaggade som avvikande", f"{len(flagged_rows)} ({len(flagged_rows) / len(rows) * 100:.0f}%)")

        st.write("**Avvikelser (sorterade efter P(anomal), högst först):**")
        if flagged_rows:
            flagged_sorted = sorted(flagged_rows, key=lambda r: -r["P(anomal)"])
            st.dataframe(flagged_sorted, use_container_width=True, hide_index=True)

            csv = pd.DataFrame(flagged_sorted).to_csv(index=False).encode("utf-8")
            st.download_button("Ladda ner avvikelser som CSV", data=csv, file_name="avvikelser_transfer.csv", mime="text/csv")
        else:
            st.success("Inga avvikelser flaggades i den uppladdade filen/filerna.")

        with st.expander("Alla rader (även normala)"):
            st.dataframe(rows, use_container_width=True, hide_index=True)
