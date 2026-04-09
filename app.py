import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import zipfile

st.set_page_config(
    page_title="Enerji Tüketimi Tahmini",
    page_icon="⚡",
    layout="wide"
)

# ── Sidebar navigation ──────────────────────────────────────────────────────

st.sidebar.title("⚡ Enerji Tahmini")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Sayfa Seç",
    ["🏠 Ana Sayfa", "📊 Keşifsel Veri Analizi", "🤖 Model & Tahmin", "📈 Model Bilgisi"],
)
st.sidebar.markdown("---")

# Model file uploader (always visible in sidebar)
st.sidebar.subheader("Model Yükle")
model_file = st.sidebar.file_uploader(
    "LSTM model dosyası (.h5 veya .keras)",
    type=["h5", "keras"],
    key="model_uploader",
)

# ── Helper functions ─────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    """Load household_power_consumption data from raw bytes."""
    if file_name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:
            txt_names = [n for n in z.namelist() if n.endswith(".txt")]
            if not txt_names:
                st.error("ZIP içinde .txt dosyası bulunamadı.")
                return pd.DataFrame()
            with z.open(txt_names[0]) as f:
                raw = f.read()
    else:
        raw = file_bytes

    df = pd.read_csv(
        io.BytesIO(raw),
        sep=";",
        encoding="latin-1",
        na_values=["?"],
        low_memory=False,
    )
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], dayfirst=True, errors="coerce"
    )
    numeric_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Datetime"] + [c for c in numeric_cols if c in df.columns])
    if "Sub_metering_3" in df.columns:
        df = df.drop(columns=["Sub_metering_3"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def get_sample_data() -> pd.DataFrame:
    """Return a small deterministic demo dataset."""
    rng = np.random.default_rng(42)
    n = 10_000
    start = pd.Timestamp("2007-01-01 00:00:00")
    datetimes = pd.date_range(start, periods=n, freq="min")
    df = pd.DataFrame(
        {
            "Datetime": datetimes,
            "Global_active_power": np.abs(rng.normal(1.2, 0.6, n)),
            "Global_reactive_power": np.abs(rng.normal(0.2, 0.05, n)),
            "Voltage": rng.normal(240, 3, n),
            "Global_intensity": np.abs(rng.normal(5.5, 2.5, n)),
            "Sub_metering_1": np.abs(rng.normal(0.3, 0.6, n)),
            "Sub_metering_2": np.abs(rng.normal(1.0, 1.8, n)),
        }
    )
    return df


@st.cache_data(show_spinner=False)
def daily_agg(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["Day"] = tmp["Datetime"].dt.date
    return tmp.groupby("Day")["Global_active_power"].mean().reset_index()


@st.cache_data(show_spinner=False)
def hourly_agg(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["Hour"] = tmp["Datetime"].dt.hour
    return tmp.groupby("Hour")["Global_active_power"].mean().reset_index()


@st.cache_data(show_spinner=False)
def corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
    ]
    cols = [c for c in numeric_cols if c in df.columns]
    return df[cols].corr()


def load_model(model_bytes: bytes, suffix: str):
    """Load a Keras/TF model from bytes."""
    try:
        import tensorflow as tf
        import tempfile, os

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(model_bytes)
            tmp_path = tmp.name
        model = tf.keras.models.load_model(tmp_path)
        os.unlink(tmp_path)
        return model
    except Exception as e:
        st.error(f"Model yüklenirken hata oluştu: {e}")
        return None


def moving_average_forecast(series: np.ndarray, lookback: int = 60, steps: int = 1) -> np.ndarray:
    """Predict next `steps` values as the mean of the last `lookback` values."""
    preds = []
    window = list(series[-lookback:])
    for _ in range(steps):
        pred = float(np.mean(window))
        preds.append(pred)
        window.pop(0)
        window.append(pred)
    return np.array(preds)


def prepare_sequences(values: np.ndarray, lookback: int = 60):
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values.reshape(-1, 1))
    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback : i, 0])
        y.append(scaled[i, 0])
    return np.array(X), np.array(y), scaler


# ── Page 1: Ana Sayfa ─────────────────────────────────────────────────────────

if page == "🏠 Ana Sayfa":
    st.title("⚡ Ev Elektrik Tüketimi Tahmini")
    st.markdown(
        """
        Bu uygulama, bir evin dakika bazlı elektrik tüketim verilerini analiz ederek
        **Global Aktif Güç** değerini **LSTM** derin öğrenme modeli ile tahmin eder.

        Veri seti: [UCI Machine Learning Repository — Individual household electric power consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
        """
    )

    st.markdown("---")
    st.subheader("📂 Veri Yükle")
    uploaded = st.file_uploader(
        "household_power_consumption.txt veya .zip dosyasını yükleyin",
        type=["txt", "zip"],
        key="data_uploader",
    )

    if uploaded is not None:
        with st.spinner("Veri yükleniyor…"):
            df = load_data(uploaded.read(), uploaded.name)
        st.session_state["df"] = df
        st.success(f"✅ {len(df):,} satır başarıyla yüklendi.")
    elif "df" not in st.session_state:
        st.info("ℹ️ Henüz dosya yüklenmedi. Demo amaçlı örnek veri (10.000 satır) kullanılıyor.")
        df = get_sample_data()
        st.session_state["df"] = df
        st.session_state["is_demo"] = True
    else:
        df = st.session_state["df"]

    if st.session_state.get("is_demo") and uploaded is None:
        st.warning("⚠️ Demo moddasınız. Gerçek veriyi görmek için dosya yükleyin.")

    st.markdown("---")
    st.subheader("🔍 İlk 10 Satır")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Özet Metrikler")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📝 Toplam Kayıt", f"{len(df):,}")
    date_range = f"{df['Datetime'].min().strftime('%d/%m/%Y')} – {df['Datetime'].max().strftime('%d/%m/%Y')}"
    col2.metric("📅 Tarih Aralığı", date_range)
    col3.metric("⚡ Ort. Global Active Power", f"{df['Global_active_power'].mean():.3f} kW")
    col4.metric("🔝 Max Global Active Power", f"{df['Global_active_power'].max():.3f} kW")

    st.markdown("---")
    st.subheader("📈 Temel İstatistikler")
    numeric_cols = [
        "Global_active_power",
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
    ]
    describe_cols = [c for c in numeric_cols if c in df.columns]
    st.dataframe(df[describe_cols].describe().T, use_container_width=True)


# ── Page 2: EDA ───────────────────────────────────────────────────────────────

elif page == "📊 Keşifsel Veri Analizi":
    st.title("📊 Keşifsel Veri Analizi")

    if "df" not in st.session_state:
        st.warning("⚠️ Lütfen önce Ana Sayfa'dan veri yükleyin.")
        st.stop()

    df = st.session_state["df"]

    # Daily average
    st.subheader("📅 Günlük Ortalama Global Active Power")
    with st.spinner("Grafik hazırlanıyor…"):
        daily = daily_agg(df)
    fig1 = px.line(
        daily,
        x="Day",
        y="Global_active_power",
        title="Günlük Ortalama Global Aktif Güç (kW)",
        color_discrete_sequence=["#FA8072"],
    )
    fig1.update_layout(xaxis_title="Tarih", yaxis_title="Ortalama Güç (kW)")
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # Hourly average
    st.subheader("🕐 Saatlik Ortalama Global Active Power")
    with st.spinner("Grafik hazırlanıyor…"):
        hourly = hourly_agg(df)
    fig2 = px.bar(
        hourly,
        x="Hour",
        y="Global_active_power",
        title="Saatlik Ortalama Global Aktif Güç (kW)",
        color_discrete_sequence=["#87CEEB"],
    )
    fig2.update_layout(xaxis_title="Saat", yaxis_title="Ortalama Güç (kW)")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # Correlation heatmap
    st.subheader("🌡️ Korelasyon Isı Haritası")
    with st.spinner("Korelasyon hesaplanıyor…"):
        corr = corr_matrix(df)
    fig3 = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.index.tolist(),
            colorscale="PiYG",
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
        )
    )
    fig3.update_layout(title="Değişkenler Arası Korelasyon Matrisi")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # Distribution histogram
    st.subheader("📉 Global Active Power Dağılımı")
    fig4 = px.histogram(
        df,
        x="Global_active_power",
        nbins=100,
        title="Global Aktif Güç Dağılımı",
        color_discrete_sequence=["#20B2AA"],
    )
    fig4.update_layout(xaxis_title="Global Aktif Güç (kW)", yaxis_title="Frekans")
    st.plotly_chart(fig4, use_container_width=True)


# ── Page 3: Model & Tahmin ────────────────────────────────────────────────────

elif page == "🤖 Model & Tahmin":
    st.title("🤖 Model & Tahmin")

    if "df" not in st.session_state:
        st.warning("⚠️ Lütfen önce Ana Sayfa'dan veri yükleyin.")
        st.stop()

    df = st.session_state["df"]
    values = df["Global_active_power"].values

    # ── Real values chart (always shown) ──────────────────────────────────────
    st.subheader("📊 Son 1000 Gerçek Değer")
    last_1000 = df.tail(1000).copy()
    fig_real = px.line(
        last_1000,
        x="Datetime",
        y="Global_active_power",
        title="Son 1000 Dakikanın Global Aktif Güç Değerleri",
        color_discrete_sequence=["#20B2AA"],
    )
    fig_real.update_layout(xaxis_title="Zaman", yaxis_title="Global Aktif Güç (kW)")
    st.plotly_chart(fig_real, use_container_width=True)

    st.markdown("---")

    # ── Model section ──────────────────────────────────────────────────────────
    if model_file is None:
        st.warning(
            "⚠️ Model dosyası yüklenmedi. LSTM modelini yeniden eğitmek için "
            "notebook'u çalıştırabilirsiniz."
        )
        st.subheader("📐 Hareketli Ortalama Tahmini (Alternatif)")
        steps = st.slider("Kaç dakika ilerisi tahmin edilsin?", 1, 120, 30)
        if st.button("Tahmin Et (Hareketli Ortalama)"):
            with st.spinner("Tahmin yapılıyor…"):
                preds = moving_average_forecast(values, lookback=60, steps=steps)
                last_dt = df["Datetime"].iloc[-1]
                pred_times = pd.date_range(last_dt, periods=steps + 1, freq="min")[1:]
                pred_df = pd.DataFrame(
                    {"Datetime": pred_times, "Tahmin (kW)": preds}
                )

            fig_ma = go.Figure()
            fig_ma.add_trace(
                go.Scatter(
                    x=df["Datetime"].tail(500),
                    y=values[-500:],
                    name="Gerçek",
                    line=dict(color="#4682B4"),
                )
            )
            fig_ma.add_trace(
                go.Scatter(
                    x=pred_df["Datetime"],
                    y=pred_df["Tahmin (kW)"],
                    name="Tahmin (Hareketli Ort.)",
                    line=dict(color="#FA8072", dash="dash"),
                )
            )
            fig_ma.update_layout(
                title="Hareketli Ortalama Tahmini",
                xaxis_title="Zaman",
                yaxis_title="Global Aktif Güç (kW)",
            )
            st.plotly_chart(fig_ma, use_container_width=True)
            st.dataframe(pred_df, use_container_width=True)
            csv = pred_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Tahminleri İndir (CSV)",
                data=csv,
                file_name="moving_avg_predictions.csv",
                mime="text/csv",
            )
    else:
        # Model loaded
        suffix = ".keras" if model_file.name.endswith(".keras") else ".h5"
        with st.spinner("Model yükleniyor…"):
            model = load_model(model_file.read(), suffix)

        if model is not None:
            st.success("✅ Model başarıyla yüklendi.")
            steps = st.slider("Kaç dakika ilerisi tahmin edilsin?", 1, 120, 30)

            if st.button("Tahmin Et"):
                with st.spinner("Tahmin yapılıyor…"):
                    LOOKBACK = 60
                    use_values = values[-10_000:]
                    X, y_true, scaler = prepare_sequences(use_values, lookback=LOOKBACK)
                    X_3d = X.reshape((X.shape[0], X.shape[1], 1))

                    # Predict on test portion
                    split = int(len(X_3d) * 0.8)
                    X_test = X_3d[split:]
                    y_test = y_true[split:]
                    y_pred_scaled = model.predict(X_test, verbose=0).flatten()

                    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
                    y_pred_inv = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

                    mse = float(np.mean((y_test_inv - y_pred_inv) ** 2))
                    mae = float(np.mean(np.abs(y_test_inv - y_pred_inv)))

                    # Future forecast
                    last_seq = X_3d[-1].copy()
                    future_preds_scaled = []
                    for _ in range(steps):
                        p = model.predict(last_seq.reshape(1, LOOKBACK, 1), verbose=0)[0, 0]
                        future_preds_scaled.append(p)
                        last_seq = np.roll(last_seq, -1, axis=0)
                        last_seq[-1, 0] = p
                    future_preds = scaler.inverse_transform(
                        np.array(future_preds_scaled).reshape(-1, 1)
                    ).flatten()
                    last_dt = df["Datetime"].iloc[-1]
                    pred_times = pd.date_range(last_dt, periods=steps + 1, freq="min")[1:]

                # Metrics
                c1, c2 = st.columns(2)
                c1.metric("📉 MSE", f"{mse:.4f}")
                c2.metric("📏 MAE", f"{mae:.4f}")

                # Plot last 500 real + predictions
                n_show = min(500, len(y_test_inv))
                fig_pred = go.Figure()
                fig_pred.add_trace(
                    go.Scatter(
                        x=df["Datetime"].tail(n_show),
                        y=y_test_inv[-n_show:],
                        name="Gerçek",
                        line=dict(color="#4682B4"),
                    )
                )
                fig_pred.add_trace(
                    go.Scatter(
                        x=df["Datetime"].tail(n_show),
                        y=y_pred_inv[-n_show:],
                        name="Model Tahmini (test)",
                        line=dict(color="#FA8072"),
                    )
                )
                fig_pred.add_trace(
                    go.Scatter(
                        x=pred_times,
                        y=future_preds,
                        name=f"Gelecek {steps} dk Tahmini",
                        line=dict(color="#20B2AA", dash="dash"),
                    )
                )
                fig_pred.update_layout(
                    title="LSTM Model Tahminleri",
                    xaxis_title="Zaman",
                    yaxis_title="Global Aktif Güç (kW)",
                )
                st.plotly_chart(fig_pred, use_container_width=True)

                # Download table
                pred_df = pd.DataFrame(
                    {"Datetime": pred_times, "Tahmin (kW)": future_preds}
                )
                st.subheader("📋 Tahmin Tablosu")
                st.dataframe(pred_df, use_container_width=True)
                csv = pred_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Tahminleri İndir (CSV)",
                    data=csv,
                    file_name="lstm_predictions.csv",
                    mime="text/csv",
                )


# ── Page 4: Model Bilgisi ─────────────────────────────────────────────────────

elif page == "📈 Model Bilgisi":
    st.title("📈 Model Bilgisi")

    st.subheader("🧠 LSTM Nedir?")
    st.markdown(
        """
        **LSTM (Long Short-Term Memory)**, tekrarlayan sinir ağlarının (RNN) özel bir türüdür.
        Klasik RNN'lerin uzun vadeli bağımlılıkları öğrenememe sorununu çözmek için
        **bellek hücreleri** ve **kapı mekanizmaları** (input, forget, output kapıları) kullanır.

        Zaman serisi tahmininde LSTM:
        - Geçmişteki örüntüleri öğrenir
        - Uzun dönemli bağımlılıkları korur
        - Enerji tüketimi gibi düzenli periyodik verilerde yüksek başarı sağlar
        """
    )

    st.markdown("---")
    st.subheader("🏗️ Model Mimarisi")
    arch_data = {
        "Katman": ["Katman 1", "Katman 2", "Katman 3", "Katman 4", "Katman 5"],
        "Tür": ["LSTM", "Dropout", "LSTM", "Dropout", "Dense"],
        "Parametreler": [
            "50 birim, relu aktivasyon, return_sequences=True",
            "Oran: 0.2",
            "50 birim, relu aktivasyon",
            "Oran: 0.2",
            "1 çıkış nöronu",
        ],
    }
    st.dataframe(pd.DataFrame(arch_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("⚙️ Eğitim Parametreleri")
    train_data = {
        "Parametre": [
            "Optimizer",
            "Loss Fonksiyonu",
            "Metrik",
            "Epoch Sayısı",
            "Batch Size",
            "Lookback (Pencere)",
            "Train / Test Oranı",
        ],
        "Değer": [
            "Adam (lr=0.001)",
            "MSE (Mean Squared Error)",
            "MAE (Mean Absolute Error)",
            "5",
            "32",
            "60 dakika",
            "%80 / %20",
        ],
    }
    st.dataframe(pd.DataFrame(train_data), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔧 Veri Ön İşleme Adımları")
    st.markdown(
        """
        1. **Veri Yükleme** — `household_power_consumption.txt`, `;` ayraçlı, `latin-1` kodlama
        2. **Datetime Birleştirme** — `Date` + `Time` sütunları tek `Datetime` sütununa dönüştürüldü
        3. **Eksik Değer Temizleme** — `?` değerleri NaN'e dönüştürülerek 25.979 satır silindi
        4. **Sütun Seçimi** — `Sub_metering_3` sütunu kaldırıldı; 6 sayısal özellik kullanıldı
        5. **Normalizasyon** — `MinMaxScaler` ile [0, 1] aralığına ölçekleme
        6. **Sequence Oluşturma** — 60 adımlı kayan pencere (lookback=60) ile X/y dizileri oluşturuldu
        7. **Train/Test Bölümü** — %80 eğitim, %20 test (zamana göre sıralı bölüm)
        """
    )

    st.markdown("---")
    st.subheader("🔗 Bağlantılar")
    st.markdown(
        """
        - 📂 [GitHub Repo](https://github.com/tugcesi/energy-consumption-prediction)
        - 📊 [UCI Veri Seti](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
        """
    )
