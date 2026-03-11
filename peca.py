import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import requests
import urllib3

# Biztonsági figyelmeztetések kikapcsolása
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Weboldal beállításai
st.set_page_config(page_title="Horgász Időgép", layout="wide", page_icon="🎣")
st.title("🎣 Valós Idejű Horgász Időgép")
st.write("Húzd el a bal oldali csúszkát, hogy lásd a következő 7 nap kapási esélyeit!")

# ==========================================
# 1. ADATOK LEKÉRÉSE (Gyorsítótárazva)
# ==========================================
@st.cache_data(ttl=3600)
def letolt_idojaras():
    szelesseg = 47.68
    hosszusag = 21.66
    api_url = f"https://api.open-meteo.com/v1/forecast?latitude={szelesseg}&longitude={hosszusag}&hourly=temperature_2m,surface_pressure&timezone=auto"
    try:
        valasz = requests.get(api_url, verify=False)
        adatok = valasz.json()
        return adatok['hourly']['time'], adatok['hourly']['temperature_2m'], adatok['hourly']['surface_pressure']
    except Exception as e:
        st.error(f"Hiba az internetes lekérésnél: {e}")
        return [f"2026-03-07T{i:02}:00" for i in range(168)], [18.0] * 168, [1012.0] * 168

idopontok, homersekletek, nyomasok = letolt_idojaras()

# ==========================================
# 2. FUZZY LOGIKA
# ==========================================
vizho = ctrl.Antecedent(np.arange(0, 41, 1), 'Hőmérséklet')
nyomas = ctrl.Antecedent(np.arange(980, 1041, 1), 'Légnyomás')
esely = ctrl.Consequent(np.arange(0, 101, 1), 'Kapási_Esély')

vizho['hideg'] = fuzz.trimf(vizho.universe, [0, 0, 15])
vizho['idealis'] = fuzz.trimf(vizho.universe, [10, 18, 25])
vizho['meleg'] = fuzz.trimf(vizho.universe, [20, 40, 40])

nyomas['alacsony'] = fuzz.trimf(nyomas.universe, [980, 980, 1010])
nyomas['idealis'] = fuzz.trimf(nyomas.universe, [1005, 1013, 1020])
nyomas['magas'] = fuzz.trimf(nyomas.universe, [1015, 1040, 1040])

esely['rossz'] = fuzz.trimf(esely.universe, [0, 0, 40])
esely['kozepes'] = fuzz.trimf(esely.universe, [30, 50, 70])
esely['kivalo'] = fuzz.trimf(esely.universe, [60, 100, 100])

szabalyok = [
    ctrl.Rule(vizho['hideg'] & nyomas['alacsony'], esely['rossz']),
    ctrl.Rule(vizho['hideg'] & nyomas['idealis'], esely['kozepes']),
    ctrl.Rule(vizho['hideg'] & nyomas['magas'], esely['rossz']),
    ctrl.Rule(vizho['idealis'] & nyomas['alacsony'], esely['kozepes']),
    ctrl.Rule(vizho['idealis'] & nyomas['idealis'], esely['kivalo']),
    ctrl.Rule(vizho['idealis'] & nyomas['magas'], esely['kozepes']),
    ctrl.Rule(vizho['meleg'] & nyomas['alacsony'], esely['rossz']),
    ctrl.Rule(vizho['meleg'] & nyomas['idealis'], esely['kozepes']),
    ctrl.Rule(vizho['meleg'] & nyomas['magas'], esely['rossz'])
]

horgasz_vezerlo = ctrl.ControlSystem(szabalyok)
szimulacio = ctrl.ControlSystemSimulation(horgasz_vezerlo)

# ==========================================
# 3. WEBES FELÜLET ÉS CSÚSZKA
# ==========================================
st.sidebar.header("⏱️ Időgép Vezérlő")
ora_idx = st.sidebar.slider("Ugrás az időben (Óra mától)", 0, 167, 0)

kivalasztott_datum = idopontok[ora_idx].replace("T", " ")
aktualis_ho = np.clip(homersekletek[ora_idx], 0, 40)
aktualis_nyomas = np.clip(nyomasok[ora_idx], 980, 1040)

szimulacio.input['Hőmérséklet'] = aktualis_ho
szimulacio.input['Légnyomás'] = aktualis_nyomas
szimulacio.compute()
kalkulalt_esely = szimulacio.output['Kapási_Esély']

if kalkulalt_esely >= 60:
    ertekeles = "KIVÁLÓ 🟢"
elif kalkulalt_esely >= 40:
    ertekeles = "KÖZEPES 🟡"
else:
    ertekeles = "ROSSZ 🔴"

col1, col2, col3, col4 = st.columns(4)
col1.metric("📅 Dátum", kivalasztott_datum)
col2.metric("🌡️ Hőmérséklet", f"{aktualis_ho:.1f} °C")
col3.metric("☁️ Légnyomás", f"{aktualis_nyomas:.1f} hPa")
col4.metric("🎣 Kapási Esély", f"{kalkulalt_esely:.1f} %", ertekeles)

# ==========================================
# 4. A 3D GRAFIKON RAJZOLÁSA
# ==========================================
st.write("---")
st.subheader("A Fuzzy Logika 3D-s Térképe")

x_ho = np.arange(0, 41, 2)
y_nyom = np.arange(980, 1041, 2)
x, y = np.meshgrid(x_ho, y_nyom)
z = np.zeros_like(x, dtype=float)

for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        szimulacio.input['Hőmérséklet'] = x[i, j]
        szimulacio.input['Légnyomás'] = y[i, j]
        szimulacio.compute()
        z[i, j] = szimulacio.output['Kapási_Esély']

fig = plt.figure(figsize=(10, 6))
# Átlátszó háttér beállítása
fig.patch.set_facecolor('none') 

ax3d = fig.add_subplot(111, projection='3d')
surf = ax3d.plot_surface(x, y, z, cmap='ocean', alpha=0.9)
ax3d.set_facecolor('none')
ax3d.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax3d.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax3d.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))

ax3d.set_xlabel('Hőmérséklet (°C)', color='white')
ax3d.set_ylabel('Légnyomás (hPa)', color='white')
ax3d.set_zlabel('Kapási Esély (%)', color='white')
ax3d.tick_params(colors='white')

ax3d.plot([aktualis_ho], [aktualis_nyomas], [kalkulalt_esely], 'ro', markersize=12, label='Kiválasztott Időpont')
ax3d.legend(facecolor='white')

# Kiküldés a weboldalra átlátszó háttérrel
st.pyplot(fig, transparent=True)