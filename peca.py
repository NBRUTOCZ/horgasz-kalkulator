import streamlit as st
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import requests
import urllib3
from datetime import datetime, date

# Biztonsági figyelmeztetések kikapcsolása
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Weboldal beállításai
st.set_page_config(page_title="Horgász Időgép", layout="wide", page_icon="🎣")
st.title("🎣 Valós Idejű Horgász Időgép")
st.write("Válaszd ki a dátumot és az időpontot az előrejelzéshez!")

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
        ma = date.today().isoformat()
        return [f"{ma}T{i:02}:00" for i in range(168)], [18.0] * 168, [1012.0] * 168

idopontok, homersekletek, nyomasok = letolt_idojaras()

elso_idopont = datetime.fromisoformat(idopontok[0])
kezdo_datum = elso_idopont.date()
utolso_datum = datetime.fromisoformat(idopontok[-1]).date()

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
# 3. WEBES VEZÉRLŐK (DÁTUM ÉS ÓRA)
# ==========================================
st.sidebar.header("📅 Időpont Kiválasztása")

valasztott_nap = st.sidebar.date_input("Válassz napot:", value=kezdo_datum, min_value=kezdo_datum, max_value=utolso_datum)
valasztott_ora = st.sidebar.slider("Válassz órát:", 0, 23, 12)

nap_kulonbseg = (valasztott_nap - kezdo_datum).days
ora_idx = min(max(0, (nap_kulonbseg * 24) + valasztott_ora), 167)

kivalasztott_datum_szoveg = f"{valasztott_nap} {valasztott_ora:02}:00"
aktualis_ho = np.clip(homersekletek[ora_idx], 0, 40)
aktualis_nyomas = np.clip(nyomasok[ora_idx], 980, 1040)

szimulacio.input['Hőmérséklet'] = aktualis_ho
szimulacio.input['Légnyomás'] = aktualis_nyomas
szimulacio.compute()
kalkulalt_esely = szimulacio.output['Kapási_Esély']

if kalkulalt_esely >= 60: ertekeles = "KIVÁLÓ 🟢"
elif kalkulalt_esely >= 40: ertekeles = "KÖZEPES 🟡"
else: ertekeles = "ROSSZ 🔴"

col1, col2, col3, col4 = st.columns(4)
col1.metric("📅 Időpont", kivalasztott_datum_szoveg)
col2.metric("🌡️ Hőmérséklet", f"{aktualis_ho:.1f} °C")
col3.metric("☁️ Légnyomás", f"{aktualis_nyomas:.1f} hPa")
col4.metric("🎣 Kapási Esély", f"{kalkulalt_esely:.1f} %", ertekeles)

# ==========================================
# 4. ÚJ: NAPI ÁTLAGOK GRAFIKONJA (2D)
# ==========================================
st.write("---")
st.subheader("📊 Napi Átlagos Kapási Esélyek (7 Napos Előrejelzés)")

# 168 óra adatának átlagolása napokra
napi_eselyek = {}
for i in range(len(idopontok)):
    ido_obj = datetime.fromisoformat(idopontok[i])
    nap_str = ido_obj.strftime("%m.%d.") # Pl. "03.07."
    
    szimulacio.input['Hőmérséklet'] = np.clip(homersekletek[i], 0, 40)
    szimulacio.input['Légnyomás'] = np.clip(nyomasok[i], 980, 1040)
    szimulacio.compute()
    
    if nap_str not in napi_eselyek:
        napi_eselyek[nap_str] = []
    napi_eselyek[nap_str].append(szimulacio.output['Kapási_Esély'])

napok = list(napi_eselyek.keys())
atlagok = [np.mean(napi_eselyek[nap]) for nap in napok]

# 2D Grafikon rajzolása
fig, ax = plt.subplots(figsize=(10, 4))
fig.patch.set_facecolor('none') 
ax.set_facecolor('none')

# Kék vonal a heti trendnek
ax.plot(napok, atlagok, color='#00d4ff', linewidth=3, marker='o', markersize=8, label='Napi átlag')

# A bal oldalt kiválasztott napot rátesszük egy hatalmas piros pöttyel!
valasztott_nap_str = valasztott_nap.strftime("%m.%d.")
if valasztott_nap_str in napok:
    idx = napok.index(valasztott_nap_str)
    ax.plot(napok[idx], atlagok[idx], color='#ff4b4b', marker='o', markersize=14, label='Kiválasztott nap')

# Formázások (Tengelyek, rács, színek)
ax.set_ylim(0, 100)
ax.set_ylabel('Átlagos Kapási Esély (%)', color='white', fontsize=12)
ax.tick_params(colors='white', labelsize=11)
ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3) # Halvány rácsvonalak

for spine in ax.spines.values():
    spine.set_edgecolor('gray')

ax.legend(facecolor='#0E1117', edgecolor='white', labelcolor='white')

# Kiküldés a weblapra
st.pyplot(fig, transparent=True)
