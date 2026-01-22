# Návrhy na vylepšení projektu

Tento dokument obsahuje návrhy na budoucí vylepšení projektu creep_rupture_fit.

---

## 1. Testování

**Problém:** Projekt nemá automatické testy. Změny v kódu mohou způsobit regrese, které se odhalí až při manuálním testování.

**Řešení:**
- Přidat pytest jako dev dependency
- Unit testy pro klíčové funkce:
  - `fit_larson_miller()` - ověřit výstup na známých datech
  - `fit_wilshire()` - dtto
  - `predict_stress_for_tr()` - ověřit inverzní výpočet
  - `detect_breakpoints()` - ověřit BIC detekci
- Integrační testy:
  - CLI s example daty
  - Ověření výstupních formátů (JSON, CSV)

**Priorita:** Vysoká

---

## 2. Diagnostika fitu - Residual plot

**Problém:** Uživatel nemá snadný způsob jak vizuálně ověřit kvalitu fitu. R² může být vysoké, ale systematické chyby zůstanou skryté.

**Řešení:**
- Nový argument `--residual-plot [FILE]`
- Graf reziduí (y_exp - y_calc) vs:
  - Predikovaná hodnota
  - Teplota
  - Napětí
- Ideálně: rezidua náhodně rozptýlená kolem nuly

**Priorita:** Střední

---

## 3. Srovnání modelů

**Problém:** Pro porovnání LM a WSH musí uživatel spustit program dvakrát a ručně porovnat výsledky.

**Řešení:**
- Nový argument `--compare` (vyžaduje --tensile-data)
- Výstup:
  ```
  === Model Comparison ===

  Metric          Larson-Miller    Wilshire
  R²              0.9367           0.9784
  MSE             0.1104           0.0089

  Predictions for tr = 100000 h:
  T = 500 C       185.2 MPa        190.9 MPa
  T = 600 C       88.1 MPa         90.6 MPa
  ```

**Priorita:** Střední

---

## 4. Export fitted křivek

**Problém:** Exportují se jen parametry modelu. Uživatel, který chce vykreslit křivky v Excelu, musí sám implementovat rovnice.

**Řešení:**
- Nový argument `--export-curve FILE.csv`
- Výstup: tabulka (T, tr, sigma_fitted) pro různé teploty a časy
- Formát:
  ```csv
  T_celsius,tr_hours,sigma_MPa
  500,1,450.2
  500,10,420.1
  500,100,385.3
  ...
  ```

**Priorita:** Nízká

---

## 5. Progress indikátor pro bootstrap

**Problém:** Bootstrap s 200+ iteracemi může trvat několik sekund. Uživatel neví, zda program běží nebo zamrzl.

**Řešení:**
- Jednoduchý progress na stderr:
  ```
  Bootstrap: 50/200 (25%)
  Bootstrap: 100/200 (50%)
  ...
  ```
- Nebo progress bar pomocí tqdm (nová dependency)
- Vypnout pomocí `--quiet`

**Priorita:** Nízká

---

## 6. Packaging (pyproject.toml)

**Problém:** Projekt nemá standardní Python packaging. Instalace vyžaduje ruční kopírování souborů.

**Řešení:**
- Vytvořit `pyproject.toml`:
  ```toml
  [project]
  name = "creep-rupture-fit"
  version = "0.2.0"
  dependencies = ["numpy", "scipy", "matplotlib"]

  [project.scripts]
  crf = "creep_rupture_fit.crf:main"
  ```
- Umožní `pip install .` a příkaz `crf` v PATH

**Priorita:** Nízká (projekt funguje i bez toho)

---

## 7. Lepší error handling

**Problém:** `parse_input_file()` tiše přeskakuje neparsovatelné řádky. Uživatel neví, že některá data byla ignorována.

**Řešení:**
- Warning při přeskočení řádku:
  ```
  Warning: Skipped line 15: could not parse '350, abc, 100'
  ```
- Souhrn na konci: "Loaded 34 points, skipped 2 lines"

**Priorita:** Střední

---

## 8. Validace vstupních dat

**Problém:** Některé kombinace dat mohou vést k nesmyslným výsledkům (např. σ > σ_TS pro Wilshire).

**Řešení:**
- Varování při podezřelých datech:
  - Stress blízko nebo nad tensile strength
  - Velmi krátké časy (< 1 h)
  - Velmi dlouhé časy (> 100000 h) - možná už extrapolace
  - Teploty mimo rozsah tensile dat

**Priorita:** Střední

---

## 9. Cross-validation

**Problém:** R² na trénovacích datech může být zavádějící. Model může být přefitovaný.

**Řešení:**
- Nový argument `--cross-validate`
- Leave-one-out nebo k-fold cross-validation
- Výstup: predikční R² (Q²), RMSE na testovacích datech

**Priorita:** Nízká (pokročilá funkce)

---

## 10. Paralelizace bootstrap

**Problém:** Bootstrap je pomalý pro velké datasety nebo mnoho iterací.

**Řešení:**
- Použít `multiprocessing` nebo `joblib`
- Nový argument `--parallel N` pro počet procesů

**Priorita:** Nízká (200 iterací je dostatečně rychlých)

---

## Shrnutí priorit

| Priorita | Vylepšení |
|----------|-----------|
| Vysoká | 1. Testování |
| Střední | 2. Residual plot, 3. Srovnání modelů, 7. Error handling, 8. Validace |
| Nízká | 4. Export křivek, 5. Progress, 6. Packaging, 9. Cross-validation, 10. Paralelizace |
