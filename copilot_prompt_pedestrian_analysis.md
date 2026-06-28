# GitHub Copilot Prompt: Pedestrian Crossing Trajectory Analysis Tool

## 🎯 Projektbeschreibung

Erstelle ein vollständiges Python-Projekt für die Analyse von Fußgänger-Überquerungsdaten aus Drohnen-BEV-Videos. Das Projekt soll folgende Hauptkomponenten enthalten:

1. **Homographie-Kalibrierung**: Kamera-zu-Boden-Transformation mit 4 markierten Punkten
2. **Trajektorienextraktion**: Aus Video via Tracking (Pixel → Meter)
3. **PedPy-Analyse**: Geschwindigkeit, Richtung, Dichte, Gruppen
4. **Behavior Labeling**: Automatische Klassifikation (waiting, approaching, crossing)
5. **Schwarmverhalten-Analyse**: Reynolds-Regeln (Cohesion, Alignment, Separation)
6. **Gradio-UI**: Parameter einstellen, Videos laden, Plots exportieren

---

## 📁 Projektstruktur

Erstelle folgende Verzeichnisstruktur:

```
pedestrian_analysis/
├── app.py                        # Gradio UI – Einstiegspunkt
├── requirements.txt
├── config.py                     # Globale Parameter (Schwellwerte, Pfade)
├── pipeline/
│   ├── __init__.py
│   ├── calibration.py            # Homographie: pixel → meter
│   ├── tracker.py                # Video → Trajektorien (ByteTrack via supervision)
│   ├── trajectory_io.py          # Laden/Speichern von Trajektoriendaten
│   ├── pedpy_analysis.py         # PedPy-Wrapper: speed, direction, density
│   ├── behavior_labeling.py      # FSM: waiting / approaching / crossing
│   ├── group_analysis.py         # Gruppenbildung (DBSCAN), Gruppengrößen, Splits
│   └── swarm_analysis.py         # Reynolds-Regeln: Cohesion, Alignment, Separation
├── visualization/
│   ├── __init__.py
│   ├── plot_trajectories.py      # Trajektorien-Plot (Plotly)
│   ├── plot_heatmap.py           # Dichteplot (Heatmap)
│   ├── plot_behavior.py          # Zeitreihe der Behavior-Labels
│   ├── plot_swarm.py             # Schwarmmetriken über Zeit
│   └── export.py                 # PDF/PNG Export für Paper
├── data/
│   ├── videos/                   # Roh-Videos (nicht eingecheckt)
│   ├── trajectories/             # Extrahierte .csv Trajektorien
│   └── calibration/              # Homographie-Matrizen (.npy)
├── outputs/
│   └── figures/                  # Exportierte Paper-Plots
└── tests/
    └── test_pipeline.py
```

---

## 📦 requirements.txt

Erstelle eine `requirements.txt` mit exakt diesen Abhängigkeiten:

```
# Core
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
scikit-learn>=1.3.0

# Computer Vision & Tracking
opencv-python>=4.8.0
supervision>=0.18.0          # ByteTrack, BoundingBox Utilities
ultralytics>=8.0.0           # YOLOv8 (Personendetektion)

# Trajektorienanalyse
pedpy>=1.1.0

# Visualisierung
plotly>=5.18.0
matplotlib>=3.7.0
seaborn>=0.12.0

# UI
gradio>=4.20.0

# Export
kaleido>=0.2.1               # Plotly → PNG/PDF Export
fpdf2>=2.7.0                 # PDF-Bericht
```

---

## 🔧 Implementierungsanweisungen

### 1. `pipeline/calibration.py`

Implementiere folgende Funktionen:

```python
def compute_homography(src_pixel_points: np.ndarray, dst_meter_points: np.ndarray) -> np.ndarray:
    """
    Berechnet Homographiematrix H von Bildpixeln zu Bodenkoordinaten in Metern.
    
    Args:
        src_pixel_points: Array shape (4, 2) – Pixel-Koordinaten der 4 Markierungen im Video
        dst_meter_points: Array shape (4, 2) – Entsprechende reale Koordinaten in Metern
    Returns:
        H: 3x3 Homographiematrix (np.ndarray)
    """

def pixel_to_meter(px: float, py: float, H: np.ndarray) -> tuple[float, float]:
    """
    Transformiert einen Fußpunkt (bottom-center der Bounding Box) von Pixel zu Meter.
    Verwendet cv2.perspectiveTransform.
    """

def validate_calibration(frame: np.ndarray, H: np.ndarray, grid_spacing_m: float = 1.0) -> np.ndarray:
    """
    Zeichnet ein metrisches Grid auf das gewarpte BEV-Bild zur visuellen Validierung.
    
    Schritte:
    1. cv2.warpPerspective(frame, H, output_size) → BEV-Bild
    2. Grüne Linien alle grid_spacing_m Meter einzeichnen
    3. Markierungen an den Gitterpunkten mit Meterangabe beschriften
    Returns:
        Annotiertes BEV-Bild (np.ndarray BGR)
    """

def save_calibration(H: np.ndarray, path: str) -> None:
    """Speichert H als .npy Datei"""

def load_calibration(path: str) -> np.ndarray:
    """Lädt H aus .npy Datei"""
```

---

### 2. `pipeline/tracker.py`

Implementiere YOLOv8 + ByteTrack Tracking mit `supervision`:

```python
def extract_trajectories_from_video(
    video_path: str,
    H: np.ndarray,
    model_name: str = "yolov8n.pt",
    confidence_threshold: float = 0.4,
    class_filter: list[int] = [0],  # COCO class 0 = person
    frame_skip: int = 1,
) -> pd.DataFrame:
    """
    Extrahiert Trajektorien aus einem Video.
    
    Pipeline:
    1. VideoCapture öffnen
    2. YOLOv8 Inferenz auf jedem Frame (class_filter=person)
    3. supervision.ByteTrack() für konsistente IDs
    4. Für jede Detection: Fußpunkt = bottom-center der BoundingBox
    5. pixel_to_meter() auf Fußpunkt anwenden
    6. Ergebnis als DataFrame: columns = ['frame', 'id', 'x', 'y', 'px', 'py']
       (px, py = originale Pixelkoordinaten, x, y = Meterkoordinaten)
    
    Returns:
        pd.DataFrame mit Spalten: frame, id, x, y (Meter), px, py (Pixel)
    """
```

---

### 3. `pipeline/pedpy_analysis.py`

PedPy-Integration für Geschwindigkeit und Richtung:

```python
def compute_kinematics(df: pd.DataFrame, fps: float, frame_step: int = 5) -> pd.DataFrame:
    """
    Berechnet Geschwindigkeit und Bewegungsrichtung mit PedPy.
    
    Steps:
    1. pedpy.TrajectoryData(data=df, frame_rate=fps) erstellen
    2. pedpy.compute_individual_speed() aufrufen
    3. Heading-Winkel aus aufeinanderfolgenden (x, y) Differenzen berechnen:
       heading = np.degrees(np.arctan2(dy, dx)) – Wertebereich [-180, 180]
    4. Ergebnisse zurück in df mergen
    
    Returns:
        df mit zusätzlichen Spalten: 'speed_ms', 'heading_deg'
    """
```

---

### 4. `pipeline/behavior_labeling.py`

Finite State Machine für Behavior-Labels:

```python
# Zustände:
# - 'waiting':    speed < SPEED_THRESHOLD_MS  UND  x < STREET_START_M
# - 'approaching': speed >= SPEED_THRESHOLD_MS  UND  x < STREET_START_M
# - 'crossing':   x >= STREET_START_M  UND  x <= STREET_END_M
# - 'crossed':    x > STREET_END_M

def label_behaviors(
    df: pd.DataFrame,
    street_start_m: float,
    street_end_m: float,
    speed_threshold_ms: float = 0.3,
    waiting_min_frames: int = 5,
) -> pd.DataFrame:
    """
    Fügt Spalte 'behavior' zu df hinzu.
    
    Wichtig:
    - Berechne Labels pro Individuum (groupby 'id')
    - Verwende Hysterese: Label wechselt nur, wenn neuer State für
      mindestens waiting_min_frames Frames stabil ist (verhindert Flicker)
    - Exportiere zusätzlich:
      - 'waiting_start_frame': Frame, ab dem Individuum wartet
      - 'crossing_start_frame': Frame, ab dem es anfängt zu überqueren
      - 'waiting_duration_s': Wartezeit in Sekunden
    
    Returns:
        df mit zusätzlichen Spalten: 'behavior', 'waiting_duration_s',
        'crossing_start_frame', 'waiting_start_frame'
    """

def compute_crossing_events(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    Extrahiert pro Individuum ein Crossing-Event mit:
    - id, group_id, crossing_start_frame, crossing_end_frame
    - crossing_duration_s
    - waiting_duration_s (Zeit am Bordstein vor dem Crossing)
    
    Returns:
        pd.DataFrame – eine Zeile pro Crossing-Event
    """
```

---

### 5. `pipeline/group_analysis.py`

Gruppenbildung via DBSCAN und Gruppenstatistiken:

```python
def detect_groups_per_frame(
    df: pd.DataFrame,
    proximity_threshold_m: float = 1.5,
    min_group_size: int = 2,
) -> pd.DataFrame:
    """
    DBSCAN-basierte Gruppenbildung pro Frame.
    
    Steps:
    1. Für jeden Frame: sklearn.cluster.DBSCAN(eps=proximity_threshold_m, min_samples=1)
       auf (x, y) Koordinaten anwenden
    2. Cluster-Labels als 'group_id_frame' in df speichern
    3. Gruppen über Zeit glätten: Individuum bleibt in Gruppe, wenn es
       in > 70% der letzten 10 Frames zur selben Gruppe gehört (Temporal Smoothing)
    
    Returns:
        df mit zusätzlicher Spalte: 'group_id' (konsistent über Frames)
    """

def compute_group_statistics(df: pd.DataFrame) -> dict:
    """
    Berechnet folgende Gruppenstatistiken:
    - Anzahl eindeutige Gruppen
    - Gruppengrößen-Histogramm (wie viele Gruppen mit N Personen)
    - Gruppenaufspaltungs-Events: Wann hat sich eine Gruppe geteilt?
      (group_id vorhanden in Frame t, aber Subgruppen in Frame t+k)
    - Median-Wartezeit pro Gruppe vor Crossing
    - Prozentsatz von Partial-Crossings (nicht alle überqueren gleichzeitig)
    
    Returns:
        dict mit Schlüsseln: 'group_count', 'size_histogram',
        'split_events', 'median_waiting_time_s', 'partial_crossing_rate'
    """
```

---

### 6. `pipeline/swarm_analysis.py`

Reynolds-Regeln auf Gruppenebene:

```python
def compute_cohesion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cohesion = mittlerer Abstand jedes Individuums zum Gruppencentroid.
    
    Berechne pro Frame und Gruppe:
    centroid = mean(x, y) der Gruppe
    cohesion_i = ||pos_i - centroid||
    
    Returns:
        df mit Spalte 'cohesion_m' (Abstand zum Centroid in Metern)
    """

def compute_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Alignment = Standardabweichung der Heading-Winkel innerhalb einer Gruppe.
    Niedriger Wert = alle gehen in dieselbe Richtung.
    
    Für Winkelstatistik: circular standard deviation verwenden:
    scipy.stats.circstd(headings, high=180, low=-180)
    
    Returns:
        df mit Spalte 'alignment_deg' (circulare Streuung in Grad)
    """

def compute_separation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Separation = minimaler paarweiser Abstand zwischen Individuen in einer Gruppe.
    
    Für jede Gruppe pro Frame:
    pdist(positions) → min distance
    
    Returns:
        df mit Spalte 'min_separation_m'
    """

def compute_leader_follower(df: pd.DataFrame) -> pd.DataFrame:
    """
    Schätzt Leader-Follower-Dynamik:
    Leader = Individuum mit dem frühesten crossing_start_frame in der Gruppe.
    Follower-Delay = crossing_start_frame_follower - crossing_start_frame_leader (in Sekunden)
    
    Returns:
        pd.DataFrame mit: group_id, leader_id, follower_ids, follower_delays_s
    """
```

---

### 7. `app.py` – Gradio UI

Baue eine vollständige Gradio-Applikation mit **drei Tabs**:

#### Tab 1: "📷 Kalibrierung"
- Komponenten:
  - `gr.Image` – Video-Frame hochladen (als Bild)
  - `gr.Dataframe` – 4 Pixelpunkte eingeben (editable, columns: ['px', 'py'])
  - `gr.Dataframe` – 4 Meterpunkte eingeben (editable, columns: ['x_m', 'y_m'])
  - `gr.Button("Kalibrierung berechnen")`
  - `gr.Image` – Ausgabe: BEV-Bild mit Grid zur Validierung
  - `gr.Slider("Grid-Auflösung (m)", 0.5, 5.0, step=0.5, value=1.0)`
  - `gr.File` – Download der `.npy` Homographiematrix

#### Tab 2: "🎬 Trajektorienextraktion"
- Komponenten:
  - `gr.File` – Video hochladen (.mp4, .mov, .avi)
  - `gr.File` – Kalibrierungsdatei (.npy) hochladen
  - `gr.Slider("Confidence Threshold", 0.1, 0.9, step=0.05, value=0.4)`
  - `gr.Slider("Frame Skip", 1, 10, step=1, value=1)`
  - `gr.Number("FPS", value=25)`
  - `gr.Number("Straßenbeginn X (m)", value=2.0)` – für Behavior Labeling
  - `gr.Number("Straßenende X (m)", value=6.0)`
  - `gr.Number("Wartezeit-Schwellwert (m/s)", value=0.3)`
  - `gr.Number("Gruppenradius DBSCAN (m)", value=1.5)`
  - `gr.Button("Analyse starten")`
  - `gr.Progress()` – Fortschrittsanzeige während Tracking
  - `gr.File` – Download der Trajektorien-CSV

#### Tab 3: "📊 Analyse & Plots"
- Komponenten:
  - `gr.File` – Trajektorien-CSV hochladen (falls bereits extrahiert)
  - `gr.Tabs` mit folgenden Sub-Tabs:
    - **"Trajektorien"**: Plotly-Scatterplot aller Trajektorien, farbkodiert nach `id`
    - **"Geschwindigkeit"**: Geschwindigkeitszeitreihe pro Individuum
    - **"Behavior Labels"**: Farbkodierter Timeline-Plot (waiting=blau, approaching=gelb, crossing=grün)
    - **"Gruppen"**: Gruppengrößen-Histogramm + Gruppenaufspaltungs-Timeline
    - **"Schwarmverhalten"**: Cohesion, Alignment, Separation über Zeit (3 Subplots)
    - **"Statistiken"**: Tabelle mit allen Kennzahlen für Paper
  - `gr.Button("Alle Plots exportieren (PNG)")`
  - `gr.File` – ZIP mit allen Plots zum Download

---

## 🎨 Visualisierungsanweisungen

### `visualization/plot_trajectories.py`

```python
def plot_trajectories(df: pd.DataFrame, street_start_m: float, street_end_m: float) -> go.Figure:
    """
    Plotly-Figure: Alle Trajektorien als Linien, farbkodiert nach Individuum-ID.
    
    Anforderungen:
    - Straße als grau schraffierte Fläche zwischen street_start_m und street_end_m
    - Startpunkt jeder Trajektorie als Dreieck-Marker
    - Endpunkt als Quadrat-Marker
    - Behavior-Label als Hover-Text
    - Gruppencentroide als gestrichelte Linie einzeichnen
    - Achsen beschriften: "X Position (m)" und "Y Position (m)"
    - Für Paper: weißer Hintergrund, Arial-Font, Schriftgröße 12pt
    """
```

### `visualization/plot_behavior.py`

```python
def plot_behavior_timeline(df: pd.DataFrame, fps: float) -> go.Figure:
    """
    Gantt-ähnlicher Timeline-Plot:
    - Y-Achse: Individuum-IDs
    - X-Achse: Zeit in Sekunden
    - Farbige Balken: waiting=steelblue, approaching=orange, crossing=green, crossed=gray
    - Vertikal gestrichelte Linien bei Crossing-Events der Gruppenleader
    """
```

---

## 📋 Weitere Anforderungen

### config.py
Erstelle eine zentrale Konfigurationsdatei:
```python
# Alle Schwellwerte als Konstanten mit Docstrings
SPEED_THRESHOLD_MS = 0.3          # m/s unter dem jemand als "wartend" gilt
PROXIMITY_THRESHOLD_M = 1.5       # m für DBSCAN-Gruppenbildung
MIN_GROUP_FRAMES = 10             # Frames bis Gruppe als stabil gilt
TEMPORAL_SMOOTH_WINDOW = 10      # Frames für Temporal Smoothing
WAITING_MIN_FRAMES = 5           # Hysterese für Behavior-Labeling
LEADER_FOLLOWER_MAX_DELAY_S = 5.0 # Max. Verzögerung für Leader-Follower-Zuordnung

# Paper-Export Einstellungen
PLOT_WIDTH_PX = 1200
PLOT_HEIGHT_PX = 800
PLOT_FONT_SIZE = 14
PLOT_FONT_FAMILY = "Arial"
PLOT_TEMPLATE = "plotly_white"
DPI = 300
```

### Error Handling
- Alle Pipeline-Funktionen sollen bei fehlerhaften Inputs sprechende Exceptions werfen
- Gradio-UI soll Fehler als `gr.Warning` oder `gr.Error` anzeigen
- Fortschrittsanzeige im Tracking-Tab mit `gr.Progress()`

### Tests (`tests/test_pipeline.py`)
Erstelle Unit-Tests für:
- `compute_homography`: Prüfe ob bekannte Punkte korrekt transformiert werden
- `label_behaviors`: Prüfe alle 4 State-Übergänge
- `detect_groups_per_frame`: Prüfe mit 2 bekannten Gruppen
- `compute_cohesion`: Prüfe gegen analytisch berechneten Wert

---

## 🚀 Start-Kommando

Füge am Ende von `app.py` hinzu:
```python
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
```

Das Projekt soll lokal mit `python app.py` startbar sein.

---

## ⚠️ Wichtige Hinweise für Copilot

1. **Verwende `supervision.ByteTrack`** für Tracking – nicht DeepSORT oder andere
2. **YOLOv8n** als Default-Modell (klein, schnell) – `ultralytics.YOLO("yolov8n.pt")`
3. **PedPy ≥ 1.1.0 API verwenden** – `TrajectoryData`, `compute_individual_speed` sind die korrekten Funktionen
4. **Keine hardkodierten Pfade** – alles über `config.py` oder Funktionsparameter
5. **Trajektorien-CSV Format**: Kompatibel mit PedPy-Format: `id, frame, x, y` (Pflicht), plus zusätzliche Spalten
6. **DBSCAN** aus `sklearn.cluster` – nicht scipy
7. **Circular statistics** für Heading-Winkel: `scipy.stats.circstd` – nicht numpy.std
8. **Gradio 4.x API** – `gr.Interface` ist veraltet, verwende `gr.Blocks`

