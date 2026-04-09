# ScalarForensic

## Konzept

## Architektur & Entwicklung

https://github.com/ScalarForensic/ScalarForensic

* Mehrkomponentensystem
    * Stage1 - Batch Embedding
        * CLI-Tool (`sfn`)
        * Konfiguration via `.env` (alle Defaults, kein CLI-Overload)
        * Anbindung
            * Datensatz (Ordner rekursiv, alle Unterordner)
            * Qdrant (URL konfigurierbar)
            * Embeddings (GPU/CPU/MPS, automatische Erkennung)
        * Berechnungsschritte
            * Lesen & Hashing (SHA-256, einmalig pro Bild)
            * Deduplizierung
                * Modi: `hash` | `filepath` | `both`
                * `filepath` sinnvoll für große Dateien (z.B. Video)
            * Normalisierung der Bilder
            * Embedding (Batching, SSCD & DINOv2 gleichzeitig möglich)
            * Qdrant Upsert
                * Metadaten
                    * Library-Versionen
                    * Hashwert des Embedding-Modells
                    * Hashwert der Bilder
                    * Absoluter Dateipfad
                    * Zeitstempel
                    * `exif` (bool) — EXIF-Daten vorhanden?
                    * `exif_geo_data` (bool) — GPS-Daten vorhanden?
        * Unterstützte Bildformate
            * `.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.tif` `.webp` `.gif` `.jp2` `.ico` `.psd`
            * `.heic` `.heif` optional (pillow-heif)
    * Stage2 - GUI Client
        * Flask
        * Embeddings (CPU oder Netzwerk-API)
        * QDRANT (lokal o. über Netzwerk)

### Technologien

* **Python**
    * uv statt pip (Geschwindigkeit, Versionskontrolle)
    * Flask?
* **Qdrant** (Vorschlag)
    * performant
    * flexibel
    * snapshots
    * Metadaten zu den Vektoren
* **Embedding-Modell**
    * https://huggingface.co/collections/facebook/dinov3 - **Vision-Encoder**
        * erforderlich wegen visueller Fähigkeiten, SOTA
    * SSCD ResNet50 - 3-4h bei 2TiB a 512*512
    * dinov2 - ca. *10, aber leistungsfähiger
    * beide Modelle können gleichzeitig in einem Lauf verwendet werden
