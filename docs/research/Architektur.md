# ScalarForensic

## Konzept

## Architektur & Entwicklung

https://github.com/ScalarForensic/ScalarForensic

* Mehrkomponentensystem
    * Stage1 - Batch Embedding
        * noch zu klären: Hardware, Netzwerk
        * CLI-Tool
        * Anbindung 
            * Datensatz
            * Qdrant
            * Embeddings (GPU?)
        * Berechnungsschritte
            * Normalisierung derBilder
            * Embedding (Batching/sequentiell)
            * Qdrant
                * + Metadaten
                    * Library-Versionen
                    * Hashwert des Embedding-Modells
                    * Hashwert der Bilder
                    * etc
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
