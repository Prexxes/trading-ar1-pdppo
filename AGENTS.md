## Projektüberblick
- "agents/": Logik der Agenten PPO und PDPPO
- "buffers/": Rollout buffer für PPO und PDPPO
- "envs/": Environments
- "eval/": Evaluierung nach der Trainingsphase
- "tests_runs/": Ergebnisse der Experimente
- "tests/": Tests
- "train/": Trainingslogik für PPO und PDPPO

## Standardbefehle
- Setup: "pip install -r requirements.txt"
- Tests: "pytest -q"

## Arbeitsregeln
- Stelle Fragen, wenn etwas unklar ist.
- Benutze die Programmiersprache Python.
- Die Bezeichnungen, Docstrings und Kommentare sollen auf Englisch sein für Python.
- Füge bei Funktionen und Klassen immer Doctrings und Kommentare hinzu.
- Befolge Google Python Style Guide.
- Keine neuen Dependencies ohne Begründung hinzufügen.
- Änderungen möglichst klein und lokal halten.
- Keine unnötigen Refactorings außerhalb des Scopes.

## Qualitätschecks
- Nach Codeänderungen immer "pytest -q" ausführen.
- Nach geänderter Trainingslogik relevante Smoke-Tests ausführen.

## Nicht tun
- Keine Secrets, Tokens oder Zugangsdaten anfassen.
- Keine API oder Dateistruktur ändern, außer wenn die Aufgabe es verlangt.

## Fertig wenn
- Die Qualitätschecks erfolgreich ausgeführt wurden.