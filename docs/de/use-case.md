[English](../use-case.md) | Deutsch

# Anwendungsfall: Automatisierung eines Kundenservice-Posteingangs

**Acme Cloud** ist ein fiktives B2B-SaaS-Unternehmen. In seinem
Support-Posteingang landen Abrechnungsfragen, Störungsmeldungen,
Vertriebsanfragen, Spam und gelegentlich rechtliche oder DSGVO-Angelegenheiten.
Jede eingehende E-Mail muss

1. **kategorisiert** werden: Kategorie, weitere Intents, Priorität, Stimmung, wird ein Mensch gebraucht?
2. **bearbeitet** werden: Kunde und Fakten nachschlagen, dann gemäß Richtlinie handeln (Erstattung, Ticket, Lead, Eskalation)
3. **beantwortet** werden: eine Antwort schreiben, sie versenden oder an einen Menschen eskalieren.

Wir haben genau diese Aufgabe mit jedem der elf Patterns umgesetzt. So lassen
sich die Patterns direkt vergleichen.

## Posteingang (Workflow-Eingabe)

`src/email_assistant/data.py` enthält sechs E-Mails, die zusammen jeden Zweig abdecken:

| ID | Von | Inhalt | Erwartet |
|---|---|---|---|
| E-1001 | Anna Schmidt (Contoso, gold) | Für Rechnung INV-2026-0815 doppelt belastet | billing, Erstattung 348 EUR, Antwort |
| E-1002 | Mark Jones (Fabrikam) | Dashboard zeigt HTTP 503, Team blockiert | technical, urgent, Ticket verknüpft mit Incident INC-7781, Antwort |
| E-1003 | Priya Natarajan (Northwind, unbekannter Absender) | Enterprise-Preise für 50 Seats + Demo | sales, Preis mit 15% Mengenrabatt, neuer Lead, Antwort |
| E-1004 | Jordan Doe (Tailspin, platinum) | Rechnung berechnet 60 statt 50 Seats **und** CSV-Export stürzt ab | **Multi-Intent**: Ticket zur Rechnungskorrektur + bekanntes technisches Problem, eine Antwort |
| E-1005 | „Prize Department“ | „You have WON an iPhone, click here“ | spam, ignorieren |
| E-1006 | Anna Schmidt | Löschantrag nach DSGVO Art. 17 + Androhung rechtlicher Schritte | braucht einen Menschen, Eskalation an den Datenschutzbeauftragten (DPO), neutrale Eingangsbestätigung |

## Mock-Datenquellen (Tools)

`src/email_assistant/tools.py` stellt LangChain-`@tool`s über In-Memory-Mock-Systeme bereit:

| Tool | Art | Gemocktes System |
|---|---|---|
| `lookup_customer(email)` | Lesen | CRM |
| `get_invoice(invoice_id)` | Lesen | Abrechnungssystem (Zahlungen, Seats) |
| `search_knowledge_base(query)` | Lesen | Richtlinien und Troubleshooting-Artikel (KB-101 ... KB-910) |
| `check_service_status(service)` | Lesen | Statusseite (Dashboard-Incident INC-7781) |
| `get_plan_pricing(plan, seats)` | Lesen | Preisrechner |
| `issue_refund(invoice_id, amount, reason)` | **Schreiben** | Zahlungen (erzwingt: nur bezahlte Rechnungen, nur den zu viel gezahlten Betrag, höchstens 500 EUR, idempotent) |
| `create_ticket(customer_id, queue, summary, priority)` | **Schreiben** | Ticketsystem |
| `create_sales_lead(...)` | **Schreiben** | CRM-Leads (weist einen Account Executive zu) |
| `escalate_to_human(customer_email, team, reason, summary)` | **Schreiben** | Eskalations-Queue (DPO, Teamleitung, Finanzen) |

Seiteneffekte landen in `data.BACKEND` (Erstattungen, Tickets, Leads,
Eskalationen, Postausgang). Dort prüfen die Tests sie. Datensatz-IDs werden aus
dem Inhalt abgeleitet. Daher erzeugen parallele Zweige unabhängig vom Timing
dieselben IDs.

## Vertrag jedes Workflows

```python
input  = {"email": Email}
output = {"resolution": EmailResolution, "delivery": Delivery}
```

`EmailResolution` (strukturierte Ausgabe, `schemas.py`) enthält Kategorie,
Priorität, Aktion (`reply` / `escalate` / `ignore`), die durchgeführten Aktionen
mit Referenz-IDs, Betreff und Text der Antwort, den Eskalationsgrund und eine
interne Notiz. Jeder Workflow endet mit dem deterministischen
**`dispatch`**-Knoten. Er legt die Antwort in den Postausgang und erfasst
Eskalationen. Das Modell versendet selbst nie etwas.

```
START -> <pattern> -> dispatch -> END
```

## Das Mock-LLM

Wir rufen keine API auf. Stattdessen verhält sich
`agentpatterns.testing.ScriptedChatModel` wie ein echtes Tool-aufrufendes
Chat-Modell:

- `bind_tools()` funktioniert. Das Modell „sieht“ also die Tools des jeweiligen Aufrufs.
- Es kann **parallele Tool-Aufrufe** und **strukturierte Ausgabe** zurückgeben,
  sowohl per Tool Calling als auch per nativer JSON-Ausgabe.
  `create_agent(response_format=...)` und `with_structured_output()`
  funktionieren daher unverändert.
  `create_mock_llm(native_structured_output=True)` deklariert native
  Unterstützung, wie es aktuelle Claude-Modelle tun; die Tests führen jeden
  Workflow auf beide Arten aus.
- Es verweigert, was ein echtes Modell nicht kann: nicht gebundene Tools
  aufrufen oder mit Text antworten, wenn ein Tool-Aufruf erzwungen wird.
- Es meldet einen geschätzten Token-Verbrauch, sodass `UsageTracker` die Patterns vergleichen kann.

Bei jedem Aufruf führt es eine *Policy* aus. `email_assistant/mock_llm.py`
enthält **eine** Policy für den gesamten Anwendungsfall. Wie ein echtes Modell
leitet sie ihre Rolle aus dem System-Prompt ab („You are the billing specialist
of Acme Cloud ...“) und handelt dann: Tools aufrufen, delegieren, per Handoff
übergeben, planen oder antworten. Das eigentliche „Reasoning“ steckt in
`brain.py`: Klassifikation per Schlüsselwort, das Vorgehen pro Domäne
(Informationen sammeln, handeln, berichten) und das Verfassen der Antwort. Der
Mock nutzt nur Informationen, die im Gespräch sichtbar sind, nie direkt die
Datenbanken. Ein Pattern, das Informationen nicht weitergibt, scheitert also
tatsächlich.

Eine Folge davon: **Jedes Pattern liefert für jede E-Mail genau dasselbe
Ergebnis**, was die Tests prüfen. Das macht den Mock ideal, um die
Orchestrierung zu testen und Aufrufe zu zählen. Qualitätsunterschiede zwischen
den Patterns kann er nicht zeigen. Echte Modelle verschlechtern sich auf
unterschiedliche Weise, wenn sie zu viele Tools, zu viel Kontext oder einen
verlustbehafteten Handoff haben. Die Berichte diskutieren diese Unterschiede auf
Basis der Recherche.

Das Evaluator-Optimizer-Pattern ist eine bewusste Ausnahme: Der Mock-Verfasser
schreibt einen nachlässigen ersten Entwurf (kein Name, keine Referenznummern,
keine Signatur), damit die Review-Schleife tatsächlich einmal iteriert.

## Ausführen

```bash
uv sync
uv run email-demo                          # comparison of all native workflows
uv run email-demo --impl library           # the same built with agentpatterns (+ composite)
uv run email-demo -p supervisor -e E-1004 -v   # replies + model calls per agent
uv run email-demo --mermaid router         # graph as a Mermaid diagram
uv run email-demo --digest                 # map-reduce over the inbox
uv run pytest                              # 178 tests (+11 opt-in real-model tests)
```

### Auf ein echtes Modell umstellen

Jeder Workflow erhält das Chat-Modell als Parameter. Für Claude installieren Sie
das Extra `anthropic` (`uv sync --extra anthropic`) und setzen
`ANTHROPIC_API_KEY`:

```python
from langchain.chat_models import init_chat_model
from email_assistant.native import PATTERNS

model = init_chat_model("anthropic:claude-opus-5")
graph = PATTERNS["supervisor"](model)
graph.invoke({"email": email})
```

Mit einem echten Modell nutzt `response_format=SomeSchema` automatisch die
native strukturierte Ausgabe des Providers (`ProviderStrategy`), wenn das Profil
des Modells diese Unterstützung deklariert. Das tut jedes aktuelle Claude-Modell.

Die Aufrufe von `with_structured_output()` erfordern mehr Sorgfalt. Für Claude
ist ihr Standard erzwungenes Tool Calling. Claude Opus 5.5, Claude Sonnet 5.5
und Claude Fable 5.1 lehnen ein erzwungenes `tool_choice` ab, ebenso jedes
Claude-Modell mit aktiviertem Extended Thinking.
`langchain-anthropic` sendet dann einen nicht erzwungenen Tool-Aufruf und löst
`OutputParserException` aus, wenn das Modell stattdessen mit Text antwortet.

- Die **bibliotheksbasierten** Workflows berücksichtigen das: Ihre internen
  strukturierten Aufrufe verwenden standardmäßig
  `structured_output_method="auto"`. Damit nutzen sie native strukturierte
  Ausgabe (`method="json_schema"`), wo das Profil sie deklariert.
- Die **handgeschriebenen** Workflows in `email_assistant/native/` rufen
  `model.with_structured_output(Schema)` mit dem Standard des Providers auf. Das
  funktioniert mit Claude Opus 5 und Sonnet 5. Ergänzen Sie für Opus 5.5,
  Sonnet 5.5 oder Fable 5.1 in diesen Aufrufen `method="json_schema"`.

Führen Sie vor dem Umstellen die Smoke-Tests gegen das Modell aus:
`AGENTPATTERNS_TEST_MODEL=anthropic:claude-opus-5 uv run pytest -m real_model`
(siehe [library.md](library.md#testen)).

## Gemessener Vergleich

`uv run email-demo` (native Implementierungen, gesamter Posteingang mit 6 E-Mails):

| Pattern | Korrekt | Modellaufrufe (gesamt) | E-1001 (1 Intent) | E-1004 (2 Intents) | E-1005 (Spam) | Tool-Aufrufe | Geschätzte Input-Token |
|---|---|---|---|---|---|---|---|
| single_agent | 6/6 | 16 | 3 | 3 | 1 | 22 | 26,794 |
| sequential | 6/6 | 21 | 4 | 4 | 1 | 21 | 16,962 |
| router | 6/6 | 30 | 5 | 8 | 2 | 23 | 25,724 |
| parallel | 6/6 | 39 | 7 | 7 | 4 | 22 | 33,874 |
| orchestrator | 6/6 | 52 | 10 | 10 | 2 | 22 | 30,489 |
| supervisor | 6/6 | 29 | 5 | 8 | 1 | 29 | 30,579 |
| hierarchical | 6/6 | 39 | 7 | 10 | 1 | 34 | 35,431 |
| swarm | 6/6 | 24 | 4 | 7 | 1 | 28 | 33,026 |
| state_machine | 6/6 | 33 | 6 | 6 | 3 | 34 | 25,933 |
| skills | 6/6 | 21 | 4 | 4 | 1 | 28 | 27,598 |
| evaluator_optimizer | 6/6 | 32 | 6 | 6 | 2 | 22 | 42,182 |

„Tool-Aufrufe“ umfassen auch Tools für Delegation, Handoff, Übergänge und das
Laden von Skills. Die Token-Zahlen sind Schätzungen (etwa 4 Zeichen pro Token
über Prompt, Tool-Schemas und Verlauf). Nutzen Sie sie, um die Patterns
untereinander zu vergleichen, nicht als absolute Kosten. Die bibliotheksbasierten
Implementierungen machen exakt dieselben Modellaufrufe. Ihre Token-Zahlen weichen
leicht ab, weil die Standard-Prompts der Bibliothek Agenten-Kataloge hinzufügen.

**Was die Zahlen zeigen**

- **Patterns mit festem Ablauf kosten für jede E-Mail gleich viel**
  (sequenziell, parallel, Orchestrator, State Machine). Dynamische Patterns
  zahlen pro Intent: Router, Supervisor, hierarchische Teams und Swarm brauchen
  3 zusätzliche Aufrufe für den zweiten Intent von E-1004.
- **Agenten, die Tool-Aufrufe bündeln, kommen mit wenigen Aufrufen aus.** Der
  Single Agent und Skills erledigen zwei Intents mit 3-4 Aufrufen, weil ihre
  Tool-Aufrufe parallel laufen. Der Preis ist ein einziger, wachsender Kontext.
- **Jede Koordinationsebene kostet Aufrufe.** Der Supervisor fügt den 3 eigenen
  Aufrufen des Spezialisten 2 hinzu (delegieren + zusammenstellen), und
  hierarchische Teams fügen pro Ebene 2 weitere hinzu.
- **Frühe Ausstiege zählen.** Gates (sequenziell, parallel, Supervisor)
  erledigen Spam mit 1 Aufruf. Die State Machine durchläuft trotzdem ihre
  Zustände (3 Aufrufe), und der parallele Fan-out bezahlt für alle vier
  Analysten.
- **Gemeinsamer Verlauf ist teuer.** Der Swarm reicht das komplette Gespräch
  weiter. Daher ist E-1004 im Swarm der Token-intensivste Lauf (etwa 11k Token),
  obwohl der Swarm weniger Aufrufe braucht als der Supervisor.
- **Die Pipeline verbraucht die wenigsten Token.** Ihre einzelnen,
  in sich abgeschlossenen LLM-Aufrufe tragen keinen wachsenden Verlauf von
  Tool-Aufrufen mit sich, nur die Fakten, die der jeweilige Schritt braucht.
