[English](../../patterns/12-other-patterns.md) | Deutsch

# 12. Weitere Patterns und Varianten

Diese Patterns tauchen in der Literatur und in anderen Frameworks auf. Wir
haben sie nicht als eigene Workflows implementiert. Hier steht aber, wie sie
mit unseren zusammenhängen und wie Sie sie mit LangGraph bauen.

## Plan-and-Execute (sowie ReWOO, LLMCompiler)

Ein Planer schreibt einen expliziten mehrstufigen Plan, Ausführende arbeiten
die Schritte ab, und ein Neuplaner überarbeitet den Plan oder schließt ab. Das
entspricht unserem [Orchestrator-Worker](05-orchestrator-workers.md) mit
`max_rounds > 1`.

- **ReWOO:** Der Plan verwendet Variablen für frühere Ergebnisse und läuft ohne
  Neuplanung.
- **LLMCompiler:** Der Plan ist ein DAG aus Aufgaben, die an parallele
  Ausführende gestreamt werden (das Paper gibt eine Beschleunigung um das
  3,6-Fache an).
- In LangChain v1 ist Planung innerhalb eines einzelnen Agenten als
  `TodoListMiddleware` verfügbar (ein `write_todos`-Tool). Deep Agents
  kombinieren Planung, Subagenten und ein virtuelles Dateisystem.

Setzen Sie es für lange Aufgaben ein, bei denen ein starker Planer günstigere
Ausführende koordiniert. Achten Sie auf veraltete Pläne und planen Sie nach
Überraschungen neu.

## Blackboard / gemeinsamer Zustand

Agenten koordinieren sich, indem sie einen gemeinsamen Speicher lesen und
beschreiben, statt einander Nachrichten zu schicken. Ein Controller (oder die
Agenten selbst) entscheidet anhand des Boards, wer als Nächstes handelt. In
LangGraph ist das Board der Graph-State (Schlüssel mit Reducern) oder ein
`Store`. Google ADK nennt den Session-State „your whiteboard“. Salemi et al.
(2025) berichten von 13-57 % relativer Verbesserung mit einem
Blackboard-Entwurf, bei dem Agenten sich für ausgeschriebene Anfragen melden.
Kein Koordinator muss die Expertise jedes Agenten kennen.

Risiken: Schreibkonflikte und inkonsistenter Zustand. Microsoft führt
gemeinsamen veränderlichen Zustand zwischen nebenläufigen Agenten als
Anti-Pattern auf. Verwenden Sie Reducer und Schlüssel mit nur einem
Schreibenden.

## Gruppenchat / Debatte / Gremium

Mehrere Agenten diskutieren in einem gemeinsamen Thread. Ein Manager wählt den
nächsten Sprecher, und eine Abbruchbedingung beendet den Chat. In
Debatten-Varianten schlagen Agenten über mehrere Runden Antworten vor und
kritisieren sie. Du et al. (2023) berichten von besserem Schlussfolgern und
höherer Faktentreue. Microsoft empfiehlt höchstens drei Agenten und Teilnehmer
mit reinem Lesezugriff.

Implementierung in LangGraph: ein `StateGraph` mit einem
`speaker_selector`-Knoten (LLM oder Round-Robin), der zu Agenten-Knoten routet,
die sich `messages` teilen, plus ein Rundenzähler. Unser
[Swarm](08-swarm-handoffs.md) ist der dezentrale Verwandte, und
[Voting](04-parallelization.md) ist der parallele, nicht interaktive
Verwandte.

Die Kosten wachsen mit Agenten × Runden. Setzen Sie es für Ideenfindung,
Reviews und adversariale Prüfungen ein, nicht für Routineverarbeitung.

## „Smart Friend“ / Berater

Ein Worker-Agent konsultiert nur bei schwierigen Entscheidungen ein stärkeres
Modell (oder eine abgezweigte Kopie mit vollem Kontext) (Cognition 2026). Es
ist ein umgekehrter Supervisor: Der Worker behält die Kontrolle. Bauen Sie es
mit `agent_as_tool(..., input_mode="fork")` um ein stärkeres Modell herum.

## Deep Agents

Das höherstufige Harness von LangChain (`deepagents`) bündelt Planung (Todos),
Subagenten (synchron und asynchron, isoliert oder abgezweigt), Skills und ein
virtuelles Dateisystem zum Auslagern von Kontext. Es ist die Option mit
„Batterien inklusive“, wenn Sie einen allgemeinen, lang laufenden Agenten statt
eines spezifischen Geschäfts-Workflows wollen.

## Composite („Custom Workflow“)

Echte Systeme kombinieren Patterns. LangChain nennt das *Custom Workflow*: eine
deterministische Graph-Struktur mit agentischen Knoten, in die andere Patterns
als Knoten eingebettet sind. Unser [Composite-Beispiel](../library.md#komposition)
verschachtelt parallele Triage, ein Spam-Gate, einen Router mit Spezialisten,
eine Evaluator-Schleife, menschliche Freigabe und Map-Reduce.
