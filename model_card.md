# Model Card: VibeFinder Retrieval-Aware Music Recommendation Assistant

## Model Name

**VibeFinder 2.0**

## Intended Use

VibeFinder is a classroom and portfolio project that recommends songs from a small local catalog based on a natural-language request. It is intended to demonstrate an applied AI workflow that combines rule-based parsing, local retrieval, class-based recommendation, validation, logging, and evaluation.

It is appropriate for:

- exploring interpretable recommendation logic
- demonstrating retrieval-augmented workflows without external APIs
- showing confidence scoring and guardrails on a small dataset

It is not appropriate for:

- production music recommendation
- large-scale personalization
- safety-critical decision making
- real-world claims about user taste, mood, or psychological state

## System Components

The current recommendation baseline is class-based and centered in `src/recommender.py`.

- `UserProfile`: structured user input with `favorite_genre`, `favorite_mood`, `target_energy`, and `likes_acoustic`
- `Recommender`: canonical recommendation engine over `Song` objects
- `Query Parser`: rule-based natural-language parser that extracts structured preferences and assumptions
- `Retriever`: local snippet retriever over markdown files in `knowledge/`
- `Validator`: confidence scoring, warnings, and validation notes
- `Logger`: JSONL run logging to `logs/runs.jsonl`
- `Evaluation Harness`: predefined end-to-end cases in `src/eval.py`

The recommender uses richer heuristics than the original version. Instead of relying only on genre, mood, and energy, it also models tempo, valence, danceability, and acousticness as supporting feature matches tied to the requested genre, mood, and target energy.

## How the System Works

1. The user enters a request such as `I want acoustic lofi songs for studying`.
2. The parser extracts structured preferences and any assumptions.
3. The retriever looks up relevant local knowledge about genres, moods, listening contexts, acousticness, and feature signals.
4. The recommender ranks songs using class-based scoring over `Song` objects.
5. The validator computes a confidence score, warnings, and validation notes.
6. The assistant prints recommendations and logs the run as structured JSON.

Every recommendation remains explainable because the ranking step returns human-readable scoring reasons.

## Strengths

- **Transparent recommendations**: every top result includes an explanation string rather than a hidden score.
- **Deterministic behavior**: the parser, retriever, recommender, validator, and evaluation harness all run locally and reproducibly.
- **Richer feature matching**: tempo, valence, danceability, and acousticness now influence ranking in addition to genre, mood, and energy.
- **Guardrails for weak cases**: contradictory or weak-fit requests lower confidence and trigger warnings instead of receiving a falsely confident answer.
- **Integrated applied AI workflow**: retrieval and validation are part of the core application logic, not side scripts.

## Limitations and Biases

- **Small catalog bias**: the dataset contains only 18 songs, so many genres have only one representative. This makes it hard to support variety or nuanced within-genre matching.
- **Western and English-language skew**: the catalog reflects a narrow slice of music styles and omits lyrics, language diversity, release era, and cultural context.
- **Rule-based parser limits**: natural-language understanding is deterministic but shallow. Requests outside the supported vocabulary may be oversimplified or partially misread.
- **Heuristic confidence**: confidence is based on explicit rules and fit checks, not learned calibration from user feedback.
- **Literal retrieval**: the retriever uses token overlap and keyword normalization rather than deeper semantic search.
- **No personalization loop**: the system does not learn from likes, skips, or long-term listening history.

These limitations mean the assistant is best understood as an explainable prototype, not a production recommender.

## Evaluation Results

The project includes both automated tests and an evaluation harness.

Current verified results:

- `python -m pytest -q`: 24 passing tests
- `python -m src.eval`: 9 out of 9 predefined evaluation cases passing
- average confidence across evaluation cases: `0.80`

Evaluation coverage includes:

- strong-fit requests such as acoustic lofi for studying and high-energy rock for the gym
- missing-coverage requests such as happy reggae songs
- contradictory requests such as ambient songs with very high energy and an acoustic feel

Common warning types observed in evaluation:

- requested genre missing from catalog coverage
- contradiction between genre expectations and target energy
- weak supporting-feature fit on energy, tempo, valence, danceability, or acousticness

## Misuse Risks and Guardrails

Potential misuse risks:

- treating the system like a universal or authoritative taste model
- assuming a recommendation is strong even when the catalog lacks coverage
- using mood labels as if they were psychological or emotional diagnoses

Current guardrails:

- contradictory or weak-fit cases reduce confidence and emit warnings
- missing genre coverage is surfaced explicitly
- retrieval stays local and inspectable
- each run is logged with parsed preferences, evidence, recommendations, confidence, and warnings
- the assistant does not claim to infer personal identity, mental health, or sensitive traits

## Helpful AI Suggestion

One helpful AI suggestion during the build was to turn the original simple recommender into a retrieval-aware assistant rather than trying to bolt on a generic chatbot. That suggestion led to a cleaner architecture: query parser, retriever, recommender, validator, logging, and evaluation around a single class-based ranking engine.

## Flawed AI Suggestion and Correction

One flawed AI suggestion early in the build was to maintain parallel functional and class-based recommendation paths. That would have duplicated logic and created drift between implementations. The project was corrected by centralizing the scoring and ranking logic in the `Recommender` class, making `Song` and `UserProfile` the main internal data structures, and building the rest of the system around that single source of truth.

## Future Improvements

- expand the catalog to reduce missing-coverage cases and thin genre representation
- improve retrieval with better snippet ranking or lightweight semantic search
- add a self-check or rerank step for explicit constraint violations
- calibrate confidence more formally using a larger evaluation set or human review
- introduce diversity-aware ranking so the top results are not overly narrow when coverage improves
