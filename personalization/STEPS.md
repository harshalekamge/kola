# Component 4: Step-by-step guide (start here)

## What this project does, in one picture

    Components 1-3 send JSON events          (simulated for now: app/simulator.py)
            |
    check each event is valid                (app/schema.py)
            |
    save raw events in SQLite                (app/store.py)
            |
    turn one session's events into 8 numbers (app/aggregator.py)
            |
    GRU reads the last 5 sessions of a user  (drift_predictor.py)
            |  "probability this user is declining"
    personalization signal                   (personalization_signal.py)
            |  gesture x1.4, speech x1.3, simple UI ...
    AI Assistant Engine applies it           (ai_assistant_engine.py)

`run_pipeline.py` runs the whole chain in one command.

## Folder layout

    RP-Personalization-Module/
      app/                      NEW: events, store, aggregator, bridge
      tests/                    NEW: 15 automatic checks
      docs/INTERFACE_CONTRACT.md   send this to Components 1-3 teammates
      data/sample_events.jsonl     example events
      context_store.py          OLD (v2): makes ready-made fake rows (kept, not used by run_pipeline)
      drift_predictor.py        OLD (v2): the GRU (used by run_pipeline)
      personalization_signal.py OLD (v2): probability -> settings (used)
      ai_assistant_engine.py    OLD (v2): mock engine (used by demo.py)
      demo.py                   OLD (v2): original quick demo, still works
      run_pipeline.py           NEW: full chain, start to end
      requirements.txt, .gitignore, STEPS.md

## Step 1. Copy into your repo
Copy everything in this folder into your PyCharm project
(RP-Personalization-Module). Keep your existing venv.

## Step 2. Install and check
In the PyCharm terminal, with the venv active:

    pip install -r requirements.txt
    python -m pytest -q          # expect: 15 passed
    python run_pipeline.py       # runs the whole chain

(If `python` is not found, use `py`.)

## Step 3. Read the output
Section 6 of the output compares two systems on users the model never saw:
- GRU (your model)
- Static threshold (one fixed rule for everyone, the "no personalization" baseline)

## Step 4. Commit in small steps (real dates, no backdating)
Run `git status` first. If `.idea` or `__pycache__` are still tracked:
`git rm -r --cached .idea __pycache__` and commit that first.

    1. docs: add interface contract for components 1-3          (docs/)
    2. feat: add event schema and validation                    (app/__init__.py, app/schema.py)
    3. feat: add SQLite context store                           (app/store.py)
    4. feat: add event simulator with 4 user profiles           (app/simulator.py)
    5. feat: add session feature aggregator                     (app/aggregator.py)
    6. test: add foundation tests                               (tests/, data/)
    7. feat: connect context store to GRU predictor             (app/bridge.py, run_pipeline.py)
    8. docs: add requirements and steps guide                   (requirements.txt, STEPS.md, .gitignore)

Spread them over several days as you actually work through and read each file.
Then `git push`.

## Step 5. Understand each file before the panel asks
Open each file and read the top comment. Be able to say, in one sentence each:
- what goes in, what comes out
- why it exists

## Step 6. Tell your teammates
Send them docs/INTERFACE_CONTRACT.md. Ask them to emit events in that shape
(or tell you what they can emit, and we adapt the schema).

## What the PP1 story is
1. Problem: users' ability changes across sessions; fixed settings don't adapt.
2. Data contract: JSON events from three components (docs/INTERFACE_CONTRACT.md).
3. Pipeline: validate, store, aggregate, GRU, signal.
4. Result: GRU vs static threshold on unseen users.
5. Honest limits: data is simulated; real validation waits for Components 1-3.
