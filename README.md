# SympAI

A personal health guide built as a hospital of AI agents. The full design is in the SympAI system design doc.

## Step 1. One call to a model in the cloud

1. `cp .env.example .env`, then paste your Groq key into `.env`
2. `uv sync`
3. `uv run python scripts/hello_groq.py`

## Step 2. An agent loop by hand, with one tool

1. `uv sync` (adds httpx, pytest, and installs the `sympai` package from `src/`)
2. `uv run pytest` (three offline tests for the MedlinePlus tool)
3. `uv run python scripts/agent_by_hand.py "I have a sore throat and fever since yesterday"`

Files
- `src/sympai/tools/medlineplus.py` is the tool. Its docstring is what the model reads.
- `scripts/agent_by_hand.py` is the loop. Model, tool call, run the tool, send the result back, repeat.
- `tests/test_medlineplus.py` tests the parsing against a saved reply, so it runs offline.

## Step 3. The hospital as a graph, with LangGraph

1. `uv sync` (adds langgraph, pyyaml, and the Studio server as a dev tool)
2. `uv run pytest` (11 offline tests, including a whole two-turn consult with a fake model)
3. `uv run python scripts/chat.py` to talk to it in the terminal
4. `uv run langgraph dev` to watch it in LangGraph Studio. Open the Studio link it prints in Chrome.

Files
- `departments/ent.yaml` is the first department. A new department is a new file.
- `src/sympai/departments.py` is the one department agent and the DepartmentReport it returns.
- `src/sympai/graph.py` is the graph. Nodes, edges and shared state.
- `scripts/chat.py` is a terminal chat that remembers the conversation.
- `langgraph.json` tells Studio where the graph is.

### Step 3.1. Fixes from the first real tests

- A lookup budget in code. After 3 rounds of tools in one turn, the coordinator must answer with what it has.
- The prompt forbids naming clinics, addresses, routes, apps or prices until a places tool exists (step 8).
- One question per turn, with no heading, and never two questions in one.
- 911 is kept for real emergencies. Urgent but not life threatening means urgent care or a same-day visit.
- MedlinePlus "no page" results now say not to search again.
- `chat.py` shows the case each department receives, its full report, and which model wrote it.

### Step 3.2. Fixes from the second test

- A department report that says "in scope" with no findings is now invalid, so the backup model retries it.
- Stroke signs and a sudden worst-ever headache always mean 911, never urgent care.
- Knowledge without a source is labeled as unchecked, never as "consensus".
- Tool results from earlier turns are cut to a stub in what the model is sent. The state keeps them in full.
- Groq calls retry up to 6 times on a rate limit.

### Step 3.3. It pauses and picks back up

- Conversations are saved after every step in `data/conversations.sqlite` on this laptop, which git ignores.
- No reply for 2 minutes, or Ctrl+C, pauses it. The next run says where you were and offers to carry on.
- A turn that failed halfway, for example on a rate limit, is finished on the next Enter or the next run.
- Try the timer quickly with `SYMPAI_IDLE_SECONDS=15 uv run python scripts/chat.py`
- SympAI no longer states facts about your own insurance plan. It points to your Summary of Benefits and Coverage.

## Step 4. The triage desk

Every message goes through `triage` before any model sees it.

1. `uv run pytest` (52 tests, plus one expected failure that shows what rules cannot do)
2. `uv run python scripts/check_prompt_guard.py` once, to see what Prompt Guard really returns
3. `uv run python scripts/chat.py`, then try "my face is drooping and my speech is slurred"

Files
- `src/sympai/triage.py` has the emergency rules, the fixed replies, and the Prompt Guard check.
- `tests/test_triage.py` has cases that must raise an alarm, and real answers from testing that must not.
- Step 4b adds a SetFit classifier for the phrasings the rules miss.

### Step 4b. The triage classifier, with SetFit

1. `uv sync` (adds SetFit and PyTorch, about 130 MB to download for PyTorch)
2. `uv run python scripts/train_triage.py` (downloads the base model once, about 470 MB, then trains in a minute or two)
3. `uv run python scripts/chat.py`, then try "it feels like an elephant is sitting on my chest"

Files
- `triage/examples.csv` is what it learns from. Add sentences the way you and your friends really talk, in any language, then train again.
- `triage/test_examples.csv` is held back for testing. Never copy its sentences into examples.csv, or the test stops meaning anything.
- `src/sympai/classifier.py` loads the trained model. Without one, triage uses the rules alone.
- Trained models go in `models/`, which git ignores.

Also in this step
- Code keeps only the first question when the coordinator asks two at once.
- An OTC medicine is suggested only after allergies, other medicines, pregnancy and organ problems are known.
- A reply typed at the welcome-back prompt now counts as your answer.

### Step 4c. Alarms that help instead of blocking

- The classifier has a fourth label, `urgent`, for "see a doctor today". Retrain after pulling this.
- A classifier hit shows a short warning and the coordinator carries on, asking whether it is happening now.
- An exact rule match still stops the turn with the fixed notice, but only the first time in a conversation.
- The same alarm is never shown twice in one conversation. The coordinator is told through the state instead.
- Warning signs described in the past ("I had chest pain last year") get a short check instead of the full stop.
- SympAI can be asked on behalf of someone else, and asks about that person.
- `uv run python scripts/train_triage.py --evaluate-only` tests the saved model without retraining. The report shows the whole desk, rules plus classifier, at several thresholds.

## Step 10. Evals, run by itself

`uv run python scripts/run_scenarios.py` plays every case in `evals/scenarios.csv` as a whole conversation.
- SympAI is the real graph, the same one `chat.py` runs.
- A second model plays the person, from the case's `patient_facts`.
- Code checks what must always hold. That means the triage level, one question at a time, no invented places or prices, and never "confirmed".
- A third model judges the transcript against `correct_if` and `wrong_if`.

Results, with every transcript, go to `evals/results/latest.md`. Run a few cases with `--only 3,7`.

## Step 6. All the departments

`departments/` now has eight files. They are general medicine, heart and lungs, neurology, ENT, eyes, skin, bones and muscles, and mental health. Heart and lungs was added to the original seven, because chest symptoms were the most common worry in testing. A test checks that every `refers_to` points at a department that exists.

## Step 5. The chart, one per person

SympAI now remembers people across conversations, in `data/charts.sqlite` on this laptop.
- Before every reply, the coordinator gets a short summary of the person's chart, so it does not ask again for their age or insurance.
- After every turn, a small "chart clerk" model folds in anything new the person said about themselves. Nothing about other people they ask for.
- `uv run python scripts/chat.py --person mom` keeps a separate chart and conversation for someone else.
- `uv run python scripts/chart.py show`, `forget` and `people` let you see or delete what it remembers.

## Step 7. The research center

- `search_pubmed` finds reviews from the last 5 years on PubMed.
- `search_clinical_trials` finds studies recruiting now on ClinicalTrials.gov, and always says a trial is a study, not a treatment.
- `uv run python -m sympai.mcp_server` publishes all the tools as an MCP server, so other agents, such as Claude in the desktop app, can use them.

## Step 8. Patient services

- `find_clinicians` lists real licensed clinicians near a ZIP code from the NPI Registry, with a Google Maps link for ratings.
- `care_options` gives map searches for urgent care and pharmacies, low-cost community health centers, and the crisis numbers.
- SympAI may now name a place, but only one a tool returned. The evals check that every address and phone number came from a tool.

## Step 9. Voice

`uv run python scripts/chat.py --voice --speak`
- Press Enter on an empty line to talk, and Enter again when you are done. Typing still works.
- Whisper on Groq writes down what you said and tells SympAI the language, so it replies in that language.
- `--speak` reads each reply aloud. English uses your Mac's own system voice. Other languages get a natural voice for that language if one is installed.
- Apple's joke voices, like Albert, Bad News and Zarvox, can never be picked. The first version picked Albert, and it was frightening.
- `SYMPAI_VOICE=Samantha` picks your own English voice. `say -v ?` lists them.
- The first time, macOS asks to let Terminal use the microphone.

### Step 9.1. When Groq's free allowance runs out

The 120b model gets 200,000 tokens a day on the free tier, and one long conversation can use 30,000. Case 13 ran out halfway.
- When Groq says the 120b model is out, the 20b model answers instead, and the reply says so. Every safety check in code still runs.
- A circuit breaker, the same idea as Resilience4j, remembers how long Groq said to wait, so later calls skip straight to the backup instead of spending 25 seconds on retries.
- If both are out, the person gets an honest "try again in about 15 minutes", with the 911 and 988 lines, instead of a crash.

## Step 11. Friends, each with their own key

Step 12 replaces the Vercel chat screen with SympAI's own page. The keys and the Tailscale address below work the same way.

Friends use the chat screen at agentchat.vercel.app. That page is only a frontend. It runs in the friend's browser and calls SympAI's API on this laptop, through a public address that forwards to `localhost:2024`. Nobody gets your IP address.

1. Make a key for each person. Add yourself as `me`, so the chat screen and the terminal share your chart.
   ```
   uv run python scripts/friends.py add me
   uv run python scripts/friends.py add asha
   ```
   Each key is printed once. Only its hash is saved, in `data/friends.json`. This works while the server is running, from a second Terminal tab.
2. Start SympAI. `--no-browser` stops it opening LangGraph Studio, which is not needed here.
   ```
   uv run langgraph dev --no-browser
   ```
3. For yourself on this laptop, open agentchat.vercel.app in Chrome and enter `http://localhost:2024`, `sympai` and your key.
4. For friends, give it a public address with Tailscale Funnel, in a second Terminal tab. Install Tailscale from tailscale.com and sign in first.
   ```
   tailscale funnel 2024
   ```
   It prints an address ending in `ts.net`, which stays the same every time. Each friend enters that address, `sympai` and their own key.

Why not `langgraph dev --tunnel`. It uses a free Cloudflare quick tunnel, and quick tunnels do not support Server-Sent Events, which is how the chat screen receives each reply. In testing SympAI answered, and the reply never reached the page.

What keeps it safe
- Every request needs a key. No key or a wrong key gets 401.
- Each conversation is stamped with its owner, and a friend only ever sees their own. Opening someone else's by its id gets 404, as if it did not exist.
- Each friend's chart is kept under their signed-in name, which the server sets itself. A friend cannot ask for someone else's.
- Friends can use SympAI but cannot change it. Everything without a rule is denied.
- `disable_studio_auth` is on. Without it, anyone who sends one special header is treated as LangGraph Studio and skips the key check. A test fails if it is ever turned off.
- Each friend gets 30 messages a day, so one person cannot use up the free allowance for everyone. Change it with `SYMPAI_DAILY_MESSAGES`.

Things to tell friends
- It is your experiment, not a medical service, and for an emergency they should call 911.
- Their chart and conversations are kept on your laptop. `uv run python scripts/friends.py remove asha --forget-chart` takes a key back and deletes the chart.
- SympAI only works while your laptop is on and running both commands.

## Step 12. SympAI's own chat page

`web/` is SympAI's own page, a static Next.js site. It replaces the Vercel chat page for friends, and it talks to the same server on your laptop.
- A welcome screen that says it is your experiment and not a doctor, and asks for the key once.
- Live lines like "Asking the bones and muscles department..." while SympAI works.
- A red banner with a Call 911 button when the triage desk flags an emergency, and a 988 banner for a crisis.
- Clinician cards with Call and Map buttons, drawn from the NPI Registry data the tool returns, not from the model's words.
- A mic button. Whisper on the server writes down what was said, and replies can be read aloud by the browser's own voice.
- A "What SympAI remembers" panel with a button to forget it all.
- An offline screen with 911, 988 and Poison Control when your laptop is off.

Three small routes in `src/sympai/web_api.py` serve the page's memory panel and voice. They sit behind the same key check as everything else.

Try it on this laptop
```
cd web
npm install
npm run dev
```
Open http://localhost:3000 with `langgraph dev --no-browser` running in another tab.

Put it online for friends, free, on Vercel
1. Push the repo to GitHub.
2. On vercel.com, choose Add New, then Project, and import the Sympai repo.
3. Set Root Directory to `web`.
4. Add the environment variable `NEXT_PUBLIC_SYMPAI_API_URL` with your `ts.net` address from `tailscale funnel 2024`.
5. Deploy. Friends open the Vercel link and paste their key. Nothing else to type.

The page has no secrets in it. Each friend's key stays in their own browser, and your laptop checks it on every request.
