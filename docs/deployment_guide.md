# Deployment Guide (Checkpoint 4)

Target: **Hugging Face Spaces (Docker SDK, free CPU Basic)**. Chosen because it runs your own Dockerfile,
needs no credit card, gives 2 vCPU / 16 GB RAM (enough for PyTorch + MiniLM), injects secrets as
environment variables, and gives a public URL. Do these in order; each step has a check.

## 0. Before you start
- A Gemini API key (`GEMINI_API_KEY`). **Treat it as a password.** Never put it in code, the README, a
  screenshot, or a GitHub commit. `.env` is already git-ignored and excluded from the image.
- Accounts: GitHub (repo), Hugging Face (free). Docker Desktop is optional but needed for step 2.
- The repo has the Checkpoint 3 + 4 files and the tests pass: `python -m pytest tests -q`.

## 1. Run locally without Docker (confirm the app works first)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # then paste your key into .env
python -m src.ingestion.pipeline                       # first run downloads MiniLM
streamlit run src/app/streamlit_app.py                 # opens http://localhost:8501
```
Check: ask "Can my employer fire me without giving a reason?" and get an answer citing Article 282.

## 2. Build and run the container locally
```bash
docker build -t labor-law-bot .                        # first build is slow (10+ min): torch + model + index
docker run --rm -p 7860:7860 -e GEMINI_API_KEY=your_key_here labor-law-bot
```
Open http://localhost:7860. Check: `docker images labor-law-bot` (record the size for the report) and
`docker ps` shows `healthy` after about a minute.

Optional limits: `-e APP_DAILY_CALL_BUDGET=15 -e APP_MAX_QUESTIONS_PER_SESSION=10 -e APP_PASSCODE=something`.

## 3. Create the Space
1. huggingface.co -> New Space -> **SDK: Docker**, template **Blank**, hardware **CPU basic (free)**.
   Visibility: Public is needed for a link a grader can open. See the quota warning below.
2. Space -> Settings -> **Variables and secrets**:
   - **Secret** `GEMINI_API_KEY` = your key. (Variables are public; secrets are not. Use a Secret.)
   - Optional Secret `APP_PASSCODE`; optional Variable `APP_DAILY_CALL_BUDGET` (default 15).

## 4. Deploy: pick ONE
**A. Manual (simplest, do this first)**
```bash
python deploy/prepare_space.py ../space-copy           # repo copy + the Space YAML header on README.md
cd ../space-copy
git init -b main && git add -A && git commit -m "Deploy"
git remote add space https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
git push --force space main                            # username = your HF username, password = an HF access token with write access
```
**B. Automatic from GitHub (the CI/CD step in `.github/workflows/ci-cd.yml`)**
GitHub repo -> Settings -> Secrets and variables -> Actions: add secret `HF_TOKEN` (HF token with write
access) and variable `HF_SPACE` (`YOUR_USERNAME/YOUR_SPACE`). Every push to `main` then runs tests, builds
and smoke-tests the image, and force-pushes a deploy copy to the Space. Without those two values the deploy
job skips itself and the tests/build still run.

## 5. Verify
The Space page shows **Building**, then **Running** (first build ~10-15 min). Then open the app and:
1. Ask a covered question and a follow-up ("In that case, how much separation pay?") to prove memory.
2. Ask "What's the minimum wage in California?" and confirm it is refused.
3. Check the sidebar shows the remaining daily AI calls.
Capture evidence in `docs/deployment_evidence.md`.

## 6. Quota and abuse warning (read this)
A public app spends **your** Gemini quota, and your own free tier is about 20 requests/day. The app limits
the whole process to `APP_DAILY_CALL_BUDGET` calls per UTC day and each visitor to 10 questions, but those
counters live in memory and reset when the Space restarts. For the live defense: keep the Space public
only as long as needed, or set `APP_PASSCODE` and share it with the panel; run the demo against a fresh
quota day; and rotate the key afterwards.

## 7. Troubleshooting
| Symptom | Likely cause / fix |
|---|---|
| Space stuck on Building / fails | Open the **Logs** tab. Check the README header has `sdk: docker` and `app_port: 7860` (the prepare script adds it). |
| App says quota used up | Daily budget reached: wait for the next UTC day or raise `APP_DAILY_CALL_BUDGET` (and check your Gemini quota). |
| "API key" error in chat | `GEMINI_API_KEY` is missing or was added as a Variable in the wrong place. Re-add it as a Secret and restart the Space. |
| Answers look generic / empty retrieval | The index is empty or stale: rebuild the image (the Dockerfile re-runs ingestion). |
| Free Space went to sleep | Free Spaces can pause after a period of inactivity. Open the URL to wake it and allow a minute. Do this before the defense. |
| Build exceeds time/disk | Confirm the Dockerfile still installs the CPU-only torch wheel. |

## 8. Updating
Change code, push (B) or rerun step 4A. The Space rebuilds. New documents: put them in `data/incoming/`,
commit, redeploy; ingestion runs during the image build. Conversation memory is in process memory and is
lost on every restart; that is a known limitation.

## 9. Alternatives (not used)
- **Streamlit Community Cloud:** easy and free, but it does not run your Dockerfile, so it does not
  demonstrate the containerization deliverable.
- **Render free tier:** runs Docker, but its free RAM (about 512 MB, as far as I know; verify) is likely too
  small for PyTorch + MiniLM.
- **Google Cloud Run / Fly.io:** work well but need a billing account.
