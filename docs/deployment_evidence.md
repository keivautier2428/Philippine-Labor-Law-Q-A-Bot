# Deployment Evidence (Checkpoint 4)

Fill this in with your own real results. Do not paste your API key into any screenshot.

| # | Evidence | Where / value | Done |
|---|---|---|---|
| 1 | Public URL of the deployed app | `https://huggingface.co/spaces/<user>/<space>` | [ ] |
| 2 | Screenshot: Space status "Running" | `docs/img/space-running.png` | [ ] |
| 3 | Screenshot: multi-turn chat with citations and the "Retrieved articles" panel | `docs/img/chat.png` | [ ] |
| 4 | Screenshot: out-of-scope question refused | `docs/img/refusal.png` | [ ] |
| 5 | Screenshot: Space Settings showing `GEMINI_API_KEY` as a secret (value hidden) | `docs/img/secrets.png` | [ ] |
| 6 | Local container: output of `docker images labor-law-bot` (image size) | `____ MB` | [ ] |
| 7 | Local container: `docker ps` showing `healthy` | | [ ] |
| 8 | GitHub Actions run: test, docker, deploy jobs | link: | [ ] |
| 9 | Date/time of the verification run and who ran it | | [ ] |

Observed results from the verification run (write what actually happened, including anything that failed):

- Cold-start time (Space wake-up to first answer): ____
- Typical answer latency (3 LLM calls, rate-limited to 5/min): ____
- Problems hit and fixes: ____
