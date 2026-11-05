# guard.media: LinkedIn assets, generated from real runs

```bash
python -m guard.media --out runs/media                       # small local runs, everything
python -m guard.media --out runs/media \
  --dataset runs/screen-injection/v1 \
  --firewall runs/firewall/teacher-none runs/firewall/teacher-rules runs/firewall/teacher-model
```

| Folder | Files | Read from |
| --- | --- | --- |
| `post1-dataset/` | `carousel.pdf` (6 slides, 1080x1350), `slide-*.png`, `grid.png` | a `guard.screen_injection` run |
| `post2-wallet/` | `carousel.pdf` (7 slides), `slide-*.png`, `terminal.mp4` | the wallet decoder's real output |
| `post3-extension/` | `extension-demo.mp4`, `before.png`, `after.png`, `phishing.png` | the extension, recorded in Chromium |
| `post4-firewall/` | `chart-by-technique.png`, `chart-by-goal.png`, `firewall-replay.mp4` | `guard.firewall` runs; the first `--firewall` dir is the baseline |

- **Numbers.** Every number on a graphic is read from the run directories, so rerun on the
  Spark's job outputs before posting real results.
- **Agent label.** Charts from the scripted agent say so on the image ("a scripted worst-case
  agent"). Use the teacher-agent runs for claims about real models.
- **The extension demo** needs Playwright's full Chromium build, which loads extensions
  (`python -m playwright install chromium`).
- **No real brands.** The demo news site and "NovaPay" are fictional. The wallet slides use the
  public USDC contract address and made-up counterparties.
- **Videos** are H.264 MP4 at 1280x720, ready for LinkedIn. Without `ffmpeg` they stay WebM.
