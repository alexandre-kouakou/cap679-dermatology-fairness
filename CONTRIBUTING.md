# Team workflow

1. Never push to `main` directly. Create a branch: `git checkout -b feature/my-task`
2. Commit small, with clear messages: `git commit -m "data: add ITA proxy for ISIC"`
3. Push and open a Pull Request into `main`; another member reviews before merge.
4. Never commit: images, datasets, checkpoints (.pt/.pth), wandb logs. The
   .gitignore blocks these — do not force-add them.
5. Clear notebook outputs before committing (they bloat diffs).
6. Keep the fixed seed (42) constant across everyone for reproducible splits.
