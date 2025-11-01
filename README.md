# Project

This repository contains a simple model script (`model.py`) and the raw dataset in `data/`.

Note: The raw CSV dataset is excluded from Git by default via `.gitignore`. If you'd like the data tracked, consider Git LFS.

Quick setup (PowerShell):

1. Configure git (only once):

   git config --global user.name "Your Name"
   git config --global user.email "you@example.com"

2. Initialize, commit and push:

   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/<username>/<repo>.git
   git push -u origin main

Or, create the repo and push with GitHub CLI:

   gh repo create <username>/<repo> --public --source=. --remote=origin --push

If using SSH, replace the remote URL with `git@github.com:<username>/<repo>.git`.

For large datasets, use Git LFS (https://git-lfs.github.com/).
