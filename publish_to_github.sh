#!/usr/bin/env bash
# Publish this folder as a new GitHub repository.
#   ./publish_to_github.sh                 # repo named "carsim", public
#   ./publish_to_github.sh my-sim private  # your own name, private repo
# Needs git, and either the GitHub CLI (https://cli.github.com, run `gh auth login` once)
# or an empty repository you created on github.com (the script then asks for its URL).
set -euo pipefail
name="${1:-carsim}"
visibility="${2:-public}"
cd "$(dirname "$0")"

if [ -z "$(git config user.name || true)" ] || [ -z "$(git config user.email || true)" ]; then
  echo "Tell git who you are first (this is what appears on your commits), then run this again:"
  echo '  git config --global user.name  "Your Name"'
  echo '  git config --global user.email "you@example.com"   # or your GitHub noreply address'
  exit 1
fi

if [ ! -d .git ]; then
  git init -q -b main
  git add -A
  git commit -q -m "carsim: analysis-first race car simulator (roadmap phases 1-12)"
  echo "Created a local git repository with one commit."
fi

if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  gh repo create "$name" --"$visibility" --source=. --remote=origin --push
  echo "Done: $(gh repo view "$name" --json url -q .url)"
else
  echo "GitHub CLI not found or not logged in."
  echo "Create an EMPTY repository at https://github.com/new (no README, no licence), then paste its URL."
  read -r -p "Repository URL (e.g. https://github.com/you/$name.git): " url
  git remote remove origin 2>/dev/null || true
  git remote add origin "$url"
  git push -u origin main
  echo "Done: $url"
fi
