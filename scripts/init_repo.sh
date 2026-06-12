#!/usr/bin/env bash
# =========================================================================
# init_repo.sh — Initialize the D10Sformer git repository
#
# Run this ONCE, from the d10sformer/ folder, on your local Mac.
# Pre-requisites:
#   - SSH key configured for GitHub (test with: ssh -T git@github.com)
#   - Repo already created on GitHub (empty, no README)
# =========================================================================

set -e  # exit on first error

# Limpiar cualquier estado parcial de git
if [ -d ".git" ]; then
  echo "⚠️  Found existing .git/ — removing for clean init"
  rm -rf .git
fi

echo "→ Initializing git repo on branch main..."
git init -b main

echo "→ Configuring user identity..."
git config user.email "victoriavazquez1995@gmail.com"
git config user.name "Vic"

echo "→ Adding remote origin..."
git remote add origin git@github.com:jose-marin-db/udesa-nlp-futbol-D10Sformer.git

echo "→ Staging files..."
git add .

echo "→ First commit..."
git commit -m "chore: initial project scaffold

- Folder structure (data, src, notebooks, tests, configs, reports)
- requirements.txt with pinned versions for Colab Pro
- README, base_config.yaml, .gitignore
- Starter EDA notebook (00_eda.ipynb)

Phase 0 of plan_de_implementacion.md"

echo ""
echo "→ Verifying SSH connection to GitHub..."
ssh -T git@github.com 2>&1 || true

echo ""
echo "→ Pushing to origin/main..."
git push -u origin main

echo ""
echo "✅ Repo initialized and pushed!"
echo "→ Verify at: https://github.com/jose-marin-db/udesa-nlp-futbol-D10Sformer"
