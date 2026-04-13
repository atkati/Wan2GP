# Synchronisation avec upstream (deepbeepmeep/Wan2GP)

Ce fork ajoute le plugin `betterclip-api` dans `plugins/`.
Le code core de Wan2GP n'est **jamais** modifie.

## Remotes

```
origin    https://github.com/atkati/Wan2GP.git     (fork)
upstream  https://github.com/deepbeepmeep/Wan2GP.git  (source)
```

## Procedure de rebase

```bash
# 1. Se placer sur la branche betterclip
git checkout betterclip

# 2. Recuperer les dernieres modifications upstream
git fetch upstream

# 3. Rebaser sur upstream/main
git rebase upstream/main

# 4. Resoudre les conflits eventuels (ne devrait pas arriver
#    car on ne touche jamais au code core)

# 5. Pousser la branche rebasee
git push origin betterclip --force-with-lease
```

## Regles

- **Ne jamais modifier** les fichiers hors de `plugins/betterclip-api/`, `SYNC.md`,
  `tests/test_betterclip_api.py` et `.gitignore`.
- Les dependances Python du plugin (fastapi, uvicorn) sont deja presentes
  dans l'environnement Wan2GP via Gradio.
- Si upstream ajoute un dossier `plugins/betterclip-api/` (improbable),
  renommer le notre en `plugins/betterclip-engine-api/`.
