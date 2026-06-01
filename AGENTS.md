<!-- ARIS-CODEX:BEGIN -->
## ARIS Codex Skill Scope
ARIS Codex packages installed in this project: skills-codex
Managed entries: 78
Manifest: `.aris/installed-skills-codex.txt`
ARIS repo root: `/home/lqz25zhj/aris_repo`
Project skill path: `.agents/skills/<skill-name>`
For ARIS Codex workflows, prefer the project-local skills under `.agents/skills/`.
When a skill needs ARIS helper scripts, resolve the repo root from the manifest or set it explicitly:
`ARIS_REPO=$(awk -F'\t' '$1=="repo_root"{print $2; exit}' "/public/zhanghaojie/MambaVision/.aris/installed-skills-codex.txt")`
Do not edit or delete symlinked skills in place; update upstream or rerun:
`bash /home/lqz25zhj/aris_repo/tools/install_aris_codex.sh "/public/zhanghaojie/MambaVision" --reconcile`
For copied Codex installs, use:
`bash /home/lqz25zhj/aris_repo/tools/smart_update_codex.sh --project "/public/zhanghaojie/MambaVision"`
<!-- ARIS-CODEX:END -->
