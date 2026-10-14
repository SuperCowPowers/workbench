# Spikes

Self-contained experiments that may not survive: each subdirectory is one spike, holding
everything it needs. A spike that wins is promoted into the package (templates into
`src/workbench/model_scripts/`, helpers into `src/workbench/`); one that loses is deleted.

A spike model template trains through the normal Workbench path with
`to_model(custom_script="spikes/<name>/<template>.template", ...)`: the template is filled
in place and its whole directory ships as the training job's source, so sibling modules
and a `requirements.txt` go with it, and the package itself needs no change or release.
