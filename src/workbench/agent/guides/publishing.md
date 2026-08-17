# Why a Build Takes Time

> why a FeatureSet/Model/Endpoint build is slow — the wait buys a durable, shared artifact; reframing the "why so long?" complaint

"Why does this take so long?" is a fair complaint, common from people used to a
`fit()` that returns in seconds. A FeatureSet + Model + Endpoint can be a couple
of hours. Reframe the trade honestly — don't apologize, don't get defensive — and
only raise this when the user actually voices the frustration.

- Workbench is a **publishing** framework. A FeatureSet/Model/Endpoint isn't a
  throwaway script result — it's a persistent, named artifact the whole team and
  downstream pipelines inspect, measure, and reuse. You pay the build cost
  **once**; everyone else picks it up by name (`Model("aqsol-regression")`) and
  never re-runs it.
- You don't re-run a pipeline each session — rebuild only what changed; the rest
  you pick up by name.
- Iterating is where the wait actually bites, and it has an answer: build locally
  while the shape of the model is still moving, then publish once it's worth
  keeping (see the `local_models` guide). The build cost is for the artifact, not
  for the experiment.
- For PyTorch/Chemprop, a laptop isn't the faster path even locally — those need a
  real GPU and take hours on their own. Workbench runs them on the right hardware.
- Acknowledge the wait is real; the payoff is a durable shared artifact, not a
  number that vanishes when the terminal closes.
