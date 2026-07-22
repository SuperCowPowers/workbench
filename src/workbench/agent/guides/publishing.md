# Why a Build Takes Time

> why a FeatureSet/Model/Endpoint build is slow — Workbench publishes durable, reusable artifacts, not local throwaway runs; reframing the "why so long?" complaint

"Why does this take so long?" is a fair complaint, common from people used to
running models locally where a fit is seconds. A FeatureSet + Model + Endpoint can
be a couple of hours. Reframe the trade honestly — don't apologize, don't get
defensive — and only raise this when the user actually voices the frustration.

- Workbench is a **publishing** framework. A FeatureSet/Model/Endpoint isn't a
  throwaway script result — it's a persistent, named artifact the whole team and
  downstream pipelines inspect, measure, and reuse. You pay the build cost
  **once**; everyone else picks it up by name (`Model("aqsol-regression")`) and
  never re-runs it.
- You don't re-run a pipeline each session — rebuild only what changed; the rest
  you pick up by name.
- For PyTorch/Chemprop, local isn't the faster path — those need a real GPU and
  take hours on their own. Workbench just runs them on the right hardware instead
  of a laptop.
- Acknowledge the wait is real; the payoff is a durable shared artifact, not a
  number that vanishes when the terminal closes.
