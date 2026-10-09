# Technical references

These references explain source provenance and implementation choices. They do
not supply a claim of whole-field accuracy or a license to redistribute assets.

| Reference | Use in this project |
| --- | --- |
| [Official field publication](https://bbs.robomaster.com/article/814728?source=8) | Pinned STEP source; identity recorded in each build |
| [Official rule centre](https://bbs.robomaster.com/wiki/20204847/809871?source=7) | Dimensions and intended field behavior; use the version recorded by the pack |
| [MuJoCo XML reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html) | Heightfields, collision geometry and model parameters |
| [MuJoCo model editing](https://mujoco.readthedocs.io/en/stable/programming/modeledit.html) | Robot/field composition through `MjSpec` |
| [MuJoCo contact parameters](https://mujoco.readthedocs.io/en/stable/modeling.html#contact-parameters) | Friction and solver-parameter precedence |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | Examples of separate visual and collision geometry |
| [Omni Physics collision guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/rigid_bodies_articulations/collision.html) | Optional Isaac static terrain contact probe |

Visual detail and collision fidelity are independent. Shared profiles retain
one physical source; local collision replacements must avoid duplicate support
and preserve adjacent routes. Single-heightfield topology limits remain explicit.

See [ROADMAP.md](ROADMAP.md) for implementation work and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for source identity and notices.
