# STATUS 

## BUILT and tested

- env_card
- data_card
- trainers
- run
- cards
- lattice_data
- bundle
- goal
- conduct
- moral_data
- behavior_eval
- __main__

Tests: 115 tests in 7 files, all passing.

## DESIGNED ONLY (not built)

- GPU and large-language-model fine-tuning trainers are not included: plug a Trainer in (see TorchTrainer). The installed torch here is CPU-only.
- Remote and cloud training servers are designed, not built.
- Real ranking scores for the lattice curriculum come from the caller (for example the galaxy SDK's convergence): none are invented here.
- The reinforcement-learning trainer that would use the conduct reward is a plug-in, not included.

## Release

Source-available under the LICENSE in this folder. Patent pending (U.S. provisional application 64/178,949 and others). The licence text is a draft that has not been reviewed by an attorney.
