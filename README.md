# Headgear vs clear-aligner elastics for full-arch distalization: 4D finite element code

Code and result summaries for the study

> Savabieh S, Sharifi ME. *Comparison of headgear-assisted fixed appliances and clear aligners with
> precision-cut, aligner-hook or metal-hook elastics for full-arch distalization: a multi-patient 4D
> finite element study.* Submitted to *BMC Oral Health*.

The study compares four ways of distalizing the right maxillary half-arch (FDI 17–11) on five
patient-specific models from the [Open-Full-Jaw](https://github.com/diku-dk/Open-Full-Jaw) dataset
(patients 8, 3, 4, 5 and 6):

| Group | Appliance | Where the 200 gf force enters |
|---|---|---|
| **HG** | MBT ceramic brackets 15–11, tubes 16/17, 0.019 × 0.025-in SS archwire sliding in the slots, cervical headgear 15° below the occlusal plane | headgear tube of 16, then the archwire |
| **PC** | 0.70-mm aligner, IZC miniscrew elastic attached through a **precision cut** over the canine | the aligner, then aligner–tooth contact |
| **CAH** | same aligner, 4-mm **hook made of aligner material** | the aligner (hook root), then contact |
| **CMH** | same aligner, 4-mm stainless-steel **hook bonded to the canine** under a window in the aligner | the canine directly |
| Control | same aligner, no elastic | — |

Aligners follow a 70-step V-pattern staging plan; the headgear group is force-driven over 140 bone-remodelling
iterations. The platform reproduces the 4D staging protocol of Mao et al. (Bioengineering 2024;11:3 and
Prog Orthod 2023;24:16) with the remodelling scheme of Hamanaka et al. (Am J Orthod Dentofacial Orthop 2017).

## Repository contents

```
ofj4d/        4D staging platform (the code used for every run in the paper)
  spec.py       all model and study parameters (materials, attachments, forces, hook and headgear geometry)
  run4d.py      aligner groups: staging loop, `--group ca_precision_cut | ca_aligner_hook | ca_metal_hook | ca_control`
  fixed4d.py    headgear group (HG): brackets, archwire and force-driven remodelling
  brackets.py   bracket reader (Parasolid text), slot/angulation/base detection, FA-point placement, tubes
  wire4d.py     co-rotational archwire with sliding slot contact, play and friction
  loads4d.py    IZC miniscrew, precision cut, aligner-material hook, bonded metal hook
  solve4d.py    contact solver (penalty contact, active set, PARDISO / optional GPU assembly via gpu4d.py)
  arch4d.py     arch form and staging prescription
  contact4d.py  frictionless tooth-to-tooth contact
  ...           geometry, aligner, model assembly, PDL stress, FEBio cross-check (febio4d.py)
ofj/          model-building code (Open-Full-Jaw mesh -> FE model, aligner shell, attachments)
build4d/study/jobs_*.txt   job definitions: main set, linear-PDL set and sensitivity set (job name, group, environment)
build4d/plan_scale.txt     collision-free plan scaling of the incisors for patient 8
out/          result summaries used in the paper
  study_stats.md / .csv          main analysis: Friedman tests, Holm-corrected paired t-tests, sign consistency
  study_stats_pdl068.*           sensitivity analysis with a linear PDL (0.68 MPa)
  study_stats_mao.* , mao_compare.md   sensitivity analysis with the original plan of the source protocol
  study_patients*.md / .csv      key landmarks for every patient and group
  force_transfer.md              how the elastic load reaches each tooth in PC, CAH and CMH
  stiffness_check.md, overlap_check.md   numerical checks
  pdl_stress.md                  PDL stress per tooth (patient 8)
  study/                         patient 8: displacement tables per step and at matched distalization, long-format path data, model renders
```

## Model settings

All settings are in `ofj4d/spec.py`, `ofj4d/solve4d.py` and `ofj4d/arch4d.py` and can be changed with the
environment variables named there. The paper used the code defaults, including:

* PDL: 0.30-mm three-layer shell, bilinear (0.05 / 0.22 MPa, knee at 7.5 % strain), Poisson's ratio 0.45
* Aligner: 0.70-mm shell, E = 1500 MPa, ν = 0.30; aligner–crown penalty contact 2000 N/mm, friction 0.20
* Attachments: 3 × 2 × 1 mm horizontal on 17/16, 2 × 3 × 1 mm vertical on 15–13, E = 20 000 MPa
* Socket kinematics: bodily translation of each aligner socket to the planned crown position (`OFJ_PRESCRIBE=translate`)
* Tooth-to-tooth contact: 500 N/mm per contact point; wire–slot penalty 10 000 N/mm, friction 0.15
* Elastic and headgear force: 200 gf; IZC miniscrew head 4 mm apical to the buccal crest and 2 mm proud of bone

The variations in `build4d/study/jobs_sens.txt` (force, linear PDL, miniscrew height, hook length and width,
headgear angle, bonded wire, wire friction) reproduce the sensitivity analyses.

## Running

Python 3.10+ with the packages in `requirements.txt`. A run needs the patient FE model built from the
Open-Full-Jaw data (`ofj/build_model.py`, see `data/README.txt`); the patient meshes and built models are
not redistributed here because they are derived from the Open-Full-Jaw dataset, which should be obtained
from its own repository.

Run from the repository root (paths such as `build4d/` are relative to it):

```bash
python ofj4d/run4d.py --group ca_precision_cut      # one aligner group, 70 steps (resumes from its checkpoint)
python ofj4d/fixed4d.py --build                     # headgear group: build brackets, tubes and archwire
python ofj4d/fixed4d.py                             # headgear group: 140 remodelling iterations
OFJ_PDL_LINEAR_E=0.68 python ofj4d/run4d.py --group ca_metal_hook   # example sensitivity run
```

## Not included

* Open-Full-Jaw patient meshes and the FE models built from them (obtain the dataset from its repository).
* The ceramic bracket CAD library read by `brackets.py` (a third-party SolidWorks assembly).
* Raw per-step run files (`*.pkl`, several hundred MB), the server batch scripts and the statistics scripts;
  the statistical results they produced are in `out/`. These are available from the corresponding author on request.

## Citation

If you use this code, please cite the paper above (see `CITATION.cff`) and the Open-Full-Jaw dataset:
Gholamalizadeh T, et al. Open-Full-Jaw: an open-access dataset and pipeline for finite element models of human
jaw. Comput Methods Programs Biomed. 2022;224:107009.

## Contact

Mohammad Emad Sharifi — emad.sharifi2001@gmail.com — ORCID 0000-0003-2846-6964

## License

MIT (see `LICENSE`).
