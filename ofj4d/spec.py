"""Specification of Mao et al., Bioengineering 2024, 11, 3 -- buccal TAD group.

'Effect of Different Anchorage Reinforcement Methods on Long-Term Maxillary Whole
Arch Distalization with Clear Aligner: A 4D Finite Element Study with Staging
Simulation.'  doi:10.3390/bioengineering11010003

Every value below is quoted from that paper (or, where it says 'following the
established methodology described in previous publications', from the companion
method paper Mao et al., Prog Orthod 2023;24:16, doi:10.1186/s40510-023-00468-1).
Anything NOT specified by either paper is marked ASSUMED.
"""
import os
import numpy as np

CITATION = ("Mao B, Tian Y, Xiao Y, Liu J, Liu D, Zhou Y, Li J. Bioengineering 2024;11(1):3. "
            "doi:10.3390/bioengineering11010003")

# ---------------------------------------------------------------- dentition --
# "To simplify the model, only the right half of the maxillary dentition was
#  involved. A contact surface was formed on the midsagittal plane for symmetric
#  analysis, and the front end of the CA, on the midsagittal plane, was
#  constrained in the x-direction."                       [Prog Orthod 2023]
HALF_ARCH_UNN = [2, 3, 4, 5, 6, 7, 8]          # FDI 17,16,15,14,13,12,11
FDI = {2: 17, 3: 16, 4: 15, 5: 14, 6: 13, 7: 12, 8: 11}
UNN = {v: k for k, v in FDI.items()}

# ---------------------------------------------------------------- materials --
# "the elastic modulus values were set at 1500 MPa and 20,000 MPa, respectively,
#  with Poisson's ratios of 0.30 for both"                          [Bioeng 2024]
E_ALIGNER, NU_ALIGNER = 1500.0, 0.30
E_ALIGNER *= float(os.environ.get('OFJ_E_ALIGNER_SCALE', 1.0))   # diagnostic only
E_ATTACH,  NU_ATTACH  = 20000.0, 0.30

# "For the PDL, the nonlinear hyperelastic model was used based on the double
#  linear stress-strain curve of Vollmer's research: when the dependent variable
#  of PDL e < 7.5%, E1 = 0.05 MPa, and when e > 7.5%, E2 = 0.22 MPa"
#                                                          [Prog Orthod 2023]
PDL_E1, PDL_E2, PDL_EPS_KNEE = 0.05, 0.22, 0.075
PDL_NU = float(os.environ.get('OFJ_PDL_NU', 0.45))
# Experiment knob (default 1.0 = the specified model): scales the PDL secant modulus of the
# listed teeth, a stand-in for the anchorage a larger root area provides -- the ligament of a
# bigger root is stiffer in the same proportion at equal thickness. Used to test whether the
# palatal-vs-buccal ordering follows molar anchorage (FINDINGS: other-patient replication).
PDL_E_SCALE = float(os.environ.get('OFJ_PDL_E_SCALE', 1.0))
PDL_E_SCALE_FDI = os.environ.get('OFJ_PDL_E_SCALE_FDI', '17,16')
# ASSUMED - neither paper states Poisson's ratio for PDL.  Overridable so the
# assumption can be tested against the paper's reported step-10 tipping.

# "The PDL was simulated with a 0.30-mm shell element"      [Prog Orthod 2023]
PDL_THICKNESS = 0.30

# "The CA was developed by making an external offset of dental crowns with a
#  thickness of 0.7 mm"                                     [Prog Orthod 2023]
ALIGNER_THICKNESS = 0.70

# Teeth and alveolar bone are RIGID: "the tooth and the alveolar bone were
# assumed to be rigid bodies during the initial displacement since their
# deformation is negligible compared with that of the PDL"   [Prog Orthod 2023]
# The bone is therefore not meshed at all; the outer PDL surface is constrained.

# -------------------------------------------------------------- attachments --
# "Conventional vertical and horizontal rectangular attachments were designed
#  according to the manufacturer's suggestion."  Fig.1a / Fig.4 of both papers
# show five attachments, on 17,16 (horizontal) and 15,14,13 (vertical).
ATTACHMENTS = {                      # unn: (label, length x height x depth mm)
    2: ('horizontal rectangular', (3.0, 2.0, 1.0)),   # 17
    3: ('horizontal rectangular', (3.0, 2.0, 1.0)),   # 16
    4: ('vertical rectangular',   (2.0, 3.0, 1.0)),   # 15
    5: ('vertical rectangular',   (2.0, 3.0, 1.0)),   # 14
    6: ('vertical rectangular',   (2.0, 3.0, 1.0)),   # 13
}

# ------------------------------------------------------------------ contact --
# "The interaction between the aligners and crowns was defined as a small-sliding
#  surface-to-surface contact, with a friction coefficient set at 0.2"
FRICTION = float(os.environ.get('OFJ_MU', 0.20))   # override for sensitivity only

# "stress and displacement converged when the element size was smaller than 0.2 mm"
TARGET_ELEM_SIZE = 0.20

# ------------------------------------------------------------ anchorage arm --
# "(1) buccal TAD group: the TAD was located in the buccal interradicular space
#  between the first and second molars 4 mm above the alveolar crest ...
#  The application of elastics on the CAs were all set at the buccal or lingual
#  mesial cervical region of the canine to simulate the precision cut.
#  ... the elastic forces for all the test groups were set as 150 g"
GROUPS = {
    'control':    dict(elastic=False),
    'buccal_tad': dict(elastic=True, side='buccal',
                       tad_between=(3, 2),       # between 16 and 17
                       tad_above_crest=4.0,
                       hook_tooth=6,             # canine 13
                       hook_site='mesial cervical, buccal'),
    'palatal_tad': dict(elastic=True, side='palatal',
                        tad_between=(3, 2), tad_above_crest=4.0,
                        hook_tooth=6, hook_site='mesial cervical, palatal'),
}
# Study variants, off by default: the precision cut on another tooth (FDI number) and the elastic force.
if os.environ.get('OFJ_HOOK_FDI'):
    for _g in ('buccal_tad', 'palatal_tad'):
        GROUPS[_g]['hook_tooth'] = UNN[int(os.environ['OFJ_HOOK_FDI'])]
ELASTIC_FORCE_GF = float(os.environ.get('OFJ_ELASTIC_GF', 150.0))
GF_TO_N = 9.80665e-3
ELASTIC_FORCE_N = ELASTIC_FORCE_GF * GF_TO_N      # 1.4710 N

# ============================================================================================
# FA-vs-CA STUDY  (Sharifi, Keshvad, Geramy: 'Comparison of Dental Effects of Full Maxillary
# Arch Distalization with Fixed Orthodontics and Headgear Versus Clear Aligner Hook Designs:
# A 4D Finite Element Method Study').  Three groups, carried over from the authors' 3D study
# (same force, same anchorage, same hook designs), on this 4D staging platform:
#
#   HG   fixed appliance (MBT ceramic brackets from the supplied Parasolid library, molar
#        tubes, passive 0.019x0.025 SS archwire bonded in the slots) + cervical headgear,
#        250 gf at the centre of the first-molar headgear tube        -> ofj4d/fixed4d.py
#   CAH  clear aligner, V-pattern whole-arch distalisation, 250 gf elastic from an IZC
#        miniscrew to a hook made of the aligner itself over the canine  (group 'ca_aligner_hook')
#   CMH  clear aligner, same staging, 250 gf elastic from the IZC miniscrew to a 4 mm metal
#        hook bonded to the canine; the aligner is trimmed around it and does not touch it
#                                                                       (group 'ca_metal_hook')
# Everything not listed here is the validated Mao et al. platform above.
# ============================================================================================
# One force for all three groups (only the appliance differs): 200 g per side, the value of the
# aligner + miniscrew distalisation FE studies (Guo et al., Eur J Orthod 2024;46:cjad077; Ma et al.,
# BMC Oral Health 2025;25:1327, upper level) and of IZC miniscrews in Kmeid et al., Dent J 2026;14:187;
# within the cervical-headgear range (150 g Kang et al. 2016, 250 g Kmeid et al. 2026).
STUDY_FORCE_GF = float(os.environ.get('OFJ_STUDY_GF', 200.0))

# IZC miniscrew: the elastic's fixed end is the screw HEAD.  Built by tools/izc_point.py from the
# bone-surface screw point at IZC_APICAL mm apical of the 16/17 buccal crest, moved IZC_HEAD mm
# out of the cortex along the buccal direction (head proud of the bone and mucosa).
# 4 mm apical of the 16/17 crest puts the head 14.7-15.6 mm above the occlusal plane on patients
# 8 and 3: the clinical IZC site of Liou et al. (14-16 mm above the occlusal plane), as used by
# Oguz & Ozden, Sci Rep 2025 (doi 10.1038/s41598-025-10035-9).  6 and 8 mm run as sensitivity.
IZC_APICAL = float(os.environ.get('OFJ_IZC_APICAL', 4.0))
IZC_HEAD = float(os.environ.get('OFJ_IZC_HEAD', 2.0))              # ASSUMED
IZC_FILE = os.environ.get('OFJ_IZC_FILE', os.path.join('build4d', 'izc.pkl'))

# Precision cut (PC): a patch of the aligner over the mesial cervical region of the canine,
# buccal.  The force goes into the aligner, which carries it to the dentition through contact.
# Metal hook (CMH): 3D study: 'The 4 mm hook arm was placed on the maxillary canine, with the
# tip of the hook positioned vertically at the level of the miniscrew ... The metal hook is
# bonded to the tooth, and the aligner is trimmed around it so they do not touch.'
# Button in the centre of the cervical third of the canine crown, buccal, mesiodistally centred
# (Domingos et al., Dental Press J Orthod 2026;30:e252564: buttons 'positioned in the center of
# the cervical-buccal region of the maxillary canines', rigid and bonded; Oh et al., Korean J
# Orthod 2023;53:420), 3 mm button (Luo et al., BMC Oral Health 2026), under a window cut in the
# aligner: the aligner's inner surface within METAL_HOOK_CUTOUT of the button (button radius +
# 0.5 mm clearance) is out of contact.  OFJ_MH_BASE_H=0 OFJ_MH_CUTOUT=0 puts it at the CEJ with no window.
METAL_HOOK_ARM = float(os.environ.get('OFJ_MH_ARM', 4.0))          # mm, apical
METAL_HOOK_TIP = os.environ.get('OFJ_MH_TIP', 'arm')               # 'arm' (4 mm) | 'screw' (tip at screw height)
METAL_HOOK_BASE_H = float(os.environ.get('OFJ_MH_BASE_H', 1.0/6.0))  # button height, fraction of crown from the CEJ (1/6 = centre of the cervical third)
METAL_HOOK_STANDOFF = float(os.environ.get('OFJ_MH_STANDOFF', 1.0))  # mm, arm clear of the gingiva (buccal)
BUTTON_D = float(os.environ.get('OFJ_BUTTON_D', 3.0))              # mm, button diameter
METAL_HOOK_CUTOUT = float(os.environ.get('OFJ_MH_CUTOUT', BUTTON_D/2 + 0.5))  # mm, aligner window radius around the button

# Aligner hook (CAH): the metal hook's arm in every respect -- button site at the CEJ, standoff,
# 4 mm length, tip -- but made of aligner material and part of the aligner (Elastic._aligner_arm):
# a cantilever of the aligner sheet, ALIGNER_THICKNESS thick and AH_WIDTH wide.  ASSUMED width.
AH_WIDTH = float(os.environ.get('OFJ_AH_WIDTH', 2.0))              # mm

GROUPS.update({
    # precision cut (PC): the elastic hooks on the aligner itself, a patch over the mesial
    # cervical region of the canine (Mao et al.'s precision cut; Guo et al. 2024)
    'ca_precision_cut': dict(elastic=True, side='buccal', tad_between=(3, 2), tad_above_crest=IZC_APICAL,
                            hook_tooth=6, hook_site='precision cut, canine, mesial cervical, buccal',
                            hook_kind='aligner', tad_file=IZC_FILE, force_gf=STUDY_FORCE_GF),
    'ca_aligner_hook': dict(elastic=True, side='buccal', tad_between=(3, 2), tad_above_crest=IZC_APICAL,
                            hook_tooth=6, hook_site='aligner-material hook (metal-hook geometry), canine, buccal',
                            hook_kind='aligner_arm', tad_file=IZC_FILE, force_gf=STUDY_FORCE_GF),
    'ca_metal_hook':   dict(elastic=True, side='buccal', tad_between=(3, 2), tad_above_crest=IZC_APICAL,
                            hook_tooth=6, hook_site='metal hook bonded to the canine, buccal',
                            hook_kind='metal', tad_file=IZC_FILE, force_gf=STUDY_FORCE_GF),
    'ca_control':      dict(elastic=False),
})

# Headgear (HG): cervical pull, 3D study: '250-gram (2.45 N) force ... vertical and distalizing
# force components were applied to the headgear tube of the molars ... at the centre of the
# molar tube, representing the attachment point of the headgear's inner bow.'
HG_FORCE_GF = float(os.environ.get('OFJ_HG_GF', STUDY_FORCE_GF))
# Cervical pull 15 deg below the occlusal plane, applied at the buccal tube of the first molar:
# Kang JM, Park JH, Bayome M, Oh M, Park CO, Kook YA. Korean J Orthod 2016;46(5):290-300,
# doi 10.4041/kjod.2016.46.5.290 (same group as Kawamura et al. 2021, the wire model).
HG_ANGLE = float(os.environ.get('OFJ_HG_ANGLE', 15.0))   # deg below the occlusal plane (cervical pull)
HG_ITER = int(os.environ.get('OFJ_HG_ITER', 2*70))       # remodelling cycles = the aligner protocol's 70 x 2

# PDL: overridable for the sensitivity run against the 3D study's linear PDL (E = 0.667 MPa)
if os.environ.get('OFJ_PDL_LINEAR_E'):
    PDL_E1 = PDL_E2 = float(os.environ['OFJ_PDL_LINEAR_E'])

# ------------------------------------------------------------------ staging --
# "The common 'V pattern' staging strategy for maxillary whole arch distalization
#  was designed. During the total 70 steps, for the movement of the canine to the
#  second molar, 0.1 mm distal movement along the arch form was prescribed for the
#  target teeth; for the movement of the incisors, 0.15 mm palatal movement was
#  designed in each relevant step."       Schedule read off Fig.4 of Bioeng 2024.
N_STEPS = 70
STEP_DISTAL = 0.10        # mm per step, canine .. second molar, along the arch form
STEP_PALATAL = 0.15       # mm per step, incisors, palatally

V_PATTERN = {             # unn: (first step, last step)  inclusive, 1-based
    2: (1, 20),           # 17
    3: (11, 30),          # 16
    4: (21, 40),          # 15
    5: (31, 50),          # 14
    6: (41, 60),          # 13
    7: (51, 70),          # 12
    8: (51, 70),          # 11
}

# Study variants, off by default. OFJ_STAGING: 'pairs' moves 17+16, then 15+14, then 13, the incisors
# keeping their V-pattern timing (70 steps); 'seq' moves one tooth at a time with no overlap (120 steps).
# OFJ_OVERCORR: planned extra distal movement of the molars (17, 16) as a fraction, 0.2 = 2.4 mm.
STAGINGS = {
    'v': V_PATTERN,
    'pairs': {2: (1, 20), 3: (1, 20), 4: (21, 40), 5: (21, 40), 6: (41, 60), 7: (51, 70), 8: (51, 70)},
    'seq': {2: (1, 20), 3: (21, 40), 4: (41, 60), 5: (61, 80), 6: (81, 100), 7: (101, 120), 8: (101, 120)},
}
STAGING = os.environ.get('OFJ_STAGING', 'v')
V_PATTERN = STAGINGS[STAGING]
N_STEPS = max(b for _, b in V_PATTERN.values())
MOLAR_OVERCORR = float(os.environ.get('OFJ_OVERCORR', 0.0))


# OFJ_PLAN_SCALE ('FDI:scale,...', e.g. '12:0.62,11:0.62'): shortens a tooth's planned movement.
# Used for the collision-free staging check (tools/plan_collision_free.py), where the V pattern's
# target positions would put adjacent teeth into each other.  Off (all 1.0) by default.
PLAN_SCALE = {}
for _kv in os.environ.get('OFJ_PLAN_SCALE', '').split(','):
    if ':' in _kv:
        _f, _s = _kv.split(':'); PLAN_SCALE[UNN[int(_f)]] = float(_s)


def step_mm(unn):
    """Prescribed movement per active step for this tooth."""
    sc = PLAN_SCALE.get(unn, 1.0)
    if unn in (7, 8):
        return STEP_PALATAL*sc
    return (STEP_DISTAL*(1 + MOLAR_OVERCORR) if unn in (2, 3) else STEP_DISTAL)*sc


def active_teeth(step):
    """Teeth prescribed to move at a given 1-based step."""
    return [u for u, (a, b) in V_PATTERN.items() if a <= step <= b]


def prescribed_increment(unn, step):
    """mm of prescribed movement for this tooth at this step (0 if not active)."""
    a, b = V_PATTERN[unn]
    if not (a <= step <= b):
        return 0.0
    return step_mm(unn)


def total_prescribed(unn):
    a, b = V_PATTERN[unn]
    return (b - a + 1) * step_mm(unn)


# ------------------------------------------------------- 4D iteration scheme --
# "each step of the CA was accompanied by two iterations of PDL to simulate the
#  clinical situation that the CA was well worn with enough time."
# Bone-remodelling loop of Hamanaka et al. (AJODO 2017;152:601):
#   phase 1 - outer PDL surface constrained, tooth moves elastically under load
#   phase 2 - tooth position retained, PDL restored to its original thickness
PDL_ITERATIONS_PER_STEP = 2

# "Prior to each iteration, the CA from the previous step was removed and
#  regenerated, followed by the application of a best-fit algorithm to match the
#  inner surface of the CA with the dental crowns, simulating a wear-in process."
WEAR_IN_BEST_FIT = True


def summary():
    lines = [f"Paper: {CITATION}", ""]
    lines.append(f"  half arch          FDI {[FDI[u] for u in HALF_ARCH_UNN]}")
    lines.append(f"  aligner            {ALIGNER_THICKNESS} mm, E={E_ALIGNER} MPa, nu={NU_ALIGNER}")
    lines.append(f"  attachments        {len(ATTACHMENTS)}, E={E_ATTACH} MPa")
    lines.append(f"  PDL                {PDL_THICKNESS} mm shell, bilinear "
                 f"E1={PDL_E1}/E2={PDL_E2} MPa at eps={PDL_EPS_KNEE}")
    lines.append(f"  teeth + bone       rigid (bone not meshed; outer PDL constrained)")
    lines.append(f"  contact            small-sliding surface-to-surface, mu={FRICTION}")
    lines.append(f"  elastic            {ELASTIC_FORCE_GF:.0f} gf = {ELASTIC_FORCE_N:.4f} N")
    lines.append(f"  staging            {N_STEPS} steps, V pattern, "
                 f"{STEP_DISTAL} mm distal / {STEP_PALATAL} mm palatal per step")
    lines.append(f"  iterations         {PDL_ITERATIONS_PER_STEP} PDL per CA step "
                 f"= {N_STEPS*PDL_ITERATIONS_PER_STEP} solves")
    lines.append("")
    lines.append("  V pattern (1-based step ranges) and total prescribed movement:")
    for u in HALF_ARCH_UNN:
        a, b = V_PATTERN[u]
        lines.append(f"    FDI {FDI[u]}   steps {a:3d}-{b:3d}   "
                     f"{total_prescribed(u):.2f} mm "
                     f"{'palatal' if u in (7, 8) else 'distal'}")
    return "\n".join(lines)


if __name__ == '__main__':
    print(summary())
