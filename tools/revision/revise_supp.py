"""Revise Additional file 2 (tracked, or clean with --clean)."""
import sys, os, json, numpy as np
from lxml import etree
HERE = os.path.dirname(os.path.abspath(__file__)); SP = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(SP, 'ana'))
from trk import Doc, q
from stats import series, at_matched, compare, holm, PAT, DATA
from scipy import stats as st

d = Doc(sys.argv[1], track='--clean' not in sys.argv)
F = d.find
def cap(m): return d.new_par(m, 'Caption', ppr_extra=[etree.Element(q('keepNext'))])
def note(m): return d.new_par(m, 'TableLegend')
def body(m): return d.new_par(m, 'BodyText')
def h1(m): return d.new_par(m, 'Heading1')
def h2(m): return d.new_par(m, 'Heading2')
tmpl = d.block([p for p in d.paras() if p.getparent().tag == q('tc')][0])
while tmpl.tag != q('tbl'): tmpl = tmpl.getparent()
mm = lambda x: f'{x:+.2f}'.replace('-', '−')

# contents
c7 = [p for p in d.paras() if d.text(p) == 'S7. Supplementary discussion'][0]
d.insert_after(c7, [d.new_par('S8. Analyses added in revision (Tables S10–S14)', d.block(c7).find(q('pPr')).find(q('pStyle')).get(q('val')))])
# S5.1
p = F('Two sensitivity analyses were run on all five patients and are reported in the main text')
d.replace(p, d.text(p).replace('Two sensitivity analyses were run on all five patients and are reported in the main text: a linear PDL (0.68 MPa) in place of the bilinear PDL (Table 4 of the main text), and',
    'Three sensitivity analyses were run on all five patients and are reported in the main text: a linear PDL (0.68 MPa) in place of the bilinear PDL for all five groups, including the aligner-only control (Table 5 of the main text); three alternative headgear force systems (section S8.4, Tables S13 and S14); and'))
# S7
p = F('Transverse effect of the headgear.')
d.replace(p, '**Transverse effect of the headgear.** The headgear force was applied at the headgear tube, buccal to the centre of resistance of the first molar, along a line fixed in space. Such a force rotates the molar mesial-out, as also reported by Kang et al. [5], which moves the distobuccal cusp palatally. When the line of action was passed through the centroid of the first molar, which represents an outer bow adjusted for bodily movement, the palatal displacement of the distobuccal cusp at the end of the protocol fell from 4.71 to 1.04 mm and canine extrusion from 2.09 to 0.93 mm (Table S13). The large transverse effect of the main model is therefore mainly the effect of the moment of a force at the tube, which a clinician would correct by adjusting the outer bow and expanding the inner bow.')
p = F('Time scale.')
d.replace(p, '**Time scale.** One remodelling iteration is not calibrated to clinical time. The headgear reached 0.8 mm of molar distalization at the equivalent of step 4.7 (about nine iterations), whereas the aligner groups needed about 19 steps, so the end-of-protocol comparison corresponds to a much longer effective treatment with headgear. The comparison at equal distalization was therefore the primary analysis, and the per-millimetre normalization is reported only descriptively. The equal-distalization point falls before the anterior teeth are activated in the aligner plan, so its anterior values reflect anchorage loss rather than planned retraction, and its sensitivity to the threshold is given in section S8.2.')
p = F('Dependence on the PDL model.')
d.replace(p, d.text(p).replace('Dependence on the PDL model. ', '**Dependence on the PDL model.** ').replace(
    'The palatal displacement of the molar distobuccal cusp and the canine extrusion with headgear, the anchorage reinforcement by the elastics and the equivalence of the hooks did not depend on the PDL model.',
    'The equivalence of the hooks and the direction of the anchorage reinforcement by the elastics were the same with both PDL models, but the size of the anchorage reinforcement was about ten times smaller with the linear PDL, in which the aligner-only control was also run (section S8.3). The palatal displacement of the distobuccal cusp and the canine extrusion with headgear occurred with both PDL models, but their size depended on how the headgear force was represented (section S8.4).'))
p = F('Equivalence of the hooks.')
d.replace(p, d.text(p).replace('Equivalence of the hooks. ', '**Equivalence of the hooks.** ').replace(
    'Fracture or fatigue of the arm, wear of the aligner and loss of aligner retention under elastic loading were outside the scope of the model and may still differ clinically.',
    'The equivalence holds for an aligner that fits perfectly, is linear-elastic without stress relaxation and is never displaced. Fracture or fatigue of the arm, wear of the aligner, lift-off of the aligner at the canine and loss of aligner retention under elastic loading were outside the scope of the model and may well differ clinically, so the result does not show that the two hooks are clinically interchangeable.'))

# ---------------------------------------------------------------- S8
P = json.load(open(os.path.join(SP, 'ana', 'prescription.json')))
rows = [['Patient', 'Tooth', 'Steps', 'Planned movement (mm)', 'Distal (mm)', 'Transverse (mm)', 'Vertical (mm)', 'MD tip (°)', 'BL tip (°)']]
for pt, rr in P.items():
    for f, r in rr.items():
        rows.append([pt, f, f"{r['steps'][0]}–{r['steps'][1]}", f"{r['total']:.2f} {'palatal' if f in ('11', '12') else 'distal'}",
                     mm(r['distal']), mm(r['buccal']), mm(r['intrusion']), f"{r['tip_md']:+.1f}".replace('-', '−'), f"{r['tip_bl']:+.1f}".replace('-', '−')])
G4 = ['hg', 'ca_precision_cut', 'ca_aligner_hook', 'ca_metal_hook']
AB = {'hg': 'HG', 'ca_precision_cut': 'PC', 'ca_aligner_hook': 'CAH', 'ca_metal_hook': 'CMH'}
SER = {g: {P_: series(P_, f'{DATA}/{P_}/run_{g}.pkl') for P_ in PAT} for g in G4 + ['ca_control', 'pc_pdl068', 'cah_pdl068', 'cmh_pdl068', 'ctl_pdl068', 'hg_izc', 'hg_cres', 'hg_level']}
def val(g, m, where, th=0.8):
    return np.array([SER[g][P_][m][-1] if where == 'final' else at_matched(SER[g][P_], th)[1][m] for P_ in PAT])
LAB = [('16 apex AP', 'First molar MB root apex, sagittal'), ('16 tipping', 'First molar distal crown tipping'), ('16 V', 'First molar mesial pit, vertical'),
       ('16 BP', 'First molar DB cusp, transverse'), ('13 crown AP', 'Canine cusp, sagittal'), ('13 V', 'Canine cusp, vertical'),
       ('11 crown BP', 'Incisor incisal edge, sagittal'), ('11 apex BP', 'Incisor apex, sagittal'), ('11 V', 'Incisor incisal edge, vertical')]
TH = [0.4, 0.6, 0.8, 1.0, 1.2]
rth = [['Outcome', 'Contrast'] + [f'{t} mm' for t in TH]]
for m, lab in LAB:
    for a, b in [('hg', 'ca_precision_cut'), ('hg', 'ca_aligner_hook'), ('ca_precision_cut', 'ca_aligner_hook')]:
        cells = []
        for t in TH:
            fr, pr = compare({g: val(g, m, 'm', t) for g in G4}, G4)
            x = [r for r in pr if r[0] == a and r[1] == b][0]
            cells.append(f'{mm(x[2])} ({max(x[4], 5 - x[4])}/5){"*" if x[3] < 0.05 else ""}')
        rth.append([lab, f'{AB[a]} − {AB[b]}'] + cells)
rct = [['Outcome', 'PDL, comparison point', 'PC − control', 'CAH − control', 'CMH − control']]
for m, lab in [('16 crown AP', 'First molar MB cusp, sagittal')] + LAB:
    for pdl, grp, ctl in [('bilinear', ['ca_precision_cut', 'ca_aligner_hook', 'ca_metal_hook'], 'ca_control'), ('linear', ['pc_pdl068', 'cah_pdl068', 'cmh_pdl068'], 'ctl_pdl068')]:
        for where in (['m', 'final'] if pdl == 'bilinear' else ['final']):
            if where == 'm' and m == '16 crown AP': continue
            c0 = val(ctl, m, where); ps = holm([st.ttest_rel(val(g, m, where), c0).pvalue for g in grp])
            cells = [f'{mm((val(g, m, where) - c0).mean())} ({int((val(g, m, where) > c0).sum())}/5; p = {ps[i]:.3f})' for i, g in enumerate(grp)]
            rct.append([lab, f"{pdl}, {'0.8 mm' if where == 'm' else 'end'}"] + cells)
rhv = [['Outcome', 'Elastic direction, 0.8 mm', 'Elastic direction, end', 'Through CRes, 0.8 mm', 'Through CRes, end', 'Levelled wire, 0.8 mm', 'Levelled wire, end']]
for m, lab in [('16 crown AP', 'First molar MB cusp, sagittal')] + LAB:
    cells = []
    for g in ['hg_izc', 'hg_cres', 'hg_level']:
        for where in ['m', 'final']:
            if where == 'm' and m == '16 crown AP': cells.append('—'); continue
            ref = np.sign(np.mean(val('hg', m, where) - val('ca_precision_cut', m, where)))
            dd = val(g, m, where) - val('ca_precision_cut', m, where)
            cells.append(f'{int((np.sign(dd) == ref).sum())}/5 ({mm(dd.mean())})')
    rhv.append([lab] + cells)

def msd(a): return f'{np.mean(a):.2f} ± {np.std(a, ddof=1):.2f}'.replace('-', '−')
GH = ['hg', 'hg_izc', 'hg_cres', 'hg_level', 'ca_precision_cut']
rhm = [['Outcome (mm)', 'HG, main model', 'HG, elastic direction', 'HG, through CRes', 'HG, levelled wire', 'PC'], ['**Equal first-molar distalization (0.8 mm)**', '', '', '', '', '']]
rhm += [[lab] + [msd(val(g, m, 'm')) for g in GH] for m, lab in LAB]
rhm += [['**End of the protocol**', '', '', '', '', '']]
rhm += [[lab] + [msd(val(g, m, 'final')) for g in GH] for m, lab in [('16 crown AP', 'First molar MB cusp, sagittal')] + LAB]
ref_h = [p for p in d.paras() if d.text(p) == 'References'][0]
new = [h1('S8. Analyses added in revision'),
       h2('S8.1 Planned tooth movement for each patient'),
       body('Table S10 gives the planned movement of every tooth at aligner step 70 for each patient: the crown-centre displacement and its components in the tooth\'s own frame, and the change in crown angulation produced by the 0.40 tipping blend. Table 2 of the main text gives the range across patients. No overcorrection was programmed.'),
       cap('**Table S10 **Planned tooth movement at aligner step 70 for each patient'),
       d.make_table(tmpl, rows, [850, 650, 800, 1600, 900, 1100, 950, 850, 850]),
       note('Distal and transverse components in the tooth\'s own frame (transverse: buccal positive, palatal positive for the incisors); vertical, intrusion positive. MD tip, mesiodistal crown tip (distal positive); BL tip, buccolingual crown tip (buccal positive).'),
       h2('S8.2 Sensitivity of the equal-distalization comparison to the threshold'),
       body('The primary comparison was repeated with the first-molar mesiobuccal cusp distalized by 0.4, 0.6, 0.8, 1.0 and 1.2 mm (Table S11); every group reached at least 1.27 mm in every patient. Fig. 4 of the main text shows the same comparison along the whole distalization path. Differences in molar extrusion and palatal displacement of the distobuccal cusp with headgear, the more distal canine position with headgear, and all differences between the precision cut and the aligner-material hook kept their sign at every threshold. Molar tipping and root-apex movement with headgear, and the canine vertical and incisor sagittal differences between headgear and aligners, depended on the threshold.'),
       cap('**Table S11 **Between-group differences at five first-molar distalization thresholds (mm)'),
       d.make_table(tmpl, rth, [2300, 1000] + [1145]*5),
       note('Mean difference (A − B) and the number of patients with the same sign as the mean; *, Holm-adjusted p < 0.05 (six pairwise tests among HG, PC, CAH and CMH). The metal hook is omitted because it differed from the aligner-material hook by 0.01 mm or less. Sign conventions as in Table 3 of the main text.'),
       h2('S8.3 Comparison of the elastic designs with the aligner-only control'),
       body('The anchorage effect of each elastic design was tested against the aligner-only control with paired t-tests, Holm-corrected over the three contrasts (Table S12). With the bilinear PDL, every elastic design moved the canine and incisor in the anchorage-preserving direction in all five patients. With the linear PDL, in which the control was run for the revision, the direction was the same in 13 of 15 patient-level comparisons for the canine and in every patient for the incisor, but the size of the effect was about one tenth.'),
       cap('**Table S12 **Elastic design minus aligner-only control (mm)'),
       d.make_table(tmpl, rct, [2300, 1600, 1720, 1720, 1720]),
       note('Mean difference, number of patients in which the elastic design exceeded the control, and Holm-adjusted p-value. Sign conventions as in Table 3 of the main text.'),
       h2('S8.4 Headgear force systems'),
       body('Table S13 gives the displacements with the three alternative headgear force systems, and Table S14 shows whether the difference between headgear and the precision cut kept the sign of the main model under each of them. The elastic-direction variant pointed the headgear force at the tube along the elastic of each patient (patient 8, 25.0°; 3, 26.3°; 4, 19.7°; 5, 20.4°; 6, 24.5° above the occlusal plane). The centre-of-resistance variant kept the cervical direction (15° below the occlusal plane) with the line of action through the centroid of the first molar. The levelled-wire variant seated every bracket and tube on one flat passive archwire, which on patient 8 required the second-molar tube to be turned 42° from its facial-axis orientation, so it is an idealized representation of a levelled arch.'),
       cap('**Table S13 **Tooth displacement with alternative headgear force systems (mean ± SD, n = 5)'),
       d.make_table(tmpl, rhm, [2526, 1300, 1300, 1300, 1300, 1300]),
       note('HG, main model: cervical force 15° below the occlusal plane at the headgear tube of 16, archwire passive in brackets placed on the unaligned teeth. Elastic direction: the same force at the tube along the elastic direction of each patient. Through CRes: 15° below the occlusal plane with the line of action through the centroid of 16, used as its centre of resistance (CRes; no moment). Levelled wire: main force system with every bracket and tube seated on one flat passive archwire. PC, aligner with precision cut, shown for reference. Sign conventions as in Table 3 of the main text.'),
       cap('**Table S14 **Consistency of the headgear − precision cut difference under alternative headgear force systems'),
       d.make_table(tmpl, rhv, [2300] + [1120]*6),
       note('Number of patients in which the headgear − precision cut difference had the same sign as in the main headgear model, and the mean difference (mm), at equal first-molar distalization (0.8 mm) and at the end of the protocol.')]
d.insert_after(ref_h.getprevious(), new)
d.save(); print('ok')
