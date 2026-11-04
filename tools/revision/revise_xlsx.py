import sys, os, json, copy, numpy as np, openpyxl
from scipy import stats as st
HERE = os.path.dirname(os.path.abspath(__file__)); SP = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SP, 'ana'))
from stats import series, at_matched, compare, holm, PAT, DATA
src, out = sys.argv[1], sys.argv[2]
wb = openpyxl.load_workbook(src); T = wb['End of protocol']
STY = {k: T[c] for k, c in dict(title='A1', sub='A2', outcome='A4', head='A5', pat='A6', num='B6', mean='A11', meanf='B11', fr='A13', pw='A14', pwrow='A15', pwnum='B15').items()}
def put(ws, r, c, v, k):
    cell = ws.cell(r, c, v); s = STY[k]
    cell.font = copy.copy(s.font); cell.fill = copy.copy(s.fill); cell.border = copy.copy(s.border)
    cell.alignment = copy.copy(s.alignment); cell.number_format = s.number_format
SER = {}
def ser(g):
    if g not in SER: SER[g] = {P: series(P, f'{DATA}/{P}/run_{g}.pkl') for P in PAT}
    return SER[g]
def val(g, m, where, th=0.8):
    return np.array([ser(g)[P][m][-1] if where == 'final' else at_matched(ser(g)[P], th)[1][m] for P in PAT])
LAB = [('16 crown AP', 'First molar MB cusp, sagittal (distal +)'), ('16 apex AP', 'First molar MB root apex, sagittal (distal +)'),
       ('16 tipping', 'First molar distal crown tipping, cusp − apex (mm)'), ('16 V', 'First molar mesial pit, vertical (intrusion +)'),
       ('16 BP', 'First molar DB cusp, transverse (buccal +)'), ('13 crown AP', 'Canine cusp, sagittal (distal +)'),
       ('13 V', 'Canine cusp, vertical (intrusion +)'), ('11 crown BP', 'Central incisor incisal edge, sagittal (palatal +)'),
       ('11 apex BP', 'Central incisor apex, sagittal (palatal +)'), ('11 V', 'Central incisor incisal edge, vertical (intrusion +)')]
NAMES = ['P8', 'P3', 'P4', 'P5', 'P6']
def block_sheet(name, title, sub, groups, heads, where, th=0.8, tests=True, ctl=None):
    if name in wb.sheetnames: del wb[name]
    ws = wb.create_sheet(name); ws.column_dimensions['A'].width = 58
    for c in 'BCDEFG': ws.column_dimensions[c].width = 18
    put(ws, 1, 1, title, 'title'); put(ws, 2, 1, sub, 'sub'); r = 4
    for m, lab in LAB:
        if where != 'final' and m == '16 crown AP': continue
        put(ws, r, 1, lab, 'outcome'); r += 1
        put(ws, r, 1, 'Patient', 'head')
        for j, h in enumerate(heads): put(ws, r, 2 + j, h, 'head')
        r += 1; r0 = r
        V = {g: val(g, m, where, th) for g in groups}
        for i, P in enumerate(NAMES):
            put(ws, r, 1, P, 'pat')
            for j, g in enumerate(groups): put(ws, r, 2 + j, round(float(V[g][i]), 3), 'num')
            r += 1
        for lab2, fn in (('Mean', 'AVERAGE'), ('SD', 'STDEV')):
            put(ws, r, 1, lab2, 'mean')
            for j in range(len(groups)):
                col = openpyxl.utils.get_column_letter(2 + j); put(ws, r, 2 + j, f'={fn}({col}{r0}:{col}{r0 + 4})', 'meanf')
            r += 1
        if tests:
            act = groups[:4]; fr, pr = compare({g: V[g] for g in act}, act)
            put(ws, r, 1, f'Friedman ({", ".join(heads[:4])}): chi-square = {fr.statistic:.2f}, p = {fr.pvalue:.4f}', 'fr'); r += 1
            put(ws, r, 1, 'Pairwise', 'pw'); put(ws, r, 2, 'Mean difference (mm)', 'pw'); put(ws, r, 3, 'Holm-adjusted p', 'pw'); put(ws, r, 4, 'Patients with A > B', 'pw'); r += 1
            H = dict(zip(groups, heads))
            rows = [(f'{H[a]} − {H[b]}', md, p, n) for a, b, md, p, n in pr]
            if ctl:
                ps = holm([st.ttest_rel(V[g], V[ctl]).pvalue for g in act[1:]])
                rows += [(f'{H[g]} − Control', float((V[g] - V[ctl]).mean()), ps[i], int((V[g] > V[ctl]).sum())) for i, g in enumerate(act[1:])]
            for lab3, md, p, n in rows:
                put(ws, r, 1, lab3, 'pwrow'); put(ws, r, 2, round(float(md), 3), 'pwnum'); put(ws, r, 3, round(float(p), 3), 'pwnum'); put(ws, r, 4, f'{n}/5', 'pwnum'); r += 1
        r += 1
    return ws
LIN = ['hg_pdl068', 'pc_pdl068', 'cah_pdl068', 'cmh_pdl068', 'ctl_pdl068']; H5 = ['HG', 'PC', 'CAH', 'CMH', 'Control']
MAIN = ['hg', 'ca_precision_cut', 'ca_aligner_hook', 'ca_metal_hook', 'ca_control']
NOTE = 'Holm correction over the six pairs of HG, PC, CAH and CMH, and separately over the three elastic-versus-control contrasts. Mean and SD rows are formulas.'
block_sheet('Linear PDL - end of protocol', 'Additional file 1. Linear PDL - end of protocol: displacement (mm) per patient and group', 'Sensitivity analysis with a linear PDL (0.68 MPa); the aligner-only control was added in revision. ' + NOTE, LIN, H5, 'final', ctl='ctl_pdl068')
block_sheet('Linear PDL - equal distalizatio', 'Additional file 1. Linear PDL - equal distalization (0.5 mm): displacement (mm) per patient and group', 'Sensitivity analysis with a linear PDL (0.68 MPa) at 0.5 mm of first-molar distalization (headgear reaches only 0.68 mm in P5); the aligner-only control was added in revision. ' + NOTE, LIN, H5, 'm', th=0.5, ctl='ctl_pdl068')
block_sheet('Control tests - equal 0.8', 'Additional file 1. Elastic designs versus aligner-only control, bilinear PDL, equal molar distalization (0.8 mm)', NOTE, MAIN, H5, 'm', ctl='ca_control')
block_sheet('Control tests - end', 'Additional file 1. Elastic designs versus aligner-only control, bilinear PDL, end of protocol', NOTE, MAIN, H5, 'final', ctl='ca_control')
HV = ['hg', 'hg_izc', 'hg_cres', 'hg_level', 'ca_precision_cut']; HH = ['HG main', 'HG elastic direction', 'HG through CRes', 'HG levelled wire', 'PC']
hvsub = 'Headgear force systems added in revision (descriptive). Elastic direction: force at the tube along the IZC elastic (P8 25.0, P3 26.3, P4 19.7, P5 20.4, P6 24.5 deg above the occlusal plane); through CRes: 15 deg below the occlusal plane through the centroid of 16; levelled wire: slots seated on one flat passive archwire. PC for reference.'
block_sheet('HG variants - equal 0.8', 'Additional file 1. Headgear force systems - equal molar distalization (0.8 mm)', hvsub, HV, HH, 'm', tests=False)
block_sheet('HG variants - end', 'Additional file 1. Headgear force systems - end of protocol', hvsub, HV, HH, 'final', tests=False)
for th in (0.4, 0.6, 1.0, 1.2):
    block_sheet(f'Equal distalization {th}', f'Additional file 1. Equal molar distalization ({th} mm): displacement (mm) per patient and group', f'Threshold sensitivity of the primary comparison (main analysis is 0.8 mm). ' + NOTE, MAIN, H5, 'm', th=th, ctl='ca_control')
# planned movement
P = json.load(open(os.path.join(SP, 'ana', 'prescription.json')))
ws = wb.create_sheet('Planned movement'); ws.column_dimensions['A'].width = 12
put(ws, 1, 1, 'Additional file 1. Planned tooth movement at aligner step 70', 'title')
put(ws, 2, 1, 'Crown-centre displacement and crown angulation change (0.40 tipping blend). Components in the tooth frame; transverse buccal + (incisors palatal +); vertical intrusion +; MD tip distal +; BL tip buccal +. No overcorrection.', 'sub')
hd = ['Patient', 'Tooth', 'First step', 'Last step', 'Planned movement (mm)', 'Distal (mm)', 'Transverse (mm)', 'Vertical (mm)', 'MD tip (deg)', 'BL tip (deg)']
for j, h in enumerate(hd): put(ws, 4, 1 + j, h, 'head'); ws.column_dimensions[openpyxl.utils.get_column_letter(1 + j)].width = 16
r = 5
for pt, rr in P.items():
    for f, x in rr.items():
        for j, v in enumerate([pt, int(f), x['steps'][0], x['steps'][1], round(x['total'], 3), round(x['distal'], 3), round(x['buccal'], 3), round(x['intrusion'], 3), round(x['tip_md'], 2), round(x['tip_bl'], 2)]):
            put(ws, r, 1 + j, v, 'pat' if j == 0 else 'num')
        r += 1
wb.save(out); print(wb.sheetnames)
