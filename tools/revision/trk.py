"""Minimal tracked-changes editor for simple WordprocessingML documents (one run = one formatting).

Markup for new text: **bold**, *italic*, ^superscript^.  TRACK=False produces a clean copy instead."""
import re, copy, difflib, os, shutil
from lxml import etree

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
NS = {'w': W, 'r': R}
def q(t): return '{%s}%s' % (W, t)
XS = '{http://www.w3.org/XML/1998/namespace}space'


class Doc:
    def __init__(self, unpacked, track=True, author='Authors (revision)', date='2026-11-04T00:00:00Z'):
        self.dir = unpacked
        self.path = os.path.join(unpacked, 'word', 'document.xml')
        self.tree = etree.parse(self.path)
        self.body = self.tree.getroot().find(q('body'))
        self.track, self.author, self.date = track, author, date
        self._id = 9000
        self.img_n = 100

    # ------------------------------------------------------------------ basics
    def nid(self):
        self._id += 1
        return str(self._id)

    def mark(self, tag):
        e = etree.Element(q(tag))
        e.set(q('id'), self.nid()); e.set(q('author'), self.author); e.set(q('date'), self.date)
        return e

    @staticmethod
    def text(p):
        return ''.join(t.text or '' for t in p.iter(q('t')))

    def paras(self):
        return [e for e in self.body.iter(q('p'))]

    def find(self, start, within=None, contains=None):
        hits = [p for p in (within if within is not None else self.body).iter(q('p'))
                if self.text(p).startswith(start) and (contains is None or contains in self.text(p))]
        if len(hits) != 1:
            raise KeyError(f'{len(hits)} paragraphs start with {start!r}')
        return hits[0]

    def block(self, p):
        """Top-level body child containing p."""
        while p.getparent() is not self.body:
            p = p.getparent()
        return p

    # ------------------------------------------------------------- markup
    @staticmethod
    def parse(markup):
        segs = []
        for tok in re.split(r'(\*\*.+?\*\*|\^.+?\^|(?<![\w*])\*[^*\s][^*]*?\*(?![\w*]))', markup):
            if not tok:
                continue
            if tok.startswith('**'):
                segs.append((tok[2:-2], 'b'))
            elif tok.startswith('^'):
                segs.append((tok[1:-1], 'sup'))
            elif tok.startswith('*') and tok.endswith('*') and len(tok) > 2:
                segs.append((tok[1:-1], 'i'))
            else:
                segs.append((tok, None))
        return segs

    @staticmethod
    def rpr_for(fmt, base=None):
        rpr = copy.deepcopy(base) if base is not None else etree.Element(q('rPr'))
        for c in list(rpr):
            if c.tag in (q('b'), q('i'), q('vertAlign'), q('ins'), q('del')):
                rpr.remove(c)
        if fmt == 'b':
            rpr.insert(0, etree.Element(q('b')))
        elif fmt == 'i':
            rpr.insert(0, etree.Element(q('i')))
        elif fmt == 'sup':
            v = etree.SubElement(rpr, q('vertAlign')); v.set(q('val'), 'superscript')
        return rpr

    def run(self, text, rpr, deleted=False):
        r = etree.Element(q('r'))
        if rpr is not None and len(rpr):
            r.append(copy.deepcopy(rpr))
        t = etree.SubElement(r, q('delText') if deleted else q('t'))
        t.text = text; t.set(XS, 'preserve')
        return r

    def wrap(self, kind, runs):
        if not self.track:
            return [] if kind == 'del' else runs
        w = self.mark(kind)
        for r in runs:
            if kind == 'del':
                for t in r.iter(q('t')):
                    t.tag = q('delText')
            w.append(r)
        return [w]

    # ------------------------------------------------------------- paragraphs
    def replace(self, p, markup, base=None):
        """Word-level tracked replacement of the paragraph's text; unchanged words keep their formatting."""
        runs = [r for r in p if r.tag == q('r') and r.find(q('t')) is not None]
        chars = []                                   # (char, rPr)
        for r in runs:
            rp = r.find(q('rPr'))
            for ch in ''.join(t.text or '' for t in r.iter(q('t'))):
                chars.append((ch, rp))
        if base is None and not runs:
            cand = [r for r in p.iter(q('r')) if r.getparent().tag != q('del') and r.find(q('t')) is not None]
            plain = [r.find(q('rPr')) for r in cand if r.find(q('rPr')) is None or r.find(q('rPr')).find(q('b')) is None]
            base = plain[-1] if plain else None
        if base is None:
            plain = [r.find(q('rPr')) for r in runs if r.find(q('rPr')) is None or r.find(q('rPr')).find(q('b')) is None]
            base = plain[-1] if plain else None
        new = []                                     # (char, fmt)
        for txt, fmt in self.parse(markup):
            new += [(c, fmt) for c in txt]
        tok = lambda s: re.findall(r'\s+|[^\s]+', s)
        old_s = ''.join(c for c, _ in chars); new_s = ''.join(c for c, _ in new)
        ot, nt = tok(old_s), tok(new_s)
        # char offsets of tokens
        def offs(ts):
            o, out = 0, []
            for t in ts:
                out.append(o); o += len(t)
            return out + [o]
        oo, no = offs(ot), offs(nt)
        out = []
        def emit_old(a, b, deleted):
            grp = []
            for i in range(a, b):
                ch, rp = chars[i]
                if grp and grp[-1][1] is rp:
                    grp[-1][0].append(ch)
                else:
                    grp.append(([ch], rp))
            rs = [self.run(''.join(c), rp) for c, rp in grp]
            return self.wrap('del', rs) if deleted else rs
        def emit_new(a, b):
            grp = []
            for i in range(a, b):
                ch, fmt = new[i]
                if grp and grp[-1][1] == fmt:
                    grp[-1][0].append(ch)
                else:
                    grp.append(([ch], fmt))
            return self.wrap('ins', [self.run(''.join(c), self.rpr_for(f, base)) for c, f in grp])
        sm = difflib.SequenceMatcher(a=ot, b=nt, autojunk=False)
        for op, a1, a2, b1, b2 in sm.get_opcodes():
            if op == 'equal':
                out += emit_old(oo[a1], oo[a2], False)
            else:
                if a2 > a1: out += emit_old(oo[a1], oo[a2], True)
                if b2 > b1: out += emit_new(no[b1], no[b2])
        for r in [c for c in p if c.tag in (q('r'), q('ins'), q('del'))]:
            p.remove(r)
        for e in out:
            p.append(e)
        return p

    def new_par(self, markup, style='BodyText', base=None, ppr_extra=None):
        p = etree.Element(q('p'))
        ppr = etree.SubElement(p, q('pPr'))
        st = etree.SubElement(ppr, q('pStyle')); st.set(q('val'), style)
        for e in (ppr_extra or []):
            ppr.append(copy.deepcopy(e))
        if self.track:
            rp = etree.SubElement(ppr, q('rPr')); rp.append(self.mark('ins'))
        for e in self.wrap('ins', [self.run(t, self.rpr_for(f, base)) for t, f in self.parse(markup)]):
            p.append(e)
        return p

    def insert_after(self, anchor, elems):
        anchor = self.block(anchor) if anchor.getparent() is not self.body else anchor
        for e in elems:
            anchor.addnext(e); anchor = e
        return anchor

    def mark_par_deleted(self, p):
        if not self.track:
            p.getparent().remove(p); return
        ppr = p.find(q('pPr'))
        if ppr is None:
            ppr = etree.Element(q('pPr')); p.insert(0, ppr)
        rp = ppr.find(q('rPr'))
        if rp is None:
            rp = etree.SubElement(ppr, q('rPr'))
        rp.insert(0, self.mark('del'))
        for r in [c for c in p if c.tag == q('r')]:
            idx = p.index(r); p.remove(r)
            for w in self.wrap('del', [r]):
                p.insert(idx, w)

    def mark_par_inserted(self, p):
        if not self.track:
            return p
        ppr = p.find(q('pPr'))
        if ppr is None:
            ppr = etree.Element(q('pPr')); p.insert(0, ppr)
        rp = ppr.find(q('rPr'))
        if rp is None:
            rp = etree.SubElement(ppr, q('rPr'))
        rp.insert(0, self.mark('ins'))
        for r in [c for c in p if c.tag == q('r')]:
            idx = p.index(r); p.remove(r)
            for w in self.wrap('ins', [r]):
                p.insert(idx, w)
        return p

    # ------------------------------------------------------------- tables
    def mark_tbl(self, tbl, kind):
        if not self.track and kind == 'del':
            tbl.getparent().remove(tbl); return
        if not self.track:
            return tbl
        for tr in tbl.iter(q('tr')):
            trpr = tr.find(q('trPr'))
            if trpr is None:
                trpr = etree.Element(q('trPr')); tr.insert(0, trpr)
            trpr.append(self.mark(kind))
            for p in tr.iter(q('p')):
                (self.mark_par_deleted if kind == 'del' else self.mark_par_inserted)(p)
        return tbl

    def make_table(self, template_tbl, rows, widths, header_rows=1, first_col_left=True):
        """Copy formatting from an existing table; rows = list of lists of markup strings."""
        tbl = etree.Element(q('tbl'))
        tp = copy.deepcopy(template_tbl.find(q('tblPr')))
        tp.find(q('tblW')).set(q('w'), str(sum(widths)))
        tbl.append(tp)
        g = etree.SubElement(tbl, q('tblGrid'))
        for wd in widths:
            gc = etree.SubElement(g, q('gridCol')); gc.set(q('w'), str(wd))
        last = len(rows) - 1
        for i, row in enumerate(rows):
            tr = etree.SubElement(tbl, q('tr'))
            if i < header_rows:
                trpr = etree.SubElement(tr, q('trPr')); etree.SubElement(trpr, q('tblHeader'))
            for j, cell in enumerate(row):
                tc = etree.SubElement(tr, q('tc'))
                tcpr = etree.SubElement(tc, q('tcPr'))
                tw = etree.SubElement(tcpr, q('tcW')); tw.set(q('w'), str(widths[j])); tw.set(q('type'), 'dxa')
                if i == header_rows - 1:
                    b = etree.SubElement(tcpr, q('tcBorders')); bb = etree.SubElement(b, q('bottom'))
                    bb.set(q('val'), 'single'); bb.set(q('sz'), '6'); bb.set(q('color'), '000000')
                p = etree.SubElement(tc, q('p'))
                ppr = etree.SubElement(p, q('pPr'))
                s = etree.SubElement(ppr, q('pStyle')); s.set(q('val'), 'TableText')
                jc = etree.SubElement(ppr, q('jc')); jc.set(q('val'), 'left' if (j == 0 and first_col_left) else 'center')
                base = etree.Element(q('rPr'))
                sz = etree.SubElement(base, q('sz')); sz.set(q('val'), '18')
                sc = etree.SubElement(base, q('szCs')); sc.set(q('val'), '18')
                mk = cell if not (i < header_rows and not cell.startswith('**') and cell) else f'**{cell}**'
                for t, f in self.parse(mk):
                    p.append(self.run(t, self.rpr_for(f, base)))
        return self.mark_tbl(tbl, 'ins')

    # ------------------------------------------------------------- figures
    def move(self, elem, after):
        """Tracked move: delete at the old place, insert a copy after `after`."""
        cp = copy.deepcopy(elem)
        if elem.tag == q('tbl'):
            self.mark_tbl(elem, 'del')
            self.insert_after(after, [cp]); self.mark_tbl(cp, 'ins')
        else:
            self.mark_par_deleted(elem)
            self.insert_after(after, [cp]); self.mark_par_inserted(cp)
        return cp

    def new_figure(self, template_fig_par, png, width_emu=5760000):
        from PIL import Image
        self.img_n += 1
        name = f'image{self.img_n}.png'
        shutil.copy(png, os.path.join(self.dir, 'word', 'media', name))
        rid = f'rIdImgRev{self.img_n}'
        rels = os.path.join(self.dir, 'word', '_rels', 'document.xml.rels')
        s = open(rels).read()
        s = s.replace('</Relationships>', f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/{name}"/></Relationships>')
        open(rels, 'w').write(s)
        wpx, hpx = Image.open(png).size
        h = int(width_emu*hpx/wpx)
        p = copy.deepcopy(template_fig_par)
        x = etree.tostring(p).decode()
        x = re.sub(r'r:embed="[^"]+"', f'r:embed="{rid}"', x)
        x = re.sub(r'cx="\d+" cy="\d+"', f'cx="{width_emu}" cy="{h}"', x)
        x = re.sub(r'(docPr id=")\d+(" name=")[^"]+', rf'\g<1>{self.img_n}\g<2>Figure {self.img_n}', x)
        x = re.sub(r'(cNvPr id=")\d+(" name=")[^"]+', rf'\g<1>{self.img_n}\g<2>{name}', x)
        p = etree.fromstring(x)
        return self.mark_par_inserted(p)

    def save(self):
        self.tree.write(self.path, xml_declaration=True, encoding='UTF-8', standalone=True)
