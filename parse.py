import re
import os
import sys
import glob
import json

input_folder = sys.argv[1]
outfile = sys.argv[2]
extension = 'x86.s' 

def extract(inp):
    sections = {}
    cloze = ""
    in_chunk = False
    in_text_section = False
    program = open(inp, 'r')
    for l in program:
        if '.section' in l:
            if '__TEXT,__text' in l:
                in_text_section = True
            else:
                in_text_section = False
        l2 = l.strip()
        label = l2.split()[0] if l2 else ""
        if label.startswith("_") and label.endswith(":") and in_text_section:
            in_chunk = True
            section = label[:-1]
            cloze += f"{{{section}}}"
            sections[section] = ""
        if in_chunk:
            sections[section] += l
            if l2.strip() in {".cfi_endproc", "jr	ra"}:
                in_chunk = False
        else:
            cloze += l
    return sections, cloze 


out = open(outfile, "w")
for f in glob.glob(os.path.join(input_folder, f"*.{extension}")):
    print(f)
    f1 = f
    f2 = f.split(f".{extension}")[0] + ".arm.s"
    if not os.path.exists(f1): continue
    if not os.path.exists(f2): continue
    d1, x86_cloze = extract(f1) #, [remove_extra_space])
    d2, arm_cloze = extract(f2) #, [remove_extra_space])

    if len(d1) != len(d2):
        print("fail", f)
        print("fail", d1)
        print("fail", d2)
        continue
    fname = os.path.basename(f)
    d = {
            "source": fname,
            "x86": open(f1).read(),
            "x86_fns": d1,
            "x86_cloze": x86_cloze,
            "arm": open(f2).read(),
            "arm_fns" : d2,
            "arm_cloze": arm_cloze
    }
    print(json.dumps(d), file=out)