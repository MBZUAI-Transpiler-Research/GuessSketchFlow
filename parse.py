import re
import os
import sys
import glob

input_folder = sys.argv[1]
outfile = sys.argv[2]
extension = 'x86.s'

def canonical_label(label: str) -> str:
    if match := re.match(r"(?:%bb\.|LBB\d_)(\d+)", label):
        return f"bb{match.group(1)}"
    if label.startswith("LCPI"):  # likely a constant pool or constant data
        return None
    return label

def extract(inp, file_modifiers=[lambda x: x]):
    sections = {}
    cloze = ""
    in_chunk = False
    section = None
    program = open(inp, 'r')

    for l in program:
        l2 = l.strip()

        # Real label (e.g., LBB0_2:)
        if l2 and l2[0] not in {'.'} and '.' not in l2 and l2.endswith(":"):
            raw_label = l2[:-1]
            canonical = canonical_label(raw_label)
            if canonical is None:
                in_chunk = False
                section = None
                continue
            section = canonical
            in_chunk = True
            cloze += f"{{{section}}}"
            sections[section] = ""

        # x86-style comment block (e.g., ## %bb.3:)
        elif l2.startswith("## %bb.") and l2.endswith(":"):
            bb_match = re.match(r"## (%bb\.\d+):", l2)
            if bb_match:
                raw_label = bb_match.group(1)
                canonical = canonical_label(raw_label)
                if canonical is None:
                    in_chunk = False
                    section = None
                    continue
                section = canonical
                in_chunk = True
                cloze += f"{{{section}}}"
                sections[section] = ""

        # ARM-style comment block (e.g., ; %bb.3:)
        elif l2.startswith("; %bb.") and l2.endswith(":"):
            bb_match = re.match(r"; (%bb\.\d+):", l2)
            if bb_match:
                raw_label = bb_match.group(1)
                canonical = canonical_label(raw_label)
                if canonical is None:
                    in_chunk = False
                    section = None
                    continue
                section = canonical
                in_chunk = True
                cloze += f"{{{section}}}"
                sections[section] = ""

        if in_chunk and section:
            sections[section] += l
        else:
            cloze += l

    return sections, cloze

def remove_comments(code, comment_prefix="//"):
    comment_regex = re.compile(f'{comment_prefix}.*\n')
    return re.sub(comment_regex, '', code)

def remove_extra_space(ass):
    return re.sub(r',\s+', ',', ass.replace('\n\t', '\n')).strip()

def remove_rawstrings(ass): 
    curr_idx = 0
    updated_prediction = ""
    while '.string' in ass[curr_idx:]:
        str_idx = re.search(r'\.string', ass[curr_idx:]).start() + curr_idx
        updated_prediction += ass[curr_idx:str_idx+len('.string')] + " RAW"
        old_line = ass[str_idx:ass.index('\n', str_idx)]

        curr_idx = str_idx + len(old_line)

    updated_prediction += ass[curr_idx:]
    return updated_prediction    

import json
out = open(outfile, "w")
for f in glob.glob(os.path.join(input_folder, f"*.{extension}")):
    #print(f)
    f1 = f
    f2 = f.replace(".x86.s", ".arm.s")
    if not os.path.exists(f1): continue
    if not os.path.exists(f2): continue
    d1, x86_cloze = extract(f1) #, [remove_extra_space])
    d2, arm_cloze = extract(f2) #, [remove_extra_space])

    #print("x86 keys:", d1.keys())
    #sprint("arm keys:", d2.keys())

    if len(d1) != len(d2):
        print("fail", f)
        print("fail", d1)
        print("fail", d2)
        continue
    fname = os.path.basename(f)
    with open(f1) as f: x86_code = f.read()
    with open(f2) as f: arm_code = f.read()
    d = {
            "source": fname,
            "x86": x86_code,
            "x86_fns": d1,
            "x86_cloze": x86_cloze,
            "arm": arm_code,
            "arm_fns" : d2,
            "arm_cloze": arm_cloze
    }
    print(json.dumps(d), file=out)