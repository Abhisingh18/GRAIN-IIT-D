"""
Spill Mask Workbench ka export text -> YOLO-seg .txt files.

Usage:
    python ingest_labels.py pasted.txt
Format:
    ### train_7__train_7_f00054
    0 0.12 0.34 0.18 0.40 ...
    ### train_6__train_6_f00054
    EMPTY
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KEY = ROOT / "dataset" / "keyframes"


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    text = Path(sys.argv[1]).read_text()

    cur, buf, n_img, n_poly, n_empty = None, [], 0, 0, 0

    def flush():
        nonlocal n_img, n_poly, n_empty
        if cur is None:
            return
        img = KEY / f"{cur}.jpg"
        if not img.exists():
            print(f"  ! unknown keyframe: {cur}"); return
        (KEY / f"{cur}.txt").write_text("\n".join(buf) + ("\n" if buf else ""))
        n_img += 1
        n_poly += len(buf)
        if not buf:
            n_empty += 1
        print(f"  {cur:34s} {len(buf) or 'EMPTY'}")

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("###"):
            flush()
            cur, buf = line[3:].strip(), []
        elif line.upper() == "EMPTY":
            pass
        elif line.startswith("0 "):
            parts = line.split()
            if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                buf.append(line)
            else:
                print(f"  ! bad polygon skipped ({len(parts)-1} coords)")
    flush()

    print(f"\n{n_img} keyframes written | {n_poly} polygons | {n_empty} empty")
    missing = [p.stem for p in sorted(KEY.glob("*.jpg"))
               if not (KEY / f"{p.stem}.txt").exists()]
    if missing:
        print(f"still missing ({len(missing)}): {', '.join(m.split('__')[0] for m in missing)}")
    else:
        print("Sab 18 keyframes ready. Ab: propagate_masks.py")


if __name__ == "__main__":
    main()
