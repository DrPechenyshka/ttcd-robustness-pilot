"""
Диагностика уровня High с расчётом IoU.
Воспроизводит искажения точно как в evaluate_stratified.py (rng=42 на каждый кадр).
Считает TP/FP/FN по порогу IoU=0.5.
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def apply_high_corruption(img, rng):
    """Идентично apply_corruption(severity=2) в evaluate_stratified.py."""
    h, w = img.shape[:2]
    img = cv2.GaussianBlur(img, (9, 9), 3.0)
    for _ in range(rng.integers(2, 5)):
        pts = rng.integers(0, min(h, w), size=(rng.integers(5, 10), 2))
        hull = cv2.convexHull(pts)
        color = (int(rng.integers(30, 100)), int(rng.integers(30, 100)), int(rng.integers(30, 100)))
        cv2.fillPoly(img, [hull], color)
    max_offset = int(min(w, h) * 0.1)
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_pts = src_pts + rng.integers(-max_offset, max_offset, size=src_pts.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img = cv2.warpPerspective(img, M, (w, h))
    return img, M


def transform_bbox(bbox, M, img_w, img_h):
    x, y, bw, bh = bbox
    pts = np.float32([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]).reshape(-1, 1, 2)
    pts_aug = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
    x_new, y_new = pts_aug.min(axis=0)
    x2_new, y2_new = pts_aug.max(axis=0)
    x_new, y_new = max(0, x_new), max(0, y_new)
    x2_new, y2_new = min(img_w, x2_new), min(img_h, y2_new)
    return (x_new, y_new, x2_new - x_new, y2_new - y_new)


def iou(a, b):
    ax1, ay1, aw, ah = a; bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def main():
    model = YOLO("results/pilot/baseline/weights/best.pt")
    test_images_dir = Path("data/processed/gtsdb_yolo/test/images")
    test_labels_dir = Path("data/processed/gtsdb_yolo/test/labels")
    out_dir = Path("results/pilot/diagnostics/high_iou")
    out_dir.mkdir(parents=True, exist_ok=True)

    cat_names = ["prohibitory", "danger", "mandatory", "other"]
    tp = fp = fn = 0
    tp_by_class = {c: 0 for c in cat_names}
    fp_by_class = {c: 0 for c in cat_names}
    fn_by_class = {c: 0 for c in cat_names}

    # ВАЖНО: тот же rng=42 на каждый кадр, как в оценке
    for img_path in sorted(test_images_dir.glob("*.jpg")):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        label_path = test_labels_dir / (img_path.stem + ".txt")
        gt = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    p = line.strip().split()
                    if len(p) != 5: continue
                    cls = int(p[0]); cx, cy, bw, bh = map(float, p[1:])
                    gt.append((cls, ((cx - bw/2)*w, (cy - bh/2)*h, bw*w, bh*h)))

        rng = np.random.default_rng(42)
        img_c, M = apply_high_corruption(img.copy(), rng)

        # Трансформируем GT
        gt_t = [(cls, transform_bbox(b, M, w, h)) for cls, b in gt]
        gt_t = [(cls, b) for cls, b in gt_t if b[2] > 1 and b[3] > 1]

        res = model(img_c, imgsz=640, conf=0.5, verbose=False)[0]
        dets = [(int(d.cls[0]), tuple(d.xyxy[0].tolist())) for d in res.boxes]
        # xyxy -> xywh
        dets = [(c, (x1, y1, x2-x1, y2-y1)) for c, (x1, y1, x2, y2) in dets]

        # Жадное сопоставление по IoU
        used = set()
        for dc, db in dets:
            best, bi = 0.5, -1
            for i, (gc, gb) in enumerate(gt_t):
                if i in used: continue
                v = iou(db, gb)
                if v >= best:
                    best, bi = v, i
            if bi >= 0:
                used.add(bi); tp += 1
                tp_by_class[cat_names[dc]] += 1
            else:
                fp += 1
                fp_by_class[cat_names[dc]] += 1
        fn += len(gt_t) - len(used)
        for i, (gc, _) in enumerate(gt_t):
            if i not in used:
                fn_by_class[cat_names[gc]] += 1

        # Визуализация (первые 12 кадров)
        if int(img_path.stem) < 12:
            viz = img_c.copy()
            for gc, gb in gt_t:
                x, y, bw, bh = [int(v) for v in gb]
                cv2.rectangle(viz, (x, y), (x+bw, y+bh), (0,255,0), 2)
            for dc, db in dets:
                x, y, bw, bh = [int(v) for v in db]
                cv2.rectangle(viz, (x, y), (x+bw, y+bh), (0,0,255), 2)
            cv2.imwrite(str(out_dir / f"{img_path.stem}_iou.jpg"), viz)

    print("="*60)
    print("РЕЗУЛЬТАТЫ ДИАГНОСТИКИ С IoU (уровень High)")
    print("="*60)
    print(f"TP={tp}  FP={fp}  FN={fn}")
    prec = tp/(tp+fp) if tp+fp else 0
    rec = tp/(tp+fn) if tp+fn else 0
    print(f"Precision={prec:.4f}  Recall={rec:.4f}")
    print(f"\nПо классам:")
    for c in cat_names:
        print(f"  {c:12s}  TP={tp_by_class[c]:3d}  FP={fp_by_class[c]:3d}  FN={fn_by_class[c]:3d}")
    print(f"\nВизуализации: {out_dir}")
    print("Зелёный=GT (трансформированный), Красный=детекция модели.")


if __name__ == "__main__":
    main()