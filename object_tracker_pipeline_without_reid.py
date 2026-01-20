"""Детекция объектов + трекинг (ассоциация в стиле ByteTrack + Калман XYAH + опциональный ReID).

Внесённые изменения:
  1) Корректный Mahalanobis-гейтинг для df=4 (по умолчанию gate_thresh=9.49; отдельный lost_gate_thresh).
  2) Обновление Калмана для всех сматченных детекций; confidence детекции масштабирует шум измерения R.
  3) Подавление дублей треков (после обновления).
  4) Улучшения ReID: EMA-фича на трек + гейтинг по внешнему виду + стадия реактивации LOST.

Пайплайн по кадру:
  1) Детекция объектов
  2) Предсказание треков (Калман)
  3) Ассоциация ACTIVE-треков с детекциями (IoU + Mahalanobis + Hungarian; ByteTrack high/low)
  4) Реактивация LOST-треков (внешний вид + IoU + гейтинг) на оставшихся HIGH-детекциях
  5) Обновление / создание / завершение треков + подавление дублей

Компенсация движения камеры (CMC) не реализована — предполагается, что камера неподвижна.
"""

from __future__ import annotations

import argparse
import subprocess as sp
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional, Iterable, Tuple, List, Dict

import numpy as np

import cv2
import torch
from scipy.optimize import linear_sum_assignment


# У трекера всегда есть компромисс:
#   - “мягче” матчи => меньше обрывов (fragmentation), но больше ID-switch/мусора
#   - “жёстче” матчи => меньше ID-switch/мусора, но больше обрывов
#
# Обозначения ошибок:
#   FRAG  = fragmentation (обрыв трека/новый ID при том же объекте)
#   IDSW  = ID switch (перепрыгивание ID на другой объект)
#   FPTRK = ложные треки (мусорные треки/дубли)
#
# Также важно: часть проблем может быть не в трекере, а в детекторе (пропуски, дубли, джиттер).


@dataclass
class TrackingConfig:
    # -------------------------
    # Детекция (YOLO: инференс + NMS)
    # -------------------------

    det_conf_thr: float = 0.25
    # Что: порог confidence детекций, которые YOLO вернёт после NMS.
    # Влияние: сколько кандидатов попадёт дальше в трекер.
    # Симптомы:
    #   - FPTRK (много мусора/ложных рамок) => ↑ det_conf_thr
    #   - FRAG (детектор “проваливается”, нет детекций на объект) => ↓ det_conf_thr
    # Типично: 0.15–0.4 (зависит от модели/сцены).

    det_iou_thr: float = 0.4
    # Что: IoU-порог NMS в YOLO (насколько агрессивно подавлять дубли детекций).
    # Влияние: число “двойных” рамок на один объект.
    # Симптомы:
    #   - дубли детекций (2 бокса на 1 объект) => ↓ det_iou_thr (строже NMS)
    #   - близкие объекты “слипаются” (один подавляется) => ↑ det_iou_thr
    # Типично: 0.35–0.6. В статичной камере часто 0.35–0.45.

    imgsz: int = 960
    # Что: размер входа для YOLO.
    # Влияние: точность/скорость и устойчивость боксов (джиттер).
    # Симптомы:
    #   - мелкие объекты пропадают / боксы скачут => ↑ imgsz (если хватает GPU)
    #   - слишком медленно => ↓ imgsz
    # Типично: 640–1280 (в зависимости от модели и железа).

    # -------------------------
    # Политика обновления трека / фильтрация
    # -------------------------

    track_conf_thr: float = 0.85
    # Что: в текущей версии НЕ участвует в матчинге / KF update.
    # Сейчас используется только для визуализации/цвета детекции (и для дебага).
    # Если позже будет введено условное обновление KF — параметр станет значимым.
    # Сейчас: имеет смысл только для визуального контроля.

    min_box_height: int = 24
    # Что: минимальная высота бокса, чтобы считать детекцию “валидной” для трекинга.
    # Влияние: отсечка шума (обычно вдали/мелкие ложные).
    # Симптомы:
    #   - FPTRK на дальних/мелких объектах => ↑ min_box_height
    #   - теряются дальние объекты => ↓ min_box_height
    # Типично: 16–48 (зависит от разрешения видео).

    # -------------------------
    # ROI (геометрический фильтр)
    # -------------------------

    roi_rel: Tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    # Что: ROI в относительных координатах (x1,y1,x2,y2), детекции вне ROI игнорируются.
    # Влияние: снижает FPTRK, повышает стабильность в зоне интереса.
    # Симптомы:
    #   - FPTRK по краям/в неважной зоне => сужать ROI
    #   - объект выходит за ROI => FRAG (трек завершится) => расширять ROI
    # Типично: (0,0,1,1) если нужно всё, иначе “полоса дороги”, “центр сцены” и т.п.

    # -------------------------
    # ByteTrack split (HIGH/LOW детекции)
    # -------------------------

    thr_high: float = 0.5
    # Что: порог “высоких” детекций.
    # Влияние:
    #   - HIGH участвуют в основном матчинге и могут порождать новые треки (spawn)
    #   - также используются для реактивации LOST (берутся оставшиеся HIGH)
    # Симптомы:
    #   - FPTRK (слишком много новых треков от шума) => ↑ thr_high
    #   - FRAG (новый трек вместо продолжения, мало кандидатов) => ↓ thr_high
    # Типично: 0.4–0.7.

    thr_low: float = 0.1
    # Что: порог “низких” детекций.
    # Влияние:
    #   - LOW участвуют во второй стадии матчинга (поддержка существующих треков)
    #   - LOW обычно не порождают новые треки
    # Симптомы:
    #   - FRAG из-за кратких падений confidence => ↓ thr_low (больше слабых детекций поддержит трек)
    #   - FPTRK/дрожание от слабых детекций => ↑ thr_low
    # Типично: 0.05–0.2.

    # -------------------------
    # Пороги ассоциации (ACTIVE matching)
    # -------------------------

    iou_thresh_high: float = 0.2
    # Что: минимальный IoU для принятия match на HIGH стадии.
    # Влияние: насколько “геометрически” должен совпасть бокс.
    # Симптомы:
    #   - FRAG (матч отваливается, хотя объект тот же) => ↓ iou_thresh_high
    #   - IDSW (перепрыгивание на соседнего, особенно при пересечении) => ↑ iou_thresh_high
    # Типично: 0.15–0.35. Для “меньше обрывов” обычно 0.15–0.25.

    iou_thresh_low: float = 0.05
    # Что: минимальный IoU для LOW стадии (поддержка).
    # Влияние: даёт “мягкий” шанс связать трек со слабой детекцией.
    # Симптомы:
    #   - FRAG при кратких просадках детектора => ↓ iou_thresh_low (но осторожно)
    #   - IDSW на близких объектах => ↑ iou_thresh_low
    # Типично: 0.03–0.15.

    # -------------------------
    # Mahalanobis motion gating (на основе Калмана)
    # -------------------------

    gate_thresh: float = 13.28
    # Что: порог χ² по Mahalanobis-дистанции между предсказанием KF и измерением (z=[x,y,a,h]).
    # df=4: 95%≈9.49, 99%≈13.28. Значение 99% делает гейтинг мягче.
    # Влияние: запрещает матчи, которые слишком далеко от предсказания движения.
    # Симптомы:
    #   - FRAG при резких движениях/джиттере => ↑ gate_thresh (мягче)
    #   - IDSW (перепрыгивания при сближениях) => ↓ gate_thresh (строже)
    # Типично: 9.49–16.0 (13.28 часто хороший компромисс при адекватном KF).

    lost_gate_thresh: float = 13.28
    # Что: отдельный motion-gate для реактивации LOST (часто держат мягче).
    # Симптомы:
    #   - LOST почти не реактивируется (появляются новые ID) => ↑ lost_gate_thresh
    #   - LOST реактивируется “не туда” (IDSW) => ↓ lost_gate_thresh

    # -------------------------
    # Веса стоимости матчинга
    # cost = alpha*(1 - IoU) + beta*d_app
    # -------------------------

    alpha: float = 0.5
    # Что: вес геометрии (IoU) в стоимости.
    # Симптомы:
    #   - если ReID надёжный, а геометрия ненадёжна (окклюзии/джиттер) => ↓ alpha
    #   - если ReID слабый/нестабилен => ↑ alpha

    beta: float = 0.5
    # Что: вес внешнего вида (ReID) в стоимости.
    # Примечание: есть эвристика “beta_eff=0 если ReID weak/off”.
    # Симптомы:
    #   - IDSW между похожими объектами => ↑ beta (и/или строже гейтинг)
    #   - FRAG из-за шума ReID => ↓ beta (или ReID будет временно отключён)
    # Типично: 0.0–0.7.

    # -------------------------
    # Реактивация LOST (ориентация на внешний вид)
    # -------------------------

    lost_alpha: float = 0.2
    # Что: вес IoU в стоимости для реактивации LOST.
    # Идея: при окклюзии геометрия менее надёжна => обычно меньше, чем в ACTIVE.
    # Симптомы:
    #   - реактивация “не туда” => ↑ lost_alpha (больше доверять геометрии)
    #   - реактивация не происходит, когда объект сильно сместился => ↓ lost_alpha

    lost_beta: float = 0.8
    # Что: вес ReID в стоимости для реактивации LOST.
    # Симптомы:
    #   - неправильная реактивация на похожего => ↓ lost_beta или строже app gate
    #   - реактивация не происходит при нормальном ReID => ↑ lost_beta

    lost_iou_thresh: float = 0.05
    # Что: минимальный IoU для реактивации LOST (жёсткий фильтр).
    # Влияние: насколько “рядом” должен быть бокс, чтобы реактивация вообще была допустима.
    # Симптомы:
    #   - FRAG (LOST не возвращается, создаются новые треки) => ↓ lost_iou_thresh
    #   - IDSW (реактивация на другого) => ↑ lost_iou_thresh
    # Типично: 0.03–0.2. Если ReID сильный — допустимо держать низко (0.05–0.1).

    reid_lost_app_gate: float = 0.45
    # Что: порог appearance-gating для реактивации LOST: если d_app > gate => запрет.
    # d_app = 1 - cos, т.е. gate=0.45 => cos >= 0.55 (мягче, чем для ACTIVE обычно).
    # Симптомы:
    #   - неправильные реактивации => ↓ reid_lost_app_gate (строже)
    #   - реактивация не происходит, хотя объект тот же => ↑ reid_lost_app_gate (мягче)
    # Типично: 0.3–0.6.

    # -------------------------
    # Масштабирование шума измерения Калмана (confidence -> R)
    # -------------------------

    kf_r_scale_min: float = 1.0
    # Что: множитель для R при “очень уверенной” детекции.
    # Чем меньше R => тем сильнее KF доверяет измерению и резче корректируется.
    # Симптомы:
    #   - трек “запаздывает” => можно ↓ kf_r_scale_min (например 0.7)
    #   - трек чрезмерно дёргается на уверенных детекциях => ↑ kf_r_scale_min

    kf_r_scale_max: float = 10.0
    # Что: множитель R для “слабых” детекций (обновление выполняется, но мягко).
    # Симптомы:
    #   - слабые детекции вносят сильный джиттер => ↑ kf_r_scale_max (10 -> 20)
    #   - при слабых детекциях трек “улетает”/отстаёт и теряет матчи => ↓ kf_r_scale_max (10 -> 5)
    # Типично: 5–30.

    # -------------------------
    # Жизненный цикл трека
    # -------------------------

    n_init: int = 3
    # Что: сколько успешных обновлений нужно, чтобы TENTATIVE стал CONFIRMED.
    # Влияние: защита от “одноразового мусора”.
    # Симптомы:
    #   - много коротких мусорных треков => ↑ n_init (3->5)
    #   - треки слишком поздно становятся confirmed => ↓ n_init (3->2)

    max_time_lost: int = 90
    # Что: сколько кадров трек может быть LOST, прежде чем REMOVED.
    # Влияние: буфер от пропусков детектора/окклюзий.
    # Симптомы:
    #   - FRAG на кратких пропусках => ↑ max_time_lost (90->150)
    #   - “призраки” (воскресает не туда) => ↓ max_time_lost (90->60)
    # Важно: измеряется в кадрах. На 30fps: 90 кадров ~ 3 сек.

    # -------------------------
    # Подавление дублей треков
    # -------------------------

    dup_iou_thresh: float = 0.85
    # Что: если два трека одного класса сильно перекрываются (IoU >= dup_iou_thresh),
    #       один удаляется по приоритету (confirmed>tentative, hits, score, age…).
    # Симптомы:
    #   - удаляется “правильный” трек => ↑ dup_iou_thresh (0.85->0.9..0.95)
    #   - дубли не убираются => ↓ dup_iou_thresh (0.85->0.8)
    # Типично: 0.8–0.95.

    # -------------------------
    # ReID (внешний вид)
    # -------------------------

    reid_enabled_classes: Tuple[int, ...] = (0,)
    # Что: для каких классов используется ReID (обычно person).
    # Влияние: для остальных классов cost_app и гейтинг отсутствуют.

    reid_ema_momentum: float = 0.90
    # Что: EMA-моментум для обновления признака трека.
    # 0.9 => медленно, устойчиво; 0.7 => быстрее адаптация, но выше риск “дрейфа личности”.
    # Симптомы:
    #   - внешний вид объекта меняется (повороты/свет), и ReID отваливается => ↓ momentum (0.9->0.8)
    #   - после окклюзии трек “переползает” на другого => ↑ momentum (0.9->0.95)

    reid_app_gate: float = 0.35
    # Что: appearance-gating на ACTIVE ассоциации: если d_app > gate => запрет.
    # gate=0.35 => cos >= 0.65.
    # Симптомы:
    #   - FRAG (матчи запрещаются из-за шума ReID) => ↑ reid_app_gate (0.35->0.45)
    #   - IDSW между похожими => ↓ reid_app_gate (0.35->0.25)
    # Типично: 0.25–0.5.

    lost_require_appearance: bool = True
    # Что: если True, реактивация LOST требует валидного признака (иначе запрет).
    # Если False, реактивация возможна и без признака, но риск IDSW выше.

    # -------------------------
    # Эвристика “слабого ReID” (auto beta=0)
    # -------------------------

    reid_min_valid_dets: int = 2
    # Что: минимальное число валидных ReID-детекций в кадре, чтобы считать ReID “полезным”.
    # Если меньше — beta_eff=0.

    reid_min_valid_ratio: float = 0.25
    # Что: минимальная доля валидных ReID-признаков среди кандидатов (по enabled classes, ROI, size, score>=thr_low).
    # Если меньше — beta_eff=0.
    # Симптомы:
    #   - ReID часто отключается как weak, но требуется чаще использовать => ↓ ratio (0.25->0.1)
    #   - ReID часто шумный, лучше отключать чаще => ↑ ratio (0.25->0.4)


COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush",
]


def resolve_ultralytics_device() -> str | int:
    # ultralytics принимает либо индекс GPU (int), либо строку 'cpu'
    return 0 if torch.cuda.is_available() else "cpu"


# -------------------------
# Video IO (ffmpeg)
# -------------------------

def get_video_info_ffprobe(path: str) -> Tuple[int, int, float]:
    """Вернуть (width, height, fps) через ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = sp.check_output(cmd).decode("utf-8").strip().split("\n")
    width = int(out[0])
    height = int(out[1])
    num, den = out[2].split("/")
    fps = float(num) / float(den)
    return width, height, fps


def ffmpeg_frame_generator(path: str, resize_to: Optional[Tuple[int, int]] = None) -> Iterable[np.ndarray]:
    """Генератор кадров BGR (np.ndarray) через ffmpeg."""
    orig_w, orig_h, _ = get_video_info_ffprobe(path)
    if resize_to is None:
        out_w, out_h = orig_w, orig_h
        vf_args: List[str] = []
    else:
        out_w, out_h = resize_to
        vf_args = ["-vf", f"scale={out_w}:{out_h}"]

    cmd = [
        "ffmpeg",
        "-i", path,
        "-loglevel", "error",
        *vf_args,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    proc = sp.Popen(cmd, stdout=sp.PIPE, bufsize=10**8)

    frame_size = out_w * out_h * 3
    try:
        assert proc.stdout is not None
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) != frame_size:
                break
            frame = np.frombuffer(raw, np.uint8).reshape((out_h, out_w, 3))
            yield frame
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
        proc.wait()


def create_ffmpeg_writer(output_path: str, width: int, height: int, fps: float) -> sp.Popen:
    """Создать subprocess с открытым stdin для записи сырых BGR-кадров."""
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", f"{fps}",
        "-i", "pipe:0",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    return sp.Popen(cmd, stdin=sp.PIPE, bufsize=10**8)


# -------------------------
# Geometry helpers
# -------------------------

def xyxy_to_xyah(boxes_xyxy: np.ndarray) -> np.ndarray:
    """(x1,y1,x2,y2) -> (x_c, y_c, a=w/h, h)."""
    if boxes_xyxy.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x1, y1, x2, y2 = boxes_xyxy[:, 0], boxes_xyxy[:, 1], boxes_xyxy[:, 2], boxes_xyxy[:, 3]
    w = x2 - x1
    h = y2 - y1
    x_c = x1 + 0.5 * w
    y_c = y1 + 0.5 * h
    a = w / (h + 1e-6)
    return np.stack([x_c, y_c, a, h], axis=-1).astype(np.float32)


def xyah_to_xyxy(boxes_xyah: np.ndarray) -> np.ndarray:
    """(x_c, y_c, a, h) -> (x1,y1,x2,y2)."""
    if boxes_xyah.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    x_c, y_c, a, h = boxes_xyah[:, 0], boxes_xyah[:, 1], boxes_xyah[:, 2], boxes_xyah[:, 3]
    w = a * h
    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c + 0.5 * h
    return np.stack([x1, y1, x2, y2], axis=-1).astype(np.float32)


def iou_matrix(boxes1_xyxy: np.ndarray, boxes2_xyxy: np.ndarray) -> np.ndarray:
    """IoU-матрица для (N1,4) и (N2,4)."""
    if boxes1_xyxy.size == 0 or boxes2_xyxy.size == 0:
        return np.zeros((boxes1_xyxy.shape[0], boxes2_xyxy.shape[0]), dtype=np.float32)

    b1 = boxes1_xyxy[:, None, :]  # (N1,1,4)
    b2 = boxes2_xyxy[None, :, :]  # (1,N2,4)

    x1 = np.maximum(b1[..., 0], b2[..., 0])
    y1 = np.maximum(b1[..., 1], b2[..., 1])
    x2 = np.minimum(b1[..., 2], b2[..., 2])
    y2 = np.minimum(b1[..., 3], b2[..., 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = (b1[..., 2] - b1[..., 0]) * (b1[..., 3] - b1[..., 1])
    area2 = (b2[..., 2] - b2[..., 0]) * (b2[..., 3] - b2[..., 1])

    union = area1 + area2 - inter + 1e-6
    return (inter / union).astype(np.float32)


def _sanitize_roi_rel(roi_rel: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Ограничить ROI в [0..1] и обеспечить (x1<=x2, y1<=y2)."""
    x1, y1, x2, y2 = roi_rel
    x1, y1, x2, y2 = [float(v) for v in (x1, y1, x2, y2)]
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if abs(x2 - x1) < 1e-6:
        x1, x2 = 0.0, 1.0
    if abs(y2 - y1) < 1e-6:
        y1, y2 = 0.0, 1.0
    return x1, y1, x2, y2


def compute_roi_rect(frame_shape: Tuple[int, int, int], roi_rel: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
    h, w, _ = frame_shape
    x1r, y1r, x2r, y2r = _sanitize_roi_rel(roi_rel)
    x1 = int(x1r * w)
    x2 = int(x2r * w)
    y1 = int(y1r * h)
    y2 = int(y2r * h)
    return x1, y1, x2, y2


# -------------------------
# Kalman filter XYAH
# -------------------------

class KalmanFilterXYAH:
    """Калман для состояния [x, y, a, h, vx, vy, va]^T с наблюдением [x, y, a, h]^T."""

    def __init__(self, dt: float):
        self.dt = float(dt)
        self.dim_x = 7
        self.dim_z = 4

        self.F = np.eye(self.dim_x, dtype=np.float32)
        self.F[0, 4] = self.dt
        self.F[1, 5] = self.dt
        self.F[2, 6] = self.dt

        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0

        q_pos = 1.0
        q_vel = 10.0
        self.Q = np.diag([q_pos, q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]).astype(np.float32)

        self.R = np.diag([10.0, 10.0, 1.0, 10.0]).astype(np.float32)
        self.R_base = self.R.copy()

        self.x = np.zeros((self.dim_x,), dtype=np.float32)
        self.initialized = False

    def initiate(self, z: np.ndarray) -> None:
        z = np.asarray(z, dtype=np.float32).reshape(-1)
        self.x[:] = 0.0
        self.x[0:4] = z[0:4]
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0
        self.initialized = True

    def predict(self) -> None:
        if not self.initialized:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def project(self, R_override: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        if not self.initialized:
            return np.zeros((self.dim_z,), dtype=np.float32), np.eye(self.dim_z, dtype=np.float32)
        z_pred = self.H @ self.x
        R = self.R if R_override is None else R_override.astype(np.float32)
        S = self.H @ self.P @ self.H.T + R
        return z_pred.astype(np.float32), S.astype(np.float32)

    def update(self, z: np.ndarray, R_override: Optional[np.ndarray] = None) -> None:
        if not self.initialized:
            self.initiate(z)
            return

        z = np.asarray(z, dtype=np.float32).reshape(-1)
        R = self.R if R_override is None else R_override.astype(np.float32)

        z_pred = (self.H @ self.x).astype(np.float32)
        S = (self.H @ self.P @ self.H.T + R).astype(np.float32)

        y = z - z_pred
        K = self.P @ self.H.T @ np.linalg.inv(S + 1e-9 * np.eye(self.dim_z, dtype=np.float32))
        self.x = self.x + K @ y
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P


# -------------------------
# Association (Mahalanobis + ByteTrack + Hungarian)
# -------------------------

def maha_distance_matrix(tracks_xyah: np.ndarray, tracks_S: np.ndarray, dets_xyah: np.ndarray) -> np.ndarray:
    T = tracks_xyah.shape[0]
    N = dets_xyah.shape[0]
    if T == 0 or N == 0:
        return np.zeros((T, N), dtype=np.float32)

    d2 = np.zeros((T, N), dtype=np.float32)
    for t in range(T):
        S_inv = np.linalg.inv(tracks_S[t] + 1e-9 * np.eye(4, dtype=np.float32))
        diff = dets_xyah - tracks_xyah[t]
        d2[t] = np.einsum("ni,ij,nj->n", diff, S_inv, diff)
    return d2


def byte_maha_associate(
    *,
    tracks_xyah: np.ndarray,
    tracks_S: np.ndarray,
    dets_xyxy: np.ndarray,
    dets_xyah: np.ndarray,
    dets_scores: np.ndarray,
    valid_size_mask: np.ndarray,
    inside_roi_mask: np.ndarray,
    thr_high: float,
    thr_low: float,
    iou_thresh_high: float,
    iou_thresh_low: float,
    gate_thresh: float,
    alpha: float,
    beta: float,
    appearance_cost: Optional[np.ndarray],        # shape (T, N)
    appearance_valid: Optional[np.ndarray],       # shape (T, N) boolean
    app_gate: float,
    track_classes: np.ndarray,
    det_classes: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Двухстадийная ассоциация в стиле ByteTrack: motion-gating + опциональный cost по внешнему виду + appearance-gating."""
    T = tracks_xyah.shape[0]
    N = dets_xyxy.shape[0]

    dets_scores = np.asarray(dets_scores, dtype=np.float32)

    # Split dets
    high_mask = (dets_scores >= thr_high) & valid_size_mask & inside_roi_mask
    low_mask = (dets_scores >= thr_low) & (dets_scores < thr_high) & valid_size_mask & inside_roi_mask

    high_idxs = np.where(high_mask)[0]
    low_idxs = np.where(low_mask)[0]

    matches_high: List[Tuple[int, int]] = []
    matches_low: List[Tuple[int, int]] = []

    all_tracks_idx = np.arange(T, dtype=np.int32)
    unmatched_tracks = all_tracks_idx.copy()
    unmatched_dets_high = high_idxs.copy()
    unmatched_dets_low = low_idxs.copy()

    LARGE = 1e6

    # A) HIGH stage
    if T > 0 and high_idxs.size > 0:
        dets_h_xyxy = dets_xyxy[high_idxs]
        dets_h_xyah = dets_xyah[high_idxs]

        tracks_xyxy = xyah_to_xyxy(tracks_xyah)
        iou_h = iou_matrix(tracks_xyxy, dets_h_xyxy)  # (T,Nh)
        iou_dist_h = 1.0 - iou_h

        maha_h = maha_distance_matrix(tracks_xyah, tracks_S, dets_h_xyah)
        gate_mask = maha_h <= gate_thresh

        if appearance_cost is None or appearance_valid is None:
            d_app_h = np.zeros_like(iou_h, dtype=np.float32)
            app_valid_h = np.zeros_like(iou_h, dtype=bool)
        else:
            d_app_h = appearance_cost[:, high_idxs].astype(np.float32)
            app_valid_h = appearance_valid[:, high_idxs].astype(bool)

        cost_h = alpha * iou_dist_h + beta * d_app_h

        # Invalidate by class + motion gate
        for t in range(T):
            for j, d_global in enumerate(high_idxs):
                if track_classes[t] != det_classes[d_global] or (not gate_mask[t, j]):
                    cost_h[t, j] = LARGE

        # Appearance gating (only when valid)
        cost_h[(app_valid_h) & (d_app_h > float(app_gate))] = LARGE

        row_ind, col_ind = linear_sum_assignment(cost_h)

        matched_tracks_mask = np.zeros(T, dtype=bool)
        matched_dets_mask_h = np.zeros(high_idxs.size, dtype=bool)

        for r, c in zip(row_ind, col_ind):
            if cost_h[r, c] >= LARGE:
                continue
            if iou_h[r, c] < iou_thresh_high:
                continue
            t_idx = int(r)
            d_idx = int(high_idxs[c])
            matches_high.append((t_idx, d_idx))
            matched_tracks_mask[t_idx] = True
            matched_dets_mask_h[c] = True

        unmatched_tracks = all_tracks_idx[~matched_tracks_mask]
        unmatched_dets_high = high_idxs[~matched_dets_mask_h]

    # B) LOW stage (только IoU + motion gating; без ReID)
    if unmatched_tracks.size > 0 and low_idxs.size > 0:
        tracks_u_xyah = tracks_xyah[unmatched_tracks]
        tracks_u_S = tracks_S[unmatched_tracks]
        tracks_u_cls = track_classes[unmatched_tracks]

        dets_l_xyxy = dets_xyxy[low_idxs]
        dets_l_xyah = dets_xyah[low_idxs]
        dets_l_cls = det_classes[low_idxs]

        tracks_u_xyxy = xyah_to_xyxy(tracks_u_xyah)
        iou_l = iou_matrix(tracks_u_xyxy, dets_l_xyxy)
        iou_dist_l = 1.0 - iou_l

        maha_l = maha_distance_matrix(tracks_u_xyah, tracks_u_S, dets_l_xyah)
        gate_mask_l = maha_l <= gate_thresh

        cost_l = iou_dist_l.copy()
        for ti in range(unmatched_tracks.size):
            for j in range(low_idxs.size):
                if tracks_u_cls[ti] != dets_l_cls[j] or (not gate_mask_l[ti, j]):
                    cost_l[ti, j] = LARGE

        row_ind_l, col_ind_l = linear_sum_assignment(cost_l)

        matched_tracks_mask_u = np.zeros(unmatched_tracks.size, dtype=bool)
        matched_dets_mask_l = np.zeros(low_idxs.size, dtype=bool)

        for r, c in zip(row_ind_l, col_ind_l):
            if cost_l[r, c] >= LARGE:
                continue
            if iou_l[r, c] < iou_thresh_low:
                continue
            t_idx = int(unmatched_tracks[r])
            d_idx = int(low_idxs[c])
            matches_low.append((t_idx, d_idx))
            matched_tracks_mask_u[r] = True
            matched_dets_mask_l[c] = True

        unmatched_tracks = unmatched_tracks[~matched_tracks_mask_u]
        unmatched_dets_low = low_idxs[~matched_dets_mask_l]

    return {
        "matches_high": np.array(matches_high, dtype=np.int32).reshape(-1, 2),
        "matches_low": np.array(matches_low, dtype=np.int32).reshape(-1, 2),
        "unmatched_tracks": unmatched_tracks,
        "unmatched_dets_high": unmatched_dets_high,
        "unmatched_dets_low": unmatched_dets_low,
    }


# -------------------------
# Tracks + tracker
# -------------------------

class TrackState(IntEnum):
    TENTATIVE = 0
    CONFIRMED = 1
    LOST = 2
    REMOVED = 3


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    n = float(np.linalg.norm(v))
    if n > 0.0:
        v = v / n
    return v


@dataclass
class Track:
    track_id: int
    kf: KalmanFilterXYAH
    class_id: int
    n_init: int
    max_time_lost: int

    state: TrackState = TrackState.TENTATIVE
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    score: float = 0.0

    last_xyah: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    last_S: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))

    trajectory_xyxy: List[np.ndarray] = field(default_factory=list)

    # ReID EMA
    feat_ema: Optional[np.ndarray] = None

    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE

    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    def is_lost(self) -> bool:
        return self.state == TrackState.LOST

    def is_removed(self) -> bool:
        return self.state == TrackState.REMOVED

    def _update_feature_ema(self, feat: Optional[np.ndarray], momentum: float) -> None:
        if feat is None:
            return
        feat = _l2_normalize(feat)
        if self.feat_ema is None:
            self.feat_ema = feat
        else:
            m = float(momentum)
            self.feat_ema = _l2_normalize(m * self.feat_ema + (1.0 - m) * feat)

    def initiate_from_detection(
        self,
        meas_xyah: np.ndarray,
        score: float,
        feature: Optional[np.ndarray] = None,
        *,
        reid_momentum: float,
    ) -> None:
        self.kf.initiate(meas_xyah)
        z_pred, S = self.kf.project()  # base R
        self.last_xyah = z_pred
        self.last_S = S
        self.score = float(score)
        self.age = 1
        self.hits = 1
        self.time_since_update = 0

        self.trajectory_xyxy.append(xyah_to_xyxy(self.last_xyah[None, :])[0])
        self._update_feature_ema(feature, reid_momentum)

    def predict(self) -> None:
        if not self.kf.initialized:
            return
        self.kf.predict()
        z_pred, S = self.kf.project()  # base R (для гейтинга)
        self.last_xyah = z_pred
        self.last_S = S
        self.age += 1
        self.time_since_update += 1
        self.trajectory_xyxy.append(xyah_to_xyxy(self.last_xyah[None, :])[0])

    def update(
        self,
        meas_xyah: np.ndarray,
        score: float,
        feature: Optional[np.ndarray] = None,
        *,
        R_scale: float,
        reid_momentum: float,
    ) -> None:
        """Всегда обновлять KF; R_scale используется, чтобы ослаблять слабые детекции."""
        self.time_since_update = 0
        self.hits += 1
        self.score = float(score)

        R_scaled = self.kf.R_base * float(R_scale)
        self.kf.update(meas_xyah, R_override=R_scaled)

        z_pred, S = self.kf.project()  # base R для гейтинга / следующего кадра
        self.last_xyah = z_pred
        self.last_S = S

        xyxy = xyah_to_xyxy(self.last_xyah[None, :])[0]
        if self.trajectory_xyxy:
            self.trajectory_xyxy[-1] = xyxy
        else:
            self.trajectory_xyxy.append(xyxy)

        self._update_feature_ema(feature, reid_momentum)

        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED
        elif self.state == TrackState.LOST:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        if self.state == TrackState.TENTATIVE:
            self.state = TrackState.REMOVED
        elif self.state in (TrackState.CONFIRMED, TrackState.LOST):
            if self.time_since_update > self.max_time_lost:
                self.state = TrackState.REMOVED
            else:
                self.state = TrackState.LOST

    def current_xyxy(self) -> np.ndarray:
        return xyah_to_xyxy(self.last_xyah[None, :])[0]


def compute_appearance_cost_matrix(
    tracks: List[Track],
    det_classes: np.ndarray,
    det_features: np.ndarray,
    det_feat_valid: np.ndarray,
    *,
    enabled_classes: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """Вернуть (cost_app, valid_app).

    cost_app(t,d) = 1 - cos_sim(track_feat_ema, det_feat)
    valid_app(t,d) == True только если:
      - класс разрешён
      - у трека есть feat_ema
      - у детекции валидный признак
      - совпадает класс
    """
    T = len(tracks)
    N = int(det_features.shape[0])
    if T == 0 or N == 0:
        return np.zeros((T, N), dtype=np.float32), np.zeros((T, N), dtype=bool)

    det_classes = np.asarray(det_classes, dtype=np.int32)
    det_features = np.asarray(det_features, dtype=np.float32)
    det_feat_valid = np.asarray(det_feat_valid, dtype=bool)

    cost_app = np.zeros((T, N), dtype=np.float32)
    valid_app = np.zeros((T, N), dtype=bool)

    enabled = set(int(c) for c in enabled_classes)

    # Нормализация детекционных признаков (безопасно даже при invalid; далее маска)
    det_feats_norm = det_features.copy().astype(np.float32)
    det_feats_norm /= (np.linalg.norm(det_feats_norm, axis=1, keepdims=True) + 1e-12)

    for ti, tr in enumerate(tracks):
        if tr.class_id not in enabled:
            continue
        if tr.feat_ema is None:
            continue

        tr_feat = _l2_normalize(tr.feat_ema)
        idxs = np.where((det_classes == tr.class_id) & det_feat_valid)[0]
        if idxs.size == 0:
            continue

        sims = det_feats_norm[idxs] @ tr_feat
        sims = np.clip(sims, -1.0, 1.0)
        cost = (1.0 - sims).astype(np.float32)

        cost_app[ti, idxs] = cost
        valid_app[ti, idxs] = True

    return cost_app, valid_app


def _r_scale_from_score(score: float, thr_low: float, thr_high: float, r_min: float, r_max: float) -> float:
    """Отобразить score в множитель шума измерения R. Чем выше score, тем меньше R (сильнее коррекция)."""
    s = float(score)
    lo = float(thr_low)
    hi = float(thr_high)
    if hi <= lo:
        return float(r_min)
    if s >= hi:
        return float(r_min)
    if s <= lo:
        return float(r_max)
    # Плавная (квадратичная) шкала
    t = (hi - s) / (hi - lo)  # 0..1
    return float(r_min + (r_max - r_min) * (t * t))


class Tracker:
    def __init__(self, fps: float, cfg: TrackingConfig):
        self.dt = 1.0 / float(fps)
        self.cfg = cfg

        self.tracks: List[Track] = []
        self._next_id = 1

        self.last_debug: Dict[str, object] = {}

    def _spawn_track(self, meas_xyah: np.ndarray, score: float, class_id: int, feature: Optional[np.ndarray]) -> None:
        kf = KalmanFilterXYAH(dt=self.dt)
        tr = Track(
            track_id=self._next_id,
            kf=kf,
            class_id=int(class_id),
            n_init=int(self.cfg.n_init),
            max_time_lost=int(self.cfg.max_time_lost),
        )
        tr.initiate_from_detection(meas_xyah, score, feature=feature, reid_momentum=float(self.cfg.reid_ema_momentum))
        self.tracks.append(tr)
        self._next_id += 1

    def _suppress_duplicates(self) -> int:
        """Удалить дублирующие треки, которые сильно перекрываются (для одного класса)."""
        # Рассматриваются только активные треки (не REMOVED и не LOST)
        idxs = [i for i, t in enumerate(self.tracks) if (not t.is_removed()) and (not t.is_lost())]
        if len(idxs) < 2:
            return 0

        boxes = np.stack([self.tracks[i].current_xyxy() for i in idxs], axis=0).astype(np.float32)
        iou = iou_matrix(boxes, boxes)
        np.fill_diagonal(iou, 0.0)

        to_remove: set[int] = set()
        thr = float(self.cfg.dup_iou_thresh)

        def keep_vs_drop(a: Track, b: Track) -> Tuple[Track, Track]:
            # confirmed > tentative
            if a.is_confirmed() and b.is_tentative():
                return a, b
            if b.is_confirmed() and a.is_tentative():
                return b, a
            # больше hits — лучше
            if a.hits != b.hits:
                return (a, b) if a.hits > b.hits else (b, a)
            # выше score — лучше
            if a.score != b.score:
                return (a, b) if a.score > b.score else (b, a)
            # старше age — лучше
            if a.age != b.age:
                return (a, b) if a.age > b.age else (b, a)
            # стабильный tie-breaker: меньший id выигрывает
            return (a, b) if a.track_id < b.track_id else (b, a)

        for ai in range(len(idxs)):
            ia = idxs[ai]
            if ia in to_remove:
                continue
            for bi in range(ai + 1, len(idxs)):
                ib = idxs[bi]
                if ib in to_remove:
                    continue
                ta = self.tracks[ia]
                tb = self.tracks[ib]
                if ta.class_id != tb.class_id:
                    continue
                if float(iou[ai, bi]) >= thr:
                    keep, drop = keep_vs_drop(ta, tb)
                    to_remove.add(drop.track_id)

        if not to_remove:
            return 0

        removed_n = 0
        for t in self.tracks:
            if t.track_id in to_remove:
                t.state = TrackState.REMOVED
                removed_n += 1
        return removed_n

    def update(self, det: Dict[str, np.ndarray], det_features: Optional[np.ndarray], det_feat_valid: Optional[np.ndarray]) -> tuple[List[Track], np.ndarray]:
        # Удалить REMOVED
        self.tracks = [t for t in self.tracks if not t.is_removed()]

        # Предсказание для всех треков
        for t in self.tracks:
            t.predict()

        T_all = len(self.tracks)

        dets_xyxy = np.asarray(det["xyxy"], dtype=np.float32)
        scores = np.asarray(det["scores"], dtype=np.float32)
        det_classes = np.asarray(det["cls"], dtype=np.int32)
        valid_size_mask = np.asarray(det["valid_size_mask"], dtype=bool)
        inside_roi_mask = np.asarray(det["inside_roi_mask"], dtype=bool)

        N = int(dets_xyxy.shape[0])
        det_consumed = np.zeros((N,), dtype=bool)  # что “поглощено” трекером (update/reactivate/spawn)

        if N == 0:
            for t in self.tracks:
                t.mark_missed()
            self.tracks = [t for t in self.tracks if not t.is_removed()]
            return self.tracks, det_consumed

        dets_xyah = xyxy_to_xyah(dets_xyxy)

        # ReID-cost для всех треков (включая LOST)
        appearance_cost = None
        appearance_valid = None
        if det_features is not None and det_feat_valid is not None and T_all > 0:
            appearance_cost, appearance_valid = compute_appearance_cost_matrix(
                self.tracks,
                det_classes,
                det_features,
                det_feat_valid,
                enabled_classes=self.cfg.reid_enabled_classes,
            )

        # Решение: использовать ли ReID в этом кадре; иначе временно отключить beta (и гейтинг)
        beta_eff = float(self.cfg.beta)
        reid_status = "on"

        use_app = (appearance_cost is not None) and (appearance_valid is not None) and (det_features is not None) and (det_feat_valid is not None)

        if not use_app:
            beta_eff = 0.0
            reid_status = "off"
        else:
            enabled = np.array(list(self.cfg.reid_enabled_classes), dtype=np.int32)
            # Кандидаты детекций (разрешённый класс + внутри ROI + размер ок + score>=thr_low)
            cand_total_mask = (
                np.isin(det_classes, enabled) &
                inside_roi_mask &
                valid_size_mask &
                (scores >= float(self.cfg.thr_low))
            )
            cand_valid_mask = cand_total_mask & det_feat_valid

            cand_total = int(cand_total_mask.sum())
            cand_valid = int(cand_valid_mask.sum())
            ratio = float(cand_valid) / float(max(1, cand_total))

            if (cand_valid < int(self.cfg.reid_min_valid_dets)) or (ratio < float(self.cfg.reid_min_valid_ratio)):
                beta_eff = 0.0
                reid_status = "weak"

        # Разделение треков: ACTIVE и LOST
        active_idxs = np.array([i for i, t in enumerate(self.tracks) if not t.is_lost()], dtype=np.int32)
        lost_idxs = np.array([i for i, t in enumerate(self.tracks) if t.is_lost()], dtype=np.int32)

        # Матрицы для ACTIVE
        if active_idxs.size > 0:
            tracks_xyah_active = np.stack([self.tracks[i].last_xyah for i in active_idxs], axis=0)
            tracks_S_active = np.stack([self.tracks[i].last_S for i in active_idxs], axis=0)
            track_classes_active = np.array([self.tracks[i].class_id for i in active_idxs], dtype=np.int32)
            if (beta_eff > 0.0) and (appearance_cost is not None) and (appearance_valid is not None):
                app_cost_active = appearance_cost[active_idxs]
                app_valid_active = appearance_valid[active_idxs]
            else:
                app_cost_active = None
                app_valid_active = None
        else:
            tracks_xyah_active = np.zeros((0, 4), dtype=np.float32)
            tracks_S_active = np.zeros((0, 4, 4), dtype=np.float32)
            track_classes_active = np.zeros((0,), dtype=np.int32)
            app_cost_active = None
            app_valid_active = None

        # Диагностика количества HIGH/LOW детекций
        det_hi = int(((scores >= float(self.cfg.thr_high)) & valid_size_mask & inside_roi_mask).sum())
        det_lo = int(((scores >= float(self.cfg.thr_low)) & (scores < float(self.cfg.thr_high)) & valid_size_mask & inside_roi_mask).sum())

        # Основная ассоциация на ACTIVE
        assoc = byte_maha_associate(
            tracks_xyah=tracks_xyah_active,
            tracks_S=tracks_S_active,
            dets_xyxy=dets_xyxy,
            dets_xyah=dets_xyah,
            dets_scores=scores,
            valid_size_mask=valid_size_mask,
            inside_roi_mask=inside_roi_mask,
            thr_high=float(self.cfg.thr_high),
            thr_low=float(self.cfg.thr_low),
            iou_thresh_high=float(self.cfg.iou_thresh_high),
            iou_thresh_low=float(self.cfg.iou_thresh_low),
            gate_thresh=float(self.cfg.gate_thresh),
            alpha=float(self.cfg.alpha),
            beta=float(beta_eff),
            appearance_cost=app_cost_active,
            appearance_valid=app_valid_active,
            app_gate=float(self.cfg.reid_app_gate),
            track_classes=track_classes_active,
            det_classes=det_classes,
        )

        matches_high = assoc["matches_high"]
        matches_low = assoc["matches_low"]
        unmatched_dets_high = assoc["unmatched_dets_high"]

        matched_active_rel = np.zeros(active_idxs.size, dtype=bool)

        # Update ACTIVE (high)
        for t_rel, d_idx in matches_high:
            t_rel_i = int(t_rel)
            d_i = int(d_idx)
            matched_active_rel[t_rel_i] = True
            det_consumed[d_i] = True

            tr = self.tracks[int(active_idxs[t_rel_i])]
            meas_xyah = dets_xyah[d_i]
            sc = float(scores[d_i])

            feat = None
            if det_features is not None and det_feat_valid is not None and bool(det_feat_valid[d_i]):
                feat = det_features[d_i]

            R_scale = _r_scale_from_score(sc, self.cfg.thr_low, self.cfg.thr_high, self.cfg.kf_r_scale_min, self.cfg.kf_r_scale_max)
            tr.update(meas_xyah, sc, feature=feat, R_scale=R_scale, reid_momentum=float(self.cfg.reid_ema_momentum))

        # Update ACTIVE (low)
        for t_rel, d_idx in matches_low:
            t_rel_i = int(t_rel)
            d_i = int(d_idx)
            matched_active_rel[t_rel_i] = True
            det_consumed[d_i] = True

            tr = self.tracks[int(active_idxs[t_rel_i])]
            meas_xyah = dets_xyah[d_i]
            sc = float(scores[d_i])

            feat = None
            if det_features is not None and det_feat_valid is not None and bool(det_feat_valid[d_i]):
                feat = det_features[d_i]

            R_scale = _r_scale_from_score(sc, self.cfg.thr_low, self.cfg.thr_high, self.cfg.kf_r_scale_min, self.cfg.kf_r_scale_max)
            tr.update(meas_xyah, sc, feature=feat, R_scale=R_scale, reid_momentum=float(self.cfg.reid_ema_momentum))

        # Missed ACTIVE
        for rel_i, tr_idx in enumerate(active_idxs.tolist()):
            if not matched_active_rel[rel_i]:
                self.tracks[tr_idx].mark_missed()

        # Реактивация LOST на оставшихся HIGH детекциях
        remaining_high = unmatched_dets_high.copy()
        used_high = np.zeros(remaining_high.size, dtype=bool)

        if lost_idxs.size > 0 and remaining_high.size > 0:
            if (appearance_cost is not None) and (appearance_valid is not None) and (det_feat_valid is not None):
                lost_tracks = [self.tracks[i] for i in lost_idxs.tolist()]
                L = len(lost_tracks)
                M = int(remaining_high.size)

                lost_xyah = np.stack([t.last_xyah for t in lost_tracks], axis=0)
                lost_S = np.stack([t.last_S for t in lost_tracks], axis=0)
                lost_cls = np.array([t.class_id for t in lost_tracks], dtype=np.int32)

                det_idx = remaining_high.astype(np.int32)
                det_xyxy_h = dets_xyxy[det_idx]
                det_xyah_h = dets_xyah[det_idx]
                det_cls_h = det_classes[det_idx]

                iou_lost = iou_matrix(xyah_to_xyxy(lost_xyah), det_xyxy_h)
                iou_dist = 1.0 - iou_lost

                maha = maha_distance_matrix(lost_xyah, lost_S, det_xyah_h)
                gate = maha <= float(self.cfg.lost_gate_thresh)

                d_app = appearance_cost[lost_idxs][:, det_idx].astype(np.float32)
                v_app = appearance_valid[lost_idxs][:, det_idx].astype(bool)

                LARGE = 1e6
                cost = float(self.cfg.lost_alpha) * iou_dist + float(self.cfg.lost_beta) * d_app

                for li in range(L):
                    for mj in range(M):
                        if (lost_cls[li] != det_cls_h[mj]) or (not gate[li, mj]):
                            cost[li, mj] = LARGE

                if bool(self.cfg.lost_require_appearance):
                    cost[~v_app] = LARGE

                cost[(v_app) & (d_app > float(self.cfg.reid_lost_app_gate))] = LARGE
                cost[iou_lost < float(self.cfg.lost_iou_thresh)] = LARGE

                row, col = linear_sum_assignment(cost)
                for r, c in zip(row, col):
                    if cost[r, c] >= LARGE:
                        continue
                    tr_idx = int(lost_idxs[int(r)])
                    d_global = int(det_idx[int(c)])

                    tr = self.tracks[tr_idx]
                    meas_xyah = dets_xyah[d_global]
                    sc = float(scores[d_global])

                    feat = None
                    if det_features is not None and det_feat_valid is not None and bool(det_feat_valid[d_global]):
                        feat = det_features[d_global]

                    R_scale = _r_scale_from_score(sc, self.cfg.thr_low, self.cfg.thr_high, self.cfg.kf_r_scale_min, self.cfg.kf_r_scale_max)
                    tr.update(meas_xyah, sc, feature=feat, R_scale=R_scale, reid_momentum=float(self.cfg.reid_ema_momentum))

                    used_high[int(c)] = True
                    det_consumed[d_global] = True

        remaining_high = remaining_high[~used_high]
        n_reactivated = int(used_high.sum())
        n_spawn = int(remaining_high.size)

        # LOST, которые так и не были реактивированы, остаются в LOST/удаляются по max_time_lost
        for i in lost_idxs.tolist():
            if self.tracks[i].is_lost():
                self.tracks[i].mark_missed()

        # Создание новых треков из оставшихся HIGH
        for d_idx in remaining_high.tolist():
            d_i = int(d_idx)
            meas_xyah = dets_xyah[d_i]
            sc = float(scores[d_i])
            class_id = int(det_classes[d_i])

            feat = None
            if det_features is not None and det_feat_valid is not None and bool(det_feat_valid[d_i]):
                feat = det_features[d_i]

            self._spawn_track(meas_xyah, sc, class_id, feature=feat)
            det_consumed[d_i] = True  # чтобы не рисовать детекцию отдельно

        # Подавление дублей
        dup_rm = int(self._suppress_duplicates())

        self.tracks = [t for t in self.tracks if not t.is_removed()]

        trk_conf = sum(1 for t in self.tracks if t.is_confirmed())
        trk_tent = sum(1 for t in self.tracks if t.is_tentative())
        trk_lost = sum(1 for t in self.tracks if t.is_lost())
        trk_active = sum(1 for t in self.tracks if (not t.is_removed()) and (not t.is_lost()))

        self.last_debug = {
            "frame": det.get("frame_idx", "-"),
            "det_n": int(N),
            "det_hi": int(det_hi),
            "det_lo": int(det_lo),
            "reid_status": reid_status,
            "beta_eff": f"{beta_eff:.2f}",
            "trk_all": int(T_all),
            "trk_active": int(trk_active),
            "trk_conf": int(trk_conf),
            "trk_tent": int(trk_tent),
            "trk_lost": int(trk_lost),
            "m_high": int(matches_high.shape[0]),
            "m_low": int(matches_low.shape[0]),
            "react": int(n_reactivated) if 'n_reactivated' in locals() else 0,
            "spawn": int(n_spawn) if 'n_spawn' in locals() else 0,
            "dup_rm": int(dup_rm),
        }

        return self.tracks, det_consumed


# -------------------------
# Detector + ReID wrappers
# -------------------------

class YoloDetector:
    def __init__(self, model_ref: str, *, device: str | int, imgsz: int):
        self.device = device
        self.imgsz = int(imgsz)

        self.model = None
        self._mode = "unknown"

        try:
            from ultralytics import YOLOv10  # type: ignore
            self.model = YOLOv10.from_pretrained(model_ref)
            self._mode = "yolov10_hf"
        except Exception:
            try:
                from ultralytics import YOLO  # type: ignore
                self.model = YOLO(model_ref)
                self._mode = "yolo_generic"
            except Exception as e:
                raise RuntimeError(
                    "Не удалось загрузить модель ultralytics. "
                    "Нужен установленный ultralytics и корректная ссылка на модель (HF repo или путь к .pt)."
                ) from e

    def predict(self, frame_bgr: np.ndarray, conf: float, iou: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            res0 = self.model.predict(
                frame_bgr,
                conf=float(conf),
                iou=float(iou),
                device=self.device,
                verbose=False,
                imgsz=self.imgsz,
            )[0]

        if getattr(res0, "boxes", None) is None or len(res0.boxes) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
            )

        boxes_xyxy = res0.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        scores = res0.boxes.conf.detach().cpu().numpy().astype(np.float32)
        cls = res0.boxes.cls.detach().cpu().numpy().astype(np.int32)
        return boxes_xyxy, scores, cls


class PersonReID:
    """Экстрактор признаков OSNet через torchreid.FeatureExtractor (опционально)."""

    def __init__(self, weights_path: str, *, device: str):
        from torchreid.utils import FeatureExtractor  # type: ignore

        wp = Path(weights_path)
        if not wp.exists():
            raise FileNotFoundError(f"Не найдены веса OSNet: {wp}")

        self.extractor = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path=str(wp),
            device=device,
        )

    def extract(
        self,
        frame_bgr: np.ndarray,
        boxes_xyxy: np.ndarray,
        cls: np.ndarray,
        person_class: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Вернуть (features[N,512], valid_mask[N])."""
        N = int(boxes_xyxy.shape[0])
        if N == 0:
            return np.zeros((0, 512), dtype=np.float32), np.zeros((0,), dtype=bool)

        H, W, _ = frame_bgr.shape
        crops: List[np.ndarray] = []
        idx_map: List[int] = []

        for i in range(N):
            if int(cls[i]) != int(person_class):
                continue
            x1, y1, x2, y2 = boxes_xyxy[i]
            x1i = max(0, min(W - 1, int(x1)))
            y1i = max(0, min(H - 1, int(y1)))
            x2i = max(0, min(W - 1, int(x2)))
            y2i = max(0, min(H - 1, int(y2)))
            if x2i <= x1i or y2i <= y1i:
                continue

            crop_bgr = frame_bgr[y1i:y2i, x1i:x2i, :]
            if crop_bgr.size == 0:
                continue

            crop_bgr = cv2.resize(crop_bgr, (128, 256), interpolation=cv2.INTER_LINEAR)
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            crops.append(crop_rgb)
            idx_map.append(i)

        features = np.zeros((N, 512), dtype=np.float32)
        valid_mask = np.zeros((N,), dtype=bool)
        if not crops:
            return features, valid_mask

        with torch.no_grad():
            feats = self.extractor(crops)
            if isinstance(feats, torch.Tensor):
                feats = feats.detach().cpu().numpy()

        for j, det_idx in enumerate(idx_map):
            v = feats[j].astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            features[det_idx] = v
            valid_mask[det_idx] = True

        return features, valid_mask


# -------------------------
# Per-frame det postproc + drawing
# -------------------------

def draw_debug_overlay(img_bgr: np.ndarray, dbg: Dict[str, object]) -> np.ndarray:
    """Компактный debug-оверлей в левом верхнем углу."""
    if not dbg:
        return img_bgr

    img = img_bgr.copy()

    lines = [
        f"f:{dbg.get('frame', '-')}"
        f" det:{dbg.get('det_n', '-')}"
        f" hi:{dbg.get('det_hi', '-')}"
        f" lo:{dbg.get('det_lo', '-')}"
        f" reid:{dbg.get('reid_status', '-')}"
        f" beta:{dbg.get('beta_eff', '-')}",
        f"trk:{dbg.get('trk_all', '-')}"
        f" act:{dbg.get('trk_active', '-')}"
        f" C:{dbg.get('trk_conf', '-')}"
        f" T:{dbg.get('trk_tent', '-')}"
        f" L:{dbg.get('trk_lost', '-')}",
        f"mH:{dbg.get('m_high', '-')}"
        f" mL:{dbg.get('m_low', '-')}"
        f" re:{dbg.get('react', '-')}"
        f" sp:{dbg.get('spawn', '-')}"
        f" dup:{dbg.get('dup_rm', '-')}",
    ]

    x, y0 = 8, 18
    line_h = 16
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.45
    th = 1

    # Фон
    w_max = 0
    for s in lines:
        (w, _h), _ = cv2.getTextSize(s, font, fs, th)
        w_max = max(w_max, w)
    box_w = w_max + 12
    box_h = len(lines) * line_h + 10

    overlay = img.copy()
    cv2.rectangle(overlay, (x - 4, y0 - 14), (x - 4 + box_w, y0 - 14 + box_h), (0, 0, 0), thickness=-1)
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    for i, s in enumerate(lines):
        y = y0 + i * line_h
        cv2.putText(img, s, (x, y), font, fs, (255, 255, 255), th, cv2.LINE_AA)

    return img


def make_det_dict(
    frame_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    cls: np.ndarray,
    *,
    cfg: TrackingConfig,
) -> Dict[str, np.ndarray]:
    roi_x1, roi_y1, roi_x2, roi_y2 = compute_roi_rect(frame_bgr.shape, cfg.roi_rel)

    h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1] if boxes_xyxy.size else np.zeros((0,), dtype=np.float32)
    valid_size_mask = h >= int(cfg.min_box_height)

    inside_roi_mask = (
        (boxes_xyxy[:, 0] >= roi_x1) &
        (boxes_xyxy[:, 1] >= roi_y1) &
        (boxes_xyxy[:, 2] <= roi_x2) &
        (boxes_xyxy[:, 3] <= roi_y2)
    ) if boxes_xyxy.size else np.zeros((0,), dtype=bool)

    return {
        "xyxy": boxes_xyxy.astype(np.float32),
        "scores": scores.astype(np.float32),
        "cls": cls.astype(np.int32),
        "valid_size_mask": valid_size_mask.astype(bool),
        "inside_roi_mask": inside_roi_mask.astype(bool),
        "roi_xyxy": np.array([roi_x1, roi_y1, roi_x2, roi_y2], dtype=np.int32),
    }


def draw_detections(frame_bgr: np.ndarray, det: Dict[str, np.ndarray], cfg: TrackingConfig, *, consumed_mask: Optional[np.ndarray] = None) -> np.ndarray:
    img = frame_bgr.copy()
    roi_x1, roi_y1, roi_x2, roi_y2 = [int(v) for v in det["roi_xyxy"]]

    overlay = img.copy()
    cv2.rectangle(overlay, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 0), thickness=-1)
    img = cv2.addWeighted(overlay, 0.15, img, 0.85, 0)

    boxes = det["xyxy"]
    scores = det["scores"]
    cls = det["cls"]
    valid_size_mask = det["valid_size_mask"]
    inside_roi_mask = det["inside_roi_mask"]

    if consumed_mask is None:
        consumed_mask = np.zeros((int(boxes.shape[0]),), dtype=bool)
    else:
        consumed_mask = np.asarray(consumed_mask, dtype=bool)
        if consumed_mask.shape[0] != int(boxes.shape[0]):
            consumed_mask = np.zeros((int(boxes.shape[0]),), dtype=bool)

    for i in range(int(boxes.shape[0])):
        if bool(consumed_mask[i]):
            continue  # не рисовать детекцию, если она уже использована трекером

        x1, y1, x2, y2 = boxes[i]
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))

        if not bool(valid_size_mask[i]):
            color = (0, 0, 255)
        else:
            if bool(inside_roi_mask[i]):
                if float(scores[i]) >= float(cfg.track_conf_thr):
                    color = (0, 255, 0)
                else:
                    color = (0, 165, 255)
            else:
                color = (255, 0, 0)

        cv2.rectangle(img, p1, p2, color, 1)

        class_id = int(cls[i])
        name = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else str(class_id)
        cv2.putText(
            img,
            f"{name} {float(scores[i]):.2f}",
            (p1[0], p1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    return img


def draw_tracks_on_frame(
    frame_bgr: np.ndarray,
    tracks: List[Track],
    *,
    draw_tentative: bool = True,
    draw_lost: bool = False,
    traj_len: int = 30,
) -> np.ndarray:
    img = frame_bgr.copy()

    for tr in tracks:
        if tr.is_removed():
            continue

        if tr.is_confirmed():
            color = (0, 255, 0)
        elif tr.is_tentative():
            if not draw_tentative:
                continue
            color = (0, 255, 255)
        elif tr.is_lost():
            if not draw_lost:
                continue
            color = (255, 0, 255)
        else:
            color = (255, 255, 255)

        x1, y1, x2, y2 = tr.current_xyxy()
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, 1)

        class_id = int(tr.class_id)
        name = COCO_CLASSES[class_id] if 0 <= class_id < len(COCO_CLASSES) else str(class_id)

        # Одна строка: класс + ID + score
        cv2.putText(
            img,
            f"{name} {tr.track_id} {tr.score:.2f}",
            (p1[0], p1[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

        # Траектория (в текущей версии хранится целиком)
        if tr.trajectory_xyxy:
            traj = tr.trajectory_xyxy
            pts = [(int(0.5 * (b[0] + b[2])), int(0.5 * (b[1] + b[3]))) for b in traj]
            if len(pts) >= 2:
                pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts_arr], isClosed=False, color=color, thickness=1)

    return img


# -------------------------
# Main video loop
# -------------------------

def process_video(
    *,
    input_path: str,
    output_path: str,
    yolo_model_ref: str,
    cfg: TrackingConfig,
    osnet_weights: Optional[str],
    disable_reid: bool,
    max_frames: Optional[int],
) -> None:
    input_path = str(Path(input_path))
    output_path = str(Path(output_path))

    w, h, fps = get_video_info_ffprobe(input_path)
    print(f"Video: {w}x{h}, fps={fps:.3f}")

    device = resolve_ultralytics_device()
    print("YOLO device:", device)

    detector = YoloDetector(yolo_model_ref, device=device, imgsz=int(cfg.imgsz))

    reid: Optional[PersonReID] = None
    if (not disable_reid) and osnet_weights:
        reid_device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            reid = PersonReID(osnet_weights, device=reid_device)
            print(f"[ReID] OSNet loaded on {reid_device}")
        except Exception as e:
            print(f"[ReID] Disabled (failed to load): {e}")
            reid = None

    tracker = Tracker(fps=fps, cfg=cfg)

    gen = ffmpeg_frame_generator(input_path, resize_to=None)
    writer = create_ffmpeg_writer(output_path, w, h, fps)

    processed = 0
    try:
        assert writer.stdin is not None
        for idx, frame in enumerate(gen):
            if max_frames is not None and idx >= int(max_frames):
                break

            boxes_xyxy, scores, cls = detector.predict(frame, conf=cfg.det_conf_thr, iou=cfg.det_iou_thr)

            allowed = np.array([0, 2, 5, 7], dtype=np.int32)  # person, car, bus, truck
            keep = np.isin(cls, allowed)
            boxes_xyxy, scores, cls = boxes_xyxy[keep], scores[keep], cls[keep]

            det = make_det_dict(frame, boxes_xyxy, scores, cls, cfg=cfg)
            det["frame_idx"] = idx

            det_features = None
            det_feat_valid = None
            if reid is not None:
                feats, valid = reid.extract(frame, boxes_xyxy, cls, person_class=0)
                det_features = feats
                det_feat_valid = valid

            tracks, det_consumed = tracker.update(det, det_features, det_feat_valid)

            vis = draw_detections(frame, det, cfg, consumed_mask=det_consumed)
            vis = draw_tracks_on_frame(vis, tracks)
            vis = draw_debug_overlay(vis, tracker.last_debug)

            try:
                writer.stdin.write(vis.tobytes())
            except BrokenPipeError:
                raise RuntimeError("ffmpeg writer pipe broken (encoder exited)")

            processed += 1
            if idx % 50 == 0:
                print(f"Processed frame {idx}")
    finally:
        if writer.stdin is not None:
            writer.stdin.close()
        ret = writer.wait()
        print("ffmpeg exited with code", ret)
        print("Output:", output_path)
        print("Frames written:", processed)


def _parse_roi(s: str) -> Tuple[float, float, float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be 'x1,y1,x2,y2' (relative, 0..1)")
    try:
        return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError as e:
        raise argparse.ArgumentTypeError("ROI must be 4 floats") from e


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLO + Kalman + ByteTrack-style association (+ optional OSNet ReID)")
    p.add_argument("--input", required=True, help="Input video path")
    p.add_argument("--output", required=True, help="Output video path")
    p.add_argument(
        "--yolo",
        default="jameslahm/yolov10m",
        help="Ultralytics model reference (HF repo or local .pt)",
    )
    p.add_argument(
        "--roi",
        type=_parse_roi,
        default="0,0.35,1,1",
        help="ROI relative coords as 'x1,y1,x2,y2' (0..1).",
    )
    p.add_argument("--osnet", default=None, help="Path to osnet_x1_0_msmt17.pth weights")
    p.add_argument("--no-reid", action="store_true", help="Disable ReID even if --osnet is provided")
    p.add_argument("--max-frames", type=int, default=None, help="Process only first N frames")

    # Common knobs
    p.add_argument("--det-conf", type=float, default=None)
    p.add_argument("--det-iou", type=float, default=None)
    p.add_argument("--track-conf", type=float, default=None)
    p.add_argument("--min-h", type=int, default=None)

    return p


def main() -> None:
    args = build_argparser().parse_args()

    cfg = TrackingConfig()
    cfg.roi_rel = _sanitize_roi_rel(tuple(args.roi))

    if args.det_conf is not None:
        cfg.det_conf_thr = float(args.det_conf)
    if args.det_iou is not None:
        cfg.det_iou_thr = float(args.det_iou)
    if args.track_conf is not None:
        cfg.track_conf_thr = float(args.track_conf)
    if args.min_h is not None:
        cfg.min_box_height = int(args.min_h)

    process_video(
        input_path=args.input,
        output_path=args.output,
        yolo_model_ref=args.yolo,
        cfg=cfg,
        osnet_weights=args.osnet,
        disable_reid=bool(args.no_reid),
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
