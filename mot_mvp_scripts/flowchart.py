from graphviz import Digraph

def build_flow():
    g = Digraph("tracking_flow", format="png")
    g.attr(rankdir="TB", fontname="Times New Roman")
    g.attr("node", fontname="Times New Roman", fontsize="10")
    g.attr("edge", fontname="Times New Roman", fontsize="10")

    # --- Верхний уровень кадра ---
    g.node("frame", "Входной кадр", shape="box")
    g.node("yolo", "Детекция YOLO\nNMS: det_conf, det_iou, imgsz", shape="box")
    g.node("filter", "Фильтрация детекций\nразрешённые классы, ROI, min высота", shape="box")

    g.node("reid_on", "ReID доступен и включён", shape="diamond")
    g.node("no_reid", "ReID выключен\nβ_eff = 0", shape="box")
    g.node("do_reid", "Извлечение признаков OSNet\nдля класса «человек»", shape="box")

    g.node("reid_weak", "ReID достаточно валиден\n(доля и число признаков)", shape="diamond")
    g.node("weak_off", "ReID слабый\nβ_eff = 0", shape="box")
    g.node("strong_on", "ReID используется\nβ_eff = β", shape="box")

    # --- Tracker.update разбит на смысловые фазы ---
    g.node("trk_prep", "Трекер: подготовка\n• удалить REMOVED\n• predict Калмана для всех треков", shape="box")

    g.node("no_det", "Детекций нет\n• отметить пропуск (mark_missed)\n• удалить REMOVED\n• выход", shape="box")
    g.node("det_xyah", "Подготовка измерений\n• преобразование боксов в XYAH\n• маски ROI и размера", shape="box")

    g.node("split", "Разделение треков\nACTIVE и LOST", shape="box")

    g.node("assoc_active", "Ассоциация ACTIVE–детекции\nByteTrack: HIGH затем LOW\nIoU + Mahalanobis + Hungarian\nReID: cost и гейтинг при β_eff>0", shape="box")

    g.node("upd_active", "Обновление ACTIVE\n• update Калмана для матчей\n• R масштабируется score\n• обновление EMA признака\n• unmatched: mark_missed", shape="box")

    g.node("react_lost", "Реактивация LOST\nпо оставшимся HIGH\nIoU + Mahalanobis + ReID-гейты\nHungarian", shape="box")

    g.node("spawn", "Создание новых треков\nиз оставшихся HIGH\n(TENTATIVE)", shape="box")

    g.node("dedup", "Подавление дублей\nпо IoU >= dup_iou_thresh\nвыбор лучшего трека", shape="box")

    g.node("out", "Конец\nТреки и маска использованных детекций", shape="box")

    # --- Рёбра верхнего уровня ---
    g.edge("frame", "yolo")
    g.edge("yolo", "filter")
    g.edge("filter", "reid_on")

    g.edge("reid_on", "no_reid", label="нет")
    g.edge("reid_on", "do_reid", label="да")

    g.edge("no_reid", "reid_weak")
    g.edge("do_reid", "reid_weak")

    g.edge("reid_weak", "weak_off", label="нет")
    g.edge("reid_weak", "strong_on", label="да")

    # --- Вход в трекер ---
    g.edge("weak_off", "trk_prep")
    g.edge("strong_on", "trk_prep")

    # --- Внутренний поток трекера ---
    g.node("det_exist", "Есть детекции", shape="diamond")
    g.edge("trk_prep", "det_exist")
    g.edge("det_exist", "no_det", label="нет")
    g.edge("det_exist", "det_xyah", label="да")

    g.edge("det_xyah", "split")
    g.edge("split", "assoc_active")
    g.edge("assoc_active", "upd_active")
    g.edge("upd_active", "react_lost")
    g.edge("react_lost", "spawn")
    g.edge("spawn", "dedup")
    g.edge("dedup", "out")

    return g

if __name__ == "__main__":
    g = build_flow()
    g.render("tracking_flow_ru", cleanup=True)
