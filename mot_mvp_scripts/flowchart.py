from graphviz import Digraph


def build_flow():
    g = Digraph("tracking_flow", format="png")

    # Глобальные атрибуты графа
    g.attr(
        rankdir="TB",
        fontname="Times New Roman",
        fontsize="12",
        nodesep="0.35",
        ranksep="0.35",
        splines="ortho",
        newrank="true",
    )
    # Узлы и рёбра: жёстко 12 pt
    g.attr("node", fontname="Times New Roman", fontsize="12")
    g.attr("edge", fontname="Times New Roman", fontsize="12")

    # --- Верхний уровень кадра ---
    g.node("init", "Начало", shape="oval", group="L")
    g.node("frame", "Входной кадр", shape="box", group="L")
    g.node("yolo", "Детекция YOLO\nNMS: det_conf, det_iou, imgsz", shape="box", group="L")
    g.node("filter", "Фильтрация детекций\nразрешённые классы, ROI, min высота", shape="box", group="L")

    g.node("reid_on", "ReID доступен и включён", shape="diamond", group="L")
    g.node("no_reid", "ReID выключен\nβ_eff = 0", shape="box", group="L")
    g.node("do_reid", "Извлечение признаков OSNet\nдля класса «человек»", shape="box", group="L")

    g.node("reid_weak", "ReID достаточно валиден\n(доля и число признаков)", shape="diamond", group="L")
    g.node("weak_off", "ReID слабый\nβ_eff = 0", shape="box", group="L")
    g.node("strong_on", "ReID используется\nβ_eff = β", shape="box", group="L")

    # --- Tracker.update разбит на смысловые фазы ---
    g.node(
        "trk_prep",
        "Трекер: подготовка\n- удалить REMOVED\n- predict Калмана для всех треков",
        shape="box",
        group="L",
    )

    g.node("det_exist", "Есть детекции", shape="diamond", group="L")

    # Левая ветка (нет детекций)
    g.node(
        "no_det",
        "Детекций нет\n- отметить пропуск (mark_missed)\n- удалить REMOVED",
        shape="box",
        group="L",
    )

    # Правая колонка (есть детекции)
    g.node(
        "det_xyah",
        "Подготовка измерений\n- преобразование боксов в XYAH\n- маски ROI и размера",
        shape="box",
        group="R",
    )
    g.node("split", "Разделение треков\nACTIVE и LOST", shape="box", group="R")
    g.node(
        "assoc_active",
        "Ассоциация ACTIVE–детекции\nByteTrack: HIGH затем LOW\nIoU + Mahalanobis + Hungarian\nReID: cost и гейтинг при β_eff>0",
        shape="box",
        group="R",
    )
    g.node(
        "upd_active",
        "Обновление ACTIVE\n- update Калмана для матчей\n- R масштабируется score\n- обновление EMA признака\n- unmatched: mark_missed",
        shape="box",
        group="R",
    )
    g.node(
        "react_lost",
        "Реактивация LOST\nпо оставшимся HIGH\nIoU + Mahalanobis + ReID-гейты\nHungarian",
        shape="box",
        group="R",
    )
    g.node("spawn", "Создание новых треков\nиз оставшихся HIGH\n(TENTATIVE)", shape="box", group="R")
    g.node("dedup", "Подавление дублей\nпо IoU >= dup_iou_thresh\nвыбор лучшего трека", shape="box", group="R")
    g.node("out", "Треки и маска использованных детекций", shape="box", group="R")

    g.node("end", "Конец", shape="oval", group="L")

    # --- Жёсткая укладка: det_xyah на уровне «Начало» (вверх вправо) ---
    with g.subgraph() as s:
        s.attr(rank="min")
        s.node("init")
        s.node("det_xyah")

    # Порядок слева направо на верхнем уровне, без влияния на ранги
    g.edge("init", "det_xyah", style="invis", constraint="false")

    # --- Рёбра верхнего уровня ---
    g.edge("init", "frame")
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

    # --- Развилка ---
    g.edge("trk_prep", "det_exist")
    g.edge("det_exist", "no_det", label="нет")  # вниз (левая колонка)

    # Ветка «да» ведёт к det_xyah вверх вправо: rank-constraint отключён
    g.edge(
    "det_exist",
    "det_xyah",
    headlabel="да",
    labeldistance="10.6",
    labelangle="420",
    constraint="false",
    minlen="3",
)

    # --- Правая колонка (нормальный поток) ---
    g.edge("det_xyah", "split")
    g.edge("split", "assoc_active")
    g.edge("assoc_active", "upd_active")
    g.edge("upd_active", "react_lost")
    g.edge("react_lost", "spawn")
    g.edge("spawn", "dedup")
    g.edge("dedup", "out")
    g.edge("out", "end")

    # --- Левая колонка (нет детекций) ---
    g.edge("no_det", "end")

    return g


if __name__ == "__main__":
    g = build_flow()
    g.render("tracking_flow_ru", cleanup=True)
