from graphviz import Digraph

def build_flow():
    g = Digraph("tracking_flow", format="png")

    g.attr(
        rankdir="TB",
        fontname="Times New Roman",
        fontsize="12",
        nodesep="0.35",
        ranksep="0.35",
        splines="ortho",
        newrank="true",
    )
    g.attr("node", fontname="Times New Roman", fontsize="12")
    g.attr("edge", fontname="Times New Roman", fontsize="12")

    g.node("init", "Начало", shape="oval")
    g.node("frame", "Входной кадр BGR\nframe_bgr", shape="box")

    g.node("preproc", "Предобработка: параметризация инференса\nimgsz, det_conf_thr, det_iou_thr", shape="box")

    g.node("yolo", "Инференс YOLOv10\nraw: boxes_xyxy, scores, cls", shape="box")

    # Твой постпроцессинг (явно)
    g.node("cls_f", "Постобработка: фильтр классов\ncls ∈ allowed", shape="box")
    g.node("roi_f", "Постобработка: ROI\nroi_rel → roi_xyxy\ninside_roi_mask", shape="box")
    g.node("sz_f", "Постобработка: min height\nbox_h ≥ min_box_height\nvalid_size_mask", shape="box")

    g.node(
        "det_pack",
        "Формирование det для трекера\nxyxy, scores, cls,\nroi_xyxy, inside_roi_mask, valid_size_mask",
        shape="box",
    )

    g.node("to_tracker", "Передача det в Tracker", shape="box")
    g.node("end", "Конец", shape="oval")

    # Рёбра
    g.edge("init", "frame")
    g.edge("frame", "preproc")
    g.edge("preproc", "yolo")
    g.edge("yolo", "cls_f")
    g.edge("cls_f", "roi_f")
    g.edge("roi_f", "sz_f")
    g.edge("sz_f", "det_pack")
    g.edge("det_pack", "to_tracker")
    g.edge("to_tracker", "end")

    return g

if __name__ == "__main__":
    g = build_flow()
    g.render("yolov10mod", cleanup=True)
