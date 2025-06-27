yolov5s.yaml表示基线方案
1表示添加ghost模块（GhostConv和C3Ghost）
2表示添加CA注意力机制
yolov5s-ghost.yaml表示最终方案，目前和yolov5s-ghost_12.yaml相同
--box-loss wiou启用WIOU损失函数，默认为CIOU损失函数
--hyp data\hyps\hyp.recommand.yaml启用推荐的数据增强超参数
