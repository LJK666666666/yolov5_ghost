#!/usr/bin/env python3
"""
Anchor optimization tool for safety vest detection
针对安全背心检测的锚框优化工具.

Usage:
    python utils/anchor_optimization.py --data data/SafetyVests.v6/data.yaml --img 640 --thr 4.0
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils.general import colorstr


def kmean_anchors(dataset, n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """
    使用k-means聚类算法为安全背心数据集计算最优锚框.

    Args:
        dataset: 数据集路径
        n: 锚框数量 (default: 9)
        img_size: 图像尺寸 (default: 640)
        thr: 锚框阈值 (default: 4.0)
        gen: 迭代次数 (default: 1000)
        verbose: 是否显示详细信息

    Returns:
        k: 最优锚框 (n, 2)
    """
    from scipy.cluster.vq import kmeans

    print(f"\n{colorstr('Analyzing anchors')}... ", end="")

    # 获取数据集中所有边界框的尺寸
    def print_results(k, verbose=True):
        k = k[np.argsort(k.prod(1))]  # sort small to large
        x, best = metric(k)
        bpr, aat = (best > thr).mean(), (x > thr).mean()  # best possible recall, anytime anchor threshold
        s = (
            f"{prefix}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.4f} anchors past thr\n"
            f"{prefix}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, "
            f"past_thr={x[x > thr].mean():.3f}-mean: "
        )
        for i, x in enumerate(k):
            s += "%i,%i, " % (round(x[0]), round(x[1]))
        if verbose:
            print(s[:-2])
        return k

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        best = x.max(1)[0]  # best_x
        (x > thr).float().sum(1).mean()  # anchors above threshold
        (best > thr).float().mean()  # best possible recall
        return x, best

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_mutation(km, anchor_fitness, gen):
        v = anchor_fitness(km)
        print(f"{prefix}gen {gen}: fitness = {v:.4f}")
        if v > fitness:
            return km, v
        else:
            return k, fitness

    # 加载数据集
    if isinstance(dataset, str):
        with open(dataset, errors="ignore") as f:
            data_dict = yaml.safe_load(f)
        dataset = data_dict["train"]

    # 获取边界框信息
    shapes = img_size * dataset_stats(dataset, verbose=False)
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in shapes])  # wh

    # 过滤小边界框
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(f"{prefix}WARNING: Extremely small objects found: {i} of {len(wh0)} labels are < 3 pixels in size")
    wh = wh0[(wh0 >= 2.0).any(1)]  # filter > 2 pixels

    # 转换为PyTorch张量
    wh = torch.tensor(wh, dtype=torch.float32)
    print(f"{len(wh)} labels, {n} anchors, {img_size} img_size")

    # Kmeans计算
    try:
        print(f"{prefix}Running kmeans for {n} anchors on {len(wh)} points...")
        assert n <= len(wh)  # apply overdetermined constraint
        s = wh.std(0)  # sigmas for whitening
        k = kmeans(wh / s, n, iter=30)[0] * s  # points
        assert n == len(k)  # kmeans may return fewer points than requested if wh is insufficient or too similar
    except Exception:
        print(f"{prefix}WARNING: switching strategies from kmeans to random init")
        k = np.sort(np.random.rand(n * 2)).reshape(n, 2) * img_size  # random init
    wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    k = print_results(k, verbose=False)

    # 进化算法优化
    prefix = colorstr("Evolving anchors: ")
    fitness = anchor_fitness(k)
    pbar = tqdm(range(gen), desc=f"{prefix}Evolving anchors with Genetic Algorithm:")
    for _ in pbar:
        v = np.ones(k.shape)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((np.random.random(k.shape) < 0.3) * random.random() * np.random.randn(*k.shape) * 0.1 + 1).clip(
                0.3, 3.0
            )
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > fitness:
            fitness, k = fg, kg.copy()
            pbar.desc = f"{prefix}Evolving anchors with Genetic Algorithm: fitness = {fitness:.4f}"
            if verbose:
                print_results(k, verbose)

    return print_results(k)


def dataset_stats(
    path="data/coco128.yaml",
    autodownload=False,
    verbose=False,
    profile=False,
    hub=False,
):
    """获取数据集统计信息，包括边界框分布."""

    def round_labels(labels):
        # Update labels to integer class and 6 decimal place floats
        return [[int(c), *(round(x, 4) for x in points)] for c, *points in labels]

    def unzip(path):
        # Unzip TODO: CONSTRAINT: path/to/file.zip MUST unzip to 'path/to/file/'
        if str(path).endswith(".zip"):  # path is data.zip
            assert Path(path).is_file(), f"Error unzipping {path}, file not found"
            ZipFile(path).extractall(path=path.parent)  # unzip
            dir = path.with_suffix("")  # dataset directory == zip name
            assert dir.is_dir(), (
                f"Error unzipping {path}, {dir} not found. path/to/file.zip MUST unzip to path/to/file/"
            )
            return str(dir), next(dir.rglob("*.yaml"))  # return dataset directory, dataset.yaml path
        else:  # path is dataset directory
            return path, next(Path(path).rglob("*.yaml"))  # return dataset directory, dataset.yaml path

    def save_dataset_cache_file(prefix, path, x):
        # Save data.yaml cache file
        if hub and Path(path + "-hub").is_dir():
            path += "-hub"
        x["path"] = path  # download 'path' directory contains dataset
        file = Path(prefix).resolve()
        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir
        np.save(str(file), x)  # save cache for next time
        path, prefix = file.as_posix(), colorstr(f"{prefix}: ")
        print(f"{prefix}New cache created: {path}")

    # 处理数据集路径
    if isinstance(path, (str, Path)) and str(path).endswith(".zip"):  # i.e. gs://bucket/dir/data.zip
        path, data_file = unzip(path)
    else:
        data_file = path
        path = extract_dir if extract_dir else Path(path).parent  # download dir

    # 读取数据配置
    with open(data_file, errors="ignore") as f:
        data = yaml.safe_load(f)  # data dict
    for k in "train", "val", "test":
        if data.get(k):  # prepend path
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith("../"):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)

    # 解析yaml
    train, val, test, s = (data.get(x) for x in ("train", "val", "test", "download"))
    if val:
        val = [Path(x).resolve() for x in (val if isinstance(val, list) else [val])]  # val path
        if not all(x.exists() for x in val):
            name = clean_str(data["path"])  # dataset name
            m = f"\nDataset '{name}' images not found ⚠️ \n"
            if s and len(s):
                m += f"Dataset download with: python {gdown} && python {s}\n"
            m += "Dataset not found."
            raise FileNotFoundError(m)
        time.time()
        for path in val:
            cache_path = (path / Path(path).name).with_suffix(".cache")  # *.cache file
            try:
                cache, exists = (
                    np.load(cache_path, allow_pickle=True).item(),
                    True,
                )  # load dict
                if cache["version"] != cache_version or cache["hash"] != get_hash(list(cache["imgs"])):
                    cache, exists = (
                        cache_stats(path, cache_path, verbose),
                        False,
                    )  # re-cache
            except (FileNotFoundError, AssertionError, AttributeError):
                cache, exists = cache_stats(path, cache_path, verbose), False  # cache

        # Display cache
        nf, nc, n, imgs, lengths = cache.pop("results")  # found, missing, empty, corrupted, total
        if verbose:
            if imgs or labels:
                s = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
            else:
                s = f"Scanning '{cache_path}'... {nf} images, {labels} backgrounds, {nc} corrupt"
            tqdm(None, desc=prefix + s, total=nd, initial=nd, bar_format=BAR_FORMAT)  # display cache results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings
        assert nf > 0 or not augment, f"{prefix}No labels found in {cache_path}, can not start training. {HELP_URL}"

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels, shapes, self.segments = zip(*cache.values())
        nl = len(np.concatenate(labels, 0))  # number of labels
        assert nl > 0 or not augment, f"{prefix}All labels empty in {cache_path}, can not start training. {HELP_URL}"
        self.labels = list(labels)

    # Check if label files exist
    cache_path = (p if p.is_file() else Path(l[0]).parent).with_suffix(".cache")
    try:
        cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
        assert cache["version"] == cache_version  # matches current version
        assert cache["hash"] == get_hash(l + files)  # identical hash
    except (FileNotFoundError, AssertionError):
        cache, exists = (
            cache_stats(dataset_path, cache_path, prefix, verbose),
            False,
        )  # re-cache

    # Display cache
    nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
    if exists and verbose:
        d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
        tqdm(None, desc=prefix + d, total=n, initial=n, bar_format=BAR_FORMAT)  # display cache results
        if cache["msgs"]:
            LOGGER.info("\n".join(cache["msgs"]))  # display warnings

    # Read cache
    [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
    labels, shapes = zip(*cache.values())

    return labels, shapes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/SafetyVests.v6/data.yaml",
        help="dataset.yaml path",
    )
    parser.add_argument("--img-size", type=int, default=640, help="image size")
    parser.add_argument("--thr", type=float, default=4.0, help="anchor-label wh ratio threshold")
    parser.add_argument(
        "--gen",
        type=int,
        default=1000,
        help="generations to evolve anchors using genetic algorithm",
    )
    opt = parser.parse_args()

    print(f"\n{colorstr('Anchor Optimization')} starting...")
    print(f"Dataset: {opt.data}")
    print(f"Image size: {opt.img_size}")
    print(f"Threshold: {opt.thr}")
    print(f"Generations: {opt.gen}")

    # 计算最优锚框
    anchors = kmean_anchors(opt.data, n=9, img_size=opt.img_size, thr=opt.thr, gen=opt.gen)

    # 格式化输出
    print(f"\n{colorstr('Optimized anchors for safety vest detection:')}")
    anchors_formatted = anchors.reshape(3, 6).astype(int)

    print("\n# 优化后的锚框尺寸 (适用于安全背心检测)")
    print("anchors:")
    for i, anchor_set in enumerate(anchors_formatted):
        level = ["P3/8", "P4/16", "P5/32"][i]
        print(f"  - {list(anchor_set)} # {level}")

    print(f"\n{colorstr('Copy the above anchors to your model config file')}")


if __name__ == "__main__":
    main()
