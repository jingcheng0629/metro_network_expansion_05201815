import folium

try:
    from folium.plugins import MeasureControl, MiniMap
except Exception:
    MeasureControl = None
    MiniMap = None
import os
import math
import time
import datetime
import random
import gc
import json
import re
from pathlib import Path
from contextlib import redirect_stdout
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys
from scipy.spatial import cKDTree
import gymnasium as gym
from torch.distributions import Categorical
from collections import deque
from typing import List, Tuple, Dict, Any

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import gurobipy as gp
from gurobipy import GRB
import networkx as nx
from itertools import combinations
from scipy.ndimage import gaussian_filter1d
from scipy.stats import t as student_t

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
CONFIG = {
    "POI_FILE": "beijing_grid_rewards_1.5km.csv",
    "PRICE_FILE": "house_price_full_idw_target.csv",
    "SKELETON_FILE": "Beijing_Skeleton_Env_260205.csv",
    "STATION_FILE": "地铁站点_北京市_2025.csv",
    "OUTPUT_DIR": "./training_results_random_seed_full",

    "PRESET_LINES": {
        "1号线": ["苹果园", "古城", "八角游乐园", "八宝山", "玉泉路", "五棵松", "万寿路", "公主坟", "军事博物馆",
                  "木樨地", "南礼士路", "复兴门", "西单", "天安门西", "天安门东", "王府井", "东单", "建国门",
                  "永安里", "国贸", "大望路", "四惠", "四惠东"],
        "4号线": ["安河桥北", "北宫门", "西苑", "圆明园", "北京大学东门", "中关村", "人民大学", "魏公村",
                  "国家图书馆", "动物园", "西直门", "新街口", "平安里", "西四", "灵境胡同", "西单", "宣武门",
                  "菜市口", "陶然亭", "北京南站", "马家堡", "角门西", "公益西桥"],
        "5号线": ["宋家庄", "刘家窑", "蒲黄榆", "天坛东门", "磁器口", "崇文门", "东单", "灯市口", "东四",
                  "张自忠路", "北新桥", "雍和宫", "和平里北街", "和平西桥", "惠新西街南口", "惠新西街北口",
                  "大屯路东", "北苑路北", "立水桥南", "立水桥", "天通苑南", "天通苑", "天通苑北"]
    },
    "PRESET_LINE_COLORS": {
        "1号线": "#A4343A", "4号线": "#008E9C", "5号线": "#AA0061",
    },

    "Grid_H": 180,
    "Grid_W": 178,
    "GRID_SIZE_KM": 1.0,

    "COVERAGE_RADIUS_KM": 1.5,
    "COVERAGE_RADIUS_GRID": 1.5,
    "ANTI_LOOP_START_RADIUS": 3.0,
    "MAX_STEPS": 30,
    "TOTAL_BUDGET": 150.0,
    "BUDGET_PER_STATION": 5.0,
    "BUDGET_PER_KM": 1.0,

    "MIN_STATION_DIST_GRID": 1.2,
    "MAX_STATION_DIST_GRID": 2.0,
    "MAX_CANDIDATES": 50,

    "MAX_TURN_ANGLE": 60,
    "MIN_TURN_ANGLE_COS": 0.5,
    "GLOBAL_DIR_MIN_COS_HARD": 0.1,

    "W_TD": 0.7,
    "W_EQ": 0.3,
    "EPSILON_INTERCHANGE": 2.0,
    "EPSILON_REGULAR": 1.0,

    "BLOCK_EXISTING_COVERAGE_AT_RESET": True,
    "TRANSFER_BONUS_ONCE": True,

    "WEIGHT_CONFIGS": [
        {"name": "W80E20", "w_td": 0.8, "w_eq": 0.2, "desc": "出行需求优先"},
        {"name": "W70E30", "w_td": 0.7, "w_eq": 0.3, "desc": "出行需求较优先"},
        {"name": "W60E40", "w_td": 0.6, "w_eq": 0.4, "desc": "需求与公平平衡"},
        {"name": "W50E50", "w_td": 0.5, "w_eq": 0.5, "desc": "完全平衡"},
    ],

    "PENALTY_FORBIDDEN": -5.0,
    "PENALTY_INVALID": -10.0,
    "PENALTY_OVER_BUDGET": -20.0,

    "TOTAL_EPISODES": 20000,
    "MINI_BATCH_SIZE": 64,
    "GAMMA": 0.99,
    "LR_ACTOR": 0.0003,
    "LR_CRITIC": 0.001,

    "UPDATE_TIMESTEPS": 1024,
    "K_EPOCHS": 10,

    "OFF_POLICY_UPDATE_INTERVAL": 8,

    "EPS_CLIP": 0.2,
    "LAMBDA_GAE": 0.95,
    "ENTROPY_COEF": 0.05,
    "MAX_GRAD_NORM": 0.5,

    "GA_POP_SIZE": 100,
    "GA_GENERATIONS": 500,
    "MP_MAX_NODES": 1000,

    "SEED": 42,
    "HIDDEN_DIM": 256,
    "CNN_CHANNELS": [32, 64, 128],
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",

    "LINE_CONFIG": [
        {"name": "Line_E1", "start": "西直门", "direction": "E", "strategy": "hub_radiation"},
        {"name": "Line_NE1", "start": "宋家庄", "direction": "NE", "strategy": "gap_filling"},
    ],
    "BUDGET_CONFIGS": [
        {"name": "B90", "label": "90预算", "total_budget": 90.0},
        {"name": "B120", "label": "120预算", "total_budget": 120.0},
        {"name": "B150", "label": "150预算", "total_budget": 150.0},
    ],
    "STATISTICAL_SEEDS": [42, 66, 142, 242, 342, 442],
}

FULL_MODEL_CONFIG = {
    "use_global_state_module": True,
    "use_six_channel_encoder": True,
    "use_spatial_cnn": True,
    "use_attention": True,
    "use_candidate_sequence_module": True,
    "use_feature_coupling": True,
    "use_geometric_candidate_rules": True,
    "use_action_masking": True,
    "use_budget_constraint": True,
    "use_ppo_update": True,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def release_runtime_memory():
    """Release Python and CUDA allocator state between independent runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_model_config() -> Dict[str, Any]:
    return FULL_MODEL_CONFIG.copy()


def get_model_name() -> str:
    return "PPO"


def make_file_tag(*parts) -> str:
    raw = "_".join(str(p) for p in parts if p is not None and str(p) != "")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_")


def apply_publication_plot_style():
    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "axes.linewidth": 1.2,
        "lines.linewidth": 2.8,
    })


def parse_direction_hint(direction_hint: str) -> Tuple[float, float]:
    if direction_hint is None: return 0.0, 0.0
    h = str(direction_hint).strip().upper()
    h = re.sub(r"[^A-Z]", "", h)
    mapping = {
        "E": (0.0, 1.0), "W": (0.0, -1.0),
        "N": (1.0, 0.0), "S": (-1.0, 0.0),
        "NE": (1.0, 1.0), "NW": (1.0, -1.0),
        "SE": (-1.0, 1.0), "SW": (-1.0, -1.0),
    }
    if h in mapping:
        dr, dc = mapping[h]
    elif len(h) >= 2 and h[:2] in mapping:
        dr, dc = mapping[h[:2]]
    elif len(h) >= 1 and h[0] in mapping:
        dr, dc = mapping[h[0]]
    else:
        return 0.0, 0.0
    norm = math.sqrt(dr * dr + dc * dc) + 1e-9
    return dr / norm, dc / norm


class DataProcessor:
    def __init__(self):
        for f in [CONFIG["POI_FILE"], CONFIG["PRICE_FILE"], CONFIG["SKELETON_FILE"], CONFIG["STATION_FILE"]]:
            if not os.path.exists(f):
                raise FileNotFoundError(f"❌ Missing file: {f}")

        self.df_poi = pd.read_csv(CONFIG["POI_FILE"])
        self.df_price = pd.read_csv(CONFIG["PRICE_FILE"])
        self.df_skeleton = pd.read_csv(CONFIG["SKELETON_FILE"])
        try:
            self.df_stations = pd.read_csv(CONFIG["STATION_FILE"], encoding='utf-8-sig')
        except:
            self.df_stations = pd.read_csv(CONFIG["STATION_FILE"], encoding='gbk')

        self.preset_stations = set()
        self.station_to_lines = {}
        for line_name, stations in CONFIG["PRESET_LINES"].items():
            for st in stations:
                self.preset_stations.add(st)
                self.station_to_lines.setdefault(st, []).append(line_name)

        self.df_stations['lng'] = pd.to_numeric(self.df_stations['lng'], errors='coerce')
        self.df_stations['lat'] = pd.to_numeric(self.df_stations['lat'], errors='coerce')
        self.df_stations.dropna(subset=['lng', 'lat'], inplace=True)
        ref_df = self.df_skeleton if 'center_lon' in self.df_skeleton.columns else self.df_poi
        self.grid_coords = ref_df[['center_lon', 'center_lat']].values
        self.grid_indices = ref_df[['grid_j', 'grid_i']].values
        self.grid_tree = cKDTree(self.grid_coords)

        self.idx_to_lonlat = {tuple(self.grid_indices[i]): tuple(self.grid_coords[i]) for i in
                              range(len(self.grid_indices))}

        self.station_to_grid = {}
        for _, row in self.df_stations.iterrows():
            dist, idx = self.grid_tree.query([row['lng'], row['lat']], k=1)
            self.station_to_grid[str(row['stop_name']).strip()] = tuple(self.grid_indices[idx])

        self.maps = self._build_maps()

    def _build_maps(self) -> Dict[str, torch.Tensor]:
        H, W, device = CONFIG["Grid_H"], CONFIG["Grid_W"], CONFIG["DEVICE"]
        maps = {
            'poi': torch.zeros((H, W), device=device),
            'price': torch.zeros((H, W), device=device),
            'skeleton': torch.zeros((H, W), device=device),
            'forbidden': torch.ones((H, W), device=device)
        }
        for _, row in self.df_poi.iterrows():
            r, c = int(row['grid_j']), int(row['grid_i'])
            if 0 <= r < H and 0 <= c < W: maps['poi'][r, c] = float(row.get('poi_weight', 0))

        for _, row in self.df_price.iterrows():
            r, c = int(row['grid_j']), int(row['grid_i'])
            if 0 <= r < H and 0 <= c < W:
                price = float(row.get('housing_price', 0))
                maps['price'][r, c] = price
                if price > 0: maps['forbidden'][r, c] = 0.0

        for stations in CONFIG["PRESET_LINES"].values():
            for st in stations:
                if st in self.station_to_grid:
                    r, c = self.station_to_grid[st]
                    if 0 <= r < H and 0 <= c < W: maps['skeleton'][int(r), int(c)] = 1.0

        price_min, price_max = maps['price'].min(), maps['price'].max()
        if price_max > price_min: maps['price'] = (maps['price'] - price_min) / (price_max - price_min)

        poi_log = torch.log1p(maps['poi'])
        poi_log_min, poi_log_max = poi_log.min(), poi_log.max()
        if poi_log_max > poi_log_min: maps['poi'] = (poi_log - poi_log_min) / (poi_log_max - poi_log_min)
        return maps

    def get_grid_by_name(self, station_name: str) -> Tuple[int, int]:
        matches = self.df_stations[self.df_stations['stop_name'].astype(str).str.contains(station_name)]
        if matches.empty: return (CONFIG["Grid_H"] // 2, CONFIG["Grid_W"] // 2)
        dist, idx = self.grid_tree.query([matches.iloc[0]['lng'], matches.iloc[0]['lat']], k=1)
        r, c = self.grid_indices[idx]
        return int(r), int(c)

    def grid_to_lonlat(self, r: int, c: int) -> Tuple[float, float]:
        return self.idx_to_lonlat.get((r, c), (0.0, 0.0))


class RewardCalculatorASOC:
    def __init__(self, maps: Dict[str, torch.Tensor], w_td: float = None, w_eq: float = None):
        self.maps = maps
        self.w_td = w_td if w_td is not None else CONFIG["W_TD"]
        self.w_eq = w_eq if w_eq is not None else CONFIG["W_EQ"]
        self._cover_offsets = self._build_cover_offsets(float(CONFIG.get("COVERAGE_RADIUS_GRID", 1.5)))
        self._served_mask = None
        self._has_connected_to_existing = False

    def reset_episode(self, start_pos: Tuple[int, int], existing_network: torch.Tensor = None):
        H, W = self.maps['poi'].shape
        self._served_mask = torch.zeros((H, W), dtype=torch.bool, device=self.maps['poi'].device)
        self._has_connected_to_existing = False
        if (existing_network is not None) and CONFIG.get("BLOCK_EXISTING_COVERAGE_AT_RESET", True):
            self._served_mask |= self._dilate_base_mask((existing_network > 0) | (self.maps['skeleton'] > 0))
        self._mark_covered(start_pos)

    def _dilate_base_mask(self, base_mask: torch.Tensor) -> torch.Tensor:
        H, W = base_mask.shape
        dilated = base_mask.clone()
        for dr, dc in self._cover_offsets:
            if dr == 0 and dc == 0: continue
            r0_t, r1_t = max(0, dr), min(H, H + dr)
            c0_t, c1_t = max(0, dc), min(W, W + dc)
            r0_s, r1_s = max(0, -dr), min(H, H - dr)
            c0_s, c1_s = max(0, -dc), min(W, W - dc)
            if (r0_t < r1_t) and (c0_t < c1_t):
                dilated[r0_t:r1_t, c0_t:c1_t] |= base_mask[r0_s:r1_s, c0_s:c1_s]
        return dilated

    def compute_step_reward(self, pos: Tuple[int, int], prev_pos: Tuple[int, int],
                            existing_network: torch.Tensor, direction_hint: str,
                            sequence: List[Tuple[int, int]] = None) -> Tuple[float, Dict[str, float]]:
        if self._served_mask is None: self.reset_episode(start_pos=prev_pos)
        r, c = pos
        is_transfer = (self.maps['skeleton'][r, c] > 0) or (existing_network[r, c] > 0)
        dist_grid = math.sqrt((r - prev_pos[0]) ** 2 + (c - prev_pos[1]) ** 2)
        cost = CONFIG["BUDGET_PER_STATION"] + (dist_grid * CONFIG["GRID_SIZE_KM"] * CONFIG["BUDGET_PER_KM"])

        new_cells = self._get_newly_covered_cells(pos)
        if is_transfer and (not CONFIG.get("TRANSFER_BONUS_ONCE", True) or not self._has_connected_to_existing):
            epsilon = CONFIG["EPSILON_INTERCHANGE"]
            self._has_connected_to_existing = True
        else:
            epsilon = CONFIG["EPSILON_REGULAR"]

        delta_td = epsilon * self.maps['poi'][new_cells].sum().item()
        delta_eq = self.maps['price'][new_cells].sum().item()
        self._mark_covered(pos)

        return (self.w_td * delta_td) + (self.w_eq * delta_eq), {
            'r_td': delta_td, 'r_eq': delta_eq, 'cost': cost, 'is_transfer': 1 if is_transfer else 0
        }

    def _build_cover_offsets(self, radius_grid: float):
        rmax = int(math.ceil(radius_grid))
        offsets = []
        for dr in range(-rmax, rmax + 1):
            for dc in range(-rmax, rmax + 1):
                if math.sqrt(dr * dr + dc * dc) <= radius_grid + 1e-9:
                    offsets.append((dr, dc))
        return offsets

    def _iter_cover_cells(self, pos: Tuple[int, int]):
        H, W = self.maps['poi'].shape
        for dr, dc in self._cover_offsets:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < H and 0 <= nc < W: yield nr, nc

    def _get_newly_covered_cells(self, pos: Tuple[int, int]):
        cover = torch.zeros(self.maps['poi'].shape, dtype=torch.bool, device=self.maps['poi'].device)
        for nr, nc in self._iter_cover_cells(pos): cover[nr, nc] = True
        return cover & (~self._served_mask)

    def _mark_covered(self, pos: Tuple[int, int]):
        for nr, nc in self._iter_cover_cells(pos): self._served_mask[nr, nc] = True


class HybridStateEncoder:
    def __init__(self, device):
        self.device = device
        self.H, self.W = CONFIG["Grid_H"], CONFIG["Grid_W"]

    def _sample_candidates(self, current_pos, sequence, maps, direction_hint):
        cr, cc = current_pos
        candidates = []
        min_d, max_d = CONFIG["MIN_STATION_DIST_GRID"], CONFIG["MAX_STATION_DIST_GRID"]
        MIN_TURN_ANGLE_COS = CONFIG["MIN_TURN_ANGLE_COS"]
        use_rules = get_model_config().get("use_geometric_candidate_rules", True)

        target_dr, target_dc = parse_direction_hint(direction_hint)
        hard_cos_threshold = float(CONFIG.get("GLOBAL_DIR_MIN_COS_HARD", 0.1))

        if not use_rules:
            radius = int(math.ceil(max_d))
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < self.H and 0 <= nc < self.W):
                        continue
                    dist = math.sqrt(dr ** 2 + dc ** 2)
                    if min_d <= dist <= max_d:
                        candidates.append((nr, nc))
            if len(candidates) > CONFIG["MAX_CANDIDATES"]:
                candidates = random.sample(candidates, CONFIG["MAX_CANDIDATES"])
            return candidates

        radius = int(math.ceil(max_d))
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < self.H and 0 <= nc < self.W):
                    continue
                if use_rules and maps['forbidden'][nr, nc] == 1.0:
                    continue
                if use_rules and (nr, nc) in sequence:
                    continue

                dist = math.sqrt(dr ** 2 + dc ** 2)
                if not (min_d <= dist <= max_d):
                    continue

                if use_rules and (target_dr != 0.0 or target_dc != 0.0):
                    step_norm = dist + 1e-9
                    cos_sim_global = (dr * target_dr + dc * target_dc) / step_norm
                    if cos_sim_global < hard_cos_threshold:
                        continue

                if use_rules and len(sequence) >= (CONFIG["MAX_STEPS"] // 2):
                    sr, sc = sequence[0]
                    dist_to_start = math.sqrt((nr - sr) ** 2 + (nc - sc) ** 2)
                    if dist_to_start < CONFIG["ANTI_LOOP_START_RADIUS"]:
                        continue

                if use_rules and len(sequence) >= 2:
                    pr, pc = sequence[-2]
                    curr_r, curr_c = sequence[-1]

                    v1 = (curr_r - pr, curr_c - pc)
                    v2 = (nr - curr_r, nc - curr_c)

                    norm1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
                    norm2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
                    if norm1 > 0 and norm2 > 0:
                        cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (norm1 * norm2)
                        if cos_angle < MIN_TURN_ANGLE_COS:
                            continue

                if use_rules and len(sequence) >= 3:
                    if self._check_segment_intersection(sequence, (nr, nc)):
                        continue

                candidates.append((nr, nc))

        if use_rules and len(candidates) > CONFIG["MAX_CANDIDATES"] and len(sequence) >= 2:
            candidates = self._sort_by_direction_consistency(candidates, sequence)
            candidates = candidates[:CONFIG["MAX_CANDIDATES"]]
        elif len(candidates) > CONFIG["MAX_CANDIDATES"]:
            candidates = random.sample(candidates, CONFIG["MAX_CANDIDATES"])

        return candidates

    def encode(self, maps, sequence, current_pos, existing_net, budget_used, direction_hint):
        cand_tensor, mask, candidates, should_terminate = self._encode_candidates(
            maps, sequence, current_pos, existing_net, budget_used, direction_hint
        )
        pos_map = torch.zeros((self.H, self.W), device=self.device)
        pos_map[current_pos] = 1.0
        hist_map = torch.zeros((self.H, self.W), device=self.device)
        for r, c in sequence: hist_map[r, c] = 1.0
        model_config = get_model_config()
        if not model_config.get("use_six_channel_encoder", True):
            spatial = maps['poi'].unsqueeze(0)
        elif not model_config.get("use_global_state_module", True):
            zero_map = torch.zeros_like(maps['poi'])
            spatial = torch.stack([zero_map, zero_map, zero_map, zero_map, zero_map, zero_map], dim=0)
        else:
            spatial = torch.stack([maps['poi'], maps['price'], maps['skeleton'] + existing_net,
                                   pos_map, hist_map, maps['forbidden']], dim=0)
        return spatial, cand_tensor, candidates, mask, should_terminate

    def _check_segment_intersection(self, sequence, new_pos):
        if len(sequence) < 3:
            return False

        p1, p2 = sequence[-1], new_pos

        for i in range(len(sequence) - 3):
            p3, p4 = sequence[i], sequence[i + 1]
            if self._segments_intersect(p1, p2, p3, p4):
                return True
        return False

    def _segments_intersect(self, p1, p2, p3, p4):
        def cross(a, b):
            return a[0] * b[1] - a[1] * b[0]

        def direction(a, b, c):
            return cross((c[0] - a[0], c[1] - a[1]), (b[0] - a[0], b[1] - a[1]))

        d1 = direction(p3, p4, p1)
        d2 = direction(p3, p4, p2)
        d3 = direction(p1, p2, p3)
        d4 = direction(p1, p2, p4)

        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
                ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False

    def _sort_by_direction_consistency(self, candidates, sequence):
        if len(sequence) < 2:
            return candidates

        pr, pc = sequence[-2]
        cr, cc = sequence[-1]

        v1 = (cr - pr, cc - pc)
        norm1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)

        if norm1 == 0:
            return candidates

        scored_candidates = []
        for (nr, nc) in candidates:
            v2 = (nr - cr, nc - cc)
            norm2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

            if norm2 == 0:
                scored_candidates.append((0, (nr, nc)))
            else:
                cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (norm1 * norm2)
                scored_candidates.append((cos_angle, (nr, nc)))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [cand for _, cand in scored_candidates]

    def _encode_candidates(self, maps, sequence, current_pos, existing_net, budget_used, direction_hint):
        candidates = []
        cr, cc = current_pos
        min_d, max_d = CONFIG["MIN_STATION_DIST_GRID"], CONFIG["MAX_STATION_DIST_GRID"]
        target_dr, target_dc = parse_direction_hint(direction_hint)
        hard_cos_threshold = float(CONFIG.get("GLOBAL_DIR_MIN_COS_HARD", 0.1))
        use_rules = get_model_config().get("use_geometric_candidate_rules", True)

        if not use_rules:
            radius = int(math.ceil(max_d))
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < self.H and 0 <= nc < self.W):
                        continue
                    dist = math.sqrt(dr ** 2 + dc ** 2)
                    if min_d <= dist <= max_d:
                        candidates.append((nr, nc))
            if len(candidates) > CONFIG["MAX_CANDIDATES"]:
                candidates = random.sample(candidates, CONFIG["MAX_CANDIDATES"])
        else:
            radius = int(math.ceil(max_d))
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < self.H and 0 <= nc < self.W):
                        continue
                    if use_rules and maps['forbidden'][nr, nc] == 1.0:
                        continue
                    if use_rules and (nr, nc) in sequence:
                        continue
                    dist = math.sqrt(dr ** 2 + dc ** 2)
                    if not (min_d <= dist <= max_d):
                        continue

                    if use_rules and (target_dr != 0.0 or target_dc != 0.0):
                        if (dr * target_dr + dc * target_dc) / (dist + 1e-9) < hard_cos_threshold:
                            continue

                    if use_rules and len(sequence) >= (CONFIG["MAX_STEPS"] // 2):
                        if math.sqrt((nr - sequence[0][0]) ** 2 + (nc - sequence[0][1]) ** 2) < CONFIG[
                            "ANTI_LOOP_START_RADIUS"]:
                            continue

                    if use_rules and len(sequence) >= 2:
                        v1 = (sequence[-1][0] - sequence[-2][0], sequence[-1][1] - sequence[-2][1])
                        v2 = (nr - sequence[-1][0], nc - sequence[-1][1])
                        n1, n2 = math.sqrt(v1[0] ** 2 + v1[1] ** 2), math.sqrt(v2[0] ** 2 + v2[1] ** 2)
                        if n1 > 0 and n2 > 0 and (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2) < CONFIG[
                            "MIN_TURN_ANGLE_COS"]:
                            continue

                    if use_rules and len(sequence) >= 3:
                        if self._check_segment_intersection(sequence, (nr, nc)):
                            continue

                    candidates.append((nr, nc))

            if use_rules and len(candidates) > CONFIG["MAX_CANDIDATES"] and len(sequence) >= 2:
                candidates = self._sort_by_direction_consistency(candidates, sequence)
                candidates = candidates[:CONFIG["MAX_CANDIDATES"]]
            elif len(candidates) > CONFIG["MAX_CANDIDATES"]:
                candidates = random.sample(candidates, CONFIG["MAX_CANDIDATES"])

        if not candidates:
            return torch.zeros((1, 5), device=self.device), torch.ones(1, dtype=torch.bool, device=self.device), [
                (-1, -1)], False

        feats, mask = [], []
        for r, c in candidates:
            dist = math.sqrt((r - current_pos[0]) ** 2 + (c - current_pos[1]) ** 2)
            est_cost = CONFIG["BUDGET_PER_STATION"] + (dist * CONFIG["GRID_SIZE_KM"] * CONFIG["BUDGET_PER_KM"])
            poi = maps['poi'][r, c].item()
            is_tr = 1.0 if (maps['skeleton'][r, c] > 0 or existing_net[r, c] > 0) else 0.0
            dir_cos = (float(r - current_pos[0]) * target_dr + float(c - current_pos[1]) * target_dc) / (dist + 1e-9)
            feats.append([(r - current_pos[0]) / 10.0, (c - current_pos[1]) / 10.0, poi, is_tr, dir_cos])
            if get_model_config().get("use_action_masking", True):
                mask.append(budget_used + est_cost <= CONFIG["TOTAL_BUDGET"])
            else:
                mask.append(True)

        cand_tensor = torch.tensor(feats, dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)
        cand_tensor = torch.cat([cand_tensor, torch.zeros((1, 5), device=self.device)], dim=0)
        mask_tensor = torch.cat([mask_tensor, torch.ones(1, dtype=torch.bool, device=self.device)], dim=0)
        candidates.append((-1, -1))

        return cand_tensor, mask_tensor, candidates, False


class MetroExpansionEnv(gym.Env):
    def __init__(self, processor, start_pos, existing_net, direction_hint, w_td: float = None, w_eq: float = None):
        self.processor, self.maps = processor, processor.maps
        self.existing_net, self.start_pos = existing_net, start_pos
        self.direction_hint = direction_hint
        self.encoder = HybridStateEncoder(CONFIG["DEVICE"])
        self.reward_calc = RewardCalculatorASOC(self.maps, w_td, w_eq)
        self.reset()

    def reset(self, seed=None):
        if seed: set_seed(seed)
        self.curr_pos, self.sequence = self.start_pos, [self.start_pos]
        self.budget_used, self.steps, self.force_terminate = 0.0, 0, False
        self.ep_stats = {'r_td': 0.0, 'r_eq': 0.0, 'transfers': 0}
        self.reward_calc.reset_episode(self.start_pos, self.existing_net)
        return self._get_obs()

    def _get_obs(self):
        spatial, cand_tensor, self.curr_candidates, self.curr_mask, self.force_terminate = self.encoder.encode(
            self.maps, self.sequence, self.curr_pos, self.existing_net, self.budget_used, self.direction_hint)
        return {
            'spatial': spatial.detach().cpu().numpy(),
            'candidates': cand_tensor.detach().cpu().numpy(),
            'mask': self.curr_mask.detach().cpu().numpy()
        }

    def step(self, action_idx):
        if self.force_terminate:
            return self._get_obs(), 0.0, True, False, {'ep_stats': self.ep_stats, 'route': self.sequence}
        if action_idx < 0 or action_idx >= len(self.curr_candidates):
            return self._get_obs(), CONFIG["PENALTY_INVALID"], True, False, {'ep_stats': self.ep_stats,
                                                                             'route': self.sequence}

        next_pos = self.curr_candidates[action_idx]
        if next_pos == (-1, -1):
            return self._get_obs(), 0.0, True, False, {'ep_stats': self.ep_stats, 'route': self.sequence,
                                                       'terminated_by_stop': True}

        est_cost = CONFIG["BUDGET_PER_STATION"] + (
                math.sqrt((next_pos[0] - self.curr_pos[0]) ** 2 + (next_pos[1] - self.curr_pos[1]) ** 2) * CONFIG[
            "GRID_SIZE_KM"] * CONFIG["BUDGET_PER_KM"])

        use_action_masking = get_model_config().get("use_action_masking", True)
        use_budget_constraint = get_model_config().get("use_budget_constraint", True)
        if use_budget_constraint and self.budget_used + est_cost > CONFIG["TOTAL_BUDGET"]:
            return self._get_obs(), CONFIG["PENALTY_OVER_BUDGET"], True, False, {'ep_stats': self.ep_stats,
                                                                                 'route': self.sequence}
        if use_action_masking and not self.curr_mask[action_idx]:
            return self._get_obs(), CONFIG["PENALTY_INVALID"], True, False, {'ep_stats': self.ep_stats,
                                                                             'route': self.sequence}

        reward, info = self.reward_calc.compute_step_reward(next_pos, self.curr_pos, self.existing_net,
                                                            self.direction_hint, self.sequence)

        self.budget_used += info['cost']
        self.steps += 1
        self.ep_stats['r_td'] += info['r_td']
        self.ep_stats['r_eq'] += info['r_eq']
        self.ep_stats['transfers'] += info['is_transfer']
        self.sequence.append(next_pos)
        self.curr_pos = next_pos

        done = (self.steps >= CONFIG["MAX_STEPS"]) or self.force_terminate
        if use_budget_constraint:
            done = done or (self.budget_used >= CONFIG["TOTAL_BUDGET"])
        return self._get_obs(), reward, done, False, {'ep_stats': self.ep_stats, 'route': self.sequence}


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.BatchNorm2d(channels)
        )

    def forward(self, x): return F.relu(x + self.block(x))


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query, self.key, self.value = nn.Conv2d(in_dim, in_dim // 8, 1), nn.Conv2d(in_dim, in_dim // 8,
                                                                                        1), nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, C, H, W = x.size()
        att = F.softmax(torch.bmm(self.query(x).view(b, -1, W * H).permute(0, 2, 1), self.key(x).view(b, -1, W * H)),
                        dim=-1)
        return self.gamma * torch.bmm(self.value(x).view(b, -1, W * H), att.permute(0, 2, 1)).view(b, C, H, W) + x


class SpatialCNN(nn.Module):
    def __init__(self, in_ch=6, channels=[32, 64, 128]):
        super().__init__()
        self.use_spatial_cnn = get_model_config().get("use_spatial_cnn", True)
        self.use_attention = get_model_config().get("use_attention", True)

        layers, curr_ch = [], in_ch
        for idx, ch in enumerate(channels):
            layers.extend([nn.Conv2d(curr_ch, ch, 3, 2, 1), nn.BatchNorm2d(ch), nn.ReLU(), ResidualBlock(ch)])
            if idx == len(channels) - 1 and self.use_attention: layers.append(SelfAttention(ch))
            curr_ch = ch
        self.net, self.out_dim = nn.Sequential(*layers), channels[-1] * 23 * 23

    def forward(self, x):
        out = self.net(x)
        return out.flatten(1)


class SpatialPoolMLP(nn.Module):

    def __init__(self, in_ch=6, hidden_dim=256, pooled_hw=(23, 23)):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pooled_hw)
        pooled_dim = in_ch * pooled_hw[0] * pooled_hw[1]
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(pooled_dim, 512),
            nn.ReLU(),
            nn.Linear(512, hidden_dim),
            nn.ReLU()
        )
        self.out_dim = hidden_dim

    def forward(self, x):
        return self.net(self.pool(x))


class CandidateLSTM(nn.Module):
    def __init__(self, in_dim=5, hidden_dim=256):
        super().__init__()
        self.use_sequence_module = get_model_config().get("use_candidate_sequence_module", True)
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU()) if self.use_sequence_module else None
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True) if self.use_sequence_module else None
        self.raw_pool = nn.Linear(in_dim, hidden_dim) if not self.use_sequence_module else None
        self.raw_action = nn.Linear(in_dim, hidden_dim) if not self.use_sequence_module else None

    def forward(self, x, mask):
        if self.use_sequence_module:
            out = self.mlp(x)
            out, _ = self.lstm(out)
        else:
            out = self.raw_action(x)
            raw_pool = (x * mask.unsqueeze(-1).float()).sum(1) / (
                    mask.unsqueeze(-1).float().sum(1) + 1e-8
            )
            return self.raw_pool(raw_pool), out
        return (out * mask.unsqueeze(-1).float()).sum(1) / (mask.unsqueeze(-1).float().sum(1) + 1e-8), out


class ReplayBuffer:
    def __init__(self, capacity=10000): self.buffer = deque(maxlen=capacity)

    def push(self, *args): self.buffer.append(args)

    def sample(self, bs): return random.sample(self.buffer, bs)

    def __len__(self): return len(self.buffer)


class HybridActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_config = get_model_config()
        self.use_global_state_module = self.model_config.get("use_global_state_module", True)
        self.use_spatial_cnn = self.model_config.get("use_spatial_cnn", True)
        self.use_feature_coupling = self.model_config.get("use_feature_coupling", True)
        self.use_action_masking = self.model_config.get("use_action_masking", True)
        self.use_spatial_branch = self.use_global_state_module
        spatial_channels = 6 if self.model_config.get("use_six_channel_encoder", True) else 1
        if self.use_spatial_branch:
            if self.use_spatial_cnn:
                self.spatial_net = SpatialCNN(
                    in_ch=spatial_channels, channels=CONFIG["CNN_CHANNELS"]
                )
            else:
                self.spatial_net = SpatialPoolMLP(
                    in_ch=spatial_channels,
                    hidden_dim=CONFIG["HIDDEN_DIM"],
                    pooled_hw=(23, 23)
                )
        else:
            self.spatial_net = None
        self.cand_net = CandidateLSTM(in_dim=5, hidden_dim=CONFIG["HIDDEN_DIM"])
        if self.use_spatial_branch and self.use_feature_coupling:
            fusion_dim = self.spatial_net.out_dim + CONFIG["HIDDEN_DIM"]
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim, 512), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(512, CONFIG["HIDDEN_DIM"]), nn.ReLU()
            )
        elif self.use_spatial_branch:
            self.spatial_value = nn.Sequential(
                nn.Linear(self.spatial_net.out_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            self.candidate_value = nn.Sequential(
                nn.Linear(CONFIG["HIDDEN_DIM"], 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
            self.fusion = None
        else:
            self.fusion = nn.Identity()
        self.critic = nn.Sequential(nn.Linear(CONFIG["HIDDEN_DIM"], 128), nn.ReLU(), nn.Linear(128, 1))
        self.actor_proj = nn.Linear(CONFIG["HIDDEN_DIM"], CONFIG["HIDDEN_DIM"])
        self.score_net = nn.Linear(CONFIG["HIDDEN_DIM"], 1)

    def forward(self, sp, cand, mask):
        cand_pool, cand_seq = self.cand_net(cand, mask)
        if self.use_action_masking and mask is not None and mask.sum(dim=1).min() == 0:
            return torch.full((sp.size(0), cand.size(1)), -1e9, device=sp.device), torch.zeros(sp.size(0), 1,
                                                                                               device=sp.device)
        if self.use_spatial_branch and self.use_feature_coupling:
            sp_feat = self.spatial_net(sp)
            fused = self.fusion(torch.cat([sp_feat, cand_pool], dim=1))
            val = self.critic(fused)
            logits = self.score_net(self.actor_proj(fused).unsqueeze(1) + cand_seq).squeeze(-1)
        elif self.use_spatial_branch:
            sp_feat = self.spatial_net(sp)
            val = self.spatial_value(sp_feat) + self.candidate_value(cand_pool)
            logits = self.score_net(cand_seq).squeeze(-1)
        else:
            fused = self.fusion(cand_pool)
            val = self.critic(fused)
            logits = self.score_net(self.actor_proj(fused).unsqueeze(1) + cand_seq).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
        if self.use_action_masking and mask is not None:
            logits = logits - logits.max(dim=-1, keepdim=True)[0]
        return logits, val


class PPOAgent:
    def __init__(self):
        self.use_ppo_update = get_model_config().get("use_ppo_update", True)
        self.policy = HybridActorCritic().to(CONFIG["DEVICE"])
        self.optimizer = optim.Adam(self.policy.parameters(), lr=CONFIG["LR_ACTOR"])
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3000, gamma=0.9)
        self.policy_old = None
        if self.use_ppo_update:
            self.policy_old = HybridActorCritic().to(CONFIG["DEVICE"])
            self.policy_old.load_state_dict(self.policy.state_dict())
        self.mse_loss = nn.MSELoss()

    def select_action(self, obs):
        sp = torch.tensor(obs['spatial'], dtype=torch.float32).unsqueeze(0).to(CONFIG["DEVICE"])
        cand = torch.tensor(obs['candidates'], dtype=torch.float32).unsqueeze(0).to(CONFIG["DEVICE"])
        mask = torch.tensor(obs['mask'], dtype=torch.bool).unsqueeze(0).to(CONFIG["DEVICE"])
        if cand.size(1) == 0:
            select_policy = self.policy_old if self.use_ppo_update else self.policy
            with torch.no_grad(): _, val = select_policy(sp, torch.zeros((1, 1, 5), device=CONFIG["DEVICE"]),
                                                         torch.zeros((1, 1), dtype=torch.bool,
                                                                     device=CONFIG["DEVICE"]))
            return -1, 0.0, val.item()
        with torch.no_grad():
            select_policy = self.policy_old if self.use_ppo_update else self.policy
            logits, val = select_policy(sp, cand, mask)
            if (logits == -1e9).all(): return -1, 0.0, val.item()
            dist = Categorical(logits=logits)
            action = dist.sample()
        return action.item(), dist.log_prob(action).item(), val.item()

    def update(self, memory):
        old_sp = torch.stack([torch.tensor(s, dtype=torch.float32) for s in memory['spatial']]).to(CONFIG["DEVICE"])
        max_cand = max(c.shape[0] for c in memory['candidates'])
        old_cand = torch.stack(
            [torch.tensor(np.pad(c, ((0, max_cand - c.shape[0]), (0, 0))), dtype=torch.float32).to(CONFIG["DEVICE"]) for
             c in memory['candidates']])
        old_mask = torch.stack(
            [torch.tensor(np.pad(m, (0, max_cand - m.shape[0])), dtype=torch.bool).to(CONFIG["DEVICE"]) for m in
             memory['masks']])
        old_act = torch.tensor(memory['actions'], device=CONFIG["DEVICE"])
        old_logp = torch.tensor(memory['logprobs'], device=CONFIG["DEVICE"])
        old_val = torch.tensor(memory['values'], device=CONFIG["DEVICE"])
        rewards, dones = memory['rewards'], memory['dones']

        advs, gae, next_v = [], 0, 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + CONFIG["GAMMA"] * next_v * (1.0 - int(dones[t])) - old_val[t].item()
            gae = delta + CONFIG["GAMMA"] * CONFIG["LAMBDA_GAE"] * (1.0 - int(dones[t])) * gae
            advs.insert(0, gae)
            next_v = old_val[t].item()
        advs = torch.tensor(advs, dtype=torch.float32, device=CONFIG["DEVICE"])
        returns = advs + old_val
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        total_actor_loss, total_critic_loss, batch_count = 0, 0, 0

        update_epochs = CONFIG["K_EPOCHS"] if self.use_ppo_update else 1
        for _ in range(update_epochs):
            indices = np.random.permutation(len(old_act))
            for start in range(0, len(old_act), CONFIG["MINI_BATCH_SIZE"]):
                idx = indices[start: start + CONFIG["MINI_BATCH_SIZE"]]
                logits, vals = self.policy(old_sp[idx], old_cand[idx], old_mask[idx])
                dist = Categorical(logits=logits)
                if self.use_ppo_update:
                    ratios = torch.exp(dist.log_prob(old_act[idx]) - old_logp[idx])
                    surr1 = ratios * advs[idx]
                    surr2 = torch.clamp(ratios, 1 - CONFIG["EPS_CLIP"], 1 + CONFIG["EPS_CLIP"]) * advs[idx]
                    loss_actor = -torch.min(surr1, surr2).mean() - CONFIG["ENTROPY_COEF"] * dist.entropy().mean()
                else:
                    logp = dist.log_prob(old_act[idx])
                    loss_actor = -(logp * advs[idx].detach()).mean() - CONFIG["ENTROPY_COEF"] * dist.entropy().mean()
                loss_critic = 0.5 * self.mse_loss(vals.squeeze(-1), returns[idx])
                loss = loss_actor + loss_critic

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), CONFIG["MAX_GRAD_NORM"])
                self.optimizer.step()

                total_actor_loss += loss_actor.item()
                total_critic_loss += loss_critic.item()
                batch_count += 1

        if self.use_ppo_update:
            self.policy_old.load_state_dict(self.policy.state_dict())
        self.scheduler.step()
        return total_actor_loss / max(1, batch_count), total_critic_loss / max(1, batch_count)


class DQN_Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_net, self.cand_net = SpatialCNN(in_ch=6, channels=CONFIG["CNN_CHANNELS"]), CandidateLSTM(in_dim=5,
                                                                                                              hidden_dim=
                                                                                                              CONFIG[
                                                                                                                  "HIDDEN_DIM"])
        self.q_net = nn.Sequential(nn.Linear(self.spatial_net.out_dim + CONFIG["HIDDEN_DIM"], 512), nn.ReLU(),
                                   nn.Linear(512, CONFIG["HIDDEN_DIM"]), nn.ReLU(), nn.Linear(CONFIG["HIDDEN_DIM"], 1))

    def forward(self, sp, cand, mask):
        sp_feat = self.spatial_net(sp)
        _, cand_seq = self.cand_net(cand, mask)
        q_values = self.q_net(
            torch.cat([sp_feat.unsqueeze(1).expand(-1, cand_seq.size(1), -1), cand_seq], dim=-1)).squeeze(-1)
        if mask is not None: q_values = q_values.masked_fill(~mask, -1e9)
        return q_values


class DQNAgent:
    def __init__(self):
        self.q_net, self.target_net = DQN_Network().to(CONFIG["DEVICE"]), DQN_Network().to(CONFIG["DEVICE"])
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=CONFIG["LR_ACTOR"])
        self.memory = ReplayBuffer(20000)
        self.epsilon, self.epsilon_decay, self.epsilon_min = 1.0, 0.9995, 0.0

    def select_action(self, obs):
        sp, cand, mask = [torch.tensor(obs[k]).unsqueeze(0).to(CONFIG["DEVICE"]) for k in
                          ['spatial', 'candidates', 'mask']]
        if cand.size(1) == 0: return -1
        if random.random() < self.epsilon:
            valid_indices = torch.where(mask[0])[0].tolist()
            return random.choice(valid_indices) if valid_indices else -1
        with torch.no_grad():
            q_values = self.q_net(sp, cand, mask)
            return -1 if (q_values == -1e9).all() else torch.argmax(q_values, dim=1).item()

    def update(self):
        if len(self.memory) < CONFIG["MINI_BATCH_SIZE"]: return 0.0
        batch = self.memory.sample(CONFIG["MINI_BATCH_SIZE"])

        sp_batch = torch.stack([torch.tensor(b[0]['spatial']) for b in batch]).to(CONFIG["DEVICE"])
        max_cand = max(b[0]['candidates'].shape[0] for b in batch)
        cand_batch = torch.stack(
            [torch.tensor(np.pad(b[0]['candidates'], ((0, max_cand - b[0]['candidates'].shape[0]), (0, 0)))) for b in
             batch]).to(
            CONFIG["DEVICE"])
        mask_batch = torch.stack(
            [torch.tensor(np.pad(b[0]['mask'], (0, max_cand - b[0]['mask'].shape[0]))) for b in batch]).to(
            CONFIG["DEVICE"])

        act_batch = torch.tensor([b[1] for b in batch], dtype=torch.int64).unsqueeze(1).to(CONFIG["DEVICE"])
        rew_batch = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(CONFIG["DEVICE"])

        nxt_sp_batch = torch.stack([torch.tensor(b[3]['spatial']) for b in batch]).to(CONFIG["DEVICE"])
        max_nxt_cand = max(b[3]['candidates'].shape[0] for b in batch)
        nxt_cand_batch = torch.stack(
            [torch.tensor(np.pad(b[3]['candidates'], ((0, max_nxt_cand - b[3]['candidates'].shape[0]), (0, 0)))) for b
             in
             batch]).to(CONFIG["DEVICE"])
        nxt_mask_batch = torch.stack(
            [torch.tensor(np.pad(b[3]['mask'], (0, max_nxt_cand - b[3]['mask'].shape[0]))) for b in batch]).to(
            CONFIG["DEVICE"])
        done_batch = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(CONFIG["DEVICE"])

        curr_q = self.q_net(sp_batch, cand_batch, mask_batch).gather(1,
                                                                     torch.clamp(act_batch, 0, max_cand - 1)).squeeze(1)

        with torch.no_grad():
            nxt_q_values = self.target_net(nxt_sp_batch, nxt_cand_batch, nxt_mask_batch)
            nxt_q_values = nxt_q_values.masked_fill(~nxt_mask_batch, -1e9)
            max_nxt_q = nxt_q_values.max(dim=1)[0]
            max_nxt_q[max_nxt_q == -1e9] = 0.0
            target_q = rew_batch + CONFIG["GAMMA"] * max_nxt_q * (1 - done_batch)

        loss = nn.MSELoss()(curr_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for tp, p in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(0.001 * p.data + 0.999 * tp.data)

        return loss.item()


class DDPG_Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_net = SpatialCNN(in_ch=6, channels=CONFIG["CNN_CHANNELS"])
        self.fc = nn.Sequential(nn.Linear(self.spatial_net.out_dim, 256), nn.ReLU(), nn.Linear(256, 5), nn.Tanh())

    def forward(self, sp):
        return self.fc(self.spatial_net(sp))


class DDPG_Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_net = SpatialCNN(in_ch=6, channels=CONFIG["CNN_CHANNELS"])
        self.cand_net = CandidateLSTM(in_dim=5, hidden_dim=CONFIG["HIDDEN_DIM"])
        self.q_net = nn.Sequential(
            nn.Linear(self.spatial_net.out_dim + CONFIG["HIDDEN_DIM"] + 5, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, sp, cand, mask, action_weights):
        sp_feat = self.spatial_net(sp)
        cand_pool, _ = self.cand_net(cand, mask)
        return self.q_net(torch.cat([sp_feat, cand_pool, action_weights], dim=-1))


class DDPGAgent:
    def __init__(self):
        self.actor, self.critic = DDPG_Actor().to(CONFIG["DEVICE"]), DDPG_Critic().to(CONFIG["DEVICE"])

        self.target_actor = DDPG_Actor().to(CONFIG["DEVICE"])
        self.target_critic = DDPG_Critic().to(CONFIG["DEVICE"])
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=CONFIG["LR_ACTOR"])
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=CONFIG["LR_CRITIC"])
        self.memory = ReplayBuffer(20000)

        self.noise_scale = 0.2
        self.noise_decay = 0.9996
        self.noise_min = 0.0

    def select_action(self, obs):
        sp = torch.tensor(obs['spatial'], dtype=torch.float32).unsqueeze(0).to(CONFIG["DEVICE"])
        candidates = obs['candidates']
        mask = obs['mask']

        if len(candidates) == 0:
            return -1, np.zeros(5)

        with torch.no_grad():
            weights = self.actor(sp).cpu().numpy()[0]

            explore_prob = self.noise_scale * 0.25

            if random.random() < explore_prob:
                continuous_weights = np.random.uniform(-0.2, 1.0, 5)
            else:
                if self.noise_scale > 0:
                    noise = np.random.normal(0, self.noise_scale, 5)
                else:
                    noise = np.zeros(5)
                continuous_weights = np.clip(weights + noise, -1.0, 1.0)

        best_idx = -1
        max_score = -float('inf')

        for i, c in enumerate(candidates):
            if not mask[i]: continue
            score = np.dot(c, continuous_weights)
            if score > max_score:
                max_score = score
                best_idx = i

        if best_idx == -1: best_idx = len(candidates) - 1

        return best_idx, continuous_weights

    def update(self):
        if len(self.memory) < CONFIG["MINI_BATCH_SIZE"]: return 0.0, 0.0
        batch = self.memory.sample(CONFIG["MINI_BATCH_SIZE"])

        sp_batch = torch.stack([torch.tensor(b[0]['spatial'], dtype=torch.float32) for b in batch]).to(CONFIG["DEVICE"])
        max_cand = max(b[0]['candidates'].shape[0] for b in batch)
        cand_batch = torch.stack([torch.tensor(
            np.pad(b[0]['candidates'], ((0, max_cand - b[0]['candidates'].shape[0]), (0, 0))), dtype=torch.float32) for
            b in batch]).to(CONFIG["DEVICE"])
        mask_batch = torch.stack(
            [torch.tensor(np.pad(b[0]['mask'], (0, max_cand - b[0]['mask'].shape[0])), dtype=torch.bool) for b in
             batch]).to(CONFIG["DEVICE"])

        act_batch = torch.stack([torch.tensor(b[1], dtype=torch.float32) for b in batch]).to(CONFIG["DEVICE"])

        rew_batch = torch.tensor([b[2] for b in batch], dtype=torch.float32).unsqueeze(1).to(CONFIG["DEVICE"])

        nxt_sp_batch = torch.stack([torch.tensor(b[3]['spatial'], dtype=torch.float32) for b in batch]).to(
            CONFIG["DEVICE"])
        max_nxt_cand = max(b[3]['candidates'].shape[0] for b in batch)
        nxt_cand_batch = torch.stack([torch.tensor(
            np.pad(b[3]['candidates'], ((0, max_nxt_cand - b[3]['candidates'].shape[0]), (0, 0))), dtype=torch.float32)
            for b in batch]).to(CONFIG["DEVICE"])
        nxt_mask_batch = torch.stack(
            [torch.tensor(np.pad(b[3]['mask'], (0, max_nxt_cand - b[3]['mask'].shape[0])), dtype=torch.bool) for b in
             batch]).to(CONFIG["DEVICE"])

        done_batch = torch.tensor([b[4] for b in batch], dtype=torch.float32).unsqueeze(1).to(CONFIG["DEVICE"])

        with torch.no_grad():
            nxt_actions = self.target_actor(nxt_sp_batch)
            nxt_q = self.target_critic(nxt_sp_batch, nxt_cand_batch, nxt_mask_batch, nxt_actions)
            target_q = rew_batch + CONFIG["GAMMA"] * nxt_q * (1 - done_batch)

        current_q = self.critic(sp_batch, cand_batch, mask_batch, act_batch)
        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_actions = self.actor(sp_batch)
        actor_loss = -self.critic(sp_batch, cand_batch, mask_batch, actor_actions).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        for tp, p in zip(self.target_actor.parameters(), self.actor.parameters()):
            tp.data.copy_(0.001 * p.data + 0.999 * tp.data)
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(0.001 * p.data + 0.999 * tp.data)

        return actor_loss.item(), critic_loss.item()


def evaluate_full_route_baseline(route, reward_calc, existing_net, dir_hint):
    tot_rt, tot_td, tot_eq, budget, tot_tr = 0.0, 0.0, 0.0, 0.0, 0

    if route:
        reward_calc.reset_episode(route[0], existing_net)

    for i in range(1, len(route)):
        rt, info = reward_calc.compute_step_reward(
            route[i], route[i - 1], existing_net, dir_hint, route[:i]
        )
        tot_rt += rt
        tot_td += info['r_td']
        tot_eq += info['r_eq']
        budget += info['cost']
        tot_tr += info['is_transfer']

    return tot_rt, tot_td, tot_eq, budget, tot_tr


class GA_Runner:
    def __init__(self, encoder: HybridStateEncoder, reward_calc: RewardCalculatorASOC,
                 start_pos: Tuple[int, int], existing_net: torch.Tensor, dir_hint: str):
        self.encoder = encoder
        self.reward_calc = reward_calc
        self.start_pos = start_pos
        self.existing_net = existing_net
        self.dir_hint = dir_hint
        self.maps = reward_calc.maps

    def _random_walk(self, start_route=None):
        route = start_route.copy() if start_route else [self.start_pos]
        budget = 0.0

        self.reward_calc.reset_episode(route[0], self.existing_net)

        if len(route) > 1:
            _, _, _, budget, _ = evaluate_full_route_baseline(route, self.reward_calc, self.existing_net, self.dir_hint)

        for _ in range(CONFIG["MAX_STEPS"] - len(route) + 1):
            _, _, candidates, mask, should_terminate = self.encoder.encode(
                self.maps, route, route[-1], self.existing_net, budget, self.dir_hint
            )

            if should_terminate or len(candidates) == 0 or not mask.any():
                break

            valid_cands = [c for c, m in zip(candidates, mask.tolist()) if m]
            if not valid_cands:
                break

            nxt = random.choice(valid_cands)
            _, info = self.reward_calc.compute_step_reward(nxt, route[-1], self.existing_net, self.dir_hint, route)

            if budget + info['cost'] > CONFIG["TOTAL_BUDGET"]:
                break

            budget += info['cost']
            route.append(nxt)
        return route

    def _is_valid_route(self, route):
        for i in range(1, len(route)):
            r1, c1 = route[i - 1]
            r2, c2 = route[i]
            dist = math.sqrt((r2 - r1) ** 2 + (c2 - c1) ** 2)
            if not (CONFIG["MIN_STATION_DIST_GRID"] <= dist <= CONFIG["MAX_STATION_DIST_GRID"]):
                return False
        return True

    def _mutate(self, route):
        if len(route) <= 2:
            return self._random_walk()
        mutate_point = random.randint(1, len(route) - 1)
        new_route = route[:mutate_point]
        new_child = self._random_walk(start_route=new_route)
        if self._is_valid_route(new_child):
            return new_child
        return route

    def run(self):
        print(f"\n[{get_timestamp()}] 🧬 开始 GA 算法演化计算...")
        pop = [self._random_walk() for _ in range(CONFIG["GA_POP_SIZE"])]
        best_overall_route = []
        best_overall_rtotal = -float('inf')

        for gen in range(1, CONFIG["GA_GENERATIONS"] + 1):
            new_pop = pop.copy()

            for _ in range(CONFIG["GA_POP_SIZE"] // 2):
                p1, p2 = random.sample(pop, 2)
                common = list(set(p1) & set(p2))
                if common:
                    cp = random.choice(common)
                    child = p1[:p1.index(cp)] + p2[p2.index(cp):]

                    if (
                            len(set(child)) == len(child)
                            and len(child) <= CONFIG["MAX_STEPS"] + 1
                            and self._is_valid_route(child)
                    ):
                        _, _, _, cost, _ = evaluate_full_route_baseline(child, self.reward_calc, self.existing_net,
                                                                        self.dir_hint)
                        if cost <= CONFIG["TOTAL_BUDGET"]:
                            new_pop.append(child)

            for i in range(len(pop)):
                if random.random() < 0.2:
                    mutated_child = self._mutate(pop[i])
                    new_pop.append(mutated_child)

            scored_pop = []
            seen_routes = set()

            for ind in new_pop:
                ind_tuple = tuple(ind)
                if ind_tuple in seen_routes:
                    continue
                seen_routes.add(ind_tuple)

                rt, td, eq, cost, tr = evaluate_full_route_baseline(ind, self.reward_calc, self.existing_net,
                                                                    self.dir_hint)
                if cost <= CONFIG["TOTAL_BUDGET"]:
                    scored_pop.append({'route': ind, 'rt': rt, 'td': td, 'eq': eq, 'tr': tr, 'cost': cost})

            scored_pop.sort(key=lambda x: x['rt'], reverse=True)

            while len(scored_pop) < CONFIG["GA_POP_SIZE"]:
                new_ind = self._random_walk()
                rt, td, eq, cost, tr = evaluate_full_route_baseline(new_ind, self.reward_calc, self.existing_net,
                                                                    self.dir_hint)
                if cost <= CONFIG["TOTAL_BUDGET"]:
                    scored_pop.append({'route': new_ind, 'rt': rt, 'td': td, 'eq': eq, 'tr': tr, 'cost': cost})

            scored_pop.sort(key=lambda x: x['rt'], reverse=True)
            pop = [x['route'] for x in scored_pop[:CONFIG["GA_POP_SIZE"]]]

            gen_best = scored_pop[0] if scored_pop else {'rt': -float('inf'), 'td': 0, 'eq': 0, 'tr': 0, 'route': []}
            if gen_best['rt'] > best_overall_rtotal:
                best_overall_rtotal = gen_best['rt']
                best_overall_route = gen_best['route']

            if gen % 20 == 0 and scored_pop:
                print(
                    f"[{get_timestamp()}] GA epoch {gen:4d} | Rtotal {gen_best['rt']:7.2f} | "
                    f"Rtd {gen_best['td']:6.2f} | Req {gen_best['eq']:6.2f} | "
                    f"Len {len(gen_best['route']):2d} | Tr {gen_best['tr']:2d} | "
                    f"Budget {gen_best['cost']:6.2f}")

        return best_overall_route if best_overall_route else [self.start_pos]


class MP_Runner:
    def __init__(self, encoder: HybridStateEncoder, reward_calc: RewardCalculatorASOC,
                 start_pos: Tuple[int, int], existing_net: torch.Tensor, dir_hint: str):
        self.encoder = encoder
        self.reward_calc = reward_calc
        self.start_pos = start_pos
        self.existing_net = existing_net
        self.dir_hint = dir_hint
        self.maps = reward_calc.maps
        self.H, self.W = self.maps['poi'].shape

    def run(self):
        print(f"\n[{get_timestamp()}] 🧮 开始 MP 算法 (严格整数线性规划 ILP - Gurobi)...")
        nodes = set([self.start_pos])
        edges = {}
        node_rewards = {}

        target_dr, target_dc = parse_direction_hint(self.dir_hint)
        hard_cos_threshold = float(CONFIG.get("GLOBAL_DIR_MIN_COS_HARD", 0.1))

        queue = [(self.start_pos, 0.0, 0)]
        MAX_NODES = CONFIG["MP_MAX_NODES"]
        visited_edges = set()

        while queue and len(nodes) < MAX_NODES:
            curr, curr_cost, steps = queue.pop(0)
            if steps >= CONFIG["MAX_STEPS"]:
                continue

            cr, cc = curr
            min_d, max_d = CONFIG["MIN_STATION_DIST_GRID"], CONFIG["MAX_STATION_DIST_GRID"]
            radius = int(math.ceil(max_d))

            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < self.H and 0 <= nc < self.W): continue
                    if self.maps['forbidden'][nr, nc] == 1.0: continue

                    dist = math.sqrt(dr ** 2 + dc ** 2)
                    if not (min_d <= dist <= max_d): continue

                    if target_dr != 0.0 or target_dc != 0.0:
                        step_norm = dist + 1e-9
                        cos_sim = (dr * target_dr + dc * target_dc) / step_norm
                        if cos_sim < hard_cos_threshold: continue

                    step_cost = CONFIG["BUDGET_PER_STATION"] + (dist * CONFIG["GRID_SIZE_KM"] * CONFIG["BUDGET_PER_KM"])
                    if curr_cost + step_cost > CONFIG["TOTAL_BUDGET"]: continue

                    nxt = (nr, nc)
                    if (curr, nxt) not in visited_edges:
                        visited_edges.add((curr, nxt))
                        edges[(curr, nxt)] = step_cost
                        nodes.add(nxt)
                        queue.append((nxt, curr_cost + step_cost, steps + 1))

                        if nxt not in node_rewards:
                            is_tr = 1.0 if (self.maps['skeleton'][nr, nc] > 0 or self.existing_net[nr, nc] > 0) else 0.0
                            epsilon = CONFIG["EPSILON_INTERCHANGE"] if is_tr else CONFIG["EPSILON_REGULAR"]
                            r_td = epsilon * self.maps['poi'][nr, nc].item() * 5.0
                            r_eq = self.maps['price'][nr, nc].item() * 5.0
                            node_rewards[nxt] = self.reward_calc.w_td * r_td + self.reward_calc.w_eq * r_eq

        print(f"[{get_timestamp()}] ILP 子图生成完毕: 包含 {len(nodes)} 个可行网格节点，{len(edges)} 条有向可行边。")
        if not edges:
            return [self.start_pos]

        try:
            m = gp.Model("Metro_Network_Expansion_ILP")
            m.setParam('OutputFlag', 0)
            m.setParam('TimeLimit', 60.0)
            m.setParam('MIPGap', 0.05)

            x_vars = m.addVars(edges.keys(), vtype=GRB.BINARY, name="x")
            y_vars = m.addVars(nodes, vtype=GRB.BINARY, name="y")
            step_vars = m.addVars(nodes, lb=0, ub=CONFIG["MAX_STEPS"], vtype=GRB.INTEGER, name="step")

            m.setObjective(gp.quicksum(y_vars[v] * node_rewards.get(v, 0.0) for v in nodes), GRB.MAXIMIZE)

            m.addConstr(y_vars[self.start_pos] == 1, name="start_node_select")
            m.addConstr(step_vars[self.start_pos] == 0, name="start_node_step")
            m.addConstr(gp.quicksum(x_vars[e] * edges[e] for e in edges.keys()) <= CONFIG["TOTAL_BUDGET"],
                        name="total_budget")
            m.addConstr(gp.quicksum(x_vars[e] for e in edges.keys()) <= CONFIG["MAX_STEPS"], name="max_steps")

            for v in nodes:
                in_edges = [e for e in edges.keys() if e[1] == v]
                out_edges = [e for e in edges.keys() if e[0] == v]

                if v == self.start_pos:
                    m.addConstr(gp.quicksum(x_vars[e] for e in in_edges) == 0, name=f"flow_in_{v}")
                    m.addConstr(gp.quicksum(x_vars[e] for e in out_edges) <= 1, name=f"flow_out_{v}")
                else:
                    m.addConstr(gp.quicksum(x_vars[e] for e in in_edges) == y_vars[v], name=f"flow_in_{v}")
                    m.addConstr(gp.quicksum(x_vars[e] for e in out_edges) <= y_vars[v], name=f"flow_out_{v}")

            M = CONFIG["MAX_STEPS"] + 1
            for (u, v) in edges.keys():
                m.addConstr(step_vars[v] >= step_vars[u] + 1 - M * (1 - x_vars[(u, v)]), name=f"mtz_{u}_{v}")

            print(f"[{get_timestamp()}] 调用 Gurobi 求解器 (限时 60 秒)......")
            m.optimize()

            if m.Status == GRB.OPTIMAL or m.Status == GRB.TIME_LIMIT or m.Status == GRB.SUBOPTIMAL:
                if m.SolCount > 0:
                    route = [self.start_pos]
                    curr = self.start_pos
                    for _ in range(CONFIG["MAX_STEPS"]):
                        next_node = None
                        for (u, v) in edges.keys():
                            if u == curr and x_vars[(u, v)].X > 0.5:
                                next_node = v
                                break
                        if next_node:
                            route.append(next_node)
                            curr = next_node
                        else:
                            break
                    return route
                else:
                    return [self.start_pos]
            else:
                return [self.start_pos]

        except gp.GurobiError as e:
            return [self.start_pos]
        except Exception as e:
            return [self.start_pos]


class NetworkEvaluator:
    def __init__(self, processor):
        self.proc = processor
        self.H = CONFIG["Grid_H"]
        self.W = CONFIG["Grid_W"]
        self.radius = CONFIG["COVERAGE_RADIUS_GRID"]

    def _build_full_network(self, new_route: List[Tuple[int, int]]):
        G = nx.Graph()
        interchange_stations = set()

        for line_name, stations in CONFIG["PRESET_LINES"].items():
            prev_grid = None
            for st in stations:
                if st in self.proc.station_to_grid:
                    grid = self.proc.station_to_grid[st]
                    G.add_node(grid, type='existing')
                    if prev_grid is not None and prev_grid != grid:
                        dist = math.sqrt((grid[0] - prev_grid[0]) ** 2 + (grid[1] - prev_grid[1]) ** 2)
                        G.add_edge(prev_grid, grid, weight=dist)
                    prev_grid = grid

        prev_grid = None
        for grid in new_route:
            if grid in G.nodes:
                interchange_stations.add(grid)
                G.nodes[grid]['type'] = 'interchange'
            else:
                G.add_node(grid, type='new')

            if prev_grid is not None and prev_grid != grid:
                dist = math.sqrt((grid[0] - prev_grid[0]) ** 2 + (grid[1] - prev_grid[1]) ** 2)
                G.add_edge(prev_grid, grid, weight=dist)
            prev_grid = grid

        return G, interchange_stations

    def eval_service_capacity(self, G: nx.Graph, interchange_stations: set):
        poi_map = self.proc.maps['poi']
        station_loads = {node: 0.0 for node in G.nodes}
        total_capacity = 0.0

        stations = list(G.nodes)
        nonzero_pois = torch.nonzero(poi_map > 0)
        for r, c in nonzero_pois:
            r, c = r.item(), c.item()
            v_j = poi_map[r, c].item()

            nearby_stations = []
            for st in stations:
                dist = math.sqrt((st[0] - r) ** 2 + (st[1] - c) ** 2)
                if dist <= self.radius:
                    epsilon = CONFIG["EPSILON_INTERCHANGE"] if st in interchange_stations else CONFIG["EPSILON_REGULAR"]
                    nearby_stations.append((st, epsilon, 1.0 / (dist + 1e-4)))

            if not nearby_stations:
                continue

            denominator = sum(item[2] for item in nearby_stations)
            for st, epsilon, inv_dist in nearby_stations:
                v_ij = epsilon * (inv_dist / denominator) * v_j
                station_loads[st] += v_ij
                total_capacity += v_ij

        return total_capacity, station_loads

    def eval_vulnerability(self, G: nx.Graph, station_loads: dict, alpha=0.3, num_attacks=3):
        if len(G) < 2: return 0.0

        def calc_efficiency_and_pairs(graph):
            pairs = 0
            efficiency = 0.0
            for component in nx.connected_components(graph):
                comp_len = len(component)
                if comp_len > 1:
                    pairs += comp_len * (comp_len - 1) / 2
                    subgraph = graph.subgraph(component)
                    efficiency += nx.global_efficiency(subgraph) * (comp_len * (comp_len - 1))

            n = len(graph)
            global_eff = efficiency / (n * (n - 1)) if n > 1 else 0
            return pairs, global_eff

        K_0, E_0 = calc_efficiency_and_pairs(G)
        if K_0 == 0 or E_0 == 0: return 1.0

        current_loads = {node: max(load, 1e-5) for node, load in station_loads.items()}
        capacities = {node: (1 + alpha) * load for node, load in current_loads.items()}

        G_attack = G.copy()
        degrees = dict(G_attack.degree())
        attack_targets = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:num_attacks]

        failed_nodes = set(attack_targets)
        newly_failed = list(attack_targets)

        while newly_failed:
            current_failed = newly_failed.copy()
            newly_failed = []

            for failed_node in current_failed:
                neighbors = [n for n in G.neighbors(failed_node) if n not in failed_nodes]
                if not neighbors: continue

                total_neighbor_load = sum(station_loads[n] for n in neighbors)
                if total_neighbor_load == 0: continue

                load_to_distribute = current_loads[failed_node]
                for n in neighbors:
                    delta_A = load_to_distribute * (station_loads[n] / total_neighbor_load)
                    current_loads[n] += delta_A

                    if current_loads[n] >= capacities[n]:
                        failed_nodes.add(n)
                        newly_failed.append(n)

            G_attack.remove_nodes_from(current_failed)

        K_a, E_a = calc_efficiency_and_pairs(G_attack)
        mu = 0.5
        TEL = mu * (1 - K_a / K_0) + (1 - mu) * (1 - E_a / E_0)

        return TEL


class ExperimentRunner:
    def __init__(self):
        apply_publication_plot_style()
        ensure_dir(CONFIG["OUTPUT_DIR"])
        self.proc = DataProcessor()
        self.global_net = self.proc.maps['skeleton'].clone()
        self.encoder = HybridStateEncoder(CONFIG["DEVICE"])

    def run_ppo(self, start_pos, dir_hint, w_td, w_eq, weight_name):
        model_name = get_model_name()
        model_tag = make_file_tag(model_name)
        print(f"[{get_timestamp()}] Start PPO training | weight={weight_name} | model={model_name}")
        print(f"[{get_timestamp()}] ⚙️ 开始训练 PPO (核心算法)...")
        env = MetroExpansionEnv(self.proc, start_pos, self.global_net, dir_hint, w_td, w_eq)
        agent, best_route, best_r = PPOAgent(), [], -float('inf')
        memory = {k: [] for k in
                  ['spatial', 'candidates', 'masks', 'actions', 'logprobs', 'values', 'rewards', 'dones']}
        timestep = 0

        history = {'episodes': [], 'rewards': [], 'actor_losses': [], 'critic_losses': []}
        last_a_loss, last_c_loss = 0.0, 0.0

        for ep in range(1, CONFIG["TOTAL_EPISODES"] + 1):
            obs, done, ep_r = env.reset(), False, 0
            info = {'ep_stats': env.ep_stats, 'route': env.sequence}
            terminated_by_budget = False
            while not done:
                timestep += 1
                act, logp, val = agent.select_action(obs)
                if act == -1:
                    terminated_by_budget = True
                    break
                nxt_obs, r, done, _, info = env.step(act)
                for k, v in zip(memory.keys(),
                                [obs['spatial'], obs['candidates'], obs['mask'], act, logp, val, r, done]):
                    memory[k].append(v)
                obs, ep_r = nxt_obs, ep_r + r

                if timestep % CONFIG["UPDATE_TIMESTEPS"] == 0:
                    last_a_loss, last_c_loss = agent.update(memory)
                    for k in memory: memory[k] = []

            history['episodes'].append(ep)
            history['rewards'].append(ep_r)
            history['actor_losses'].append(last_a_loss)
            history['critic_losses'].append(last_c_loss)

            if ep_r > best_r: best_r, best_route = ep_r, info['route']
            if ep % 50 == 0:
                stats = info['ep_stats']
                status = "BUDGET" if terminated_by_budget else (
                    "STOP" if ("terminated_by_stop" in info) else "MAX_STEP")
                print(f"[{get_timestamp()}] PPO[{weight_name}/{model_name}] epoch {ep:4d} | Rtotal {ep_r:7.2f} | "
                      f"Rtd {stats['r_td']:6.2f} | Req {stats['r_eq']:6.2f} | "
                      f"Len {len(info['route']):2d} | Tr {stats['transfers']:2d} | "
                      f"Budget {env.budget_used:6.2f} | {status}")

        self._plot_single_alg_curves(history, f"PPO_{model_tag}", weight_name)

        return best_route, best_r, history

    def _plot_single_alg_curves(self, history, alg_name, weight_name):
        output_dir = CONFIG["OUTPUT_DIR"]
        eps = history['episodes']
        sigma = min(100, max(1, len(history['rewards']) // 10))

        plt.figure(figsize=(12, 7.5))
        plt.plot(eps, history['rewards'], color='royalblue', alpha=0.25, linewidth=1.2, label='Episode reward')
        trend = gaussian_filter1d(history['rewards'], sigma=sigma)
        plt.plot(eps, trend, color='darkorange', linewidth=3.2, label='Smoothed trend')
        plt.xlabel('Episode', fontsize=18)
        plt.ylabel('Average reward', fontsize=18)
        plt.title(f'{alg_name} Training Reward Performance ({weight_name})', fontsize=18)
        plt.tick_params(axis='both', labelsize=15)
        plt.legend(frameon=True, fontsize=15)
        plt.grid(True, linestyle='--', alpha=0.45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Fig9_{alg_name}_Reward_{weight_name}.png"), dpi=600, bbox_inches='tight')
        plt.close()

        fig, ax1 = plt.subplots(figsize=(12, 7.5))

        if alg_name == "DQN":
            color_loss = 'crimson'
            ax1.plot(eps, history['losses'], color=color_loss, alpha=0.85, linewidth=2.8, label='Q-network loss')
            ax1.set_xlabel('Episode', fontsize=18)
            ax1.set_ylabel('Loss value', fontsize=18, color=color_loss)
            ax1.tick_params(axis='both', labelsize=15)
            ax1.tick_params(axis='y', labelcolor=color_loss)
            ax1.grid(True, linestyle='--', alpha=0.45)
            ax1.legend(loc='upper right', frameon=True, fontsize=15)
            plt.title(f'{alg_name} Loss Convergence ({weight_name})', fontsize=18)
        else:
            color_critic = 'crimson'
            line1 = ax1.plot(eps, history['critic_losses'], color=color_critic, alpha=0.85, linewidth=2.8,
                             label='Critic loss')
            ax1.set_xlabel('Episode', fontsize=18)
            ax1.set_ylabel('Critic loss value', fontsize=18, color=color_critic)
            ax1.tick_params(axis='both', labelsize=15)
            ax1.tick_params(axis='y', labelcolor=color_critic)
            ax1.grid(True, linestyle='--', alpha=0.45)

            ax2 = ax1.twinx()
            color_actor = 'forestgreen'
            line2 = ax2.plot(eps, history['actor_losses'], color=color_actor, alpha=0.85, linewidth=2.8,
                             label='Actor loss')
            ax2.set_ylabel('Actor loss value', fontsize=18, color=color_actor)
            ax2.tick_params(axis='y', labelsize=15, labelcolor=color_actor)

            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right', frameon=True, fontsize=15)
            plt.title(f'{alg_name} Actor-Critic Loss Convergence ({weight_name})', fontsize=18)

        fig.tight_layout()
        plt.savefig(os.path.join(output_dir, f"Fig10_{alg_name}_Loss_{weight_name}.png"), dpi=600, bbox_inches='tight')
        plt.close()

    def _plot_combined_rewards(self, histories_dict, weight_name):

        plt.figure(figsize=(12, 7.5))
        colors = {'PPO': 'darkorange', 'DQN': 'seagreen', 'DDPG': 'crimson'}

        for alg, hist in histories_dict.items():
            eps = hist['episodes']
            rewards_raw = hist['rewards']

            sigma = min(100, max(1, len(rewards_raw) // 10))
            trend = gaussian_filter1d(rewards_raw, sigma=sigma)
            plt.plot(eps, rewards_raw, color=colors[alg], linewidth=0.8, alpha=0.12)
            plt.plot(eps, trend, color=colors[alg], linewidth=3.2, label=f'{alg} trend')

        plt.xlabel('Episode', fontsize=18)
        plt.ylabel('Average reward', fontsize=18)
        plt.title(f'RL Algorithms Reward Comparison ({weight_name})', fontsize=18)
        plt.tick_params(axis='both', labelsize=15)
        plt.legend(loc='lower right', frameon=True, fontsize=15)
        plt.grid(True, linestyle='--', alpha=0.45)
        plt.tight_layout()
        plt.savefig(os.path.join(CONFIG["OUTPUT_DIR"], f"Fig_Combined_Reward_{weight_name}.png"), dpi=600,
                    bbox_inches='tight')
        plt.close()
        print(f"[{get_timestamp()}] Combined reward comparison figure saved for weight {weight_name}.")

    def _plot_algorithm_metric_comparison(self, results, line_name, weight_name):
        algorithms = ["PPO", "DQN", "DDPG", "GA", "MP"]
        labels = ["Rtotal", "Rtd", "Req", "Service capacity", "TEL vulnerability"]
        keys = ["rtotal", "rtd", "req", "service_cap", "tel_vul"]
        values = []
        for key in keys:
            values.append([float(results.get(alg, {}).get(key, np.nan)) for alg in algorithms])

        fig, axes = plt.subplots(1, len(keys), figsize=(22, 6.5))
        colors = ["#d95f02", "#1b9e77", "#7570b3", "#66a61e", "#e7298a"]
        for ax, title, row in zip(axes, labels, values):
            plot_values = [v if np.isfinite(v) else 0.0 for v in row]
            bars = ax.bar(algorithms, plot_values, color=colors, edgecolor="#333333", linewidth=0.8)
            ax.set_title(title, fontsize=15)
            ax.tick_params(axis="x", labelrotation=35, labelsize=11)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            for bar, value in zip(bars, row):
                if np.isfinite(value):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{value:.2f}", ha="center", va="bottom", fontsize=9)
        fig.suptitle(f"Algorithm Comparison ({line_name}, {weight_name})", fontsize=18)
        fig.tight_layout()
        path = os.path.join(CONFIG["OUTPUT_DIR"],
                            f"Fig_Algorithm_Metric_Comparison_{make_file_tag(line_name)}_{weight_name}.png")
        fig.savefig(path, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"[{get_timestamp()}] Algorithm metric comparison figure saved to: {path}")

    def run_dqn(self, start_pos, dir_hint, w_td, w_eq, weight_name):
        print(f"[{get_timestamp()}] ⚙️ 开始训练 DQN (对比基线)...")
        env = MetroExpansionEnv(self.proc, start_pos, self.global_net, dir_hint, w_td, w_eq)
        agent, best_route, best_r = DQNAgent(), [], -float('inf')

        timestep = 0
        history = {'episodes': [], 'rewards': [], 'losses': []}
        last_loss = 0.0

        for ep in range(1, CONFIG["TOTAL_EPISODES"] + 1):
            obs, done, ep_r = env.reset(), False, 0
            while not done:
                timestep += 1
                act = agent.select_action(obs)
                nxt_obs, r, done, _, info = env.step(act)
                agent.memory.push(obs, act, r, nxt_obs, done)
                obs, ep_r = nxt_obs, ep_r + r

                if len(agent.memory) >= CONFIG["MINI_BATCH_SIZE"] and timestep % CONFIG[
                    "OFF_POLICY_UPDATE_INTERVAL"] == 0:
                    last_loss = agent.update()

            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            history['episodes'].append(ep)
            history['rewards'].append(ep_r)
            history['losses'].append(last_loss)

            if ep_r > best_r: best_r, best_route = ep_r, info['route']
            if ep % 50 == 0:
                stats = info['ep_stats']
                print(f"[{get_timestamp()}] DQN[{weight_name}] epoch {ep:4d} | Rtotal {ep_r:7.2f} | "
                      f"Rtd {stats['r_td']:6.2f} | Req {stats['r_eq']:6.2f} | "
                      f"Len {len(info['route']):2d} | Tr {stats['transfers']:2d} | "
                      f"Budget {env.budget_used:6.2f} | EPS {agent.epsilon:.2f}")

        self._plot_single_alg_curves(history, "DQN", weight_name)

        return best_route, best_r, history

    def run_ddpg(self, start_pos, dir_hint, w_td, w_eq, weight_name):
        print(f"[{get_timestamp()}] ⚙️ 开始训练 DDPG (权重自适应重构版)...")
        env = MetroExpansionEnv(self.proc, start_pos, self.global_net, dir_hint, w_td, w_eq)
        agent, best_route, best_r = DDPGAgent(), [], -float('inf')

        timestep = 0
        history = {'episodes': [], 'rewards': [], 'actor_losses': [], 'critic_losses': []}
        last_a_loss, last_c_loss = 0.0, 0.0

        for ep in range(1, CONFIG["TOTAL_EPISODES"] + 1):
            obs, done, ep_r = env.reset(), False, 0
            while not done:
                timestep += 1
                act_idx, act_weights = agent.select_action(obs)

                nxt_obs, r, done, _, info = env.step(act_idx)

                agent.memory.push(obs, act_weights, r, nxt_obs, done)

                obs, ep_r = nxt_obs, ep_r + r

                if len(agent.memory) >= CONFIG["MINI_BATCH_SIZE"] and timestep % CONFIG[
                    "OFF_POLICY_UPDATE_INTERVAL"] == 0:
                    a_l, c_l = agent.update()
                    last_a_loss = a_l
                    last_c_loss = c_l
            agent.noise_scale = max(agent.noise_min, agent.noise_scale * agent.noise_decay)

            history['episodes'].append(ep)
            history['rewards'].append(ep_r)
            history['actor_losses'].append(last_a_loss)
            history['critic_losses'].append(last_c_loss)

            if ep_r > best_r: best_r, best_route = ep_r, info['route']
            if ep % 50 == 0:
                stats = info['ep_stats']
                print(f"[{get_timestamp()}] DDPG[{weight_name}] epoch {ep:4d} | Rtotal {ep_r:7.2f} | "
                      f"Rtd {stats['r_td']:6.2f} | Req {stats['r_eq']:6.2f} | "
                      f"Len {len(info['route']):2d} | Tr {stats['transfers']:2d} | "
                      f"Budget {env.budget_used:6.2f} | Noise {agent.noise_scale:.2f}")

        self._plot_single_alg_curves(history, "DDPG", weight_name)

        return best_route, best_r, history

    def run_single_weight_config(self, line_cfg: Dict, weight_cfg: Dict) -> Dict[str, Any]:
        w_name, w_td, w_eq, w_desc = weight_cfg['name'], weight_cfg['w_td'], weight_cfg['w_eq'], weight_cfg['desc']
        print(
            f"\n{'=' * 80}\n[{get_timestamp()}] >>> 权重配置: {w_name} ({w_desc}): w_td={w_td}, w_eq={w_eq}\n{'=' * 80}")
        start_pos, dir_hint = self.proc.get_grid_by_name(line_cfg['start']), line_cfg['direction']
        reward_calc = RewardCalculatorASOC(self.proc.maps, w_td, w_eq)

        set_seed(CONFIG["SEED"])
        route_ppo, _, hist_ppo = self.run_ppo(start_pos, dir_hint, w_td, w_eq, w_name)
        set_seed(CONFIG["SEED"])
        route_dqn, _, hist_dqn = self.run_dqn(start_pos, dir_hint, w_td, w_eq, w_name)
        set_seed(CONFIG["SEED"])
        route_ddpg, _, hist_ddpg = self.run_ddpg(start_pos, dir_hint, w_td, w_eq, w_name)
        set_seed(CONFIG["SEED"])
        route_ga = GA_Runner(self.encoder, reward_calc, start_pos, self.global_net, dir_hint).run()
        set_seed(CONFIG["SEED"])
        route_mp = MP_Runner(self.encoder, reward_calc, start_pos, self.global_net, dir_hint).run()

        training_histories = {"PPO": hist_ppo, "DQN": hist_dqn, "DDPG": hist_ddpg}
        self._plot_combined_rewards(training_histories, w_name)

        results = {
            "weight_name": w_name, "w_td": w_td, "w_eq": w_eq,
            "PPO": {"route": route_ppo, "source": "PPO_Core"},
            "DQN": {"route": route_dqn, "source": "DQN"},
            "DDPG": {"route": route_ddpg, "source": "DDPG"},
            "GA": {"route": route_ga, "source": "GA"},
            "MP": {"route": route_mp, "source": "MP"},
            "training_histories": training_histories,
        }

        evaluator = NetworkEvaluator(self.proc)

        print(f"\n[{get_timestamp()}] 🏆 权重配置 [{w_name}] 的综合评估报告")
        print(
            f"{'-' * 125}\n| {'Algorithm':<10} | {'Rtotal':<9} | {'Rtd':<8} | {'Req':<8} | {'Len':<4} | {'Tr':<3} | {'Budget':<8} | {'Service':<9} | {'TEL(Vul)':<8} | {'Status':<8} |\n{'-' * 125}")

        for alg in ["PPO", "DQN", "DDPG", "GA", "MP"]:
            route = results[alg]["route"]

            rt, td, eq, cost, tr = evaluate_full_route_baseline(route, reward_calc, self.global_net, dir_hint)
            budget_ok = cost <= CONFIG["TOTAL_BUDGET"]

            if len(route) > 1:
                G_net, interchange_sts = evaluator._build_full_network(route)
                service_cap, station_loads = evaluator.eval_service_capacity(G_net, interchange_sts)
                tel_vul = evaluator.eval_vulnerability(G_net, station_loads, alpha=0.3, num_attacks=3)
            else:
                service_cap, tel_vul = 0.0, 1.0

            results[alg].update({
                "rtotal": rt, "rtd": td, "req": eq, "cost": cost, "transfers": tr,
                "len": len(route), "budget_ok": budget_ok,
                "service_cap": service_cap, "tel_vul": tel_vul
            })

            print(
                f"| {alg:<10} | {rt:9.2f} | {td:8.2f} | {eq:8.2f} | {len(route):4d} | {tr:3d} | {cost:8.2f} | {service_cap:9.1f} | {tel_vul:8.3f} | {'✓ OK' if budget_ok else '✗ OVER!':<8} |")

            self._save_route_files(route, f"{line_cfg['name']}_{alg}_{w_name}", w_name, w_td, w_eq, cost, budget_ok)

        print(f"{'-' * 125}")
        self._plot_algorithm_metric_comparison(results, line_cfg['name'], w_name)
        return results

    def _get_route_style_colors(self, name: str):
        if '3' in name or 'NE1' in name:
            return '#ff7f0e', '#b35900', '#ffbc80'
        if '5' in name or 'NE2' in name:
            return '#2ca02c', '#145c14', '#8cd98c'
        return '#ff7f0e', '#b35900', '#ffbc80'

    def run_all(self):
        print(f"\n[{get_timestamp()}] Starting complete-algorithm experiment")
        all_results = []
        for line_cfg in CONFIG["LINE_CONFIG"]:
            print(f"\n[{get_timestamp()}] Line: {line_cfg['name']}")
            line_results = [self.run_single_weight_config(line_cfg, w_cfg) for w_cfg in CONFIG["WEIGHT_CONFIGS"]]
            all_results.append({"line": line_cfg["name"], "weight_results": line_results})
        return all_results

    def _save_route_files(self, sequence, name: str, weight_name: str = "", w_td: float = None,
                          w_eq: float = None, budget_used: float = None, budget_ok: bool = None):
        geojson_coords = []
        folium_coords = []

        for r, c in sequence:
            lon, lat = self.proc.grid_to_lonlat(r, c)
            if lon != 0.0 and lat != 0.0:
                geojson_coords.append([lon, lat])
                folium_coords.append([lat, lon])

        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": geojson_coords},
                "properties": {
                    "line_name": name, "weight_config": weight_name, "w_td": w_td, "w_eq": w_eq,
                    "station_count": len(sequence), "budget_used": budget_used,
                    "budget_limit": CONFIG["TOTAL_BUDGET"], "budget_ok": budget_ok
                }
            }]
        }
        with open(os.path.join(CONFIG["OUTPUT_DIR"], f"{name}_route.geojson"), 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False)

        if folium_coords:
            avg_lat = sum(c[0] for c in folium_coords) / len(folium_coords)
            avg_lon = sum(c[1] for c in folium_coords) / len(folium_coords)
            center = [avg_lat, avg_lon]

            def draw_folium_base(m_obj):
                for line_name, stations in CONFIG["PRESET_LINES"].items():
                    line_color = CONFIG["PRESET_LINE_COLORS"].get(line_name, "#000000")
                    line_coords = []
                    for st in stations:
                        if st in self.proc.station_to_grid:
                            r, c = self.proc.station_to_grid[st]
                            s_lon, s_lat = self.proc.grid_to_lonlat(r, c)
                            if s_lon != 0.0 and s_lat != 0.0:
                                line_coords.append([s_lat, s_lon])
                                folium.CircleMarker(
                                    location=[s_lat, s_lon], radius=3.5, color=line_color,
                                    weight=1.8, fill=True, fill_color=line_color, fill_opacity=0.9,
                                    tooltip=f"Existing station: {st} ({line_name})"
                                ).add_to(m_obj)
                    if len(line_coords) >= 2:
                        folium.PolyLine(
                            line_coords, color=line_color, weight=4, opacity=0.75,
                            tooltip=f"Existing metro corridor: {line_name}"
                        ).add_to(m_obj)

                route_color, fill_color = '#808080', '#cccccc'
                if "PPO" in name:
                    route_color, fill_color = '#ff5733', '#ff8c00'

                folium.PolyLine(
                    folium_coords, color=route_color, weight=8, opacity=0.95,
                    tooltip=f"Generated route: {name}"
                ).add_to(m_obj)
                for i, (lat, lon) in enumerate(folium_coords):
                    folium.CircleMarker(
                        location=[lat, lon], radius=6.5, color='white', weight=2.0, fill=True,
                        fill_color=fill_color, fill_opacity=1.0, tooltip=f'Generated station {i}'
                    ).add_to(m_obj)

            m_bg = folium.Map(location=center, zoom_start=11, tiles='OpenStreetMap', control_scale=True)
            if MiniMap is not None:
                MiniMap(toggle_display=True, position='bottomright').add_to(m_bg)
            if MeasureControl is not None:
                MeasureControl(position='topleft', primary_length_unit='kilometers').add_to(m_bg)
            draw_folium_base(m_bg)
            folium.map.Marker(
                center,
                icon=folium.DivIcon(html="""
                <div style="font-size:14px; background:white; padding:6px 8px; border:1px solid #555; box-shadow:0 1px 4px rgba(0,0,0,0.25);">
                Spatial reference: WGS84 (EPSG:4326)<br>
                Grid resolution: 1 km x 1 km<br>
                Scale bar: bottom-left map control
                </div>
                """)
            ).add_to(m_bg)
            folium.LayerControl(collapsed=False).add_to(m_bg)
            m_bg.save(os.path.join(CONFIG["OUTPUT_DIR"], f"{name}_map_with_bg.html"))

            m_nobg = folium.Map(location=center, zoom_start=11, tiles=None, control_scale=True)
            if MeasureControl is not None:
                MeasureControl(position='topleft', primary_length_unit='kilometers').add_to(m_nobg)
            draw_folium_base(m_nobg)
            m_nobg.save(os.path.join(CONFIG["OUTPUT_DIR"], f"{name}_map_pure_network.html"))

        all_r_coords = []
        all_c_coords = []

        for line_name, stations in CONFIG["PRESET_LINES"].items():
            for st in stations:
                if st in self.proc.station_to_grid:
                    r, c = self.proc.station_to_grid[st]
                    all_r_coords.append(r)
                    all_c_coords.append(c)

        if sequence:
            for r, c in sequence:
                all_r_coords.append(r)
                all_c_coords.append(c)

        padding = 2
        if all_r_coords and all_c_coords:
            min_r = max(0, min(all_r_coords) - padding)
            max_r = min(CONFIG["Grid_H"], max(all_r_coords) + padding)
            min_c = max(0, min(all_c_coords) - padding)
            max_c = min(CONFIG["Grid_W"], max(all_c_coords) + padding)
        else:
            min_r, max_r = 0, CONFIG["Grid_H"]
            min_c, max_c = 0, CONFIG["Grid_W"]

        def draw_matplotlib_img(show_grid=True):
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.set_xlim(min_c, max_c)
            ax.set_ylim(min_r, max_r)

            if show_grid:
                ax.xaxis.set_major_locator(MultipleLocator(5))
                ax.xaxis.set_minor_locator(MultipleLocator(1))
                ax.yaxis.set_major_locator(MultipleLocator(5))
                ax.yaxis.set_minor_locator(MultipleLocator(1))
                ax.grid(which='major', color='#00bfff', linestyle='-', linewidth=0.9)
                ax.grid(which='minor', color='#00bfff', linestyle='-', linewidth=0.35)
                ax.set_xlabel("Grid X (1 km cells)", fontsize=18)
                ax.set_ylabel("Grid Y (1 km cells)", fontsize=18)
                ax.set_title(f"Metro Network Expansion - {name}\nWeight: {weight_name} (w_td={w_td}, w_eq={w_eq})",
                             fontsize=18)
                ax.tick_params(axis='both', labelsize=15)
                for spine in ax.spines.values():
                    spine.set_color('#333333')
                    spine.set_linewidth(1.6)
            else:
                ax.axis('off')

            existing_plotted = False
            for line_name, stations in CONFIG["PRESET_LINES"].items():
                r_coords = []
                c_coords = []
                for st in stations:
                    if st in self.proc.station_to_grid:
                        r, c = self.proc.station_to_grid[st]
                        r_coords.append(r)
                        c_coords.append(c)

                if r_coords:
                    label = 'Existing Network' if not existing_plotted else ""
                    base_color, edge_color, mid_color = '#1f77b4', '#0d4a75', '#73b3e6'
                    ax.plot(c_coords, r_coords, color=base_color, marker='o', markersize=7.5,
                            linestyle='-', linewidth=1.5, markerfacecolor=base_color,
                            markeredgecolor=edge_color, markeredgewidth=0.8, zorder=2, label=label)
                    ax.plot(c_coords, r_coords, linestyle='none', marker='o', markersize=4.5,
                            markerfacecolor=mid_color, markeredgecolor='none', zorder=3)
                    ax.plot(c_coords, r_coords, linestyle='none', marker='o', markersize=1.5,
                            markerfacecolor='white', markeredgecolor='none', zorder=4)
                    existing_plotted = True

            if sequence:
                gen_r = []
                gen_c = []
                for r, c in sequence:
                    gen_r.append(r)
                    gen_c.append(c)
                route_color, edge_color, mid_color = self._get_route_style_colors(name)
                ax.plot(gen_c, gen_r, color=route_color, marker='o', markersize=7.5,
                        linestyle='-', linewidth=1.5, markerfacecolor=route_color,
                        markeredgecolor=edge_color, markeredgewidth=0.8, zorder=5, label='Generated Route')
                ax.plot(gen_c, gen_r, linestyle='none', marker='o', markersize=4.5,
                        markerfacecolor=mid_color, markeredgecolor='none', zorder=6)
                ax.plot(gen_c, gen_r, linestyle='none', marker='o', markersize=1.5,
                        markerfacecolor='white', markeredgecolor='none', zorder=7)

            if show_grid:
                x_span, y_span = max_c - min_c, max_r - min_r
                scale_len_km = 5
                x0 = min_c + 0.06 * x_span
                y0 = min_r + 0.07 * y_span
                ax.plot([x0, x0 + scale_len_km], [y0, y0], color='black', linewidth=4, solid_capstyle='butt')
                ax.text(x0 + scale_len_km / 2, y0 + 0.025 * y_span, f"{scale_len_km} km",
                        ha='center', va='bottom', fontsize=15, fontweight='bold')
                ax.text(min_c + 0.02 * x_span, max_r - 0.04 * y_span,
                        "Spatial reference: grid coordinates\nGrid resolution: 1 km x 1 km",
                        ha='left', va='top', fontsize=13,
                        bbox=dict(facecolor='white', edgecolor='#555555', alpha=0.85, boxstyle='round,pad=0.35'))
                ax.legend(loc='upper right', fontsize=15, frameon=True, framealpha=0.92)

            suffix = "_grid_map" if show_grid else "_pure_network"
            plt.savefig(os.path.join(CONFIG["OUTPUT_DIR"], f"{name}{suffix}.png"), dpi=600, bbox_inches='tight',
                        transparent=not show_grid)
            plt.close(fig)

        draw_matplotlib_img(show_grid=True)
        draw_matplotlib_img(show_grid=False)


def _collect_result_rows(all_rows, budget, line_name, seed, weight_cfg, result):
    for algorithm in ["PPO", "DQN", "DDPG", "GA", "MP"]:
        metrics = result.get(algorithm, {})
        all_rows.append({
            "Budget": budget["name"], "Budget_Label": budget["label"],
            "Total_Budget": budget["total_budget"], "Line": line_name,
            "Seed": int(seed), "Weight": weight_cfg["name"],
            "w_td": weight_cfg["w_td"], "w_eq": weight_cfg["w_eq"],
            "Algorithm": algorithm,
            "Rtotal": metrics.get("rtotal", np.nan),
            "Rtd": metrics.get("rtd", np.nan), "Req": metrics.get("req", np.nan),
            "Service_Cap": metrics.get("service_cap", np.nan),
            "TEL_Vul": metrics.get("tel_vul", np.nan),
            "Length": metrics.get("len", np.nan),
            "Transfers": metrics.get("transfers", np.nan),
            "Cost": metrics.get("cost", np.nan),
        })


def _build_statistics(rows):
    data = pd.DataFrame(rows)
    metrics = ["Rtotal", "Rtd", "Req", "Service_Cap", "TEL_Vul", "Length", "Transfers", "Cost"]
    group_cols = ["Budget", "Budget_Label", "Total_Budget", "Line", "Weight", "w_td", "w_eq", "Algorithm"]
    statistics = []
    for keys, group in data.groupby(group_cols, dropna=False):
        record = dict(zip(group_cols, keys))
        record["N_Seeds"] = int(group["Seed"].nunique())
        record["Seeds_Used"] = sorted(group["Seed"].astype(int).unique().tolist())
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(dtype=float)
            n = len(values)
            mean = float(np.mean(values)) if n else np.nan
            std = float(np.std(values, ddof=1)) if n > 1 else np.nan
            minimum = float(np.min(values)) if n else np.nan
            maximum = float(np.max(values)) if n else np.nan
            median = float(np.median(values)) if n else np.nan
            ci_low = ci_high = np.nan
            if n > 1 and np.isfinite(std):
                margin = float(student_t.ppf(0.975, n - 1) * std / np.sqrt(n))
                ci_low, ci_high = mean - margin, mean + margin
            record.update({
                f"{metric}_Mean": mean, f"{metric}_Std": std,
                f"{metric}_CI95_Low": ci_low, f"{metric}_CI95_High": ci_high,
                f"{metric}_Min": minimum, f"{metric}_Max": maximum,
                f"{metric}_Range": maximum - minimum if n else np.nan,
                f"{metric}_Median": median,
            })
        statistics.append(record)
    return data, pd.DataFrame(statistics)


def _json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    return value


def _save_statistics_plots(summary, output_root):
    metrics = {
        "Rtotal": "Total reward", "Rtd": "Travel-demand reward", "Req": "Equity reward",
        "Service_Cap": "Service capacity", "TEL_Vul": "Network vulnerability",
    }
    plot_root = Path(output_root) / "统计汇总图"
    for (budget, line, weight), subset in summary.groupby(["Budget", "Line", "Weight"], dropna=False):
        plot_dir = plot_root / str(budget) / str(line) / str(weight)
        plot_dir.mkdir(parents=True, exist_ok=True)
        algorithms = [a for a in ["PPO", "DQN", "DDPG", "GA", "MP"] if a in set(subset["Algorithm"])]
        if not algorithms:
            continue
        for metric, ylabel in metrics.items():
            means = [subset.loc[subset["Algorithm"] == a, f"{metric}_Mean"].iloc[0] for a in algorithms]
            stds = [subset.loc[subset["Algorithm"] == a, f"{metric}_Std"].iloc[0] for a in algorithms]
            mins = [subset.loc[subset["Algorithm"] == a, f"{metric}_Min"].iloc[0] for a in algorithms]
            maxs = [subset.loc[subset["Algorithm"] == a, f"{metric}_Max"].iloc[0] for a in algorithms]
            means = np.asarray(means, dtype=float)
            stds = np.nan_to_num(np.asarray(stds, dtype=float), nan=0.0)
            mins = np.asarray(mins, dtype=float);
            maxs = np.asarray(maxs, dtype=float)
            x = np.arange(len(algorithms))
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(x, means, yerr=stds, capsize=4, color="#d95f02", alpha=0.82)
            ax.vlines(x, mins, maxs, color="#1b4f72", linewidth=3, label="Min-max")
            ax.set_xticks(x, algorithms);
            ax.set_ylabel(ylabel)
            ax.set_title(f"{budget} | {line} | {weight} | {metric}")
            ax.grid(axis="y", linestyle="--", alpha=0.35);
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_dir / f"{metric}_mean_std_range.png", dpi=300)
            plt.close(fig)


def _run_suite():
    base_output = Path(CONFIG["OUTPUT_DIR"]).resolve()
    base_output.mkdir(parents=True, exist_ok=True)
    seeds = list(CONFIG["STATISTICAL_SEEDS"])
    rows = []

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for stream in self.streams: stream.write(data)

        def flush(self):
            for stream in self.streams: stream.flush()

    for budget in CONFIG["BUDGET_CONFIGS"]:
        CONFIG["TOTAL_BUDGET"] = float(budget["total_budget"])
        for line_cfg in CONFIG["LINE_CONFIG"]:
            for seed in seeds:
                CONFIG["SEED"] = int(seed)
                set_seed(seed)
                line_dir_name = str(line_cfg["name"]).removeprefix("Line_")
                run_dir = base_output / budget["label"] / line_dir_name / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                CONFIG["OUTPUT_DIR"] = str(run_dir)
                log_path = run_dir / f"experiment_log_seed{seed}.txt"
                original_stdout = sys.stdout
                with log_path.open("w", encoding="utf-8") as log_file, redirect_stdout(_Tee(original_stdout, log_file)):
                    print(f"[{get_timestamp()}] Budget={budget['label']} Line={line_cfg['name']} Seed={seed}")
                    for weight_cfg in CONFIG["WEIGHT_CONFIGS"]:
                        set_seed(seed)
                        runner = ExperimentRunner()
                        try:
                            result = runner.run_single_weight_config(line_cfg, weight_cfg)
                            _collect_result_rows(rows, budget, line_cfg["name"], seed, weight_cfg, result)
                        finally:
                            del runner
                            if "result" in locals():
                                del result
                            release_runtime_memory()

    data, summary = _build_statistics(rows)
    CONFIG["OUTPUT_DIR"] = str(base_output)
    _save_statistics_plots(summary, base_output)
    records = [{k: _json_safe(v) for k, v in row.items()} for row in summary.to_dict(orient="records")]
    (base_output / "FINAL_STATISTICS_MEAN_STD_CI.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (base_output / "FINAL_ALL_RESULTS_SUMMARY.txt").write_text(
        summary.to_string(index=False), encoding="utf-8"
    )
    config_report = {
        "seeds": seeds,
        "budgets": CONFIG["BUDGET_CONFIGS"],
        "lines": CONFIG["LINE_CONFIG"],
        "weights": CONFIG["WEIGHT_CONFIGS"],
        "algorithms": ["PPO", "DQN", "DDPG", "GA", "MP"],
        "total_episodes": CONFIG["TOTAL_EPISODES"],
        "statistics": ["mean", "sample_std", "95_percent_t_ci", "min", "max", "range", "median"],
        "csv_output": False,
    }
    (base_output / "EXPERIMENT_CONFIGURATION.json").write_text(
        json.dumps(config_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[{get_timestamp()}] Completed {len(seeds)} seeds across {len(CONFIG['BUDGET_CONFIGS'])} budgets, "
          f"{len(CONFIG['LINE_CONFIG'])} lines, and {len(CONFIG['WEIGHT_CONFIGS'])} weights.")
    print(f"[{get_timestamp()}] No experiment CSV files were written.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    _run_suite()
