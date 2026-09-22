"""Compatibility facade for the original monolithic module."""

from training.train_model_v8_exp import UNIVERSE_PATH, MIN_CROSS_SECTION
from modules.macro_features import load_macro_slim5
from modules.cross_sectional_features import valid_cross_section_dates
from _feature_cache_utils import (get_full_feature_pool_df, get_full_macro_pool_df,
    get_margin_pool_df, get_sector_relative_pool_df)
from _eval_utils import (compute_regime_labels_expanding, evaluate_daily_topn)

from .config import *
from .runtime import *
from .feature_catalog import *
from .losses import *
from .samplers import *
from .targets import *
from .data_builder import *
from .model_factory import *
from .trainer import *
from .inference import *
from .diagnostics import *
from .cache import *
