"""
MDK-LRP1 Nanobody Design Pipeline
Core src module for nanobody design against Midkine
"""

__version__ = "1.0.0"
__author__ = "MDK Nanobody Design Team"

from .prepare_target import (
    TargetProtein,
    EpitopeDefinition,
    prepare_mdk_target,
    define_epitope,
    expand_epitope_region
)

from .design_binders import (
    BinderDesigner,
    BindCraftRunner,
    design_de_novo_binders,
    optimize_sequences,
    filter_designs
)

from .format_nanobody import (
    NanobodyFormatter,
    extract_paratope,
    graft_cdrs,
    create_nanobody,
    optimize_framework
)

from .validate_designs import (
    DesignValidator,
    calculate_developability,
    predict_expression,
    assess_specificity,
    rank_designs
)

__all__ = [
    # Target preparation
    'TargetProtein',
    'EpitopeDefinition',
    'prepare_mdk_target',
    'define_epitope',
    'expand_epitope_region',
    
    # Binder design
    'BinderDesigner',
    'BindCraftRunner',
    'design_de_novo_binders',
    'optimize_sequences',
    'filter_designs',
    
    # Nanobody formatting
    'NanobodyFormatter',
    'extract_paratope',
    'graft_cdrs',
    'create_nanobody',
    'optimize_framework',
    
    # Validation
    'DesignValidator',
    'calculate_developability',
    'predict_expression',
    'assess_specificity',
    'rank_designs'
]