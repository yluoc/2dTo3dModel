"""
Utility functions for the CNN-based 2D to 3D model project.
"""

from .common_utils import (
    load_obj,
    save_obj,
    preprocess_image,
    tensor_to_numpy,
    numpy_to_tensor,
    create_directories,
    plot_training_curves,
    calculate_vertex_count,
    get_device_info
)

from .data_utils import (
    ShapeDataset,
    get_data_transforms,
    create_dataloader,
    prepare_batch_data,
    calculate_output_vertices,
    get_dataset_info
)

from .model_utils import (
    create_optimizer,
    create_scheduler,
    save_checkpoint,
    load_checkpoint,
    save_model,
    load_model,
    count_parameters,
    count_trainable_parameters,
    get_model_info,
    set_model_mode
)

__all__ = [
    # Common utilities
    'load_obj',
    'save_obj',
    'preprocess_image',
    'tensor_to_numpy',
    'numpy_to_tensor',
    'create_directories',
    'plot_training_curves',
    'calculate_vertex_count',
    'get_device_info',
    
    # Data utilities
    'ShapeDataset',
    'get_data_transforms',
    'create_dataloader',
    'prepare_batch_data',
    'calculate_output_vertices',
    'get_dataset_info',
    
    # Model utilities
    'create_optimizer',
    'create_scheduler',
    'save_checkpoint',
    'load_checkpoint',
    'save_model',
    'load_model',
    'count_parameters',
    'count_trainable_parameters',
    'get_model_info',
    'set_model_mode'
]
