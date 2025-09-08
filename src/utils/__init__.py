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

# data_utils module not available - commenting out
# from .data_utils import (
#     ShapeDataset,
#     get_data_transforms,
#     create_dataloader,
#     prepare_batch_data,
#     calculate_output_vertices,
#     get_dataset_info
# )

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

from .frame_capture import (
    ModelFrameCapture
)

from .dataset_2d3d import (
    Image3DDataset,
    get_data_transforms_2d3d,
    create_2d3d_dataloader,
    collate_fn_2d3d,
    get_dataset_info_2d3d,
    split_dataset
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
    
    # Data utilities (commented out - module not available)
    # 'ShapeDataset',
    # 'get_data_transforms',
    # 'create_dataloader',
    # 'prepare_batch_data',
    # 'calculate_output_vertices',
    # 'get_dataset_info',
    
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
    'set_model_mode',
    
    # Frame capture utilities
    'ModelFrameCapture',
    
    # 2D-3D dataset utilities
    'Image3DDataset',
    'get_data_transforms_2d3d',
    'create_2d3d_dataloader',
    'collate_fn_2d3d',
    'get_dataset_info_2d3d',
    'split_dataset'
]
