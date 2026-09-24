def task_render(cfg):
    """Run the render task with render-only dependencies imported lazily.

    Args:
        cfg: Hydra config.

    Returns:
        Render task result.
    """
    from .render import task_render as _task_render

    return _task_render(cfg)


def task_visualize(cfg):
    """Run the visualize task with visualize-only dependencies imported lazily.

    Args:
        cfg: Hydra config.

    Returns:
        Visualize task result.
    """
    from .visualize import task_visualize as _task_visualize

    return _task_visualize(cfg)


def task_visualize_prior_path(cfg):
    """Run the prior-path visualizer with visualize-only dependencies imported lazily.

    Args:
        cfg: Hydra config.

    Returns:
        Prior-path visualize task result.
    """
    from .visualize_prior_path import task_visualize_prior_path as _task_visualize_prior_path

    return _task_visualize_prior_path(cfg)


def task_synthesis(cfg):
    """Run synthesis with synthesis-only dependencies imported lazily.

    Args:
        cfg: Hydra config.

    Returns:
        Synthesis task result.
    """
    from .synthesis import task_synthesis as _task_synthesis

    return _task_synthesis(cfg)
