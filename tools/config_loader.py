#!/usr/bin/env python3
"""
Configuration Loader for Drivable Area Detector
================================================
Centralized parameter loading from params.yaml for all Python tools.

ALWAYS loads from params.yaml (required). Historical defaults are kept
for comparison/tracking improvements over time.

Usage:
    from tools.config_loader import ConfigLoader, HISTORICAL_DEFAULTS

    config = ConfigLoader()
    max_intensity = config.max_road_intensity
    max_range = config.max_range

Or load specific node parameters:
    road_params = config.get_road_extraction_params()
    ground_params = config.get_ground_segmentation_params()
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


# ============================================================================
# HISTORICAL DEFAULTS - Previous parameter values for comparison/tracking
# ============================================================================
# These are the original default values before tuning. Keep for comparison
# to measure improvements when changing params.yaml values.

HISTORICAL_DEFAULTS = {
    'road_points_extraction_node': {
        'max_road_intensity': 6.0,        # Original value (before tuning)
        'max_range': 15.0,                # Original value
        'min_neighbors': 5,               # Original value
        'neighbor_radius': 0.8,           # Original value
        'dbscan_eps': 0.052,              # Original value
        'dbscan_min_samples': 30,         # Original value
        'dbscan_min_cluster_size': 50,    # Original value
    },
    'ring_boundary_node': {
        'max_boundary_width': 4.0,        # Original value
        'y_threshold': 0.4,               # Original value
    }
}


# ============================================================================
# Load params.yaml at module import time (REQUIRED)
# ============================================================================

def _load_params_yaml() -> Dict[str, Any]:
    """
    Load params.yaml at module import time.

    Returns the ros__parameters section. Raises FileNotFoundError if params.yaml
    is not found (params.yaml is REQUIRED, not optional).
    """
    default_path = Path('/home/ws/src/drivable_area_detector/config/params.yaml')

    if not default_path.exists():
        raise FileNotFoundError(
            f"params.yaml REQUIRED but not found at {default_path}\n"
            "Ensure the file exists before importing config_loader."
        )

    with open(default_path, 'r') as f:
        raw = yaml.safe_load(f)

    ros_params = raw.get('/**', {}).get('ros__parameters', {})
    return ros_params


# Load params.yaml once at module import
_PARAMS_YAML = _load_params_yaml()
_ROAD_PARAMS = _PARAMS_YAML.get('road_points_extraction_node', {})
_BOUNDARY_PARAMS = _PARAMS_YAML.get('ring_boundary_node', {})


@dataclass
class RoadExtractionParams:
    """
    Parameters for road_points_extraction_node (DBSCAN-based pipeline).

    Values are loaded from params.yaml at import time.
    Use HISTORICAL_DEFAULTS to compare with original values.
    """
    enable_extraction: bool = _ROAD_PARAMS.get('enable_extraction', True)
    min_road_intensity: float = _ROAD_PARAMS.get('min_road_intensity', 0.0)
    max_road_intensity: float = _ROAD_PARAMS.get('max_road_intensity', 8.0)
    max_range: float = _ROAD_PARAMS.get('max_range', 20.0)
    min_neighbors: int = _ROAD_PARAMS.get('min_neighbors', 50)
    neighbor_radius: float = _ROAD_PARAMS.get('neighbor_radius', 0.5)
    # Temporal buffer parameters
    buffer_frames: int = _ROAD_PARAMS.get('buffer_frames', 5)
    min_confirmations: int = _ROAD_PARAMS.get('min_confirmations', 3)
    bin_size: float = _ROAD_PARAMS.get('bin_size', 0.3)
    # DBSCAN clustering parameters
    dbscan_eps: float = _ROAD_PARAMS.get('dbscan_eps', 0.06)
    dbscan_min_samples: int = _ROAD_PARAMS.get('dbscan_min_samples', 10)
    dbscan_min_cluster_size: int = _ROAD_PARAMS.get('dbscan_min_cluster_size', 80)

    def __str__(self):
        return (f"RoadExtractionParams(max_intensity={self.max_road_intensity}, "
                f"max_range={self.max_range}, min_neighbors={self.min_neighbors}, "
                f"neighbor_radius={self.neighbor_radius}, dbscan_eps={self.dbscan_eps})")


@dataclass
class GroundSegmentationParams:
    """Parameters for ground_segmentation_node."""
    ransac_distance_threshold: float = 0.05
    ransac_max_iterations: int = 100
    min_ground_points: int = 50
    z_tolerance: float = 0.3
    
    def __str__(self):
        return (f"GroundSegmentationParams(distance_threshold={self.ransac_distance_threshold}, "
                f"max_iterations={self.ransac_max_iterations})")


@dataclass
class BoundaryDetectionParams:
    """Parameters for boundary_detection_node."""
    grid_resolution: float = 0.5
    min_points_per_cell: int = 5
    boundary_threshold: float = 0.5
    
    def __str__(self):
        return (f"BoundaryDetectionParams(grid_resolution={self.grid_resolution}, "
                f"min_points_per_cell={self.min_points_per_cell})")


class ConfigLoader:
    """
    Centralized configuration loader for all drivable area detector tools.
    
    Automatically finds and loads params.yaml from the standard location.
    Provides type-safe access to all node parameters.
    """
    
    # Default paths to search for params.yaml
    DEFAULT_PATHS = [
        Path('/home/ws/src/drivable_area_detector/config/params.yaml'),
        Path('config/params.yaml'),
        Path('../config/params.yaml'),
        Path('../../config/params.yaml'),
    ]
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the config loader.
        
        Args:
            config_path: Optional explicit path to params.yaml.
                        If None, searches default locations.
        """
        self.config_path = self._find_config_path(config_path)
        self.raw_config: Dict[str, Any] = {}
        self.ros_params: Dict[str, Any] = {}
        
        if self.config_path:
            self._load_config()
        else:
            print("Warning: params.yaml not found, using default parameters")
    
    def _find_config_path(self, explicit_path: Optional[Path]) -> Optional[Path]:
        """Find params.yaml in default locations or use explicit path."""
        if explicit_path:
            if explicit_path.exists():
                return explicit_path
            else:
                print(f"Warning: Specified config path not found: {explicit_path}")
        
        # Search default paths
        for path in self.DEFAULT_PATHS:
            if path.exists():
                return path
        
        return None
    
    def _load_config(self):
        """Load and parse the YAML configuration."""
        try:
            with open(self.config_path, 'r') as f:
                self.raw_config = yaml.safe_load(f)
            
            # Extract ROS parameters
            self.ros_params = self.raw_config.get('/**', {}).get('ros__parameters', {})
            
        except Exception as e:
            print(f"Error loading config from {self.config_path}: {e}")
            self.raw_config = {}
            self.ros_params = {}
    
    def get_road_extraction_params(self) -> RoadExtractionParams:
        """Get parameters for road points extraction node (DBSCAN-based pipeline)."""
        node_params = self.ros_params.get('road_points_extraction_node', {})

        return RoadExtractionParams(
            enable_extraction=node_params.get('enable_extraction', True),
            min_road_intensity=float(node_params.get('min_road_intensity', 0.0)),
            max_road_intensity=float(node_params.get('max_road_intensity', 8.0)),
            max_range=float(node_params.get('max_range', 20.0)),
            min_neighbors=int(node_params.get('min_neighbors', 2)),
            neighbor_radius=float(node_params.get('neighbor_radius', 1.2)),
            # Temporal buffer parameters
            buffer_frames=int(node_params.get('buffer_frames', 5)),
            min_confirmations=int(node_params.get('min_confirmations', 3)),
            bin_size=float(node_params.get('bin_size', 0.3)),
            # DBSCAN clustering parameters
            dbscan_eps=float(node_params.get('dbscan_eps', 0.052)),
            dbscan_min_samples=int(node_params.get('dbscan_min_samples', 30)),
            dbscan_min_cluster_size=int(node_params.get('dbscan_min_cluster_size', 50))
        )
    
    def get_ground_segmentation_params(self) -> GroundSegmentationParams:
        """Get parameters for ground segmentation node."""
        node_params = self.ros_params.get('ground_segmentation_node', {})
        
        return GroundSegmentationParams(
            ransac_distance_threshold=float(node_params.get('ransac_distance_threshold', 0.05)),
            ransac_max_iterations=int(node_params.get('ransac_max_iterations', 100)),
            min_ground_points=int(node_params.get('min_ground_points', 50)),
            z_tolerance=float(node_params.get('z_tolerance', 0.3))
        )
    
    def get_boundary_detection_params(self) -> BoundaryDetectionParams:
        """Get parameters for boundary detection node."""
        node_params = self.ros_params.get('boundary_detection_node', {})
        
        return BoundaryDetectionParams(
            grid_resolution=float(node_params.get('grid_resolution', 0.5)),
            min_points_per_cell=int(node_params.get('min_points_per_cell', 5)),
            boundary_threshold=float(node_params.get('boundary_threshold', 0.5))
        )
    
    def get_node_params(self, node_name: str) -> Dict[str, Any]:
        """
        Get raw parameters for any node.
        
        Args:
            node_name: Name of the ROS2 node (e.g., 'road_points_extraction_node')
        
        Returns:
            Dictionary of parameters for that node
        """
        return self.ros_params.get(node_name, {})
    
    def reload(self):
        """Reload configuration from disk (useful if params.yaml changed)."""
        if self.config_path:
            self._load_config()
    
    # Convenience properties for commonly used parameters
    @property
    def max_road_intensity(self) -> float:
        """Quick access to max_road_intensity parameter."""
        return self.get_road_extraction_params().max_road_intensity
    
    @property
    def max_range(self) -> float:
        """Quick access to max_range parameter."""
        return self.get_road_extraction_params().max_range
    
    @property
    def min_neighbors(self) -> int:
        """Quick access to min_neighbors parameter."""
        return self.get_road_extraction_params().min_neighbors
    
    @property
    def neighbor_radius(self) -> float:
        """Quick access to neighbor_radius parameter."""
        return self.get_road_extraction_params().neighbor_radius
    
    def __repr__(self):
        status = "loaded" if self.config_path else "not found"
        path = self.config_path if self.config_path else "N/A"
        return f"ConfigLoader(status={status}, path={path})"


# Global singleton instance for convenience
_global_config: Optional[ConfigLoader] = None


def get_config(reload: bool = False) -> ConfigLoader:
    """
    Get the global ConfigLoader instance (singleton pattern).
    
    Args:
        reload: If True, reload configuration from disk
    
    Returns:
        ConfigLoader instance
    
    Example:
        from tools.config_loader import get_config
        
        config = get_config()
        max_intensity = config.max_road_intensity
    """
    global _global_config
    
    if _global_config is None or reload:
        _global_config = ConfigLoader()
    
    return _global_config


# Convenience function for quick parameter extraction
def load_road_extraction_params() -> RoadExtractionParams:
    """Load road extraction parameters (convenience function)."""
    return get_config().get_road_extraction_params()


if __name__ == '__main__':
    # Test the config loader
    print("="*60)
    print("Configuration Loader Test")
    print("="*60)

    config = ConfigLoader()
    print(f"\n{config}")

    if config.config_path:
        print(f"\nRoad Extraction Parameters (from params.yaml):")
        road_params = config.get_road_extraction_params()
        print(f"  {road_params}")

        print(f"\nGround Segmentation Parameters:")
        ground_params = config.get_ground_segmentation_params()
        print(f"  {ground_params}")

        print(f"\nQuick access properties:")
        print(f"  max_road_intensity: {config.max_road_intensity}")
        print(f"  max_range: {config.max_range}")
        print(f"  min_neighbors: {config.min_neighbors}")
        print(f"  neighbor_radius: {config.neighbor_radius}")

        # Show comparison with historical defaults
        print(f"\n{'='*60}")
        print("COMPARISON: Current (params.yaml) vs Historical Defaults")
        print("="*60)

        hist_road = HISTORICAL_DEFAULTS['road_points_extraction_node']
        print(f"\nRoad Extraction Parameters:")
        print(f"  {'Parameter':<25} {'Current':<12} {'Historical':<12} {'Change':<10}")
        print(f"  {'-'*59}")
        print(f"  {'max_road_intensity':<25} {road_params.max_road_intensity:<12} {hist_road['max_road_intensity']:<12} {'↑' if road_params.max_road_intensity > hist_road['max_road_intensity'] else '↓' if road_params.max_road_intensity < hist_road['max_road_intensity'] else '='}")
        print(f"  {'max_range':<25} {road_params.max_range:<12} {hist_road['max_range']:<12} {'↑' if road_params.max_range > hist_road['max_range'] else '↓' if road_params.max_range < hist_road['max_range'] else '='}")
        print(f"  {'min_neighbors':<25} {road_params.min_neighbors:<12} {hist_road['min_neighbors']:<12} {'↑' if road_params.min_neighbors > hist_road['min_neighbors'] else '↓' if road_params.min_neighbors < hist_road['min_neighbors'] else '='}")
        print(f"  {'neighbor_radius':<25} {road_params.neighbor_radius:<12} {hist_road['neighbor_radius']:<12} {'↑' if road_params.neighbor_radius > hist_road['neighbor_radius'] else '↓' if road_params.neighbor_radius < hist_road['neighbor_radius'] else '='}")
        print(f"  {'dbscan_eps':<25} {road_params.dbscan_eps:<12} {hist_road['dbscan_eps']:<12} {'↑' if road_params.dbscan_eps > hist_road['dbscan_eps'] else '↓' if road_params.dbscan_eps < hist_road['dbscan_eps'] else '='}")
        print(f"  {'dbscan_min_samples':<25} {road_params.dbscan_min_samples:<12} {hist_road['dbscan_min_samples']:<12} {'↑' if road_params.dbscan_min_samples > hist_road['dbscan_min_samples'] else '↓' if road_params.dbscan_min_samples < hist_road['dbscan_min_samples'] else '='}")
        print(f"  {'dbscan_min_cluster_size':<25} {road_params.dbscan_min_cluster_size:<12} {hist_road['dbscan_min_cluster_size']:<12} {'↑' if road_params.dbscan_min_cluster_size > hist_road['dbscan_min_cluster_size'] else '↓' if road_params.dbscan_min_cluster_size < hist_road['dbscan_min_cluster_size'] else '='}")

    print("\n" + "="*60)