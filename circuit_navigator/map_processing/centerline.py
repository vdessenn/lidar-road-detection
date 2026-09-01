#!/usr/bin/env python3
"""
Centerline Extraction - Extract track centerline using skeletonization

Extracts the centerline from a binary navigable grid using morphological
skeletonization, then orders and smooths the points for path following.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from scipy import ndimage
from skimage.morphology import skeletonize
from collections import deque


@dataclass
class Centerline:
    """Container for extracted centerline data."""
    points_pixel: np.ndarray      # (N, 2) pixel coordinates [row, col]
    points_world: np.ndarray      # (N, 2) world coordinates [x, y]
    length_m: float               # Total centerline length in meters
    is_closed: bool               # Whether the track is a closed loop
    resolution: float             # Grid resolution (m/cell)
    origin: Tuple[float, float]   # World origin (x, y)


class CenterlineExtractor:
    """Extract and process track centerline from navigable grid."""

    def __init__(self, min_branch_length: int = 10, smoothing_window: int = 5, verbose: bool = False):
        """
        Args:
            min_branch_length: Minimum skeleton branch length to keep (pixels)
            smoothing_window: Window size for moving average smoothing
            verbose: If True, print progress information
        """
        self.min_branch_length = min_branch_length
        self.smoothing_window = smoothing_window
        self.verbose = verbose
        self.skeleton: Optional[np.ndarray] = None
        self._navigable_grid: Optional[np.ndarray] = None

    def _log(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def extract(
        self,
        navigable_grid: np.ndarray,
        resolution: float,
        origin: Tuple[float, float],
        method: str = "ridge",
        skeleton_grid: np.ndarray = None
    ) -> Centerline:
        """
        Extract centerline from navigable grid.

        Args:
            navigable_grid: Binary grid (True = navigable)
            resolution: Grid resolution in meters per cell
            origin: World origin (x, y) of grid
            method: "skeleton" for traditional skeletonization,
                    "ridge" for distance transform ridge following
            skeleton_grid: Optional pre-dilated grid for skeletonization
                          (if None, uses navigable_grid)

        Returns:
            Centerline object with ordered, smoothed points
        """
        self._log("Extracting centerline...")

        # Store navigable grid for distance transform calculations
        self._navigable_grid = navigable_grid

        if method == "ridge":
            ordered_pixels, is_closed = self._extract_ridge_centerline_v2(navigable_grid, resolution)
            self.skeleton = None  # Not used in ridge method
        else:
            # Use skeleton_grid if provided (should be wider to avoid fragmentation)
            grid_for_skeleton = skeleton_grid if skeleton_grid is not None else navigable_grid

            # Step 1: Skeletonize
            self._log("  Skeletonizing navigable area...")
            self.skeleton = skeletonize(grid_for_skeleton)
            skeleton_points = np.sum(self.skeleton)
            self._log(f"  Skeleton pixels: {skeleton_points}")

            # Step 2: Prune short branches
            self._log("  Pruning short branches...")
            pruned = self._prune_branches(self.skeleton)
            pruned_points = np.sum(pruned)
            self._log(f"  After pruning: {pruned_points} pixels")

            # Step 3: Order skeleton points by traversal
            self._log("  Ordering points...")
            ordered_pixels = self._order_skeleton_points(pruned)
            self._log(f"  Ordered points: {len(ordered_pixels)}")

            # Step 4: Check if closed loop
            is_closed = self._check_closed_loop(ordered_pixels, pruned)

        if len(ordered_pixels) == 0:
            raise ValueError("Failed to extract centerline - no points found")

        self._log(f"  Closed loop: {is_closed}")

        # Step 5: Smooth the centerline
        self._log("  Smoothing centerline...")
        smoothed_pixels = self._smooth_centerline(ordered_pixels, is_closed)

        # Step 6: Convert to world coordinates
        points_world = self._to_world_coordinates(smoothed_pixels, resolution, origin)

        # Calculate total length
        if len(points_world) > 1:
            diffs = np.diff(points_world, axis=0)
            segment_lengths = np.linalg.norm(diffs, axis=1)
            total_length = np.sum(segment_lengths)
            if is_closed:
                # Add closing segment
                total_length += np.linalg.norm(points_world[0] - points_world[-1])
        else:
            total_length = 0.0

        self._log(f"  Centerline length: {total_length:.1f} m")

        return Centerline(
            points_pixel=smoothed_pixels,
            points_world=points_world,
            length_m=total_length,
            is_closed=is_closed,
            resolution=resolution,
            origin=origin
        )

    def _extract_ridge_centerline_v2(self, navigable_grid: np.ndarray, resolution: float) -> Tuple[np.ndarray, bool]:
        """
        Extract centerline by following the ridge (maximum distance from boundaries).
        Uses direction-aware path ordering with aggressive gap jumping for sparse tracks.

        Args:
            navigable_grid: Binary navigable grid
            resolution: Grid resolution in m/cell

        Returns:
            Tuple of (ordered_pixels, is_closed)
        """
        self._log("  Computing distance transform...")
        dist_transform = ndimage.distance_transform_edt(navigable_grid)

        # Find local maxima of distance transform (ridge points)
        self._log("  Finding ridge points...")
        from scipy.ndimage import maximum_filter

        # Use larger neighborhood for smoother ridge
        neighborhood_size = 5
        local_max = maximum_filter(dist_transform, size=neighborhood_size)

        # Ridge points: where distance equals local maximum and distance > minimum track width
        min_track_width_m = 1.0
        min_dist_threshold = min_track_width_m / (2 * resolution)
        ridge_mask = (dist_transform == local_max) & (dist_transform >= min_dist_threshold)

        # Get ridge coordinates
        ridge_coords = np.argwhere(ridge_mask)
        self._log(f"    Found {len(ridge_coords)} ridge points")

        if len(ridge_coords) == 0:
            return np.array([]), False

        ridge_set = {tuple(c) for c in ridge_coords}

        # Start from the point with minimum row (bottom of track)
        start_idx = np.argmin(ridge_coords[:, 0])
        start = tuple(ridge_coords[start_idx])

        # Order points with direction-aware traversal and aggressive gap jumping
        self._log("  Ordering ridge points with direction-aware traversal...")
        ordered = [start]
        remaining = ridge_set - {start}
        current = start
        prev_dir = (1.0, 0.0)  # Initial direction: going up

        max_gap_m = 30.0  # Allow jumps up to 30m for sparse tracks
        max_gap_px = int(max_gap_m / resolution)

        while remaining:
            # Find nearest point, preferring direction continuity
            best_score = -float('inf')
            best_point = None

            for candidate in remaining:
                dr = candidate[0] - current[0]
                dc = candidate[1] - current[1]
                dist = np.sqrt(dr*dr + dc*dc)

                if dist > max_gap_px:
                    continue

                # Direction continuity score
                if dist > 0:
                    new_dir = (dr / dist, dc / dist)
                    dir_score = prev_dir[0] * new_dir[0] + prev_dir[1] * new_dir[1]
                else:
                    dir_score = 1.0

                # Distance penalty (prefer closer points)
                dist_penalty = dist / max_gap_px

                # Ridge strength (prefer points further from boundary = wider track)
                ridge_strength = dist_transform[candidate[0], candidate[1]] / (dist_transform.max() + 1)

                # Combined score
                score = dir_score * 2.0 - dist_penalty + ridge_strength * 0.5

                if score > best_score:
                    best_score = score
                    best_point = candidate

            if best_point is None:
                # No point found within max_gap - path ends
                break

            # Update direction
            dr = best_point[0] - current[0]
            dc = best_point[1] - current[1]
            dist = np.sqrt(dr*dr + dc*dc)
            if dist > 0:
                prev_dir = (dr / dist, dc / dist)

            ordered.append(best_point)
            remaining.discard(best_point)
            current = best_point

        ordered_pixels = np.array(ordered)
        self._log(f"    Ordered {len(ordered_pixels)} points")

        # Check if closed loop
        if len(ordered_pixels) > 50:
            start_pt = ordered_pixels[0]
            end_pt = ordered_pixels[-1]
            dist = np.sqrt((start_pt[0] - end_pt[0])**2 + (start_pt[1] - end_pt[1])**2)
            is_closed = dist * resolution < 15.0  # Within 15m
            self._log(f"    Start-end distance: {dist * resolution:.1f}m, closed: {is_closed}")
        else:
            is_closed = False

        return ordered_pixels, is_closed

    def _get_neighbors(self, skeleton: np.ndarray, row: int, col: int) -> List[Tuple[int, int]]:
        """Get 8-connected skeleton neighbors of a pixel."""
        neighbors = []
        rows, cols = skeleton.shape
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    if skeleton[nr, nc]:
                        neighbors.append((nr, nc))
        return neighbors

    def _count_neighbors(self, skeleton: np.ndarray) -> np.ndarray:
        """Count 8-connected neighbors for each pixel."""
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        return ndimage.convolve(skeleton.astype(int), kernel, mode='constant')

    def _prune_branches(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Remove short branches from skeleton.
        Keeps only the main path(s).
        """
        result = skeleton.copy()
        neighbor_count = self._count_neighbors(result)

        # Find endpoints (pixels with exactly 1 neighbor)
        endpoints = result & (neighbor_count == 1)
        endpoint_coords = np.argwhere(endpoints)

        for ep in endpoint_coords:
            branch = self._trace_branch(result, tuple(ep))
            if len(branch) < self.min_branch_length:
                # Remove this short branch
                for px in branch:
                    result[px[0], px[1]] = False

        return result

    def _remove_roundabout_circles(self, skeleton: np.ndarray, navigable_grid: np.ndarray) -> np.ndarray:
        """
        Remove skeleton points that form roundabout inner circles.
        These are skeleton points close to holes (grass zones) in the navigable grid.
        """
        from scipy import ndimage as ndi

        # Find holes in the navigable grid (roundabout grass zones)
        # These are regions of False surrounded by True
        filled = ndi.binary_fill_holes(navigable_grid)
        holes = filled & ~navigable_grid

        if not np.any(holes):
            return skeleton

        # Dilate holes to create a "danger zone"
        # Skeleton points in this zone are part of roundabout inner circles
        hole_distance = ndi.distance_transform_edt(~holes)

        # Main track skeleton points should be far from holes
        # Roundabout inner circle points are close to holes
        # Threshold: skeleton points within 2 pixels of a hole are roundabout circles
        # Using a smaller threshold to avoid breaking the main track
        danger_zone = hole_distance < 2.5

        # Remove skeleton points in danger zone
        result = skeleton & ~danger_zone

        removed = np.sum(skeleton) - np.sum(result)
        self._log(f"    Removed {removed} roundabout circle pixels")

        return result

    def _trace_branch(self, skeleton: np.ndarray, start: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Trace a branch from endpoint until junction or another endpoint."""
        branch = [start]
        current = start
        prev = None

        while True:
            neighbors = self._get_neighbors(skeleton, current[0], current[1])

            # Exclude previous pixel
            if prev is not None:
                neighbors = [n for n in neighbors if n != prev]

            if len(neighbors) == 0:
                # Dead end
                break
            elif len(neighbors) == 1:
                # Continue along branch
                prev = current
                current = neighbors[0]
                branch.append(current)
            else:
                # Junction reached (multiple neighbors)
                break

        return branch

    def _find_dead_end_branches(self, skeleton: np.ndarray, endpoint_coords: np.ndarray,
                                 max_branch_length: int = 30) -> set:
        """
        Find skeleton points that are part of dead-end branches.
        These are likely mapping errors (spurious exits) that should be avoided.

        Args:
            skeleton: Binary skeleton
            endpoint_coords: Array of endpoint coordinates
            max_branch_length: Maximum branch length to consider as dead-end (pixels)

        Returns:
            Set of (row, col) tuples that are part of dead-end branches
        """
        dead_end_points = set()
        dead_end_branches = []

        for ep in endpoint_coords:
            branch = self._trace_branch(skeleton, tuple(ep))
            # If branch is short, mark all points as dead-end
            if len(branch) < max_branch_length:
                dead_end_branches.append((tuple(ep), len(branch)))
                for point in branch:
                    dead_end_points.add(point)

        if dead_end_branches:
            self._log(f"    Found {len(dead_end_branches)} dead-end branches ({len(dead_end_points)} points)")

        return dead_end_points

    def _order_skeleton_points(self, skeleton: np.ndarray, max_gap: int = 15, allow_return: bool = True) -> np.ndarray:
        """
        Order skeleton points for circuit traversal.
        At junctions, prefer the path that stays on the main track (first exit strategy).
        When stuck (no neighbors), attempt to jump gaps to continue.
        For out-and-back circuits, allows returning on the same path.

        Scoring at junctions:
        1. Direction continuity (strongly prefer going straight)
        2. Right turn preference (European roundabout: go right)
        3. Distance from boundary (prefer main track over small junction loops)
        4. Avoid dead-end branches (penalize paths leading to endpoints)

        Args:
            skeleton: Binary skeleton array
            max_gap: Maximum pixel distance to jump when stuck (default 100 = ~50m at 0.5m resolution)
            allow_return: If True, after hitting a dead end, allow returning on visited path
        """
        skeleton_coords = np.argwhere(skeleton)
        if len(skeleton_coords) == 0:
            return np.array([])

        # Build set of all skeleton points for fast lookup
        skeleton_set = {tuple(c) for c in skeleton_coords}

        # Compute distance from track boundary for each point
        dist_transform = ndimage.distance_transform_edt(self._navigable_grid)

        neighbor_count = self._count_neighbors(skeleton)
        endpoints = skeleton & (neighbor_count == 1)
        endpoint_coords = np.argwhere(endpoints)

        # Build a map of which points lead to dead ends (short branches ending at endpoints)
        # This helps avoid mapping errors like spurious exits
        dead_end_points = self._find_dead_end_branches(skeleton, endpoint_coords, max_branch_length=30)

        # Find starting point - prefer bottom of track (lowest row)
        if len(endpoint_coords) > 0:
            # Start from the endpoint with lowest row (bottom of track)
            start_idx = np.argmin(endpoint_coords[:, 0])
            start = tuple(endpoint_coords[start_idx])
        else:
            # No endpoints - start from bottom-most skeleton point
            start_idx = np.argmin(skeleton_coords[:, 0])
            start = tuple(skeleton_coords[start_idx])

        # Greedy traversal with direction continuity, gap jumping, and outer-path preference
        ordered = [start]
        visited = {start}
        current = start
        prev_dir = (1, 0)  # Initial direction: going up (increasing row)

        while True:
            neighbors = self._get_neighbors(skeleton, current[0], current[1])
            unvisited = [n for n in neighbors if n not in visited]

            if len(unvisited) == 0:
                # No unvisited neighbors
                # Option 1: Try small gap jump to continue on unvisited points
                jump_target = self._find_jump_target(
                    current, prev_dir, skeleton_set, visited, max_gap, dist_transform
                )
                if jump_target is not None:
                    dist_to_target = np.sqrt((jump_target[0] - current[0])**2 + (jump_target[1] - current[1])**2)
                    self._log(f"    Gap jump: {dist_to_target:.1f}px from ({current[0]}, {current[1]}) to ({jump_target[0]}, {jump_target[1]})")
                    next_point = jump_target
                    visited.add(next_point)
                    ordered.append(next_point)
                    current = next_point
                    prev_dir = (next_point[0] - ordered[-2][0], next_point[1] - ordered[-2][1])
                    length = np.sqrt(prev_dir[0]**2 + prev_dir[1]**2)
                    if length > 0:
                        prev_dir = (prev_dir[0] / length, prev_dir[1] / length)
                    continue

                # Option 2: At roundabout exit - allow revisiting to continue the loop
                if allow_return:
                    prev_point = ordered[-2] if len(ordered) > 1 else None
                    # Allow any neighbor except the one we just came from
                    revisitable = [n for n in neighbors if n != prev_point]
                    if revisitable:
                        # Prefer continuing in same direction
                        best_score = -float('inf')
                        next_point = revisitable[0]
                        for n in revisitable:
                            new_dir = (n[0] - current[0], n[1] - current[1])
                            score = prev_dir[0] * new_dir[0] + prev_dir[1] * new_dir[1]
                            if score > best_score:
                                best_score = score
                                next_point = n
                        self._log(f"    Crossing back at ({current[0]}, {current[1]}) to ({next_point[0]}, {next_point[1]})")
                        prev_dir = (next_point[0] - current[0], next_point[1] - current[1])
                        ordered.append(next_point)
                        current = next_point
                        # Mark that we've crossed back - now track visited differently
                        visited = set(ordered)  # Reset visited to current path
                        continue

                self._log(f"    Path ended at pixel=({current[0]}, {current[1]}), {len(visited)}/{len(skeleton_set)} visited")
                break

            elif len(unvisited) == 1:
                next_point = unvisited[0]
            else:
                # Multiple choices - junction point (likely roundabout entry/exit)
                # Rule: at roundabouts, always go RIGHT (first exit strategy)
                # Also: prefer main track over small junction loops (higher distance from boundary)
                best_score = -float('inf')
                next_point = unvisited[0]

                # Get distance values for scoring
                current_dist = dist_transform[current[0], current[1]]

                for neighbor in unvisited:
                    new_dir = (neighbor[0] - current[0], neighbor[1] - current[1])

                    # Direction continuity score (dot product)
                    straight_score = prev_dir[0] * new_dir[0] + prev_dir[1] * new_dir[1]

                    # Right turn preference score (cross product)
                    # In image coords with origin='lower': right of (dr, dc) is (-dc, dr)
                    # This means: if going up (1,0), right is (0,1) = increasing column
                    right_dir = (-prev_dir[1], prev_dir[0])
                    right_score = right_dir[0] * new_dir[0] + right_dir[1] * new_dir[1]

                    # Distance from boundary score - prefer paths on main track
                    # Small junction loops are closer to boundary (lower distance)
                    neighbor_dist = dist_transform[neighbor[0], neighbor[1]]
                    # Normalize by current distance to get relative improvement
                    dist_score = neighbor_dist / (current_dist + 1.0)

                    # Dead-end penalty - strongly penalize paths leading to dead ends
                    # These are likely mapping errors (spurious exits)
                    dead_end_penalty = -50.0 if neighbor in dead_end_points else 0.0

                    # Combined scoring:
                    # - Direction continuity is most important (stay straight)
                    # - Distance from boundary helps avoid small junction loops
                    # - Right turn is tiebreaker for roundabouts
                    # - Dead-end penalty avoids spurious exits (mapping errors)
                    score = straight_score * 10.0 + dist_score * 3.0 + right_score * 1.5 + dead_end_penalty

                    if score > best_score:
                        best_score = score
                        next_point = neighbor

            # Update direction
            prev_dir = (next_point[0] - current[0], next_point[1] - current[1])
            # Normalize direction for gap jumps
            length = np.sqrt(prev_dir[0]**2 + prev_dir[1]**2)
            if length > 1.5:  # This was a gap jump, normalize direction
                prev_dir = (prev_dir[0] / length, prev_dir[1] / length)

            visited.add(next_point)
            ordered.append(next_point)
            current = next_point

        return np.array(ordered)

    def _find_jump_target(
        self,
        current: Tuple[int, int],
        prev_dir: Tuple[float, float],
        skeleton_set: set,
        visited: set,
        max_gap: int,
        dist_transform: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """
        Find nearest unvisited skeleton point when stuck.

        Uses tiered search:
        1. First look for points with good direction continuity (forward)
        2. If none found, find the largest contiguous unvisited region and jump there

        Args:
            current: Current position
            prev_dir: Previous movement direction
            skeleton_set: Set of all skeleton points
            visited: Set of visited points
            max_gap: Maximum distance to search
            dist_transform: Distance transform for scoring

        Returns:
            Best jump target or None if no suitable target found
        """
        # Get all unvisited skeleton points
        unvisited = skeleton_set - visited
        if not unvisited:
            return None

        candidates = []
        for candidate in unvisited:
            dr = candidate[0] - current[0]
            dc = candidate[1] - current[1]
            dist = np.sqrt(dr*dr + dc*dc)

            if dist > max_gap:
                continue

            # Score by direction continuity
            if dist > 0:
                new_dir = (dr / dist, dc / dist)
                dir_score = prev_dir[0] * new_dir[0] + prev_dir[1] * new_dir[1]
            else:
                dir_score = 0

            # Distance from boundary (prefer main track)
            boundary_score = dist_transform[candidate[0], candidate[1]]

            # Count how many unvisited neighbors this point has (connectivity)
            neighbor_count = 0
            for dr2 in [-1, 0, 1]:
                for dc2 in [-1, 0, 1]:
                    if dr2 == 0 and dc2 == 0:
                        continue
                    neighbor = (candidate[0] + dr2, candidate[1] + dc2)
                    if neighbor in unvisited:
                        neighbor_count += 1

            candidates.append((candidate, dir_score, dist, boundary_score, neighbor_count))

        if not candidates:
            return None

        # Tier 1: Try to find points with positive direction score (forward)
        forward_candidates = [c for c in candidates if c[1] > 0.3]
        if forward_candidates:
            # Among forward candidates, prefer closer and on main track
            forward_candidates.sort(key=lambda x: (-x[3] / 10.0 + x[2] * 0.1))
            return forward_candidates[0][0]

        # Tier 2: Find the point that leads to the most unvisited skeleton
        # This helps find the "main" unvisited region (return path) rather than
        # scattered isolated points
        # Prefer: high connectivity, high boundary score (main track), lower distance
        candidates.sort(key=lambda x: (-x[4] * 5.0 - x[3] + x[2] * 0.1))
        return candidates[0][0]

    def _check_closed_loop(self, ordered_pixels: np.ndarray, skeleton: np.ndarray) -> bool:
        """Check if the ordered points form a closed loop."""
        if len(ordered_pixels) < 10:
            return False

        start = tuple(ordered_pixels[0])
        end = tuple(ordered_pixels[-1])

        # Only closed if start and end are immediate 8-connected neighbors
        neighbors = self._get_neighbors(skeleton, end[0], end[1])
        return start in neighbors

    def _smooth_centerline(self, points: np.ndarray, is_closed: bool) -> np.ndarray:
        """Apply moving average smoothing to centerline."""
        if len(points) < self.smoothing_window:
            return points.astype(float)

        n = len(points)
        w = self.smoothing_window
        half_w = w // 2
        smoothed = np.zeros_like(points, dtype=float)

        for i in range(n):
            if is_closed:
                # Circular indexing for closed loop
                indices = [(i + j - half_w) % n for j in range(w)]
            else:
                # Clamp indices for open path
                indices = [max(0, min(n - 1, i + j - half_w)) for j in range(w)]

            smoothed[i] = np.mean(points[indices], axis=0)

        return smoothed

    def _to_world_coordinates(
        self,
        points_pixel: np.ndarray,
        resolution: float,
        origin: Tuple[float, float]
    ) -> np.ndarray:
        """
        Convert pixel coordinates to world coordinates.

        Pixel [row, col] -> World [x, y]:
            x = origin_x + col * resolution
            y = origin_y + row * resolution
        """
        if len(points_pixel) == 0:
            return np.array([])

        points_world = np.zeros((len(points_pixel), 2), dtype=float)
        points_world[:, 0] = origin[0] + points_pixel[:, 1] * resolution  # x from col
        points_world[:, 1] = origin[1] + points_pixel[:, 0] * resolution  # y from row

        return points_world

    def resample_by_distance(
        self,
        centerline: Centerline,
        spacing_m: float = 1.0
    ) -> Centerline:
        """
        Resample centerline to have uniform spacing between points.

        Args:
            centerline: Original centerline
            spacing_m: Desired spacing between points in meters

        Returns:
            New Centerline with uniformly spaced points
        """
        points = centerline.points_world
        if len(points) < 2:
            return centerline

        # Calculate cumulative distance along path
        diffs = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)

        if centerline.is_closed:
            # Add closing segment
            closing_length = np.linalg.norm(points[0] - points[-1])
            segment_lengths = np.append(segment_lengths, closing_length)

        cumulative_dist = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_dist[-1]

        # Generate uniformly spaced distances
        num_points = int(total_length / spacing_m) + 1
        target_distances = np.linspace(0, total_length, num_points)

        # Interpolate points at target distances
        if centerline.is_closed:
            # Extend points array for interpolation across closure
            extended_points = np.vstack([points, points[0:1]])
        else:
            extended_points = points

        resampled = np.zeros((num_points, 2))
        for i, target_dist in enumerate(target_distances):
            # Find segment containing this distance
            idx = np.searchsorted(cumulative_dist, target_dist) - 1
            idx = max(0, min(idx, len(extended_points) - 2))

            # Interpolate within segment
            segment_start_dist = cumulative_dist[idx]
            segment_length = segment_lengths[idx] if idx < len(segment_lengths) else 0

            if segment_length > 0:
                t = (target_dist - segment_start_dist) / segment_length
                t = max(0, min(1, t))
                resampled[i] = (1 - t) * extended_points[idx] + t * extended_points[idx + 1]
            else:
                resampled[i] = extended_points[idx]

        # Convert back to pixel coordinates
        resampled_pixel = np.zeros_like(resampled)
        resampled_pixel[:, 0] = (resampled[:, 1] - centerline.origin[1]) / centerline.resolution
        resampled_pixel[:, 1] = (resampled[:, 0] - centerline.origin[0]) / centerline.resolution

        return Centerline(
            points_pixel=resampled_pixel,
            points_world=resampled,
            length_m=total_length,
            is_closed=centerline.is_closed,
            resolution=centerline.resolution,
            origin=centerline.origin
        )


if __name__ == "__main__":
    # Test with simple example
    print("Testing CenterlineExtractor...")

    # Create a simple test grid (oval track)
    grid = np.zeros((100, 100), dtype=bool)

    # Draw oval track
    center = (50, 50)
    outer_a, outer_b = 40, 25
    inner_a, inner_b = 30, 15

    for i in range(100):
        for j in range(100):
            # Distance from center (ellipse equation)
            dist_outer = ((i - center[0]) / outer_a) ** 2 + ((j - center[1]) / outer_b) ** 2
            dist_inner = ((i - center[0]) / inner_a) ** 2 + ((j - center[1]) / inner_b) ** 2

            # Inside outer, outside inner = track
            if dist_outer <= 1 and dist_inner >= 1:
                grid[i, j] = True

    extractor = CenterlineExtractor()
    centerline = extractor.extract(grid, resolution=1.0, origin=(0, 0))

    print(f"\nResult:")
    print(f"  Points: {len(centerline.points_world)}")
    print(f"  Length: {centerline.length_m:.1f} m")
    print(f"  Closed: {centerline.is_closed}")
