#!/usr/bin/env python3
"""
Validation Report Generator
===========================
Analyzes validation session files and generates a report for algorithm improvement.

Usage:
    python3 analyze_validation.py validation_YYYYMMDD_HHMMSS.json

Output:
    - Summary statistics
    - Patterns in false detections (timestamps, point counts)
    - Recommendations for parameter tuning
"""

import json
import sys
import os
from typing import List, Dict, Any
from collections import defaultdict
import statistics


def load_session(filepath: str) -> Dict[str, Any]:
    """Load a validation session file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_session(data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a validation session for patterns."""
    
    records = data.get('records', [])
    
    good_records = [r for r in records if r['label'] == 'good']
    bad_records = [r for r in records if r['label'] == 'bad']
    
    analysis = {
        'summary': {
            'total_labeled': len(records),
            'good_count': len(good_records),
            'bad_count': len(bad_records),
            'accuracy': len(good_records) / len(records) * 100 if records else 0
        },
        'good_patterns': {},
        'bad_patterns': {},
        'recommendations': []
    }
    
    # Analyze point counts for good detections
    if good_records:
        good_left = [r['num_left_points'] for r in good_records]
        good_right = [r['num_right_points'] for r in good_records]
        analysis['good_patterns'] = {
            'left_points': {
                'min': min(good_left),
                'max': max(good_left),
                'mean': statistics.mean(good_left),
                'median': statistics.median(good_left)
            },
            'right_points': {
                'min': min(good_right),
                'max': max(good_right),
                'mean': statistics.mean(good_right),
                'median': statistics.median(good_right)
            }
        }
    
    # Analyze point counts for bad detections
    if bad_records:
        bad_left = [r['num_left_points'] for r in bad_records]
        bad_right = [r['num_right_points'] for r in bad_records]
        analysis['bad_patterns'] = {
            'left_points': {
                'min': min(bad_left),
                'max': max(bad_left),
                'mean': statistics.mean(bad_left),
                'median': statistics.median(bad_left)
            },
            'right_points': {
                'min': min(bad_right),
                'max': max(bad_right),
                'mean': statistics.mean(bad_right),
                'median': statistics.median(bad_right)
            },
            'timestamps': [
                f"{r['timestamp_sec']}.{r['timestamp_nanosec']:09d}"
                for r in bad_records
            ]
        }
        
        # Generate recommendations based on patterns
        if good_records:
            good_total = statistics.mean([r['num_left_points'] + r['num_right_points'] for r in good_records])
            bad_total = statistics.mean([r['num_left_points'] + r['num_right_points'] for r in bad_records])
            
            if bad_total > good_total * 1.5:
                analysis['recommendations'].append(
                    "Bad detections have MORE points than good ones. "
                    "Consider TIGHTENING intensity thresholds (lower max_road_intensity) "
                    "or INCREASING min_boundary_points."
                )
            elif bad_total < good_total * 0.5:
                analysis['recommendations'].append(
                    "Bad detections have FEWER points than good ones. "
                    "Consider LOOSENING intensity thresholds (higher max_road_intensity) "
                    "or DECREASING min_boundary_points."
                )
            
            # Check for one-sided failures
            good_ratio = statistics.mean(good_left) / max(statistics.mean(good_right), 0.1)
            bad_left_mean = statistics.mean(bad_left) if bad_left else 0
            bad_right_mean = statistics.mean(bad_right) if bad_right else 0
            bad_ratio = bad_left_mean / max(bad_right_mean, 0.1)
            
            if abs(bad_ratio - good_ratio) > 1.0:
                if bad_left_mean < bad_right_mean:
                    analysis['recommendations'].append(
                        "Left boundary detection is weaker in bad frames. "
                        "Check for asymmetric road features or calibration issues."
                    )
                else:
                    analysis['recommendations'].append(
                        "Right boundary detection is weaker in bad frames. "
                        "Check for asymmetric road features or calibration issues."
                    )
    
    return analysis


def print_report(analysis: Dict[str, Any], filepath: str):
    """Print formatted analysis report."""
    
    print("\n" + "="*60)
    print("VALIDATION ANALYSIS REPORT")
    print("="*60)
    print(f"File: {filepath}")
    print()
    
    summary = analysis['summary']
    print("SUMMARY")
    print("-"*40)
    print(f"  Total labeled frames: {summary['total_labeled']}")
    print(f"  Good detections:      {summary['good_count']}")
    print(f"  Bad detections:       {summary['bad_count']}")
    print(f"  Accuracy:             {summary['accuracy']:.1f}%")
    print()
    
    if analysis['good_patterns']:
        print("GOOD DETECTION PATTERNS")
        print("-"*40)
        gp = analysis['good_patterns']
        print(f"  Left points:  min={gp['left_points']['min']}, "
              f"max={gp['left_points']['max']}, "
              f"median={gp['left_points']['median']:.0f}")
        print(f"  Right points: min={gp['right_points']['min']}, "
              f"max={gp['right_points']['max']}, "
              f"median={gp['right_points']['median']:.0f}")
        print()
    
    if analysis['bad_patterns']:
        print("BAD DETECTION PATTERNS")
        print("-"*40)
        bp = analysis['bad_patterns']
        print(f"  Left points:  min={bp['left_points']['min']}, "
              f"max={bp['left_points']['max']}, "
              f"median={bp['left_points']['median']:.0f}")
        print(f"  Right points: min={bp['right_points']['min']}, "
              f"max={bp['right_points']['max']}, "
              f"median={bp['right_points']['median']:.0f}")
        print()
        print("  Bad frame timestamps:")
        for ts in bp['timestamps'][:10]:  # Show first 10
            print(f"    - {ts}")
        if len(bp['timestamps']) > 10:
            print(f"    ... and {len(bp['timestamps']) - 10} more")
        print()
    
    if analysis['recommendations']:
        print("RECOMMENDATIONS")
        print("-"*40)
        for i, rec in enumerate(analysis['recommendations'], 1):
            print(f"  {i}. {rec}")
        print()
    
    print("="*60)


def export_bad_timestamps(analysis: Dict[str, Any], output_file: str):
    """Export bad detection timestamps for replay analysis."""
    if 'bad_patterns' in analysis and 'timestamps' in analysis['bad_patterns']:
        with open(output_file, 'w') as f:
            for ts in analysis['bad_patterns']['timestamps']:
                f.write(ts + '\n')
        print(f"Bad timestamps exported to: {output_file}")


def main():
    if len(sys.argv) < 2:
        # Find most recent validation file
        validation_dir = '/home/ws/tools/validation_output'
        if os.path.exists(validation_dir):
            files = sorted([f for f in os.listdir(validation_dir) if f.endswith('.json')])
            if files:
                filepath = os.path.join(validation_dir, files[-1])
                print(f"Using most recent: {filepath}")
            else:
                print("No validation files found.")
                print(f"Usage: {sys.argv[0]} <validation_file.json>")
                sys.exit(1)
        else:
            print(f"Usage: {sys.argv[0]} <validation_file.json>")
            sys.exit(1)
    else:
        filepath = sys.argv[1]
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    data = load_session(filepath)
    analysis = analyze_session(data)
    print_report(analysis, filepath)
    
    # Export bad timestamps
    if analysis['bad_patterns']:
        base = os.path.splitext(filepath)[0]
        export_bad_timestamps(analysis, f"{base}_bad_timestamps.txt")


if __name__ == '__main__':
    main()
