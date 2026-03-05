#!/usr/bin/env python3
"""
Compare GPU Benchmark Results Against Expected Performance

This script loads benchmark results and compares them against expected
performance specifications for different GPU architectures.

Usage:
    python compare_results.py benchmark_results.yaml expected_performance_A100_80GB.yaml
    python compare_results.py --auto benchmark_results.yaml
"""

import yaml
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

def load_yaml(filepath: str) -> Dict[Any, Any]:
    """Load YAML file"""
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def detect_gpu_from_name(gpu_name: str) -> str:
    """Detect GPU architecture from name"""
    gpu_name_lower = gpu_name.lower()
    
    if 'a100' in gpu_name_lower:
        return 'A100'
    elif 'h100' in gpu_name_lower:
        return 'H100'
    elif 'l40s' in gpu_name_lower:
        return 'L40S'
    elif 'b300' in gpu_name_lower:
        return 'B300'
    elif 'b200' in gpu_name_lower:
        return 'B200'
    else:
        return None

def find_expected_file(gpu_type: str) -> str:
    """Find expected performance file for GPU type"""
    expected_files = {
        'A100': 'expected_performance_A100_80GB.yaml',
        'H100': 'expected_performance_H100_80GB.yaml',
        'L40S': 'expected_performance_L40S.yaml',
        'B200': 'expected_performance_B200.yaml',
        'B300': 'expected_performance_B300.yaml'
    }
    
    return expected_files.get(gpu_type)

def extract_results_by_test(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group results by test type"""
    grouped = {}
    for result in results:
        test_name = result['test_name']
        if test_name not in grouped:
            grouped[test_name] = []
        grouped[test_name].append(result)
    return grouped

def compare_gemm_performance(actual_results: List[Dict], expected: Dict) -> List[Tuple[str, str, float, float, float, str]]:
    """Compare GEMM performance against expected values"""
    comparisons = []
    
    for result in actual_results:
        dtype = result['dtype']
        shape = result['shape']
        actual_tflops = result['throughput_tflops']
        
        # Determine size category
        dims = [int(x) for x in shape.split('x')]
        max_dim = max(dims)
        
        if max_dim <= 4096:
            size_key = 'small_4096x4096x4096'
        elif max_dim <= 8192:
            size_key = 'large_8192x8192x8192'
        else:
            size_key = 'xlarge_16384x16384x16384'
        
        # Get expected range
        if dtype in expected and size_key in expected[dtype]:
            exp = expected[dtype][size_key]
            min_tflops = exp['min_tflops']
            max_tflops = exp['max_tflops']
            typical_tflops = exp['typical_tflops']
            
            # Determine status
            if min_tflops <= actual_tflops <= max_tflops:
                status = '✓ OK'
            elif actual_tflops < min_tflops:
                status = '⚠ BELOW'
            else:
                status = '✓ ABOVE'
            
            comparisons.append((
                f"GEMM {dtype} {size_key}",
                result['operation'],
                actual_tflops,
                min_tflops,
                max_tflops,
                status
            ))
    
    return comparisons

def compare_memory_bandwidth(actual_results: List[Dict], expected: Dict) -> List[Tuple[str, str, float, float, float, str]]:
    """Compare memory bandwidth against expected values"""
    comparisons = []
    
    for result in actual_results:
        operation = result['operation']
        actual_bw = result['memory_bandwidth_gb_s']
        
        # Map operation to expected key
        if 'Device-to-Device' in operation:
            exp_key = 'device_to_device'
        elif 'Host-to-Device' in operation:
            # Try to determine PCIe gen or SXM
            exp_key = 'host_to_device_pcie_gen4'  # Default assumption
        elif 'Device-to-Host' in operation:
            exp_key = 'device_to_host_pcie_gen4'  # Default assumption
        else:
            continue
        
        if exp_key in expected:
            exp = expected[exp_key]
            min_bw = exp['min_bandwidth_gbs']
            max_bw = exp['max_bandwidth_gbs']
            
            # Determine status
            if min_bw <= actual_bw <= max_bw:
                status = '✓ OK'
            elif actual_bw < min_bw:
                status = '⚠ BELOW'
            else:
                status = '✓ ABOVE'
            
            comparisons.append((
                operation,
                operation,
                actual_bw,
                min_bw,
                max_bw,
                status
            ))
    
    return comparisons

def compare_multi_gpu(actual_results: List[Dict], expected: Dict) -> List[Tuple[str, str, float, float, float, str]]:
    """Compare multi-GPU performance"""
    comparisons = []
    
    # Try to detect NVLink vs PCIe from actual bandwidth
    max_bw = max((r['memory_bandwidth_gb_s'] for r in actual_results), default=0)
    
    if max_bw > 100:
        interconnect_key = list(expected.keys())[0]  # Use first (typically NVLink)
    else:
        # Use PCIe key
        pcie_keys = [k for k in expected.keys() if 'pcie' in k.lower()]
        interconnect_key = pcie_keys[0] if pcie_keys else list(expected.keys())[-1]
    
    for result in actual_results:
        operation = result['operation']
        actual_bw = result['memory_bandwidth_gb_s']
        
        if 'All-Reduce' in operation:
            exp = expected[interconnect_key]['all_reduce_bandwidth_gbs']
        elif 'P2P' in operation:
            exp = expected[interconnect_key]['p2p_bandwidth_gbs']
        else:
            continue
        
        min_bw = exp['min']
        max_bw = exp['max']
        
        # Determine status
        if min_bw <= actual_bw <= max_bw:
            status = '✓ OK'
        elif actual_bw < min_bw:
            status = '⚠ BELOW'
        else:
            status = '✓ ABOVE'
        
        comparisons.append((
            operation,
            operation,
            actual_bw,
            min_bw,
            max_bw,
            status
        ))
    
    return comparisons

def print_comparison_table(comparisons: List[Tuple[str, str, float, float, float, str]], metric: str):
    """Print comparison table"""
    if not comparisons:
        return
    
    print(f"\n{'='*100}")
    print(f"{comparisons[0][0].split()[0]} Performance Comparison")
    print(f"{'='*100}")
    print(f"{'Test':<40} {'Actual':<15} {'Min Expected':<15} {'Max Expected':<15} {'Status':<10}")
    print(f"{'-'*100}")
    
    for test_name, operation, actual, min_exp, max_exp, status in comparisons:
        print(f"{test_name:<40} {actual:<15.2f} {min_exp:<15.2f} {max_exp:<15.2f} {status:<10}")

def main():
    parser = argparse.ArgumentParser(
        description='Compare GPU benchmark results against expected performance'
    )
    parser.add_argument('results_file', type=str,
                       help='Benchmark results YAML file')
    parser.add_argument('expected_file', type=str, nargs='?',
                       help='Expected performance YAML file (optional with --auto)')
    parser.add_argument('--auto', action='store_true',
                       help='Auto-detect GPU and find expected file')
    
    args = parser.parse_args()
    
    # Load benchmark results
    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}")
        sys.exit(1)
    
    results_data = load_yaml(args.results_file)
    
    # Auto-detect GPU if requested
    if args.auto or not args.expected_file:
        gpu_name = results_data['gpu_info']['name']
        gpu_type = detect_gpu_from_name(gpu_name)
        
        if not gpu_type:
            print(f"Error: Could not detect GPU type from name: {gpu_name}")
            print("Please specify expected performance file manually")
            sys.exit(1)
        
        expected_filename = find_expected_file(gpu_type)
        print(f"Auto-detected GPU: {gpu_type}")
        print(f"Using expected performance file: {expected_filename}")
        
        if not Path(expected_filename).exists():
            print(f"Error: Expected performance file not found: {expected_filename}")
            sys.exit(1)
        
        args.expected_file = expected_filename
    
    # Load expected performance
    expected_data = load_yaml(args.expected_file)
    
    # Print header
    print(f"\n{'='*100}")
    print(f"GPU BENCHMARK COMPARISON REPORT")
    print(f"{'='*100}")
    print(f"Tested GPU: {results_data['gpu_info']['name']}")
    print(f"Expected GPU: {expected_data['gpu_info']['name']}")
    print(f"Architecture: {expected_data['gpu_info']['architecture']}")
    print(f"{'='*100}")
    
    # Group results by test type
    grouped_results = extract_results_by_test(results_data['results'])
    
    # Compare GEMM performance
    if 'GEMM' in grouped_results and 'gemm' in expected_data['expected_performance']:
        gemm_comparisons = compare_gemm_performance(
            grouped_results['GEMM'],
            expected_data['expected_performance']['gemm']
        )
        print_comparison_table(gemm_comparisons, 'TFLOPS')
    
    # Compare Memory bandwidth
    if 'Memory' in grouped_results and 'memory' in expected_data['expected_performance']:
        mem_comparisons = compare_memory_bandwidth(
            grouped_results['Memory'],
            expected_data['expected_performance']['memory']
        )
        print_comparison_table(mem_comparisons, 'GB/s')
    
    # Compare Multi-GPU performance
    multi_gpu_tests = [k for k in grouped_results.keys() if k in ['AllReduce', 'P2P']]
    if multi_gpu_tests and 'multi_gpu' in expected_data['expected_performance']:
        all_multi_gpu_results = []
        for test in multi_gpu_tests:
            all_multi_gpu_results.extend(grouped_results[test])
        
        if all_multi_gpu_results:
            multi_gpu_comparisons = compare_multi_gpu(
                all_multi_gpu_results,
                expected_data['expected_performance']['multi_gpu']
            )
            print_comparison_table(multi_gpu_comparisons, 'GB/s')
    
    # Print summary
    print(f"\n{'='*100}")
    print("SUMMARY")
    print(f"{'='*100}")
    
    all_results = []
    for test_type in grouped_results.values():
        for result in test_type:
            all_results.append(result)
    
    print(f"Total tests run: {len(all_results)}")
    
    # Check if any tests are below expected
    below_count = 0
    for test_name, results in grouped_results.items():
        # This is a simplified check - actual implementation would need full comparison
        pass
    
    print(f"\n{'='*100}")
    print("NOTES")
    print(f"{'='*100}")
    if 'notes' in expected_data:
        for note in expected_data['notes']:
            print(f"  - {note}")
    
    print(f"\n{'='*100}")

if __name__ == "__main__":
    main()
